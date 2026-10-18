import os
import io
import time
import modal
from huggingface_hub import get_token

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "datasets",
        "pyarrow",
        "pandas",
        "scikit-learn",
        "huggingface_hub",
        "safetensors",
        "onnx",
        "onnxruntime",
    )
)

app = modal.App("train-production-twi-restorer", image=image)

hf_token = get_token() or ""
hf_secret = modal.Secret.from_dict({"HF_TOKEN": hf_token})

@app.function(gpu="L4", timeout=2400, secrets=[hf_secret])
def train_and_export():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoConfig, AutoModel, get_cosine_schedule_with_warmup
    from datasets import load_dataset
    from sklearn.metrics import classification_report, f1_score
    from safetensors.torch import save_file
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    base_model_name = "Ghana-NLP/distilabena-base-v2-akuapem-twi-cased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    config = AutoConfig.from_pretrained(base_model_name)

    # 1. Load dataset
    print("Loading dataset ghanaopenai/twi-grapheme-comma-fixed...")
    ds = load_dataset("ghanaopenai/twi-grapheme-comma-fixed", split="train")
    print(f"Loaded {len(ds)} examples.")

    def parse_sentence(text):
        raw_tokens = text.split()
        words = []
        c_tags = []
        p_tags = []
        for tok in raw_tokens:
            p = 0
            clean = tok
            if clean.endswith(','):
                p = 1
                clean = clean[:-1]
            elif clean.endswith('.'):
                p = 2
                clean = clean[:-1]
            elif clean.endswith('?'):
                p = 3
                clean = clean[:-1]
            elif clean.endswith('!'):
                p = 4
                clean = clean[:-1]

            if clean.isupper() and len(clean) > 1:
                c = 2
            elif clean and (clean[0].isupper() or any(part and part[0].isupper() for part in clean.split('-'))):
                c = 1
            else:
                c = 0

            words.append(clean.lower())
            c_tags.append(c)
            p_tags.append(p)
        return words, c_tags, p_tags

    parsed_data = []
    for row in ds:
        text = row["comma_fixed_text"]
        words, c_tags, p_tags = parse_sentence(text)
        if words:
            parsed_data.append((words, c_tags, p_tags))

    # Split train / val (90% / 10%)
    np.random.seed(42)
    indices = np.random.permutation(len(parsed_data))
    split = int(0.9 * len(parsed_data))
    train_data = [parsed_data[i] for i in indices[:split]]
    val_data = [parsed_data[i] for i in indices[split:]]
    print(f"Train sentences: {len(train_data)}, Validation sentences: {len(val_data)}")

    # PyTorch Dataset
    class PunctCapDataset(Dataset):
        def __init__(self, data, max_len=128):
            self.data = data
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            words, c_tags, p_tags = self.data[idx]
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            word_ids = encoding.word_ids(batch_index=0)
            cap_labels = []
            punct_labels = []
            current_word = None

            for w_id in word_ids:
                if w_id is None:
                    cap_labels.append(-100)
                    punct_labels.append(-100)
                elif w_id != current_word:
                    if w_id < len(c_tags):
                        cap_labels.append(c_tags[w_id])
                        punct_labels.append(p_tags[w_id])
                    else:
                        cap_labels.append(-100)
                        punct_labels.append(-100)
                    current_word = w_id
                else:
                    cap_labels.append(-100)
                    punct_labels.append(-100)

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "cap_labels": torch.tensor(cap_labels, dtype=torch.long),
                "punct_labels": torch.tensor(punct_labels, dtype=torch.long),
            }

    train_ds = PunctCapDataset(train_data)
    val_ds = PunctCapDataset(val_data)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Multi-task architecture
    class TwiPunctCapModel(nn.Module):
        def __init__(self, config, base_model_name):
            super().__init__()
            self.config = config
            self.distilbert = AutoModel.from_pretrained(base_model_name, config=config)
            hidden_size = config.hidden_size # 768
            self.dropout = nn.Dropout(0.2)
            self.cap_head = nn.Linear(hidden_size, 3)
            self.punct_head = nn.Linear(hidden_size, 5)

        def forward(self, input_ids, attention_mask, cap_labels=None, punct_labels=None):
            outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            seq_out = self.dropout(outputs[0])
            cap_logits = self.cap_head(seq_out)
            punct_logits = self.punct_head(seq_out)

            loss = None
            if cap_labels is not None and punct_labels is not None:
                # Class weights: upweight commas (class 1) so pauses are accurately caught
                cap_weights = torch.tensor([1.0, 1.5, 2.0], device=input_ids.device)
                punct_weights = torch.tensor([0.6, 2.2, 1.8, 2.5, 2.5], device=input_ids.device)

                loss_cap_fn = nn.CrossEntropyLoss(weight=cap_weights, ignore_index=-100)
                loss_punct_fn = nn.CrossEntropyLoss(weight=punct_weights, ignore_index=-100)

                loss_cap = loss_cap_fn(cap_logits.view(-1, 3), cap_labels.view(-1))
                loss_punct = loss_punct_fn(punct_logits.view(-1, 5), punct_labels.view(-1))
                loss = loss_cap + loss_punct

            return {
                "loss": loss,
                "cap_logits": cap_logits,
                "punct_logits": punct_logits,
            }

    model = TwiPunctCapModel(config, base_model_name).to(device)

    epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    print(f"\n--- Training for {epochs} Epochs ---")
    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cap_labels = batch["cap_labels"].to(device)
            punct_labels = batch["punct_labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, cap_labels, punct_labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        epoch_dur = time.time() - t0

        # Evaluate on validation split
        model.eval()
        all_cap_preds, all_cap_trues = [], []
        all_punct_preds, all_punct_trues = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                cap_labels = batch["cap_labels"].to(device)
                punct_labels = batch["punct_labels"].to(device)

                outputs = model(input_ids, attention_mask)
                cap_pred = torch.argmax(outputs["cap_logits"], dim=-1)
                punct_pred = torch.argmax(outputs["punct_logits"], dim=-1)

                mask = cap_labels != -100
                all_cap_preds.extend(cap_pred[mask].cpu().numpy())
                all_cap_trues.extend(cap_labels[mask].cpu().numpy())
                all_punct_preds.extend(punct_pred[mask].cpu().numpy())
                all_punct_trues.extend(punct_labels[mask].cpu().numpy())

        cap_f1 = f1_score(all_cap_trues, all_cap_preds, average="macro")
        punct_f1 = f1_score(all_punct_trues, all_punct_preds, average="macro")

        # Comma specific precision / recall / F1
        c_p = sum(1 for p, t in zip(all_punct_preds, all_punct_trues) if p == 1 and t == 1) / max(1, sum(1 for p in all_punct_preds if p == 1))
        c_r = sum(1 for p, t in zip(all_punct_preds, all_punct_trues) if p == 1 and t == 1) / max(1, sum(1 for t in all_punct_trues if t == 1))
        c_f1 = 2 * c_p * c_r / max(1e-6, c_p + c_r)

        # Period specific
        dot_p = sum(1 for p, t in zip(all_punct_preds, all_punct_trues) if p == 2 and t == 2) / max(1, sum(1 for p in all_punct_preds if p == 2))
        dot_r = sum(1 for p, t in zip(all_punct_preds, all_punct_trues) if p == 2 and t == 2) / max(1, sum(1 for t in all_punct_trues if t == 2))

        print(f"Epoch {epoch+1}/{epochs} ({epoch_dur:.1f}s) | Loss: {train_loss:.4f} | Cap F1: {cap_f1:.4f} | Comma Prec: {c_p:.3f}, Rec: {c_r:.3f}, F1: {c_f1:.3f} | Period Prec: {dot_p:.3f}, Rec: {dot_r:.3f}")

    # 3. Save PyTorch & Safetensors weights
    save_dir = "/tmp/twi-punct-cap-production"
    os.makedirs(save_dir, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
    save_file(state_dict, os.path.join(save_dir, "model.safetensors"))
    config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"PyTorch and Safetensors models saved to {save_dir}")

    # 4. ONNX Export and INT8 Quantization
    print("\n--- Exporting to ONNX ---")
    model.eval().to("cpu")

    class OnnxWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids, attention_mask)
            return out["cap_logits"], out["punct_logits"]

    onnx_wrapper = OnnxWrapper(model)
    dummy_input = tokenizer("ahene nanahene aka se ese se wobebo dawuro", return_tensors="pt")
    onnx_path = os.path.join(save_dir, "model.onnx")

    torch.onnx.export(
        onnx_wrapper,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["cap_logits", "punct_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "cap_logits": {0: "batch_size", 1: "sequence_length"},
            "punct_logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14
    )
    print(f"ONNX exported: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    # INT8 Quantization using onnxruntime
    print("\n--- Quantizing ONNX to INT8 for ultra-fast CPU inference ---")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quant_path = os.path.join(save_dir, "model_quantized.onnx")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8
    )
    print(f"INT8 Quantized ONNX saved: {quant_path} ({os.path.getsize(quant_path)/1e6:.1f} MB)")

    # 5. Benchmark ONNX INT8 vs PyTorch on CPU
    print("\n--- Benchmarking CPU Latency ---")
    import onnxruntime as ort
    ort_session = ort.InferenceSession(quant_path, providers=["CPUExecutionProvider"])

    test_sentence = "ahene nanahene nana kwame mensah bonsu aka se ese se wobebo dawuro ama chief of staff julius debrah se onkasa nkyere president john dramani mahama ne vice president professor naana jane opoku-agyemang se wonntena nkyere dormaahene osagyefo oseadeeyo agyeman badu ii wo ne nneyee ne n’akasa a ereka tia asanteman no ho"
    test_words = test_sentence.split()
    inputs = tokenizer(test_words, is_split_into_words=True, return_tensors="pt")

    # PyTorch CPU timing
    t0 = time.time()
    for _ in range(20):
        with torch.no_grad():
            _ = model(inputs["input_ids"], inputs["attention_mask"])
    torch_lat = (time.time() - t0) / 20 * 1000

    # ONNX INT8 timing
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    t0 = time.time()
    for _ in range(20):
        _ = ort_session.run(None, ort_inputs)
    onnx_lat = (time.time() - t0) / 20 * 1000

    print(f"PyTorch CPU Latency:    {torch_lat:.1f} ms")
    print(f"ONNX INT8 CPU Latency:  {onnx_lat:.1f} ms  ({torch_lat/onnx_lat:.1f}x speedup!)")

    # 6. Verification Test Cases with INT8 Model
    print("\n--- Verifying Predictions with ONNX INT8 ---")
    ort_outputs = ort_session.run(None, ort_inputs)
    cap_preds = np.argmax(ort_outputs[0][0], axis=-1)
    punct_preds = np.argmax(ort_outputs[1][0], axis=-1)

    word_ids = inputs.word_ids(batch_index=0)
    current_word = None
    word_caps, word_puncts = [], []
    for token_idx, w_id in enumerate(word_ids):
        if w_id is not None and w_id != current_word:
            word_caps.append(cap_preds[token_idx])
            word_puncts.append(punct_preds[token_idx])
            current_word = w_id

    REV_PUNCT = {0: "", 1: ",", 2: ".", 3: "?", 4: "!"}
    reconstructed = []
    for w, c, p in zip(test_words, word_caps, word_puncts):
        if c == 2:
            word_str = w.upper()
        elif c == 1:
            word_str = "-".join(part.capitalize() for part in w.split("-"))
        else:
            word_str = w.lower()
        word_str += REV_PUNCT.get(p, "")
        reconstructed.append(word_str)

    print("INPUT:\n", test_sentence)
    print("RESTORED:\n", " ".join(reconstructed))

    # 7. Upload to Hugging Face
    repo_id = "ghanaopenai/twi-punct-cap-restoration"
    print(f"\nPushing all models and files to Hugging Face: {repo_id}...")
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add fine-tuned Twi capitalization, punctuation, safetensors, and INT8 ONNX models"
    )
    print(f"Successfully uploaded to https://huggingface.co/{repo_id}!")

    # Update README
    readme_content = f"""---
language:
- tw
- en
license: apache-2.0
tags:
- twi
- akan
- punctuation-restoration
- capitalization
- truecasing
- asr-post-processing
- distilbert
- onnx
- ghana-nlp
widget:
- text: "ahene nanahene nana kwame mensah bonsu aka sɛ ɛsɛ sɛ wɔbɛbɔ dawuro ama chief of staff julius debrah sɛ ɔnkasa nkyerɛ president john dramani mahama ne vice president professor naana jane opoku-agyemang"
- text: "me nim sɛ ebi te sɛ nea ɛyɛ asɛm ketewa nanso sɛ wobɛhwɛ danso debrah beduako a ɔyɛ action people’s party app national communications director no anohyiamu"
- text: "bere a liverpool nnamfoo virgil van dijk ne andy robertson kɔɔ asieeɛ no de nhwiren kɔhyɛɛ rute ne ne mma nkuran no ɛkaa nnipa pii koma sɛ cristiano ronaldo captain no annya akwanya anaa ɔpɛ sɛ ɔba hɔ"
---

# Twi Punctuation and Capitalization Restoration Model

A fast, lightweight CPU model specifically fine-tuned for Akan/Twi to simultaneously restore **capitalization** (lowercase, Title Case, ALL CAPS) and **punctuation** (commas, sentence-ending periods, question marks).

## Highlights

- **Backbone**: `Ghana-NLP/distilabena-base-v2-akuapem-twi-cased` (Ghana NLP Twi DistilBERT, 6 layers, 768 hidden).
- **Multi-Task Architecture**: Dual classification heads for simultaneous token-level Capitalization (3 classes) and Punctuation (5 classes).
- **Fast CPU Inference**:
  - **PyTorch CPU**: ~{torch_lat:.1f} ms
  - **ONNX INT8 Quantized**: **~{onnx_lat:.1f} ms** per sentence (runs on any standard CPU).
- **Zero Hallucination Guarantee**: Non-autoregressive token classification preserves all input words in their original order.

## Model Formats Available

- `model.safetensors` (539 MB)
- `pytorch_model.bin` (539 MB)
- `model.onnx` (537 MB)
- `model_quantized.onnx` (**135 MB**, INT8 quantized for ultra-fast CPU inference)
- Complete tokenizer files (`vocab.txt`, `tokenizer.json`, `tokenizer_config.json`)

## Performance Metrics

- **Capitalization Macro F1**: {cap_f1:.4f}
- **Comma Precision**: {c_p:.3f} | **Comma Recall**: {c_r:.3f} | **Comma F1**: {c_f1:.3f}
- **Period Precision**: {dot_p:.3f} | **Period Recall**: {dot_r:.3f}

## Quick Usage

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

class TwiPunctCapModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.distilbert = AutoModel.from_pretrained("Ghana-NLP/distilabena-base-v2-akuapem-twi-cased", config=config)
        self.cap_head = nn.Linear(config.hidden_size, 3)
        self.punct_head = nn.Linear(config.hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return self.cap_head(out), self.punct_head(out)

# Load
model_id = "ghanaopenai/twi-punct-cap-restoration"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")

model = TwiPunctCapModel(config)
model.load_state_dict(load_file(weights_path))
model.eval()
```

## Ultra-Fast CPU Inference (ONNX INT8)

```python
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

tokenizer = AutoTokenizer.from_pretrained("ghanaopenai/twi-punct-cap-restoration")
model_path = hf_hub_download("ghanaopenai/twi-punct-cap-restoration", "model_quantized.onnx")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
```
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card with ONNX benchmarks and metrics"
    )
    print("Model card updated successfully!")

@app.local_entrypoint()
def main():
    train_and_export.remote()
