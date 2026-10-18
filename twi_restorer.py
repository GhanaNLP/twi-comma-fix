"""
Twi Punctuation and Capitalization Restorer
Fast CPU inference tool for Akan/Twi text.
Restores capitalization (Title Case, ALL CAPS) and punctuation (commas, periods, question marks).
Supports ONNX INT8 runtime (~34ms on CPU) and standard PyTorch.
"""

import sys
import os
import re
import argparse
import time
import numpy as np


class TwiRestorer:
    def __init__(self, model_id="ghanaopenai/twi-punct-cap-restoration", backend="auto", device="cpu"):
        """
        backend: "auto", "onnx", or "torch"
        device: "cpu" or "cuda"
        """
        self.model_id = model_id
        self.device = device
        self.rev_punct = {0: "", 1: ",", 2: ".", 3: "?", 4: "!"}

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Decide backend
        use_onnx = False
        if backend in ["auto", "onnx"]:
            try:
                import onnxruntime as ort
                use_onnx = True
            except ImportError:
                if backend == "onnx":
                    raise ImportError("onnxruntime is required for backend='onnx'. Install with: pip install onnxruntime")
                use_onnx = False

        if use_onnx:
            self.backend = "onnx"
            print(f"Loading quantized ONNX model from {model_id} for ultra-fast CPU inference...")
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort
            onnx_path = hf_hub_download(repo_id=model_id, filename="model_quantized.onnx")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = os.cpu_count() or 4
            self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            print("ONNX INT8 session initialized!")
        else:
            self.backend = "torch"
            print(f"Loading PyTorch model from {model_id} on {device}...")
            import torch
            import torch.nn as nn
            from transformers import AutoConfig, AutoModel
            from huggingface_hub import hf_hub_download

            class TwiPunctCapModel(nn.Module):
                def __init__(self, config, base_model_name="Ghana-NLP/distilabena-base-v2-akuapem-twi-cased"):
                    super().__init__()
                    self.distilbert = AutoModel.from_pretrained(base_model_name, config=config)
                    hidden_size = config.hidden_size
                    self.dropout = nn.Dropout(0.2)
                    self.cap_head = nn.Linear(hidden_size, 3)
                    self.punct_head = nn.Linear(hidden_size, 5)

                def forward(self, input_ids, attention_mask):
                    outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                    seq_out = self.dropout(outputs[0])
                    return self.cap_head(seq_out), self.punct_head(seq_out)

            config = AutoConfig.from_pretrained(model_id)
            weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
            self.model = TwiPunctCapModel(config)
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
            self.torch = torch
            print("PyTorch model initialized!")

    def restore(self, text: str) -> str:
        """
        Takes raw or unpunctuated Twi text, strips any existing punctuation,
        and restores proper capitalization and punctuation (commas and periods).
        """
        raw_tokens = text.split()
        if not raw_tokens:
            return ""

        # Strip all leading and trailing punctuation from words
        clean_words = []
        for w in raw_tokens:
            # Preserve internal characters and hyphens, strip outer punctuation
            cw = re.sub(r'^[^\w\u0190\u0186\u025B\u0254]+|[^\w\u0190\u0186\u025B\u0254]+$', '', w)
            clean_words.append(cw.lower() if cw else w.lower())

        encoding = self.tokenizer(
            clean_words,
            is_split_into_words=True,
            return_tensors="np" if self.backend == "onnx" else "pt"
        )
        word_ids = encoding.word_ids(batch_index=0)

        if self.backend == "onnx":
            ort_inputs = {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            }
            ort_outputs = self.session.run(None, ort_inputs)
            cap_preds = np.argmax(ort_outputs[0][0], axis=-1)
            punct_preds = np.argmax(ort_outputs[1][0], axis=-1)
        else:
            with self.torch.no_grad():
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                cap_logits, punct_logits = self.model(input_ids, attention_mask)
            cap_preds = self.torch.argmax(cap_logits[0], dim=-1).cpu().numpy()
            punct_preds = self.torch.argmax(punct_logits[0], dim=-1).cpu().numpy()

        # Map predictions to original words (first subtoken of each word)
        current_word = None
        word_caps, word_puncts = [], []
        for token_idx, w_id in enumerate(word_ids):
            if w_id is not None and w_id != current_word:
                word_caps.append(cap_preds[token_idx])
                word_puncts.append(punct_preds[token_idx])
                current_word = w_id

        # Reconstruct sentence with capitalization and punctuation
        reconstructed = []
        for w, c, p in zip(clean_words, word_caps, word_puncts):
            if c == 2:
                word_str = w.upper()
            elif c == 1:
                # Handle hyphenated names like Opoku-Agyemang
                word_str = "-".join(part.capitalize() for part in w.split("-"))
            else:
                word_str = w.lower()

            word_str += self.rev_punct.get(p, "")
            reconstructed.append(word_str)

        return " ".join(reconstructed)


def main():
    parser = argparse.ArgumentParser(description="Twi Punctuation & Capitalization Restorer")
    parser.add_argument("text", nargs="?", help="Input Twi text (uncapitalized or unpunctuated)")
    parser.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"], help="Inference backend (default: auto -> onnx if available)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    restorer = TwiRestorer(backend=args.backend, device=args.device)

    if args.text:
        t0 = time.time()
        result = restorer.restore(args.text)
        lat = (time.time() - t0) * 1000
        print("\n--- Output ---")
        print(result)
        print(f"\nLatency: {lat:.1f}ms ({restorer.backend})")
    else:
        print("\nInteractive Twi Restorer (Type text and press Enter, Ctrl+C to exit):")
        try:
            while True:
                user_in = input("\n>> ")
                if not user_in.strip():
                    continue
                t0 = time.time()
                out = restorer.restore(user_in)
                lat = (time.time() - t0) * 1000
                print(f"Output: {out}")
                print(f"({lat:.1f}ms)")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")


if __name__ == "__main__":
    main()
