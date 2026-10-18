import os
import io
import time
import modal
from huggingface_hub import get_token

# Pre-download MMS_FA model into image so containers boot with zero model download time
def download_mms():
    from torchaudio.pipelines import MMS_FA as bundle
    print("Pre-downloading MMS_FA model into image layer...")
    bundle.get_model()
    print("MMS_FA model cached successfully!")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "soundfile",
        "datasets",
        "pyarrow",
        "pandas",
        "huggingface_hub",
    )
    .run_function(download_mms)
)

app = modal.App("twi-comma-fix-pipeline", image=image)

# Pass user HF_TOKEN for pushing dataset to Hugging Face
hf_token = get_token() or ""
hf_secret = modal.Secret.from_dict({"HF_TOKEN": hf_token})

@app.cls(gpu="L4", timeout=1200, secrets=[hf_secret])
class ShardProcessor:
    @modal.enter()
    def setup(self):
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA as bundle

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = bundle
        self.model = bundle.get_model().to(self.device)
        self.model.eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.dictionary = bundle.get_dict()
        self.valid_chars = set(c for c in self.dictionary if c != '-')
        print(f"ShardProcessor initialized on {self.device}")

    def align_single(self, text: str, norm_text: str, audio_bytes: bytes, threshold: float = 0.35):
        import soundfile as sf
        import torch
        import torchaudio

        try:
            audio_arr, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            return text, norm_text, 0, 0.0

        dur = len(audio_arr) / sr
        waveform = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0)
        if sr != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.bundle.sample_rate)

        raw_words = norm_text.split()
        orig_words = text.split()

        cleaned_words = []
        valid_indices = []
        for i, w in enumerate(raw_words):
            cw = "".join([c for c in w.lower() if c in self.valid_chars])
            if cw:
                cleaned_words.append(cw)
                valid_indices.append(i)

        if len(cleaned_words) < 2:
            return text, norm_text, 0, dur

        tokens = self.tokenizer(cleaned_words)

        with torch.no_grad():
            emission, _ = self.model(waveform.to(self.device))
        emission_cpu = emission[0].cpu()

        try:
            token_spans = self.aligner(emission_cpu, tokens)
        except Exception:
            return text, norm_text, 0, dur

        num_frames = emission.shape[1]
        duration = waveform.shape[1] / self.bundle.sample_rate
        frame_to_sec = duration / num_frames

        word_spans = []
        for orig_idx, spans in zip(valid_indices, token_spans):
            if spans:
                start_sec = spans[0].start * frame_to_sec
                end_sec = spans[-1].end * frame_to_sec
                word_spans.append({
                    "idx": orig_idx,
                    "start": start_sec,
                    "end": end_sec,
                })

        result_words = list(orig_words)
        result_norm_words = list(raw_words)
        added_count = 0
        punct_chars = (",", ".", "!", "?", ";", ":", "—", "-")

        for i in range(len(word_spans) - 1):
            w1 = word_spans[i]
            w2 = word_spans[i + 1]
            gap = w2["start"] - w1["end"]
            if gap >= threshold:
                target_idx = w2["idx"] - 1
                if target_idx < len(result_words):
                    cur_w = result_words[target_idx]
                    if not cur_w.endswith(punct_chars):
                        result_words[target_idx] = cur_w + ","
                        added_count += 1
                if target_idx < len(result_norm_words):
                    cur_nw = result_norm_words[target_idx]
                    if not cur_nw.endswith(punct_chars):
                        result_norm_words[target_idx] = cur_nw + ","

        return " ".join(result_words), " ".join(result_norm_words), added_count, round(dur, 2)

    @modal.method()
    def process_shard(self, shard_info: dict) -> list[dict]:
        from datasets import load_dataset, Audio

        shard_idx = shard_info["shard_idx"]
        shard_path = shard_info["shard_path"]
        print(f"[{shard_idx}] Loading shard: {shard_path}...")
        
        shard_url = f"https://huggingface.co/datasets/ghanaopenai/twi-zephyr-synthetic-speech/resolve/main/{shard_path}"
        ds = load_dataset("parquet", data_files=shard_url, split="train")
        ds = ds.cast_column("audio", Audio(decode=False))

        print(f"[{shard_idx}] Loaded {len(ds)} rows. Aligning...")
        t0 = time.time()
        processed_rows = []

        for row_i, row in enumerate(ds):
            text = row["text"]
            norm_text = row["normalised_text"]
            audio_bytes = row["audio"]["bytes"]

            fixed_text, fixed_norm, added, dur = self.align_single(text, norm_text, audio_bytes, threshold=0.35)

            processed_rows.append({
                "shard": shard_path,
                "row_in_shard": row_i,
                "text": text,
                "comma_fixed_text": fixed_text,
                "normalised_text": norm_text,
                "normalised_comma_fixed_text": fixed_norm,
                "commas_added": added,
                "audio_duration_sec": dur,
            })

            if (row_i + 1) % 100 == 0 or (row_i + 1) == len(ds):
                rate = (row_i + 1) / (time.time() - t0)
                print(f"[{shard_idx}] Progress: {row_i + 1}/{len(ds)} rows ({rate:.1f} rows/s)")

        total_time = time.time() - t0
        print(f"[{shard_idx}] Done! Processed {len(processed_rows)} rows in {total_time:.1f}s ({len(processed_rows)/total_time:.1f} rows/s).")
        return processed_rows

@app.function(timeout=1800, secrets=[hf_secret])
def run_and_push():
    from huggingface_hub import HfApi
    import pandas as pd
    from datasets import Dataset

    api = HfApi()
    repo_files = api.list_repo_files("ghanaopenai/twi-zephyr-synthetic-speech", repo_type="dataset")
    parquet_shards = sorted([f for f in repo_files if f.startswith("data/train-") and f.endswith(".parquet")])
    print(f"Found {len(parquet_shards)} parquet shards to process:")
    for s in parquet_shards:
        print(f"  {s}")

    shards_to_process = [
        {"shard_idx": i, "shard_path": path} for i, path in enumerate(parquet_shards)
    ]

    processor = ShardProcessor()

    print(f"\nLaunching {len(shards_to_process)} parallel GPU shard workers on Modal...")
    t_start = time.time()
    all_shard_results = list(processor.process_shard.map(shards_to_process))
    elapsed = time.time() - t_start
    print(f"\nAll {len(parquet_shards)} shards completed in {elapsed:.1f}s!")

    # Flatten results in exact shard order
    all_rows = []
    global_idx = 0
    total_commas_added = 0
    for shard_res in all_shard_results:
        for r in shard_res:
            r["index"] = global_idx
            global_idx += 1
            total_commas_added += r["commas_added"]
            all_rows.append(r)

    print(f"Total rows gathered: {len(all_rows)}")
    print(f"Total commas added across dataset: {total_commas_added}")

    # Inspect first 3 samples
    print("\n--- Verification: First 3 Samples ---")
    for r in all_rows[:3]:
        print(f"\n[Index {r['index']}]:")
        print(f"ORIGINAL:    {r['text']}")
        print(f"COMMA-FIXED: {r['comma_fixed_text']}")
        print(f"NORM-FIXED:  {r['normalised_comma_fixed_text']}")
        print(f"Commas added: {r['commas_added']}, Duration: {r['audio_duration_sec']}s")

    # Build DataFrame and Dataset
    df = pd.DataFrame(all_rows)
    # Order columns cleanly
    df = df[[
        "index",
        "text",
        "comma_fixed_text",
        "normalised_text",
        "normalised_comma_fixed_text",
        "commas_added",
        "audio_duration_sec",
        "shard",
        "row_in_shard"
    ]]

    dataset = Dataset.from_pandas(df)
    print(f"\nCreated Dataset with features: {dataset.features}")

    # Push to Hugging Face
    target_repo = "ghanaopenai/twi-grapheme-comma-fixed"
    print(f"\nPushing dataset to Hugging Face: {target_repo}...")
    dataset.push_to_hub(
        repo_id=target_repo,
        split="train",
        commit_message="Add comma-fixed parallel text aligned with MMS forced aligner"
    )
    print(f"Successfully pushed dataset to {target_repo}!")

    # Upload README.md dataset card
    readme_content = f"""---
language:
- tw
- en
license: cc-by-nc-4.0
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- twi
- parallel-corpus
- punctuation
- comma-restoration
- alignment
- africa-g2p
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/train-*.parquet"
size_categories:
- 10K<n<100K
---

# Twi Grapheme Comma-Fixed Parallel Dataset

This dataset provides parallel texts for Twi speech where natural pauses in synthesized speech have been acoustically aligned and inserted as commas.

## Background

The source speech was synthesized with **Gemini 3.1 Flash TTS** (`gemini-3.1-flash-tts-preview`, voice *Zephyr*) from [ghanaopenai/twi-zephyr-synthetic-speech](https://huggingface.co/datasets/ghanaopenai/twi-zephyr-synthetic-speech). While the synthesized speech features natural phrase, clause, and list pauses, many of these pauses were not represented as commas in the original transcript text.

Using **Meta MMS Forced Aligner** (`torchaudio.pipelines.MMS_FA`), speech waveforms were aligned at the word level with universal grapheme representations. Acoustic pauses (silent gaps $\ge 350$ms) between words were identified and resolved into punctuation commas.

## Columns

- `index` (*int64*): Global index of the sentence (0 to {len(all_rows) - 1}).
- `text` (*string*): The original Twi text.
- `comma_fixed_text` (*string*): The original Twi text with commas inserted at acoustic pause locations.
- `normalised_text` (*string*): Africa-G2P universal grapheme conversion of `text`.
- `normalised_comma_fixed_text` (*string*): Africa-G2P universal grapheme text with commas inserted at identical pause positions.
- `commas_added` (*int64*): Number of commas newly inserted for this sentence.
- `audio_duration_sec` (*float64*): Duration of the synthesized speech in seconds.
- `shard` (*string*): Original parquet shard path.
- `row_in_shard` (*int64*): Row index within the original shard.

## Statistics

- Total sentences: **{len(all_rows):,}**
- Total new commas added: **{total_commas_added:,}**
- Alignment model: `Meta MMS Forced Aligner (ctc_alignment_mling_uroman)`
- Pause threshold: **350 ms**
"""

    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=target_repo,
        repo_type="dataset",
        commit_message="Add dataset card"
    )
    print(f"Dataset card uploaded to {target_repo}!")

@app.local_entrypoint()
def main():
    run_and_push.remote()
