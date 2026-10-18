# Twi Punctuation & Capitalization Restoration

End-to-end pipeline for **acoustic speech-pause alignment** and **fast CPU-based punctuation & capitalization restoration** for Akan/Twi.

Developed by **Ghana NLP** & **Ghana OpenAI**.

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/ghananlpcommunity/twi-punct-cap-restorer)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-twi--punct--cap--restoration-yellow)](https://huggingface.co/ghanaopenai/twi-punct-cap-restoration)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-twi--grapheme--comma--fixed-green)](https://huggingface.co/datasets/ghanaopenai/twi-grapheme-comma-fixed)

---

## Overview

When processing Akan/Twi audio (e.g. from Automatic Speech Recognition or Text-to-Speech datasets like [ghanaopenai/twi-zephyr-synthetic-speech](https://huggingface.co/datasets/ghanaopenai/twi-zephyr-synthetic-speech)), speech audio contains natural pauses (e.g. at clause boundaries, list items, and parentheticals) that are frequently missing from transcript texts.

This repository provides:
1. **Acoustic Pause Alignment Pipeline (`data/run_audio_pause_alignment.py`)**: Uses Meta's Massively Multilingual Speech (MMS) Forced Aligner on Modal to align synthesized speech audio with text at the word level, detecting pauses ($\ge 350$ms) and inserting commas into parallel text columns.
2. **Multi-Task Restoration Model (`training/train_model.py`)**: Fine-tunes `Ghana-NLP/distilabena-base-v2-akuapem-twi-cased` with dual heads for simultaneous **Capitalization** (3 classes) and **Punctuation** (5 classes: commas, periods, question marks, exclamations).
3. **Ultra-Fast CPU Inference (`twi_restorer.py`)**: Powered by an INT8 quantized ONNX engine running in **~11–28 ms** per sentence on any standard CPU without needing PyTorch.
4. **Interactive Web Interface (`app.py`)**: Gradio application deployed to [Hugging Face Spaces](https://huggingface.co/spaces/ghananlpcommunity/twi-punct-cap-restorer).

---

## Project Structure

```
├── twi_restorer.py                  # Standalone CPU inference library and CLI tool
├── app.py                           # Gradio interactive web interface
├── requirements.txt                 # Project dependencies
├── data/
│   └── run_audio_pause_alignment.py # Modal pipeline for parallel MMS forced alignment on 10,562 audio samples
├── training/
│   └── train_model.py               # Modal pipeline for fine-tuning DistilAbena-v2, ONNX export & INT8 quantization
└── scripts/
    ├── test_alignment.py            # Verification of MMS alignment on GPU
    └── verify_inference.py          # Benchmark of ONNX INT8 vs PyTorch latency on CPU
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/GhanaNLP/twi-comma-fix.git
cd twi-comma-fix
pip install -r requirements.txt
```

### 2. Python Usage

```python
from twi_restorer import TwiRestorer

# Automatically uses ultra-fast quantized ONNX if onnxruntime is available
restorer = TwiRestorer(backend="auto")

raw_text = "ahene nanahene nana kwame mensah bonsu aka se ese se wobebo dawuro ama chief of staff julius debrah se onkasa nkyere president john dramani mahama ne vice president professor naana jane opoku-agyemang se wonntena nkyere dormaahene osagyefo oseadeeyo agyeman badu ii wo ne nneyee ne n’akasa a ereka tia asanteman no ho"

restored = restorer.restore(raw_text)
print(restored)
```

**Output:**
> *Ahene Nanahene Nana Kwame Mensah Bonsu, aka sɛ, ɛsɛ sɛ wɔbɛbɔ dawuro ama Chief of Staff, Julius Debrah, sɛ ɔnkasa nkyerɛ President John Dramani Mahama, ne Vice President Professor Naana Jane Opoku-Agyemang, sɛ, wɔnntena nkyerɛ Dormaahene, Osagyefo Oseadeeyo Agyeman Badu Ii, wɔ ne nneyɛɛ, ne n’akasa a ɛreka tia Asanteman no ho.*

### 3. CLI Usage

```bash
# Run on a single sentence
python3 twi_restorer.py "wo ofa foforo nso mu ministry of lands and natural resources ananmusini paa kwesi schandorf yii n'ano kaa se"

# Interactive REPL mode
python3 twi_restorer.py
```

### 4. Interactive Web App (Gradio)

```bash
python3 app.py
```

---

## Pipelines

### Data Preparation: Acoustic Pause Alignment (`data/`)

The script `data/run_audio_pause_alignment.py` processes the 10,562 audio samples from `ghanaopenai/twi-zephyr-synthetic-speech` in parallel using Modal GPU workers:

```bash
modal run data/run_audio_pause_alignment.py
```

1. Shards the 19 Parquet files across 19 concurrent Modal L4 GPU workers.
2. Runs Meta MMS CTC alignment (`torchaudio.pipelines.MMS_FA`) to obtain exact millisecond word timestamps.
3. Detects silent intervals ($\ge 350$ms) between adjacent words.
4. Inserts 7,996 commas into parallel columns: `text`, `comma_fixed_text`, `normalised_text`, `normalised_comma_fixed_text`.
5. Pushes directly to [ghanaopenai/twi-grapheme-comma-fixed](https://huggingface.co/datasets/ghanaopenai/twi-grapheme-comma-fixed).

### Model Training (`training/`)

The script `training/train_model.py` fine-tunes `Ghana-NLP/distilabena-base-v2-akuapem-twi-cased` with multi-task dual heads on Modal:

```bash
modal run training/train_model.py
```

1. **Pre-processing**: Strips all punctuation and casing from the input text so the model learns to restore punctuation from raw text (e.g. ASR transcripts).
2. **Subword Alignment**: Subword tokens are mapped back to whole words; loss is computed on the first subtoken of each word.
3. **Class Weighting**: Upweights comma and period classes to balance the dominant non-punct class (`O`).
4. **ONNX & INT8 Quantization**: Exports the trained PyTorch model to ONNX (`model.onnx`) and performs dynamic INT8 quantization (`model_quantized.onnx`), cutting the model size from **539 MB** to **135 MB**.
5. **Publishing**: Pushes weights and model card to [ghanaopenai/twi-punct-cap-restoration](https://huggingface.co/ghanaopenai/twi-punct-cap-restoration).

---

## Benchmarks & Performance

| Metric | PyTorch (FP32) | ONNX (INT8 Quantized) |
|---|---|---|
| **Model Size** | 539 MB | **135 MB** (4x smaller) |
| **CPU Latency (2-core)** | ~117 ms | **~11–28 ms** (3.5x–10x faster) |
| **Capitalization Macro F1** | 0.8725 | 0.8725 |
| **Comma Recall** | 80.6% | 80.6% |
| **Period Precision** | 92.4% | 92.4% |

---

## Citation & Credits

* Base Model: [Ghana-NLP/distilabena-base-v2-akuapem-twi-cased](https://huggingface.co/Ghana-NLP/distilabena-base-v2-akuapem-twi-cased)
* Alignment Model: [Meta MMS Forced Aligner](https://pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html)
* Organization: [Ghana NLP](https://ghananlp.org) & [Ghana OpenAI](https://huggingface.co/ghanaopenai)
