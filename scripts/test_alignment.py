import modal
import time

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "soundfile",
        "datasets",
    )
)

app = modal.App("test-mms-gpu", image=image)

@app.cls(gpu="L4", timeout=300)
class AlignerGPU:
    @modal.enter()
    def setup(self):
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA as bundle
        print("Loading MMS_FA model on GPU...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = bundle
        self.model = bundle.get_model().to(self.device)
        self.model.eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.dictionary = bundle.get_dict()
        print(f"Model ready on {self.device}!")

    @modal.method()
    def align_and_punctuate(self, text: str, norm_text: str, audio_bytes: bytes, threshold: float = 0.35):
        import io
        import torch
        import torchaudio
        import soundfile as sf

        t0 = time.time()
        audio_arr, sr = sf.read(io.BytesIO(audio_bytes))
        waveform = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0)
        if sr != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.bundle.sample_rate)

        raw_words = norm_text.split()
        orig_words = text.split()

        cleaned_words = []
        valid_indices = []
        for i, w in enumerate(raw_words):
            cw = "".join([c for c in w.lower() if c in self.dictionary and c != '-'])
            if cw:
                cleaned_words.append(cw)
                valid_indices.append(i)

        if not cleaned_words:
            return text, 0.0

        tokens = self.tokenizer(cleaned_words)

        t_infer_start = time.time()
        waveform = waveform.to(self.device)
        with torch.no_grad():
            emission, _ = self.model(waveform)
        emission_cpu = emission[0].cpu()
        t_emission = time.time() - t_infer_start

        t_align_start = time.time()
        token_spans = self.aligner(emission_cpu, tokens)
        t_align = time.time() - t_align_start

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
                    "orig_word": orig_words[orig_idx],
                    "start": start_sec,
                    "end": end_sec,
                })

        # Insert commas at pauses >= threshold
        result_words = list(orig_words)
        pauses = []
        for i in range(len(word_spans) - 1):
            w1 = word_spans[i]
            w2 = word_spans[i+1]
            gap = w2["start"] - w1["end"]
            if gap >= threshold:
                w_idx = w1["idx"]
                cur = result_words[w_idx]
                pauses.append((gap, w1["orig_word"], w2["orig_word"]))
                if not cur.endswith((",", ".", "!", "?", ";", ":")):
                    result_words[w_idx] = cur + ","

        punctuated_text = " ".join(result_words)
        total_time = time.time() - t0
        return {
            "punctuated_text": punctuated_text,
            "pauses": pauses,
            "duration": duration,
            "total_time": total_time,
            "emission_time": t_emission,
            "align_time": t_align,
        }

@app.local_entrypoint()
def main():
    from datasets import load_dataset, Audio

    print("Fetching first 5 samples...")
    ds = load_dataset("ghanaopenai/twi-zephyr-synthetic-speech", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    samples = []
    it = iter(ds)
    for _ in range(5):
        samples.append(next(it))

    service = AlignerGPU()

    print("Running GPU alignment benchmark on 5 samples...")
    # Warmup
    _ = service.align_and_punctuate.remote(
        samples[0]["text"], samples[0]["normalised_text"], samples[0]["audio"]["bytes"], 0.35
    )

    t_start = time.time()
    for i, sample in enumerate(samples):
        res = service.align_and_punctuate.remote(
            sample["text"],
            sample["normalised_text"],
            sample["audio"]["bytes"],
            threshold=0.35
        )
        print(f"\n================ Sample {i} (audio duration: {res['duration']:.2f}s, total time: {res['total_time']*1000:.1f}ms, emission: {res['emission_time']*1000:.1f}ms, align: {res['align_time']*1000:.1f}ms) ================")
        print("ORIGINAL:")
        print(sample["text"])
        print("PUNCTUATED:")
        print(res["punctuated_text"])
        print(f"Detected {len(res['pauses'])} pauses (>=0.35s):")
        for gap, w1, w2 in res["pauses"]:
            print(f"  {gap:.2f}s after '{w1}' -> '{w2}'")

    print(f"\nTotal benchmark time for 5 samples: {time.time() - t_start:.2f}s")
