import os
import re
import time
import numpy as np
import gradio as gr
from transformers import AutoTokenizer
import onnxruntime as ort
from huggingface_hub import hf_hub_download

MODEL_ID = "ghanaopenai/twi-punct-cap-restoration"
print(f"Loading model {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
onnx_path = hf_hub_download(repo_id=MODEL_ID, filename="model_quantized.onnx")

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
print("Model initialized successfully!")

REV_PUNCT = {0: "", 1: ",", 2: ".", 3: "?", 4: "!"}

def restore_twi(text: str):
    if not text or not text.strip():
        return "", "0 ms", "0", "0", "0"

    t0 = time.time()
    raw_tokens = text.strip().split()
    if not raw_tokens:
        return "", "0 ms", "0", "0", "0"

    # Clean outer punctuation to treat input as raw words
    clean_words = []
    for w in raw_tokens:
        cw = re.sub(r'^[^\w\u0190\u0186\u025B\u0254]+|[^\w\u0190\u0186\u025B\u0254]+$', '', w)
        clean_words.append(cw.lower() if cw else w.lower())

    # Tokenize with subword mapping
    encoding = tokenizer(clean_words, is_split_into_words=True, return_tensors="np")
    word_ids = encoding.word_ids(batch_index=0)

    ort_inputs = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    }
    ort_outputs = session.run(None, ort_inputs)
    cap_preds = np.argmax(ort_outputs[0][0], axis=-1)
    punct_preds = np.argmax(ort_outputs[1][0], axis=-1)

    # First subtoken per word
    current_word = None
    word_caps, word_puncts = [], []
    for token_idx, w_id in enumerate(word_ids):
        if w_id is not None and w_id != current_word:
            word_caps.append(cap_preds[token_idx])
            word_puncts.append(punct_preds[token_idx])
            current_word = w_id

    commas_added = 0
    capitals_added = 0
    reconstructed = []

    for w, c, p in zip(clean_words, word_caps, word_puncts):
        if c == 2:
            word_str = w.upper()
            capitals_added += 1
        elif c == 1:
            word_str = "-".join(part.capitalize() for part in w.split("-"))
            capitals_added += 1
        else:
            word_str = w.lower()

        punct = REV_PUNCT.get(p, "")
        if punct == ",":
            commas_added += 1
        word_str += punct
        reconstructed.append(word_str)

    latency_ms = (time.time() - t0) * 1000
    restored_text = " ".join(reconstructed)

    return (
        restored_text,
        f"{latency_ms:.1f} ms",
        f"{len(clean_words)}",
        f"{commas_added}",
        f"{capitals_added}"
    )

examples = [
    ["wɔ may 6 no abansoro dan bi a ɛyɛ fɛ a ɛwɔ abansoro anum bubui wɔ george south africa anyɛ yiye koraa no asɛm a esii no kunkum adansi adwumayɛfo aduasa abiɛsa"],
    ["nea ɛyɛ awerɛhow no mmɔden a wɔbɔe sɛ wobegye nkurɔfo no baa awiei a na nnipa dunkron a wɔda so ara mmu wɔn ho akontaa"],
    ["sɛnea nhwehwɛmu asɛm no kyerɛ no lf a wɔde siw ano no gyina akwan abien so titiriw yare a wɔde siw ano ne nea wɔde kɔ nnipa mu a wɔtew so"],
    ["dɛn na ɛtoo george wɔ south africa wɔ may da a ɛto so asia"],
    ["afei wɔde ɔdan no bɛhyɛ nhwehwɛmufo nsa ma wɔahu nea ɛde akwanhyia no bae"],
    ["ahene nanahene nana kwame mensah bonsu aka se ese se wobebo dawuro ama chief of staff julius debrah se onkasa nkyere president john dramani mahama ne vice president professor naana jane opoku-agyemang"]
]

custom_css = """
.title-box { text-align: center; margin-bottom: 1rem; }
.stat-box { font-weight: bold; }
"""

with gr.Blocks(title="Twi Punctuation & Capitalization Restorer", css=custom_css) as demo:
    gr.Markdown(
        """        # ✍️ Twi Punctuation & Capitalization Restorer
        Restore **commas**, **sentence-ending periods**, and **proper capitalization** (Title Case, ALL CAPS) in Akan/Twi text.
        
        * **Why this matters for Text-to-Speech (TTS):** Commas are routinely omitted when writing Twi, despite their clear presence as pauses in normal spoken Twi. This model compensates for these conventionally ignored commas, enabling natural breath pauses and prosody for TTS engines.
        * **Target Use-Cases:** Text-to-Speech (TTS) script pre-processing, ASR / Speech-to-Text transcript formatting, and conversational text cleaning.
        * **Model:** Fine-tuned `Ghana-NLP/distilabena-base-v2-akuapem-twi-cased` running on **quantized INT8 ONNX** (~15–28 ms CPU latency)."""
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Raw Unpunctuated / Lowercase Twi Text",
                placeholder="Type or paste unpunctuated Twi text here...",
                lines=5,
            )
            submit_btn = gr.Button("✨ Restore Punctuation & Capitals", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Punctuated & Capitalized Text",
                lines=5,
                show_copy_button=True,
            )
            with gr.Row():
                latency_stat = gr.Textbox(label="CPU Latency", scale=1, interactive=False)
                words_stat = gr.Textbox(label="Words", scale=1, interactive=False)
                commas_stat = gr.Textbox(label="Commas Restored", scale=1, interactive=False)
                caps_stat = gr.Textbox(label="Words Capitalized", scale=1, interactive=False)

    submit_btn.click(
        fn=restore_twi,
        inputs=[input_text],
        outputs=[output_text, latency_stat, words_stat, commas_stat, caps_stat]
    )

    gr.Examples(
        examples=examples,
        inputs=[input_text],
        outputs=[output_text, latency_stat, words_stat, commas_stat, caps_stat],
        fn=restore_twi,
        cache_examples=False,
    )

    gr.Markdown(
        """
        ---
        Developed by **Ghana NLP** & **Ghana OpenAI**. Model available on Hugging Face: [`ghanaopenai/twi-punct-cap-restoration`](https://huggingface.co/ghanaopenai/twi-punct-cap-restoration).
        """
    )

if __name__ == "__main__":
    demo.launch()
