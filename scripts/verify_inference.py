import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "onnxruntime",
        "huggingface_hub",
        "numpy",
    )
    .add_local_file("/home/owusus/Documents/GitHub/twi-comma-fix/twi_restorer.py", remote_path="/root/twi_restorer.py")
)

app = modal.App("verify-twi-restorer", image=image)

@app.function(cpu=2.0)
def test_stripped_inputs():
    import sys
    sys.path.append("/root")
    from twi_restorer import TwiRestorer
    import time

    restorer = TwiRestorer(backend="onnx")

    samples = [
        # Sample with completely stripped punctuation and lowercase
        "ahene nanahene nana kwame mensah bonsu aka se ese se wobebo dawuro ama chief of staff julius debrah se onkasa nkyere president john dramani mahama ne vice president professor naana jane opoku-agyemang se wonntena nkyere dormaahene osagyefo oseadeeyo agyeman badu ii wo ne nneyee ne n’akasa a ereka tia asanteman no ho",
        
        # Sample with standard Twi orthography
        "ahene nanahene nana kwame mensah bonsu aka sɛ ɛsɛ sɛ wɔbɛbɔ dawuro ama chief of staff julius debrah sɛ ɔnkasa nkyerɛ president john dramani mahama ne vice president professor naana jane opoku-agyemang sɛ wɔnntena nkyerɛ dormaahene osagyefo oseadeeyo agyeman badu ii wɔ ne nneyɛɛ ne n’akasa a ɛreka tia asanteman no ho",
        
        # Sample with short clause
        "wo ofa foforo nso mu ministry of lands and natural resources ananmusini paa kwesi schandorf yii n'ano kaa se ewom se nkurofuo refre se womfa state of emergency nsi ho mfa nko tia galamsey de nanso enye adwene nko na ese na mmom ese se yewo actionable policies"
    ]

    for idx, text in enumerate(samples):
        t0 = time.time()
        output = restorer.restore(text)
        lat = (time.time() - t0) * 1000
        print(f"\n================ Sample {idx+1} (ONNX CPU Latency: {lat:.1f}ms) ================")
        print(f"RAW INPUT:\n{text}")
        print(f"RESTORED:\n{output}")

@app.local_entrypoint()
def main():
    test_stripped_inputs.remote()
