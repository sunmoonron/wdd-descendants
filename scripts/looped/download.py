from huggingface_hub import snapshot_download
for m in ["ByteDance/Ouro-1.4B", "HuggingFaceTB/SmolLM2-1.7B"]:
    print(m, snapshot_download(m), flush=True)
