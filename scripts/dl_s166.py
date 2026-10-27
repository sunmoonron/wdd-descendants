from huggingface_hub import snapshot_download
import datasets
for m in ("openai-community/gpt2", "openai-community/gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-135M", "openai-community/gpt2-large", "EleutherAI/pythia-1b", "Qwen/Qwen2.5-1.5B", "allenai/OLMo-1B-0724-hf", "HuggingFaceTB/SmolLM2-360M", "EleutherAI/pythia-70m", "roneneldan/TinyStories-33M", "roneneldan/TinyStories-8M", "Qwen/Qwen3-4B", "Qwen/Qwen3.5-4B"):
    for attempt in range(4):
        try: print(m, snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "*.tiktoken", "merges.txt", "vocab.json"]), flush=True); break
        except Exception as e: print(m, "retry", attempt, str(e)[:120], flush=True)
for d, c in (("Salesforce/wikitext", "wikitext-2-raw-v1"), ("NeelNanda/pile-10k", None)):
    try: datasets.load_dataset(d, c) if c else datasets.load_dataset(d); print("dataset", d, "ok", flush=True)
    except Exception as e: print("dataset", d, "FAILED", str(e)[:120], flush=True)
print("ALL DOWNLOADS DONE", flush=True)
