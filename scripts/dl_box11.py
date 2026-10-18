"""Pre-download every model, checkpoint and dataset the e680-e691 batch needs (box 11, session 155). Small models first so
the smokes can start, then the Pythia checkpoints, then the big models. Safetensors preferred; the Pythia seed repos
carry only .bin. Each download is logged with its size; failures are logged and skipped."""
import os, sys, time, json
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download, HfApi
t0 = time.time(); api = HfApi(); LOG = open("/workspace/wdd/logs/dl.log", "a")
def log(s): print(s, flush=True); LOG.write(f"[{time.strftime('%H:%M:%S')}] {s}\n"); LOG.flush()
BASE = ["*.json", "*.txt", "*.model", "*.tiktoken", "*.py"]
def get(repo, rev=None):
    try:
        files = api.list_repo_files(repo, revision=rev); pats = BASE + (["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"])
        p = snapshot_download(repo, revision=rev, allow_patterns=pats); sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs) / 1e9
        log(f"ok {repo}@{rev or 'main'} {sz:.2f} GB ({time.time() - t0:.0f}s)")
    except Exception as e: log(f"FAIL {repo}@{rev or 'main'}: {type(e).__name__}: {str(e)[:200]}")
SMALL = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-135M", "openai-community/gpt2", "openai-community/gpt2-medium", "HuggingFaceTB/SmolLM2-360M", "facebook/opt-125m", "allenai/OLMo-1B-0724-hf", "EleutherAI/pythia-1b", "Qwen/Qwen2.5-1.5B"]
CK160 = list(range(1000, 16001, 1000)) + [32000, 64000]
BIG = ["EleutherAI/pythia-1.4b", "openai-community/gpt2-large", "openai-community/gpt2-xl", "HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen2.5-3B", "facebook/opt-1.3b", "EleutherAI/pythia-2.8b", "Qwen/Qwen2.5-7B"]
STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
if STAGE in ("all", "small"):
    for r in SMALL: get(r)
    import datasets
    for args in [("Salesforce/wikitext", "wikitext-2-raw-v1"), ("NeelNanda/pile-10k",), ("allenai/ai2_arc", "ARC-Easy"), ("Rowan/hellaswag",)]:
        try: d = datasets.load_dataset(*args); log(f"ok dataset {args} {list(d.keys())}")
        except Exception as e: log(f"FAIL dataset {args}: {type(e).__name__}: {str(e)[:200]}")
    for split in ("duped.410m", "duped.160m", "duped.1b"):
        try: d = datasets.load_dataset("EleutherAI/pythia-memorized-evals", split=split); log(f"ok memorized-evals {split}: {len(d)} rows, columns {d.column_names}")
        except Exception as e: log(f"FAIL memorized-evals {split}: {type(e).__name__}: {str(e)[:300]}")
if STAGE in ("all", "ckpts"):
    for rep in ("EleutherAI/pythia-160m", "EleutherAI/pythia-160m-seed1", "EleutherAI/pythia-160m-seed2"):
        for n in (CK160 if rep.endswith("160m") else CK160[:16]): get(rep, f"step{n}")
    get("EleutherAI/pythia-160m", "step8000")
if STAGE in ("all", "big"):
    for r in BIG: get(r)
if STAGE == "ck410":
    for n in range(1000, 16001, 1000): get("EleutherAI/pythia-410m", f"step{n}")
log(f"stage {STAGE} done ({time.time() - t0:.0f}s)")
