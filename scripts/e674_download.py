import os, re, time, json, concurrent.futures as cf
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download, list_repo_refs, list_repo_files
t0 = time.time(); log = lambda s: print(f"[dl] {s} ({time.time() - t0:.0f}s)", flush=True)
PY = ["step128", "step256", "step512", "step1000", "step2000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
OL = "allenai/OLMo-1B-0724-hf"
refs = {int(re.search(r"step(\d+)", b.name).group(1)): b.name for b in list_repo_refs(OL).branches if re.search(r"step(\d+)-tokens", b.name)}
ks = sorted(refs); log(f"OLMo refs: {len(ks)} from {ks[0]} to {ks[-1]}")
want = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]
pick = sorted({min(ks, key=lambda k: abs(k - w)) for w in want}); olmo = {k: refs[k] for k in pick}
json.dump(dict(olmo_revisions={str(k): v for k, v in olmo.items()}, olmo_last=refs[ks[-1]]), open("/workspace/wdd/olmo_refs.json", "w"), indent=1)
jobs = [("EleutherAI/pythia-160m", r) for r in PY] + [("EleutherAI/pythia-410m", r) for r in PY] + [(OL, v) for v in olmo.values()] + [(OL, "main")]
def get(job):
    repo, rev = job
    try:
        files = list_repo_files(repo, revision=rev); pats = ["*.json", "*.txt", "tokenizer*"] + (["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"])
        snapshot_download(repo, revision=rev, allow_patterns=pats); return f"{repo}@{rev} ok"
    except Exception as e: return f"{repo}@{rev} FAILED {type(e).__name__}: {str(e)[:200]}"
with cf.ThreadPoolExecutor(8) as ex:
    for r in ex.map(get, jobs): log(r)
import datasets
for split in ("train", "test"): datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
log("wikitext ok"); log("done")
