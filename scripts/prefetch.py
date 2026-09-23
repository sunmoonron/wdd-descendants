"""Checkpoint prefetcher (CPU/network). Downloads the model weights and config of each revision listed in
prefetch_<repo-tag>.txt, in order, keeping at most MAXDISK revisions of that repo in the cache at a time; a revision is
deleted from the cache once its result files (results/<prefix>_<rev>.json) exist. Runs until the list is exhausted."""
import sys, os, time, glob
from huggingface_hub import snapshot_download, scan_cache_dir
repo, lst, prefix, MAXDISK = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
revs = [l.strip() for l in open(lst) if l.strip()]; RES = "/workspace/wdd/results"; got = []
def protected():
    """revisions named by a running or pending scheduler job must not be deleted"""
    import json
    try: st = json.load(open("/workspace/wdd/logs/sched_state.json")); cmds = st.get("running", []) + st.get("pending_all", [])
    except Exception: return None
    if "pending_all" not in st: return None
    return set(tok for cmd in cmds for tok in cmd.split("#")[0].split())
def cleanup():
    prot = protected()
    if prot is None: return []
    info = scan_cache_dir(); dele = []
    for r in info.repos:
        if r.repo_id != repo: continue
        for rv in r.revisions:
            refs = set(rv.refs)
            if any(glob.glob(f"{RES}/{prefix}_{ref}.json") for ref in refs) and not (refs & prot): dele.append(rv.commit_hash)
    if dele: info.delete_revisions(*dele).execute(); print(time.strftime("%H:%M:%S"), "deleted", len(dele), flush=True)
    return dele
for rev in revs:
    while True:
        cleanup(); ondisk = [x for x in got if not glob.glob(f"{RES}/{prefix}_{x}.json")]
        if len(ondisk) < MAXDISK: break
        time.sleep(10)
    if glob.glob(f"{RES}/{prefix}_{rev}.json"): continue
    t0 = time.time()
    for attempt in range(5):
        try: snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt"]); break
        except Exception as e: print("retry", rev, e, flush=True); time.sleep(15 * (attempt + 1))
    got.append(rev); open(f"/workspace/wdd/logs/prefetched_{prefix}.txt", "a").write(rev + "\n"); print(time.strftime("%H:%M:%S"), "fetched", rev, f"{time.time() - t0:.0f}s", flush=True)
while True:
    if not cleanup(): pass
    if all(glob.glob(f"{RES}/{prefix}_{x}.json") for x in revs): break
    time.sleep(20)
print("prefetch complete", flush=True)
