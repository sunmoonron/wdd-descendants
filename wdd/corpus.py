"""Token streams and windows: wikitext-2, pile-10k, and evenly spaced windows of a stream."""
import os, numpy as np, torch
from .model import DEV
CACHED = os.environ.get("WDD_CACHE", os.path.expanduser("~/.cache/wdd"))
def wikitext_ids(M, split):
    import datasets
    text = "\n".join(t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    return M["tok"](text, return_tensors="pt", add_special_tokens=False).input_ids[0]
def pile_ids(M, ndocs=3000, skip=0, tag=None):
    """pile-10k documents skip..skip+ndocs tokenised into one stream with the eos between documents (cached per tokenizer family)"""
    import datasets
    key = tag or M["hf"].replace("/", "_"); f = f"{CACHED}/pile_{key}_{skip}_{ndocs}.pt"
    if os.path.exists(f): return torch.load(f)
    ds = datasets.load_dataset("NeelNanda/pile-10k", split="train"); out = []
    for k in range(skip, min(skip + ndocs, len(ds))): out += M["tok"](ds[k]["text"], add_special_tokens=False)["input_ids"] + [M["tok"].eos_token_id]
    ids = torch.tensor(out); os.makedirs(CACHED, exist_ok=True); tmp = f + f".{os.getpid()}.tmp"; torch.save(ids, tmp); os.replace(tmp, f); return ids
def windows_from(ids, nwin, T, seed=None):
    if seed is None: starts = np.linspace(0, len(ids) - T - 1, nwin).astype(int)
    else: starts = np.sort(np.random.default_rng(seed).choice(len(ids) - T - 1, nwin, replace=False))
    return torch.stack([ids[s:s + T] for s in starts])
def sinkmask(X, f=10.0): n = X.norm(dim=-1); return n > f * n.median()
