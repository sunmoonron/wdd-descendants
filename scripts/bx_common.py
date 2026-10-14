"""Shared helpers for e653-e655 (session 142): the model's answer as the boundary readout of a running total of native
MLP write rows. Standalone (no repo imports) for the Dell. The dictionary is the unit MLP write rows of every block (the
s101 convention, rows only); OMP is the repo's wdd_common.omp (|corr| pick, Cholesky refit, ridge 1e-5); the rotation is
wdd_common.rotate (a random orthogonal map, seed 7)."""
import sys, os, time, json, math
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
MODELS = {"pythia160": "EleutherAI/pythia-160m", "pythia410": "EleutherAI/pythia-410m", "qwen05": "Qwen/Qwen2.5-0.5B", "smollm135": "HuggingFaceTB/SmolLM2-135M"}
K = 16


def load(name):
    tok = AutoTokenizer.from_pretrained(MODELS[name]); model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.float32).to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    fam = "neox" if hasattr(model, "gpt_neox") else "llama"
    layers, norm = (model.gpt_neox.layers, model.gpt_neox.final_layer_norm) if fam == "neox" else (model.model.layers, model.model.norm)
    WU, WE = model.get_output_embeddings().weight, model.get_input_embeddings().weight  # version-proof (Pythia's embed_out is renamed in newer transformers)
    assert getattr(model.get_output_embeddings(), "bias", None) is None, "unembedding with a bias"
    return dict(name=name, tok=tok, model=model, fam=fam, layers=layers, norm=norm, WU=WU.detach().float(), WE=WE.detach().float(), NB=len(layers), D=model.config.hidden_size, gamma=norm.weight.detach().float())


def down_w(M, b):
    l = M["layers"][b].mlp; return (l.dense_4h_to_h if M["fam"] == "neox" else l.down_proj).weight.detach().float()  # [D, DFF]: column j is neuron j's write


def dictionary(M):
    """unit MLP write rows of every block [m, D] and each row's block"""
    W = [down_w(M, b) for b in range(M["NB"])]
    R = torch.cat([w.T for w in W]); blk = torch.cat([torch.full((w.shape[1],), b, dtype=torch.long) for b, w in enumerate(W)])
    return R / R.norm(dim=1, keepdim=True).clamp_min(1e-12), blk


def rotate(A, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed); Q, _ = torch.linalg.qr(torch.randn(A.shape[1], A.shape[1], generator=g)); return A @ Q.to(A.device)


def unitr(X): return X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def omp(X, A, k=K, batch=256, ridge=1e-5):
    """the repo's batched OMP: sel [N,k], cof [N,k] at the final support"""
    N, NA = X.shape[0], A.shape[0]
    sel = torch.zeros(N, k, dtype=torch.long, device=X.device); cof = torch.zeros(N, k, device=X.device); eye = torch.eye(k, device=X.device)
    for s in range(0, N, batch):
        x = X[s:s + batch]; n = x.shape[0]; r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=X.device); taken = torch.zeros(n, NA, dtype=torch.bool, device=X.device)
        for step in range(k):
            corr = (r @ A.T).abs_().masked_fill_(taken, -1.0); pick = corr.argmax(-1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
            As = A[S]; G = As @ As.transpose(1, 2) + ridge * eye[:step + 1, :step + 1]; c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (c.transpose(1, 2) @ As)[:, 0]
        sel[s:s + n] = S; cof[s:s + n] = c[:, :, 0]; del taken, corr
    return sel, cof


def gain(M, A):
    """an atom's direction as the unembedding reads it: the final norm's linear part (centring for LayerNorm, then the gain)"""
    if M["fam"] == "neox": A = A - A.mean(-1, keepdim=True)
    return A * M["gamma"]


def final_states(M, ids):
    """the raw residual after the last block [B, T, D] (before the final norm) and the logits [B, T, V]"""
    cap = {}
    h = M["layers"][-1].register_forward_hook(lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    try: lg = M["model"](ids).logits.float()
    finally: h.remove()
    return cap["x"], lg


def readout(M, h): return M["norm"](h) @ M["WU"].T


def wikitext(M, split, n, T):
    import datasets
    out = []
    for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]:
        if len(t.split()) <= 40: continue
        ids = M["tok"](t, return_tensors="pt").input_ids[0]
        if ids.shape[0] >= T: out.append(ids[:T])
        if len(out) >= n: break
    return torch.stack(out).to(DEV)


def train_token_ids(M, nlines=None):
    """every line of wikitext-2 train tokenised (for frequency and bigram tables: zero model passes)"""
    import datasets
    lines = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip()]
    if nlines: lines = lines[:nlines]
    return [x for x in M["tok"](lines)["input_ids"] if len(x) > 1]


def sinkmask(X): n = X.norm(dim=-1); return n > 10 * n.median()


def save(tag, res, summ):
    os.makedirs("results", exist_ok=True); res["_exp"] = tag; res["_time"] = time.strftime("%Y-%m-%d %H:%M:%S"); res["summary"] = summ
    json.dump(res, open(f"results/{tag}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print(summ, flush=True)  # numpy scalars and bools
