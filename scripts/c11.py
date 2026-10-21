"""c11: shared helpers for the session-155 batch (e680-e691, box 11). Standalone (torch, transformers, datasets, numpy).
Families: neox (Pythia: parallel residual, exact GELU, LayerNorm with affine), gpt2 (Conv1D, gelu_new, LayerNorm),
gated (Qwen2.5, SmolLM2: SwiGLU with RMSNorm; OLMo-1B-0724: SwiGLU with a parameter-free LayerNorm). The floor is
e666's extreme-value calibration (verbatim); OMP is the repo's (bx_common). Nothing here reads the record."""
import os, sys, time, json, math, collections, random
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.activations import ACT2FN
torch.set_grad_enabled(False); DEV = "cuda:0" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(int(os.environ.get("C11_THREADS", "8")))
MODELS = {"pythia70": "EleutherAI/pythia-70m", "pythia160": "EleutherAI/pythia-160m", "pythia160s1": "EleutherAI/pythia-160m-seed1", "pythia160s2": "EleutherAI/pythia-160m-seed2", "pythia410": "EleutherAI/pythia-410m", "pythia1b": "EleutherAI/pythia-1b", "pythia14b": "EleutherAI/pythia-1.4b", "pythia28b": "EleutherAI/pythia-2.8b",
          "gpt2": "openai-community/gpt2", "gpt2m": "openai-community/gpt2-medium", "gpt2l": "openai-community/gpt2-large", "gpt2xl": "openai-community/gpt2-xl",
          "smollm135": "HuggingFaceTB/SmolLM2-135M", "smollm360": "HuggingFaceTB/SmolLM2-360M", "smollm17b": "HuggingFaceTB/SmolLM2-1.7B",
          "qwen05": "Qwen/Qwen2.5-0.5B", "qwen15": "Qwen/Qwen2.5-1.5B", "qwen3b": "Qwen/Qwen2.5-3B", "qwen7b": "Qwen/Qwen2.5-7B", "qwen14b": "Qwen/Qwen2.5-14B", "qwen32b": "Qwen/Qwen2.5-32B", "olmo1b": "allenai/OLMo-1B-0724-hf"}
RESD = os.environ.get("C11_RESULTS", "/workspace/wdd/results"); CACHED = os.environ.get("C11_CACHE", "/workspace/wdd/cache"); T0 = time.time()
def log(tag, s): print(f"[{tag} {time.strftime('%H:%M:%S')}] {s} ({time.time() - T0:.0f}s)", flush=True)
def med(v): v = [float(x) for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]; return float(np.median(v)) if v else None
def mean(v): v = [float(x) for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]; return float(np.mean(v)) if v else None
def binom_p(k, n): return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else None
def jsonable(o):
    if hasattr(o, "item"): return o.item()
    if torch.is_tensor(o): return o.tolist()
    if isinstance(o, (np.ndarray,)): return o.tolist()
    return str(o)
def save(tag, res, summ):
    tag = tag + os.environ.get("WDD_TAG_SUFFIX", ""); os.makedirs(RESD, exist_ok=True); res["_exp"] = tag; res["_time"] = time.strftime("%Y-%m-%d %H:%M:%S"); res["seconds"] = time.time() - T0
    if "summary" in res and not isinstance(res["summary"], str): res["summary_stats"] = res.pop("summary")   # a script's summary dictionary keeps its place; the sentence goes under 'summary'
    res["summary"] = summ
    json.dump(res, open(f"{RESD}/{tag}.json", "w"), indent=1, default=jsonable); print(summ, flush=True); print("done", flush=True)
# ---------------- models
def load(name, revision=None, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(MODELS[name]); model = AutoModelForCausalLM.from_pretrained(MODELS[name], revision=revision, dtype=dtype).to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    cfg = model.config
    if hasattr(model, "gpt_neox"): fam, base, layers, fnorm = "neox", model.gpt_neox, model.gpt_neox.layers, model.gpt_neox.final_layer_norm
    elif hasattr(model, "transformer"): fam, base, layers, fnorm = "gpt2", model.transformer, model.transformer.h, model.transformer.ln_f
    else: fam, base, layers, fnorm = "gated", model.model, model.model.layers, model.model.norm
    M = dict(name=name, hf=MODELS[name], revision=revision, tok=tok, model=model, cfg=cfg, fam=fam, base=base, layers=layers, fnorm=fnorm, NB=len(layers), D=getattr(cfg, "hidden_size", None) or cfg.n_embd, dtype=dtype, V=model.get_output_embeddings().weight.shape[0])
    M["DFF"] = write_w(M, 0).shape[1]; M["parallel"] = bool(fam == "neox" and getattr(cfg, "use_parallel_residual", True))
    M["gamma"] = fnorm.weight.detach().float() if getattr(fnorm, "weight", None) is not None else torch.ones(M["D"], device=DEV)
    M["fnorm_rms"] = "RMS" in type(fnorm).__name__; M["act"] = ACT2FN[getattr(cfg, "hidden_act", None) or getattr(cfg, "activation_function", "gelu")]; M["WU"] = model.get_output_embeddings().weight.detach().float()
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    return M
def mlp(M, b): return M["layers"][b].mlp
def write_mod(M, b): return mlp(M, b).dense_4h_to_h if M["fam"] == "neox" else (mlp(M, b).c_proj if M["fam"] == "gpt2" else mlp(M, b).down_proj)
def write_w(M, b):
    """[D, DFF]: column j is neuron j's write vector"""
    w = write_mod(M, b).weight.detach().float(); return w.T if M["fam"] == "gpt2" else w
def set_write_col(M, b, j, v):
    w = write_mod(M, b).weight
    if M["fam"] == "gpt2": w[j] = v.to(w.dtype)
    else: w[:, j] = v.to(w.dtype)
def write_bias(M, b):
    bb = getattr(write_mod(M, b), "bias", None); return bb.detach().float() if bb is not None else torch.zeros(M["D"], device=DEV)
def read_w(M, b):
    """dict of [DFF, D] read matrices: neox and gpt2 {'h'}; gated {'g', 'u'}"""
    if M["fam"] == "neox": return {"h": mlp(M, b).dense_h_to_4h.weight.detach().float()}
    if M["fam"] == "gpt2": return {"h": mlp(M, b).c_fc.weight.detach().float().T}
    return {"g": mlp(M, b).gate_proj.weight.detach().float(), "u": mlp(M, b).up_proj.weight.detach().float()}
def read_bias(M, b):
    if M["fam"] == "neox": return {"h": mlp(M, b).dense_h_to_4h.bias.detach().float()}
    if M["fam"] == "gpt2": return {"h": mlp(M, b).c_fc.bias.detach().float()}
    return {"g": torch.zeros(M["DFF"], device=DEV), "u": torch.zeros(M["DFF"], device=DEV)}
def read_mods(M, b):
    if M["fam"] == "neox": return {"h": mlp(M, b).dense_h_to_4h}
    if M["fam"] == "gpt2": return {"h": mlp(M, b).c_fc}
    return {"g": mlp(M, b).gate_proj, "u": mlp(M, b).up_proj}
def ln_mlp(M, b): return M["layers"][b].ln_2 if M["fam"] == "gpt2" else M["layers"][b].post_attention_layernorm
def ln_params(M, b):
    """the MLP's normalisation: gamma, beta, eps, and whether it centres (LayerNorm) or not (RMSNorm)"""
    ln = ln_mlp(M, b); D = M["D"]
    gam = ln.weight.detach().float() if getattr(ln, "weight", None) is not None else torch.ones(D, device=DEV); bet = ln.bias.detach().float() if getattr(ln, "bias", None) is not None else torch.zeros(D, device=DEV)
    eps = getattr(ln, "eps", None) or getattr(ln, "variance_epsilon", None) or 1e-5; return dict(gamma=gam, beta=bet, eps=float(eps), centre="RMS" not in type(ln).__name__)
def ln_apply(M, b, xin):
    """the normalised MLP input (after the affine) and the per-position scale sigma"""
    P = ln_params(M, b)
    if P["centre"]: mu = xin.mean(-1, keepdim=True); sig = (xin.var(-1, unbiased=False) + P["eps"]).sqrt(); return (xin - mu) / sig[:, None] * P["gamma"] + P["beta"], sig
    sig = (xin.pow(2).mean(-1) + P["eps"]).sqrt(); return xin / sig[:, None] * P["gamma"] + P["beta"], sig
def read_tilde(M, b, R):
    """a read matrix [DFF, D] folded through the normalisation's gamma (and centred for LayerNorm): a vector v in the MLP input contributes v . r_tilde / sigma to the pre-activation"""
    P = ln_params(M, b); Rt = R * P["gamma"][None]; return Rt - Rt.mean(-1, keepdim=True) if P["centre"] else Rt
def read_const(M, b, R, bias):
    """the constant part of the pre-activation: beta . r + b"""
    P = ln_params(M, b); return R @ P["beta"] + bias
def attn_mod(M, b): return M["layers"][b].attention if M["fam"] == "neox" else (M["layers"][b].attn if M["fam"] == "gpt2" else M["layers"][b].self_attn)
def final_gain(M, A):
    """a direction as the unembedding reads it: the final norm's linear part (centred for LayerNorm, then the gain)"""
    if not M["fnorm_rms"]: A = A - A.mean(-1, keepdim=True)
    return A * M["gamma"]
def n_att_feeding(M, b): return b if M["parallel"] else b + 1   # attention outputs that feed block b's MLP input: parallel residual excludes the block's own
class Stop(Exception): pass
class Capture:
    """forward hooks on a model; cap[(kind, b)] filled per forward. kinds: act (write input), x (block output), xin (MLP-norm input), att (attention output), pre (read outputs: h, or g and u), emb (block-0 input).
    stop_after=b raises Stop after block b (the caller catches it)."""
    def __init__(self, M, blocks, kinds, stop_after=None, fn=None):
        self.M, self.blocks, self.kinds, self.stop_after, self.fn, self.cap, self.hs = M, blocks, kinds, stop_after, (fn or (lambda t: t)), {}, []
    def __enter__(self):
        M, cap, fn = self.M, self.cap, self.fn; L = M["layers"]
        for b in self.blocks:
            if "act" in self.kinds: self.hs.append(write_mod(M, b).register_forward_pre_hook(lambda m, a, b=b: cap.__setitem__(("act", b), fn(a[0]))))
            if "x" in self.kinds: self.hs.append(L[b].register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(("x", b), fn(o[0] if isinstance(o, tuple) else o))))
            if "xin" in self.kinds: self.hs.append(ln_mlp(M, b).register_forward_pre_hook(lambda m, a, b=b: cap.__setitem__(("xin", b), fn(a[0]))))
            if "att" in self.kinds: self.hs.append(attn_mod(M, b).register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(("att", b), fn(o[0] if isinstance(o, tuple) else o))))
            if "pre" in self.kinds:
                for k, mod in read_mods(M, b).items(): self.hs.append(mod.register_forward_hook(lambda m, a, o, b=b, k=k: cap.__setitem__(("pre_" + k, b), fn(o))))
        if "emb" in self.kinds: self.hs.append(L[0].register_forward_pre_hook(lambda m, a, k: cap.__setitem__(("emb", 0), fn(a[0] if a else k["hidden_states"])), with_kwargs=True))
        if self.stop_after is not None:
            def stop(m, a, o): raise Stop
            self.hs.append(L[self.stop_after].register_forward_hook(stop))
        return self
    def __exit__(self, *e):
        for h in self.hs: h.remove()
        return False
def run_base(M, ids, **kw):
    try: return M["base"](input_ids=ids, use_cache=False, **kw)
    except Stop: return None
def logprobs_at(M, h, bs_rows=4096, normed=True):
    """log-softmax of the unembedding applied to final residuals h [..., D], in row chunks. The HF base model returns last_hidden_state after its
    final norm, so by default the norm is not applied again (normed=True); pass normed=False for a residual taken before the final norm.
    Until 2026-10-20 (session 165) this applied the norm unconditionally, which doubled it for every caller: see docs/ERRATA.md."""
    H = h.reshape(-1, h.shape[-1]); out = []; head = M["model"].get_output_embeddings()
    for s in range(0, H.shape[0], bs_rows): x = H[s:s + bs_rows]; out.append(head(x if normed else M["fnorm"](x)).float().log_softmax(-1))
    return torch.cat(out).reshape(*h.shape[:-1], -1)
def kl(P, Q): return (P.exp() * (P - Q)).sum(-1).clamp_min(0)
# ---------------- corpora
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
# ---------------- the dictionary, the rotation, the floor (e666 verbatim), OMP (bx_common)
def unitr(X): return X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
def rotate(A, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed); Q, _ = torch.linalg.qr(torch.randn(A.shape[1], A.shape[1], generator=g)); return A @ Q.to(A.device)
def rows_dictionary(M, upto):
    """unit MLP write rows of blocks 0..upto [m, D], each row's block and neuron, and the row norms"""
    W = [write_w(M, b) for b in range(upto + 1)]; R = torch.cat([w.T for w in W]); blk = torch.cat([torch.full((w.shape[1],), b, dtype=torch.long) for b, w in enumerate(W)]); neu = torch.cat([torch.arange(w.shape[1]) for w in W])
    nr = R.norm(dim=1); return R / nr[:, None].clamp_min(1e-12), blk.to(DEV), neu.to(DEV), nr
def gabs(m, Tt=14.0, n=28001):
    t = torch.linspace(0, Tt, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def floor_calibration(U, A, seed=11):
    """U unit centred states [N, D]; A unit atoms [m, D]. The competitor maximum of a rotated copy, calibrated: returns what floor_of needs"""
    m = A.shape[0]; Ar = unitr(rotate(A, seed=seed)); CA = A.T @ A / m; CR = Ar.T @ Ar / m; gm = gabs(m); N = U.shape[0]
    s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 256] @ Ar.T).abs().max(1).values for s in range(0, N, 256)])
    r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); return dict(r_cal=r_cal, gm=gm, CA=CA, m=m)
def floor_of(U, cal): return (((U @ cal["CA"]) * U).sum(1)).clamp_min(1e-12).sqrt() * cal["gm"] * cal["r_cal"]
def omp(X, A, k=16, batch=256, ridge=1e-5):
    N, NA = X.shape[0], A.shape[0]
    sel = torch.zeros(N, k, dtype=torch.long, device=X.device); cof = torch.zeros(N, k, device=X.device); eye = torch.eye(k, device=X.device)
    for s in range(0, N, batch):
        x = X[s:s + batch]; n = x.shape[0]; r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=X.device); taken = torch.zeros(n, NA, dtype=torch.bool, device=X.device)
        for step in range(k):
            corr = (r @ A.T).abs_().masked_fill_(taken, -1.0); pick = corr.argmax(-1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
            As = A[S]; G = As @ As.transpose(1, 2) + ridge * eye[:step + 1, :step + 1]; c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (c.transpose(1, 2) @ As)[:, 0]
        sel[s:s + n] = S; cof[s:s + n] = c[:, :, 0]; del taken, corr
    return sel, cof
def kmeans(X, k, iters=30, inits=5, seed=0):
    """cosine k-means on unit rows; returns labels"""
    g = torch.Generator(device="cpu").manual_seed(seed); best = None; X = X.cpu().float()
    for _ in range(inits):
        Cc = X[torch.randperm(X.shape[0], generator=g)[:k]].clone()
        for _ in range(iters):
            lab = (X @ Cc.T).argmax(1); Cc = torch.stack([X[lab == c].mean(0) if (lab == c).any() else Cc[c] for c in range(k)]); Cc = Cc / Cc.norm(dim=1, keepdim=True).clamp_min(1e-9)
        lab = (X @ Cc.T).argmax(1); obj = float((X @ Cc.T).max(1).values.sum())
        if best is None or obj > best[0]: best = (obj, lab.clone())
    return best[1]
def silhouette2(X, lab):
    Dm = 1 - X @ X.T; Mo = torch.nn.functional.one_hot(lab, 2).float(); S = Dm @ Mo; n = Mo.sum(0)
    if (n < 2).any(): return 0.0
    own = S.gather(1, lab[:, None])[:, 0] / (n[lab] - 1); oth = S.gather(1, (1 - lab)[:, None])[:, 0] / n[1 - lab]
    return float(((oth - own) / torch.maximum(own, oth).clamp_min(1e-9)).mean())
def null_silhouettes(Xraw, seed, nnull=20):
    """e672's unimodal null: the centred firings mixed by a random rotation keep their mean and covariance"""
    g = torch.Generator().manual_seed(seed); mu = Xraw.mean(0, keepdim=True); Xc = Xraw - mu; out = []
    for _ in range(nnull):
        Q, R = torch.linalg.qr(torch.randn(Xraw.shape[0], Xraw.shape[0], generator=g)); Q = Q * torch.sign(torch.diagonal(R))[None]
        Xn = unitr(mu + Q @ Xc); out.append(silhouette2(Xn, kmeans(Xn, 2, inits=5)))
    return out
