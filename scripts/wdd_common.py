"""wdd_common: shared cache + solvers for the WDD mapping sprint.

Conventions (match the paper): H[l] = residual state entering block l (H[0] = embeddings,
H[NB] = output of the last block, raw, NOT post-final-norm). "Level L" state = H[L+1] = state
after block L; its dictionary = embeddings + atoms of blocks 0..L. Ledger coefficient of MLP
neuron j in block b = act_j * ||w_j|| (unit atom convention). Sink states: norm > 10x median.
"""
import os, sys, time, json, math
import numpy as np
import torch

DEV = "cuda:0"
CACHE = os.environ.get("WDD_CACHE", "/workspace/wdd/cache")
RESULTS = os.environ.get("WDD_RESULTS", "/workspace/wdd/results")
CTX = 512
N_EVAL = int(os.environ.get("WDD_N_EVAL", 32))
N_CEN = 64
torch.set_grad_enabled(False)

MODELS = {
    "gpt2": ("openai-community/gpt2", "gpt2"),
    "smollm2": ("HuggingFaceTB/SmolLM2-135M", "llama"),
    "qwen05": ("Qwen/Qwen2.5-0.5B", "llama"),
    "pythia410": ("EleutherAI/pythia-410m", "neox"),
    "olmo1b": ("allenai/OLMo-1B-0724-hf", "llama"),
}
OG5 = ["gpt2", "smollm2", "qwen05", "pythia410", "olmo1b"]
# atom types
T_TOK, T_POS, T_MLP, T_ATT, T_BIAS = 0, 1, 2, 3, 4


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- model access
def load_model(name, revision=None, random_init=False, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    hf, fam = MODELS[name]
    tok = AutoTokenizer.from_pretrained(hf)
    if random_init:
        cfg = AutoConfig.from_pretrained(hf)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype).to(DEV).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(hf, revision=revision, dtype=dtype, device_map=DEV).eval()
    return model, tok, fam


class Arch:
    """Uniform accessors over the three HF families."""
    def __init__(self, model, fam):
        self.model, self.fam, cfg = model, fam, model.config
        if fam == "gpt2":
            self.layers = model.transformer.h; self.NB = cfg.n_layer; self.D = cfg.n_embd; self.NH = cfg.n_head
            self.emb = [model.transformer.wte.weight, model.transformer.wpe.weight]
            self.parallel = False
        elif fam == "llama":
            self.layers = model.model.layers; self.NB = cfg.num_hidden_layers; self.D = cfg.hidden_size; self.NH = cfg.num_attention_heads
            self.emb = [model.model.embed_tokens.weight]
            self.parallel = False
        else:
            self.layers = model.gpt_neox.layers; self.NB = cfg.num_hidden_layers; self.D = cfg.hidden_size; self.NH = cfg.num_attention_heads
            self.emb = [model.gpt_neox.embed_in.weight]
            self.parallel = bool(getattr(cfg, "use_parallel_residual", True))
        self.HD = self.D // self.NH
        self.DFF = self.wdir(0).shape[0]

    # write directions of the MLP: rows [DFF, D]
    def wdir(self, b):
        l = self.layers[b]
        if self.fam == "gpt2": return l.mlp.c_proj.weight.detach().float()
        if self.fam == "llama": return l.mlp.down_proj.weight.detach().float().T
        return l.mlp.dense_4h_to_h.weight.detach().float().T

    def mlp_bias(self, b):
        l = self.layers[b]
        if self.fam == "gpt2": return l.mlp.c_proj.bias.detach().float()
        if self.fam == "neox": return l.mlp.dense_4h_to_h.bias.detach().float()
        return None

    # read directions of the MLP (input weights): rows [DFF, D] (gate/up for gated; fc for gelu)
    def rdir(self, b, which="up"):
        l = self.layers[b]
        if self.fam == "gpt2": return l.mlp.c_fc.weight.detach().float().T          # Conv1D [D, DFF] -> [DFF, D]
        if self.fam == "llama": return (l.mlp.up_proj if which == "up" else l.mlp.gate_proj).weight.detach().float()
        return l.mlp.dense_h_to_4h.weight.detach().float()

    # attention output projection as [NH*HD, D]: row block h*HD:(h+1)*HD spans head h's write subspace
    def wo(self, b):
        l = self.layers[b]
        if self.fam == "gpt2": return l.attn.c_proj.weight.detach().float()
        if self.fam == "llama": return l.self_attn.o_proj.weight.detach().float().T
        return l.attention.dense.weight.detach().float().T

    def attn_bias(self, b):
        l = self.layers[b]
        if self.fam == "gpt2": return l.attn.c_proj.bias.detach().float()
        if self.fam == "neox": return l.attention.dense.bias.detach().float()
        return None

    def mlp_lin(self, b):
        l = self.layers[b]
        return l.mlp.c_proj if self.fam == "gpt2" else (l.mlp.down_proj if self.fam == "llama" else l.mlp.dense_4h_to_h)

    def attn_lin(self, b):
        l = self.layers[b]
        return l.attn.c_proj if self.fam == "gpt2" else (l.self_attn.o_proj if self.fam == "llama" else l.attention.dense)


def build_dictionary(arch, blocks=None):
    """Unit atoms [NA, D] + label arrays (type, block, index, rank). Order: tok emb, pos emb, then per block:
    MLP rows, per-head SVD basis (HD atoms each), biases (attn, mlp) when present."""
    blocks = list(range(arch.NB)) if blocks is None else blocks
    atoms, typ, blk, idx, rank = [], [], [], [], []
    e = arch.emb[0].detach().float(); atoms.append(e); n = e.shape[0]
    typ += [T_TOK] * n; blk += [-1] * n; idx += list(range(n)); rank += [0] * n
    if len(arch.emb) > 1:
        e = arch.emb[1].detach().float(); atoms.append(e); n = e.shape[0]
        typ += [T_POS] * n; blk += [-1] * n; idx += list(range(n)); rank += [0] * n
    for b in blocks:
        W = arch.wdir(b); atoms.append(W); n = W.shape[0]
        typ += [T_MLP] * n; blk += [b] * n; idx += list(range(n)); rank += [0] * n
        Wo = arch.wo(b)
        for h in range(arch.NH):
            Vh = torch.linalg.svd(Wo[h * arch.HD:(h + 1) * arch.HD, :], full_matrices=False).Vh
            atoms.append(Vh); typ += [T_ATT] * arch.HD; blk += [b] * arch.HD; idx += [h] * arch.HD; rank += list(range(arch.HD))
        ab, mb = arch.attn_bias(b), arch.mlp_bias(b)
        for j, v in enumerate([ab, mb]):
            if v is not None:
                atoms.append(v[None]); typ.append(T_BIAS); blk.append(b); idx.append(j); rank.append(0)
    A = torch.cat([a.to(DEV) for a in atoms])
    A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    lab = dict(type=torch.tensor(typ), block=torch.tensor(blk), index=torch.tensor(idx), rank=torch.tensor(rank))
    return A, lab


def corpus_ids(tok, corpus="wikitext", split="test"):
    from datasets import load_dataset
    if corpus == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    else:
        ds = load_dataset("NeelNanda/pile-10k", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return tok(text, return_tensors="pt").input_ids[0]


# ----------------------------------------------------------------------------- cache build
def build_cache(name, tag=None, revision=None, random_init=False, n_eval=N_EVAL, corpus="wikitext"):
    """Runs the model once on the centering and evaluation slices with hooks; saves everything."""
    tag = tag or name
    out = os.path.join(CACHE, tag); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    model, tok, fam = load_model(name, revision=revision, random_init=random_init)
    arch = Arch(model, fam)
    NB, D, DFF, NH, HD = arch.NB, arch.D, arch.DFF, arch.NH, arch.HD
    ids_eval = corpus_ids(tok, corpus, "test" if corpus == "wikitext" else None)
    ids_cen = corpus_ids(tok, "wikitext", "train")
    if corpus != "wikitext":
        ids_cen = ids_eval[(n_eval) * CTX:(n_eval + N_CEN) * CTX]
    eval_ids = ids_eval[:n_eval * CTX].view(n_eval, CTX)
    cen_ids = ids_cen[:N_CEN * CTX].view(N_CEN, CTX)

    # hooks: block outputs (residual after block b), MLP activations, attention out-proj input/output
    store = {}
    def blk_hook(b):
        def f(m, inp, outp):
            o = outp[0] if isinstance(outp, tuple) else outp
            store.setdefault("H", {}).setdefault(b, []).append(o.detach().to(torch.float16).cpu())
        return f
    def mlp_pre(b):
        def f(m, inp):
            store.setdefault("acts", {}).setdefault(b, []).append(inp[0].detach().to(torch.float16).cpu())
        return f
    def att_hook(b):
        def f(m, inp, outp):
            store.setdefault("HI", {}).setdefault(b, []).append(inp[0].detach().to(torch.float16).cpu())
            store.setdefault("ATT", {}).setdefault(b, []).append(outp.detach().to(torch.float16).cpu())
        return f

    def run(ids, want_hooks):
        hs = []
        if want_hooks:
            for b in range(NB):
                hs.append(arch.layers[b].register_forward_hook(blk_hook(b)))
                hs.append(arch.mlp_lin(b).register_forward_pre_hook(mlp_pre(b)))
                hs.append(arch.attn_lin(b).register_forward_hook(att_hook(b)))
        H0 = []
        for i in range(0, len(ids), 4):
            o = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
            H0.append(o.hidden_states[0].detach().to(torch.float16).cpu())
            if not want_hooks:
                for l in range(1, NB + 1):
                    store.setdefault("Hc", {}).setdefault(l, []).append(o.hidden_states[l].detach().float().sum((0, 1)).cpu())
            del o
        for h in hs: h.remove()
        return torch.cat(H0)

    # centering means per level from the disjoint slice (level 0..NB; level NB uses block hooks below)
    store.clear()
    hs = [arch.layers[b].register_forward_hook(blk_hook(b)) for b in range(NB)]
    H0c = run(cen_ids, False)
    for h in hs: h.remove()
    mu = torch.zeros(NB + 1, D)
    mu[0] = H0c.float().view(-1, D).mean(0)
    for b in range(NB):
        mu[b + 1] = torch.cat(store["H"][b]).float().view(-1, D).mean(0)
    store.clear()

    H0 = run(eval_ids, True)
    NT = n_eval * CTX
    H = torch.zeros(NB + 1, NT, D, dtype=torch.float16)
    H[0] = H0.view(NT, D)
    ATT = torch.zeros(NB, NT, D, dtype=torch.float16); HI = torch.zeros(NB, NT, NH * HD, dtype=torch.float16)
    acts = {}
    for b in range(NB):
        H[b + 1] = torch.cat(store["H"][b]).view(NT, D)
        ATT[b] = torch.cat(store["ATT"][b]).view(NT, D)
        HI[b] = torch.cat(store["HI"][b]).view(NT, -1)
        acts[b] = torch.cat(store["acts"][b]).view(NT, DFF)
    store.clear()
    # self-check of the ledger identity on block NB//2: H[b+1] - H[b] = ATT + acts @ Wdown (+ biases)
    b = NB // 2
    Wd = arch.wdir(b).cpu()
    mlpw = acts[b][:2048].float() @ Wd
    mb = arch.mlp_bias(b)
    if mb is not None: mlpw += mb.cpu()
    delta = H[b + 1][:2048].float() - H[b][:2048].float()
    resid = delta - ATT[b][:2048].float() - mlpw
    rel = (resid.norm() / delta.norm()).item()
    log(f"{tag}: ledger identity check block {b}: rel err {rel:.4f} (fp16 storage; expect < 0.02)")

    A, lab = build_dictionary(arch)
    WN = torch.stack([arch.wdir(b).norm(dim=-1).cpu() for b in range(NB)])          # [NB, DFF]
    WO = torch.stack([arch.wo(b).cpu() for b in range(NB)])                          # [NB, NH*HD, D]
    RD = torch.stack([arch.rdir(b).cpu() for b in range(NB)])                        # [NB, DFF, D] read dirs (up/fc)
    RG = torch.stack([arch.rdir(b, "gate").cpu() for b in range(NB)]) if fam == "llama" else None
    meta = dict(name=name, tag=tag, hf=MODELS[name][0], fam=fam, NB=NB, D=D, DFF=DFF, NH=NH, HD=HD, NT=NT, n_eval=n_eval,
                ctx=CTX, parallel=arch.parallel, NA=A.shape[0], n_tok_emb=arch.emb[0].shape[0],
                n_pos_emb=(arch.emb[1].shape[0] if len(arch.emb) > 1 else 0), revision=revision, random_init=random_init,
                corpus=corpus, ledger_check=rel)
    torch.save(dict(H=H, ATT=ATT, HI=HI, mu=mu, eval_ids=eval_ids, cen_ids=cen_ids), os.path.join(out, "states.pt"))
    torch.save(acts, os.path.join(out, "acts.pt"))
    torch.save(dict(A=A.cpu(), lab=lab, WN=WN, WO=WO, RD=RD, RG=RG, emb=[e.detach().float().cpu() for e in arch.emb],
                    mlp_bias=[arch.mlp_bias(b) for b in range(NB)], attn_bias=[arch.attn_bias(b) for b in range(NB)]),
               os.path.join(out, "dict.pt"))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    del model; torch.cuda.empty_cache()
    log(f"{tag}: cache built in {time.time() - t0:.0f}s: NT={NT} D={D} DFF={DFF} NB={NB} atoms={A.shape[0]}")
    return meta


class Cache:
    """Lazy loader for a cached model."""
    def __init__(self, tag):
        self.dir = os.path.join(CACHE, tag)
        self.meta = json.load(open(os.path.join(self.dir, "meta.json")))
        self._s = self._a = self._d = None
        for k, v in self.meta.items(): setattr(self, k, v)

    @property
    def s(self):
        if self._s is None: self._s = torch.load(os.path.join(self.dir, "states.pt"))
        return self._s

    @property
    def acts(self):
        if self._a is None: self._a = torch.load(os.path.join(self.dir, "acts.pt"))
        return self._a

    @property
    def d(self):
        if self._d is None: self._d = torch.load(os.path.join(self.dir, "dict.pt"))
        return self._d

    def X(self, L, center=True, dev=DEV):
        """State after block L (raw fp32 on device), centered with the disjoint-slice mean of that level."""
        x = self.s["H"][L + 1].float().to(dev)
        if center: x = x - self.s["mu"][L + 1].to(dev)
        return x

    def dictionary(self, L, types=(T_TOK, T_POS, T_MLP, T_ATT, T_BIAS), blocks=None, dev=DEV):
        """Atoms for level L: embeddings + blocks 0..L (or an explicit block list), filtered by type."""
        lab = self.d["lab"]
        keep = torch.zeros(len(lab["type"]), dtype=torch.bool)
        for t in types: keep |= lab["type"] == t
        blk = lab["block"]
        if blocks is None: keep &= (blk <= L)
        else: keep &= torch.isin(blk, torch.tensor(list(blocks))) | (blk < 0)
        A = self.d["A"][keep].to(dev)
        sub = {k: v[keep] for k, v in lab.items()}
        sub["orig"] = torch.nonzero(keep)[:, 0]
        return A, sub

    def ledger(self, L, blocks=None):
        """Signed ledger coefficients of every MLP write in blocks 0..L: dict b -> [NT, DFF] fp32 CPU (act * ||w||)."""
        blocks = range(L + 1) if blocks is None else blocks
        return {b: self.acts[b].float() * self.d["WN"][b][None] for b in blocks}

    def top_writes(self, L, n=3):
        """Per token: top-n MLP writes (by |coef|) over blocks 0..L: returns blocks [NT,n], neurons [NT,n], coefs [NT,n]."""
        led = self.ledger(L)
        C = torch.cat([led[b] for b in range(L + 1)], 1)                                   # [NT, (L+1)*DFF]
        v, i = C.abs().topk(n, dim=1)
        coef = torch.gather(C, 1, i)
        return i // self.DFF, i % self.DFF, coef

    def atom_index(self, L, blk, neu):
        """Dictionary row (within dictionary(L) with all types) of MLP neuron neu in block blk."""
        lab = self.d["lab"]
        # atoms of level L: all rows with block <= L; we need the position within that subset
        keep = lab["block"] <= L
        pos = torch.cumsum(keep.long(), 0) - 1
        mask = (lab["type"] == T_MLP)
        # build lookup table [NB, DFF] -> row
        if not hasattr(self, "_lut"):
            lut = torch.full((self.NB, self.DFF), -1, dtype=torch.long)
            rows = torch.nonzero(mask)[:, 0]
            lut[lab["block"][rows], lab["index"][rows]] = rows
            self._lut = lut
        return pos[self._lut[blk, neu]]

    def head_write(self, b, h, tokens=None):
        """Exact write of head h in block b: HI[:, h*HD:(h+1)*HD] @ WO[h rows] (no bias). [NT, D] fp32 CPU."""
        HI = self.s["HI"][b] if tokens is None else self.s["HI"][b][tokens]
        return HI[:, h * self.HD:(h + 1) * self.HD].float() @ self.d["WO"][b][h * self.HD:(h + 1) * self.HD]

    def mlp_write(self, b, tokens=None):
        a = self.acts[b] if tokens is None else self.acts[b][tokens]
        w = a.float() @ (self.d["A"][0:0].new_zeros(0) if False else self.wdir_cpu(b))
        return w

    def wdir_cpu(self, b):
        lab = self.d["lab"]; rows = (lab["type"] == T_MLP) & (lab["block"] == b)
        return self.d["A"][rows] * self.d["WN"][b][:, None]


# ----------------------------------------------------------------------------- solvers
def omp(X, A, k, batch=2048, nonneg=False, ridge=1e-5, record_err=True):
    """Batched OMP. Returns sel [N,k], cof [N,k] (coefficients at the final support), err [N, k] residual
    energy after each step (if record_err). nonneg: pick by positive correlation only and clamp the
    refit (projected refit, NNLS-lite)."""
    N, NA = X.shape[0], A.shape[0]
    sel = torch.zeros(N, k, dtype=torch.long, device=X.device); cof = torch.zeros(N, k, device=X.device)
    err = torch.zeros(N, k, device=X.device) if record_err else None
    eye = torch.eye(k, device=X.device)
    for s in range(0, N, batch):
        x = X[s:s + batch]; n = x.shape[0]
        r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=X.device)
        taken = torch.zeros(n, NA, dtype=torch.bool, device=X.device)
        for step in range(k):
            corr = r @ A.T
            if nonneg: corr = corr.masked_fill_(taken, -1e9)
            else: corr = corr.abs_().masked_fill_(taken, -1.0)
            pick = corr.argmax(-1, keepdim=True)
            taken.scatter_(1, pick, True)
            S = torch.cat([S, pick], 1)
            As = A[S]
            G = As @ As.transpose(1, 2) + ridge * eye[:step + 1, :step + 1]
            c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G))
            if nonneg: c = c.clamp_min(0)
            r = x - (c.transpose(1, 2) @ As)[:, 0]
            if record_err: err[s:s + n, step] = (r ** 2).sum(-1)
        sel[s:s + n] = S; cof[s:s + n] = c[:, :, 0]
        del taken, corr
    return sel, cof, err


def mp(X, A, k, batch=2048, record_err=True):
    """Plain matching pursuit (Mallat-Zhang 1993): no orthogonalization, atoms may repeat.
    Returns sel [N,k] (with repeats), cof [N,k] (per-step increments), err [N,k]."""
    N, NA = X.shape[0], A.shape[0]
    sel = torch.zeros(N, k, dtype=torch.long, device=X.device); cof = torch.zeros(N, k, device=X.device)
    err = torch.zeros(N, k, device=X.device) if record_err else None
    for s in range(0, N, batch):
        x = X[s:s + batch]; n = x.shape[0]; r = x.clone()
        for step in range(k):
            corr = r @ A.T
            pick = corr.abs().argmax(-1)
            c = corr.gather(1, pick[:, None])[:, 0]
            r = r - c[:, None] * A[pick]
            sel[s:s + n, step] = pick; cof[s:s + n, step] = c
            if record_err: err[s:s + n, step] = (r ** 2).sum(-1)
    return sel, cof, err


def refit(X, A, sel, ridge=1e-5):
    """Least-squares refit on a given support sel [N,k]. Returns cof [N,k] and residual energy [N]."""
    As = A[sel]; k = sel.shape[1]
    G = As @ As.transpose(1, 2) + ridge * torch.eye(k, device=X.device)
    c = torch.cholesky_solve(As @ X[:, :, None], torch.linalg.cholesky(G))[:, :, 0]
    r = X - torch.einsum("nk,nkd->nd", c, As)
    return c, (r ** 2).sum(-1)


def oneshot(X, A, k, batch=1024, whiten=None):
    """Top-k atoms by |correlation| (optionally with a whitening matrix W: score = (W x) . a), refit."""
    N = X.shape[0]
    sel = torch.zeros(N, k, dtype=torch.long, device=X.device)
    for s in range(0, N, batch):
        x = X[s:s + batch]
        sc = (x @ whiten.T if whiten is not None else x) @ A.T
        sel[s:s + x.shape[0]] = sc.abs().topk(k, dim=1).indices
    cof, err = refit(X, A, sel)
    return sel, cof, err


def fista_l1(X, A, lam, iters=300, batch=1024, nonneg=False, L=None):
    """Basis pursuit denoising / LASSO via FISTA (Beck-Teboulle 2009) on the unit-atom dictionary.
    min_c 0.5||x - A^T c||^2 + lam ||c||_1. Returns dense coefficients [N, NA] (CPU fp16 to save memory)."""
    N, NA = X.shape[0], A.shape[0]
    if L is None:
        # Lipschitz constant = largest eigenvalue of A A^T (d x d)
        L = torch.linalg.eigvalsh(A.T @ A)[-1].item()
    out = torch.zeros(N, NA, dtype=torch.float16)
    for s in range(0, N, batch):
        x = X[s:s + batch]; n = x.shape[0]
        c = torch.zeros(n, NA, device=X.device); y = c.clone(); t = 1.0
        for it in range(iters):
            grad = (y @ A - x) @ A.T
            z = y - grad / L
            if nonneg: z = (z - lam / L).clamp_min(0)
            else: z = torch.sign(z) * (z.abs() - lam / L).clamp_min(0)
            t2 = (1 + math.sqrt(1 + 4 * t * t)) / 2
            y = z + ((t - 1) / t2) * (z - c)
            c, t = z, t2
        out[s:s + n] = c.to(torch.float16).cpu()
    return out


def cosamp(X, A, k, iters=10, batch=1024):
    """CoSaMP (Needell-Tropp 2009): identify 2k, merge, least squares, prune to k."""
    N, NA = X.shape[0], A.shape[0]
    sel = torch.zeros(N, k, dtype=torch.long, device=X.device); cof = torch.zeros(N, k, device=X.device)
    err = torch.zeros(N, device=X.device)
    for s in range(0, N, batch):
        x = X[s:s + batch]; n = x.shape[0]
        S = torch.zeros(n, 0, dtype=torch.long, device=X.device); r = x.clone()
        for it in range(iters):
            omega = (r @ A.T).abs().topk(2 * k, dim=1).indices
            T = torch.cat([S, omega], 1)
            # dedupe per row: sort and mask duplicates by pushing them to a dummy (use unique via sort)
            T, _ = T.sort(1)
            dup = torch.zeros_like(T, dtype=torch.bool); dup[:, 1:] = T[:, 1:] == T[:, :-1]
            # replace duplicates with random unused atoms (rare); simpler: keep duplicates, ridge handles singular Gram
            At = A[T]
            G = At @ At.transpose(1, 2) + 1e-4 * torch.eye(T.shape[1], device=X.device)
            c = torch.cholesky_solve(At @ x[:, :, None], torch.linalg.cholesky(G))[:, :, 0]
            c = c.masked_fill(dup, 0)
            top = c.abs().topk(k, dim=1).indices
            S = torch.gather(T, 1, top)
            cS, e = refit(x, A, S)
            r = x - torch.einsum("nk,nkd->nd", cS, A[S])
        sel[s:s + n] = S; cof[s:s + n] = cS; err[s:s + n] = e
    return sel, cof, err


# ----------------------------------------------------------------------------- controls & stats
def rotate(A, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(A.shape[1], A.shape[1], generator=g))
    return A @ Q.to(A.device)


def random_dict(NA, D, seed=11, dev=DEV):
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(NA, D, generator=g).to(dev)
    return R / R.norm(dim=-1, keepdim=True)


def typical_mask(Xraw):
    n = Xraw.norm(dim=-1)
    return n <= 10 * n.median()


def fvu(err, X, mask=None):
    """Fraction of variance unexplained: sum residual energy / sum energy (X already centered)."""
    e = (X ** 2).sum(-1)
    if mask is not None: err, e = err[mask], e[mask]
    return (err.sum() / e.sum()).item()


def recall_curve(sel, true_rows, ks=(8, 16, 32, 64)):
    """sel [N,k] support in selection order; true_rows [N] dictionary rows. Returns dict k -> recall."""
    out = {}
    hit = sel == true_rows[:, None]
    for k in ks:
        if k <= sel.shape[1]: out[k] = hit[:, :k].any(1).float().mean().item()
    return out


def boot_ci(per_seq_vals, B=1000, seed=0):
    """per_seq_vals: [n_seq] tensor of per-sequence means -> (lo, hi) 95% percentile bootstrap."""
    g = torch.Generator().manual_seed(seed); v = per_seq_vals.float()
    n = len(v); idx = torch.randint(0, n, (B, n), generator=g)
    m = v[idx].mean(1)
    return (m.quantile(0.025).item(), m.quantile(0.975).item())


def record(exp, result, note=""):
    """Write results/<exp>.json and append one line to FINDINGS.log."""
    os.makedirs(RESULTS, exist_ok=True)
    result = dict(result); result["_exp"] = exp; result["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(result, open(os.path.join(RESULTS, f"{exp}.json"), "w"), indent=1, default=float)
    with open(os.path.join(RESULTS, "FINDINGS.log"), "a") as f:
        f.write(f"{result['_time']} {exp}: {note}\n")
    log(f"RECORDED {exp}: {note}")


def jdump(o):
    if isinstance(o, torch.Tensor): return o.tolist()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {str(k): jdump(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jdump(v) for v in o]
    return o


# ----------------------------------------------------------------------------- OMP output cache
def get_omp(c, L, k=64, A=None, X=None, extra=""):
    """OMP outputs (sel, cof, err) at level L for the full evaluation set, cached on disk."""
    p = os.path.join(RESULTS, "omp_cache", f"{c.tag}_L{L}_k{k}{extra}.pt")
    if os.path.exists(p):
        o = torch.load(p); return o["sel"].to(DEV), o["cof"].to(DEV), o["err"].to(DEV)
    if X is None: X = c.X(L)
    if A is None: A, _ = c.dictionary(L)
    sel, cof, err = omp(X, A, k)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f".{os.getpid()}.tmp"; torch.save(dict(sel=sel.cpu(), cof=cof.cpu(), err=err.cpu()), tmp); os.replace(tmp, p)
    return sel, cof, err


def mid(c):
    """The paper's middle layer: last block included in the state (GPT-2: 6, else NB//2)."""
    return 6 if c.fam == "gpt2" else c.NB // 2


def sub(N, n, seed=0):
    """Deterministic subsample of n indices out of N (sorted)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(N, generator=g)[:n].sort().values if n < N else torch.arange(N)
