"""Shared code for the looped-WDD study: an fp32 forward for Ouro-1.4B (24 layers looped 4 times, sandwich RMSNorms,
RMSNorm with gain between loops, exit gate) and for a standard Llama-style model (SmolLM2-1.7B), with a callback at
every (loop, layer) point that can read or replace the residual stream; the weight dictionary with the branch-norm
gains folded into the atoms; batched OMP, one-shot ranking and least-squares refit (after wdd_common.py)."""
import os, sys, json, time, math
import torch, torch.nn.functional as F
from safetensors.torch import load_file

ROOT = "/data/loopwdd"
DEVM = torch.device(os.environ.get("LW_DEVM", "cuda:1"))   # model
DEVD = torch.device(os.environ.get("LW_DEVD", "cuda:0"))   # dictionary and pursuit
if os.environ.get("LW_THREADS"): torch.set_num_threads(int(os.environ["LW_THREADS"]))
REPOS = dict(ouro="ByteDance/Ouro-1.4B", smol="HuggingFaceTB/SmolLM2-1.7B")
T_EMB, T_MLP, T_ATT = 0, 1, 2
torch.backends.cuda.matmul.allow_tf32 = False


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(o, path):
    def conv(x):
        if isinstance(x, torch.Tensor): return x.tolist()
        if isinstance(x, dict): return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [conv(v) for v in x]
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return None
        return x
    with open(path, "w") as f: json.dump(conv(o), f, indent=1)


def rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


class LM:
    """kind 'ouro': x += g1b*rms(attn(g1*rms(x))); x += g2b*rms(mlp(g2*rms(x))); after each pass of the stack
    x = gf*rms(x), which is both the next loop's input and that loop's exit state. kind 'llama': pre-norm only."""
    def __init__(self, kind, dev=DEVM):
        from huggingface_hub import snapshot_download
        self.kind, self.dev = kind, dev
        p = snapshot_download(REPOS[kind]); cfg = json.load(open(os.path.join(p, "config.json"))); self.cfg = cfg
        sd = {}
        for f in sorted(os.listdir(p)):
            if f.endswith(".safetensors"): sd.update(load_file(os.path.join(p, f)))
        g = lambda k: sd.pop(k).to(dev, torch.float32)
        self.L, self.d = cfg["num_hidden_layers"], cfg["hidden_size"]
        self.nh = cfg["num_attention_heads"]; self.hd = cfg.get("head_dim") or self.d // self.nh
        self.nkv = cfg.get("num_key_value_heads", self.nh); self.eps = cfg["rms_norm_eps"]
        self.T = cfg.get("total_ut_steps", 1); self.ffn = cfg["intermediate_size"]
        self.E = g("model.embed_tokens.weight")
        self.H = g("lm_head.weight") if "lm_head.weight" in sd else self.E
        self.nf = g("model.norm.weight")
        if kind == "ouro": self.gw = g("model.early_exit_gate.weight"); self.gb = g("model.early_exit_gate.bias")
        self.layers = []
        for l in range(self.L):
            q = f"model.layers.{l}."
            ly = dict(q=g(q + "self_attn.q_proj.weight"), k=g(q + "self_attn.k_proj.weight"), v=g(q + "self_attn.v_proj.weight"),
                      o=g(q + "self_attn.o_proj.weight"), n1=g(q + "input_layernorm.weight"), n2=g(q + "post_attention_layernorm.weight"),
                      gate=g(q + "mlp.gate_proj.weight"), up=g(q + "mlp.up_proj.weight"), down=g(q + "mlp.down_proj.weight"))
            if kind == "ouro": ly.update(n1b=g(q + "input_layernorm_2.weight"), n2b=g(q + "post_attention_layernorm_2.weight"))
            self.layers.append(ly)
        self.left = sorted(sd.keys())   # anything not used (checked by the validation script)
        theta = cfg.get("rope_theta", 10000.0)
        self.inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2, device=dev, dtype=torch.float32) / self.hd))
        self._rope = {}
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(REPOS[kind])

    def rope(self, T):
        if T not in self._rope:
            f = torch.arange(T, device=self.dev, dtype=torch.float32)[:, None] * self.inv[None]
            e = torch.cat([f, f], -1); self._rope[T] = (e.cos(), e.sin())
        return self._rope[T]

    def layer(self, x, l, rec=None):
        ly, (B, T, d) = self.layers[l], x.shape
        h = rms(x, self.eps) * ly["n1"]
        q = (h @ ly["q"].T).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = (h @ ly["k"].T).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = (h @ ly["v"].T).view(B, T, self.nkv, self.hd).transpose(1, 2)
        cos, sin = self.rope(T)
        rh = lambda u: torch.cat([-u[..., self.hd // 2:], u[..., :self.hd // 2]], -1)
        q, k = q * cos + rh(q) * sin, k * cos + rh(k) * sin
        if self.nkv != self.nh: k, v = k.repeat_interleave(self.nh // self.nkv, 1), v.repeat_interleave(self.nh // self.nkv, 1)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, self.nh * self.hd)
        a = z @ ly["o"].T
        if self.kind == "ouro": a = rms(a, self.eps) * ly["n1b"]
        x = x + a
        h = rms(x, self.eps) * ly["n2"]
        act = F.silu(h @ ly["gate"].T) * (h @ ly["up"].T)
        m = act @ ly["down"].T
        if self.kind == "ouro":
            s = torch.rsqrt(m.pow(2).mean(-1, keepdim=True) + self.eps)
            if rec is not None: rec(l, act * s)      # neuron j's write is act_j * s * (n2b * down[:, j])
            m = m * s * ly["n2b"]
        elif rec is not None: rec(l, act)
        return x + m

    def run(self, ids, loops=None, cb=None, rec=None):
        """cb(t, l, x) is called before layer l of loop t (l = L: after the last layer, before the between-loop norm);
        it may return a replacement x. rec(t, l, coef) receives the per-neuron write coefficients. Returns the normed
        exit state of every loop (the input of the next loop)."""
        return self.run_from(self.E[ids], 0, 0, loops, cb, rec, first=True)

    def run_from(self, x, t0, l0, loops=None, cb=None, rec=None, first=False):
        """Continue from the residual state x before layer l0 of loop t0 (the state at a point determines everything
        downstream). cb is not called at the starting point unless first. Returns the exits of loops t0..loops-1."""
        loops = loops or self.T; res = []
        for t in range(t0, loops):
            for l in range(l0 if t == t0 else 0, self.L + 1):
                if cb is not None and (first or t != t0 or l != l0):
                    y = cb(t, l, x)
                    if y is not None: x = y
                if l < self.L: x = self.layer(x, l, rec=(lambda l_, c, t=t: rec(t, l_, c)) if rec is not None else None)
            x = rms(x, self.eps) * self.nf
            res.append(x)
        return res

    def ce(self, h, ids):
        """Per-position next-token cross-entropy from an exit state h [B, T, d]; returns [B, T-1]."""
        B, T, _ = h.shape; out = torch.empty(B, T - 1, device=h.device)
        for b in range(B):
            lg = h[b, :-1] @ self.H.T
            out[b] = F.cross_entropy(lg, ids[b, 1:], reduction="none")
        return out

    def gate(self, h):
        return torch.sigmoid(h @ self.gw.T + self.gb)[..., 0]

    # ---- the weight dictionary ----
    def atoms(self, layers, tilt=0, emb=True, heads=True, dev=DEVD):
        """Unit atoms and labels. MLP atom (l, j) = n2b_l * down_l[:, j]; head h of layer l: orthonormal basis of the
        column space of n1b_l * o_l[:, h]; embedding rows. tilt: multiply every atom by gf**tilt elementwise
        (the direction a write of that atom has after `tilt` between-loop norms)."""
        A, typ, lay, idx = [], [], [], []
        gt = self.nf.to(dev) ** tilt if tilt else None
        def add(M, ty, l, ix):
            M = M.to(dev)
            if gt is not None: M = M * gt[None]
            A.append(unit(M)); typ.extend([ty] * M.shape[0]); lay.extend([l] * M.shape[0]); idx.extend(ix)
        if emb: add(self.E, T_EMB, -1, list(range(self.E.shape[0])))
        for l in layers:
            ly = self.layers[l]
            W = ly["down"].T                                  # [ffn, d]
            if self.kind == "ouro": W = W * ly["n2b"][None]
            add(W, T_MLP, l, list(range(W.shape[0])))
            if heads:
                O = ly["o"] * ly["n1b"][:, None] if self.kind == "ouro" else ly["o"]
                for hh in range(self.nh):
                    U = torch.linalg.svd(O[:, hh * self.hd:(hh + 1) * self.hd], full_matrices=False)[0]   # [d, hd]
                    add(U.T, T_ATT, l, [hh * self.hd + r for r in range(self.hd)])
        lab = dict(type=torch.tensor(typ), layer=torch.tensor(lay), index=torch.tensor(idx))
        return torch.cat(A), lab


def corpus(tok, n_seq, T=256, start=0, split="test"):
    """WikiText-2 raw, concatenated, cut into sequences of T tokens with BOS (id 0) first."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = tok("".join(ds["text"]), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n = T - 1; out = []
    for i in range(start, start + n_seq):
        out.append(torch.cat([torch.tensor([0]), ids[i * n:(i + 1) * n]]))
    return torch.stack(out)


# ---- pursuit ----
def omp(X, A, k, batch=256, ridge=1e-5):
    """Batched OMP (as wdd_common.omp). X [N, d] on A's device. Returns sel [N, k], cof [N, k], err [N, k]."""
    N, NA = X.shape[0], A.shape[0]; dev = A.device
    sel = torch.zeros(N, k, dtype=torch.long, device=dev); cof = torch.zeros(N, k, device=dev); err = torch.zeros(N, k, device=dev)
    eye = torch.eye(k, device=dev)
    for s in range(0, N, batch):
        x = X[s:s + batch].to(dev); n = x.shape[0]; r = x.clone()
        S = torch.zeros(n, 0, dtype=torch.long, device=dev); taken = torch.zeros(n, NA, dtype=torch.bool, device=dev)
        for step in range(k):
            corr = (r @ A.T).abs_().masked_fill_(taken, -1.0)
            pick = corr.argmax(-1, keepdim=True); del corr
            taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1); As = A[S]
            G = As @ As.transpose(1, 2) + ridge * eye[:step + 1, :step + 1]
            c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G))
            r = x - (c.transpose(1, 2) @ As)[:, 0]
            err[s:s + n, step] = (r ** 2).sum(-1)
        sel[s:s + n] = S; cof[s:s + n] = c[:, :, 0]; del taken
    return sel, cof, err


def refit(X, A, sel, ridge=1e-5, chunk=1024):
    """Least-squares refit on a given support sel [N, k] (chunked). Returns cof [N, k] and residual energy [N]."""
    k = sel.shape[1]; eye = ridge * torch.eye(k, device=A.device); cs, es = [], []
    for s in range(0, sel.shape[0], chunk):
        As = A[sel[s:s + chunk]]; x = X[s:s + chunk].to(A.device)
        c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(As @ As.transpose(1, 2) + eye))[:, :, 0]
        cs.append(c); es.append((x - torch.einsum("nk,nkd->nd", c, As)).pow(2).sum(-1))
    return torch.cat(cs), torch.cat(es)


def oneshot(X, A, k, batch=512):
    sel = torch.zeros(X.shape[0], k, dtype=torch.long, device=A.device)
    for s in range(0, X.shape[0], batch):
        sel[s:s + batch] = (X[s:s + batch].to(A.device) @ A.T).abs().topk(k, dim=1).indices
    c, e = refit(X, A, sel)
    return sel, c, e


def recon(A, sel, cof, chunk=1024):
    return torch.cat([torch.einsum("nk,nkd->nd", cof[s:s + chunk], A[sel[s:s + chunk]]) for s in range(0, sel.shape[0], chunk)])


def rotation(d, seed=7, dev=DEVD):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, d, generator=g))[0].to(dev)
