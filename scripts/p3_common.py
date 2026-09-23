"""Shared helpers for phase 3 (e374 onward): per-head query/key/value input maps for the three families, the reader's
frozen-scale input norm, per-head output capture, and head-scale hooks."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *

def attn_mod(arch, l):
    L = arch.layers[l]
    return L.attn if arch.fam == "gpt2" else (L.self_attn if arch.fam == "llama" else L.attention)

def in_norm(arch, l):
    L = arch.layers[l]
    return L.ln_1 if arch.fam == "gpt2" else L.input_layernorm

def head_in_map(arch, l, h, which):
    """[D, HD] map from the normed residual to head h's query / key / value (bias excluded, before any rotary)."""
    D, HD, NH = arch.D, arch.HD, arch.NH; A = attn_mod(arch, l)
    if arch.fam == "gpt2":
        W = A.c_attn.weight.detach().float(); o = {"Q": 0, "K": D, "V": 2 * D}[which]; return W[:, o + h * HD:o + (h + 1) * HD]
    if arch.fam == "llama":
        mod = {"Q": A.q_proj, "K": A.k_proj, "V": A.v_proj}[which]; W = mod.weight.detach().float()
        if which == "Q": return W[h * HD:(h + 1) * HD].T
        nkv = W.shape[0] // HD; g = h // (NH // nkv); return W[g * HD:(g + 1) * HD].T
    W = A.query_key_value.weight.detach().float(); o = {"Q": 0, "K": HD, "V": 2 * HD}[which]; return W[h * 3 * HD + o:h * 3 * HD + o + HD].T

def norm_frozen(arch, l, X):
    """Per-row frozen-scale linearisation of the reader's input norm at states X [P, D]: returns (gamma [D], inv_scale [P], centre flag)
    so that norm_lin(v) = gamma * (v - centre * mean(v)) * inv_scale."""
    m = in_norm(arch, l); name = type(m).__name__.lower(); eps = getattr(m, "eps", getattr(m, "variance_epsilon", 1e-5))
    if isinstance(eps, (tuple, list)): eps = eps[0]
    g = getattr(m, "weight", None); g = g.detach().float() if g is not None else torch.ones(X.shape[1], device=X.device)
    if "rms" in name: return g, (X.pow(2).mean(-1) + eps).rsqrt(), 0.0
    Xc = X - X.mean(-1, keepdim=True); return g, (Xc.pow(2).mean(-1) + eps).rsqrt(), 1.0

def reader_dirs(arch, l, h, which, X):
    """r_n such that <v, r_n> = <W^T norm_lin_n(v), u_n> / ||u_n||^2 with u_n = W^T norm_lin_n(x_n): the share of head h's
    query / key / value vector at position n contributed by a residual component v (additive over any decomposition of x_n)."""
    W = head_in_map(arch, l, h, which).to(X.device); g, inv, cen = norm_frozen(arch, l, X)
    Xl = g[None] * (X - cen * X.mean(-1, keepdim=True)) * inv[:, None]; u = Xl @ W; w = (u @ W.T) / u.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
    r = g[None] * w * inv[:, None]
    if cen: r = r - r.mean(-1, keepdim=True)
    return r, u

class Capture:
    """Forward-pass capture: residual input of every block, per-head attention inputs (z) and MLP outputs."""
    def __init__(self, model, arch):
        self.model, self.arch = model, arch
    def __call__(self, ids, blocks=None):
        arch = self.arch; blocks = range(arch.NB) if blocks is None else blocks; X, Z, M, hs = {}, {}, {}, []
        def mkx(l):
            def pre(m, args, kwargs):
                h = args[0] if len(args) > 0 else kwargs["hidden_states"]; X[l] = h.detach().float(); return None
            return pre
        for l in blocks:
            hs.append(arch.layers[l].register_forward_pre_hook(mkx(l), with_kwargs=True))
            hs.append(arch.attn_lin(l).register_forward_pre_hook((lambda l_: lambda m, a: Z.__setitem__(l_, a[0].detach().float()))(l)))
            L = arch.layers[l]; hs.append(L.mlp.register_forward_hook((lambda l_: lambda m, i, o: M.__setitem__(l_, (o[0] if isinstance(o, tuple) else o).detach().float()))(l)))
        try:
            with torch.no_grad(): out = self.model(ids, output_attentions=True)
        finally: [h.remove() for h in hs]
        return X, Z, M, out

def scale_hooks(arch, a):
    """Multiply every head's write by a[l * NH + h] (a on device); returns hook handles."""
    NH, hs = arch.NH, []
    for l in range(arch.NB):
        def pre(m, args, l=l):
            x = args[0]; B, T, W = x.shape; x = (x.reshape(B, T, NH, W // NH) * a[l * NH:(l + 1) * NH].to(x.dtype)[None, None, :, None]).reshape(B, T, W); return (x,) + tuple(args[1:])
        hs.append(arch.attn_lin(l).register_forward_pre_hook(pre))
    return hs
