"""e216: is the 1/||x|| dependence of the contraction caused by the normalization's per-token scale? Freeze both
normalizations of block B (input and post-attention) to a constant per-token scale (the median over typical
tokens) and re-measure the generic gain along random directions by state-norm quartile. Prediction: the
Spearman(gain, log norm) of e213 (+0.44 to +0.62) vanishes while the median gain is unchanged. Kill rule: if the
correlation stays above 0.3 with frozen scales, the norm dependence is not the normalization."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX
def norms(B):
    l = arch.layers[B]; return [l.ln_1, l.ln_2] if fam == "gpt2" else [l.input_layernorm, l.post_attention_layernorm]
def scale_of(mod, x):
    eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-5)); eps = eps if isinstance(eps, float) else 1e-5
    if "RMS" in type(mod).__name__: return torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    xm = x.float() - x.float().mean(-1, keepdim=True); return torch.rsqrt(xm.pow(2).mean(-1, keepdim=True) + eps)
def run(B, delta=None, frozen=None):
    store = {}; hs = [arch.layers[B].register_forward_hook(lambda m, i, o: store.__setitem__("out", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D))), arch.layers[B].register_forward_pre_hook(lambda m, args, kwargs: store.__setitem__("in", (args[0] if len(args) > 0 else kwargs["hidden_states"]).detach().float().reshape(-1, c.D)), with_kwargs=True)]
    if delta is not None:
        def pre(m, args, kwargs):
            if len(args) > 0: return (args[0] + delta.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + delta.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        hs.append(arch.layers[B].register_forward_pre_hook(pre, with_kwargs=True))
    if frozen is not None:
        for mod, sc in zip(norms(B), frozen):
            def fh(m, i, o, sc=sc):
                x = i[0]; s = scale_of(m, x); bias = getattr(m, "bias", None)
                if bias is None: return o * (sc / s).to(o.dtype)
                return (o - bias) * (sc / s).to(o.dtype) + bias
            hs.append(mod.register_forward_hook(fh))
    model(ids_seq); [h.remove() for h in hs]; return store
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
torch.manual_seed(0); out = {}
for B in [bb for bb in (3, 6, 9) if bb < c.NB]:
    S0 = run(B); x = S0["in"]; typ = typical_mask(x); U = torch.randn_like(x); U = U / U.norm(dim=1, keepdim=True); eps = 0.1 * x.norm(dim=1, keepdim=True); nrm = x.norm(dim=1)
    # constant scales: median per-token scale of each norm over typical tokens (input of the second norm = state after attention; approximate with the block input for the median)
    consts = [scale_of(mod, x)[typ].median() for mod in norms(B)]
    rec = {}
    for nm, fr in (("real", None), ("frozen", consts)):
        S0b = run(B, None, fr); S1 = run(B, eps * U, fr); gain = ((S1["out"] - S0b["out"]) * U).sum(1) / eps[:, 0] - 1.0
        q = nrm[typ].quantile(torch.tensor([0.25, 0.5, 0.75], device=DEV)); qi = (nrm[:, None] > q[None]).sum(1)
        rec[nm] = dict(median=gain[typ].median().item(), rho_lognorm=spearman(gain[typ], nrm[typ].log()), by_quartile=[gain[typ & (qi == k)].median().item() for k in range(4)])
    out[B] = rec
    log(f"{tag} block {B}: real norm: gain {rec['real']['median']:+.2f}, Spearman(gain, log norm) {rec['real']['rho_lognorm']:+.2f}, by quartile " + " ".join(f"{v:+.2f}" for v in rec['real']['by_quartile']) + f" | frozen scale: gain {rec['frozen']['median']:+.2f}, Spearman {rec['frozen']['rho_lognorm']:+.2f}, by quartile " + " ".join(f"{v:+.2f}" for v in rec['frozen']['by_quartile']))
import numpy as np
record(f"e216_frozen_{tag}", dict(model=tag, per_block={str(k): v for k, v in out.items()}), f"Spearman(gain, log norm): real {np.mean([out[b]['real']['rho_lognorm'] for b in out]):+.2f} -> frozen scale {np.mean([out[b]['frozen']['rho_lognorm'] for b in out]):+.2f} | median gain real {np.mean([out[b]['real']['median'] for b in out]):+.2f} vs frozen {np.mean([out[b]['frozen']['median'] for b in out]):+.2f} | quartile spread real {np.mean([out[b]['real']['by_quartile'][3] - out[b]['real']['by_quartile'][0] for b in out]):+.2f} frozen {np.mean([out[b]['frozen']['by_quartile'][3] - out[b]['frozen']['by_quartile'][0] for b in out]):+.2f}")
