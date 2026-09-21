"""e214: direction lottery and norm sweep. Perturb the input of block B = b+1 (b = 2, 5, 8) along nine direction
classes at eps = 0.1 ||x||: random; the token's dominant block-b MLP write; another active block-b write; the
block-b attention write; the top principal direction of the state; a middle and a bottom principal direction;
the token's own state direction (prediction: gain 0, normalization is scale-invariant); the token's embedding.
Norm sweep for random and dominant-write directions at relative sizes 0.001, 0.01, 0.1, 1. Gains are the change
of the block output along the direction per unit perturbation (minus the identity), MLP and attention parts.
Kill rule: the classes are 'direction-independent' if their block gains agree within 0.05."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; lab = c.d["lab"]; A = c.d["A"]
def rows(b): return A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
def mlp_module(b):
    lin = arch.mlp_lin(b)
    for name, mod in arch.layers[b].named_modules():
        if any(ch is lin for ch in mod.children()): return mod
def run(B, delta=None):
    store = {}; hs = [arch.layers[B].register_forward_hook(lambda m, i, o: store.__setitem__("out", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D))), mlp_module(B).register_forward_hook(lambda m, i, o: store.__setitem__("mlp", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D))), arch.layers[B].register_forward_pre_hook(lambda m, args, kwargs: store.__setitem__("in", (args[0] if len(args) > 0 else kwargs["hidden_states"]).detach().float().reshape(-1, c.D)), with_kwargs=True)]
    if delta is not None:
        def pre(m, args, kwargs):
            if len(args) > 0: return (args[0] + delta.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + delta.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        hs.append(arch.layers[B].register_forward_pre_hook(pre, with_kwargs=True))
    model(ids_seq); [h.remove() for h in hs]; return store
def unit(v): return v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
torch.manual_seed(0); out = {}; emb = arch.emb[0].detach().float()[ids_seq.reshape(-1)]
for b in [bb for bb in (2, 5, 8) if bb + 1 < c.NB]:
    B = b + 1; S0 = run(B); x = S0["in"]; typ = typical_mask(x); nrm = x.norm(dim=1, keepdim=True)
    led = c.acts[b][:NT].float().to(DEV) * c.d["WN"][b].to(DEV)[None]; tn = led.abs().argmax(1); active = led.abs() >= 0.05 * led.abs().max(1, keepdim=True).values; active[torch.arange(NT), tn] = False
    other = (active.float() * torch.rand(NT, c.DFF, device=DEV)).argmax(1); R = rows(b)
    Xs = c.X(b)[sub(c.NT, 8192, seed=3)]; Sig = Xs.T @ Xs / len(Xs); ev, V = torch.linalg.eigh(Sig)
    dirs = {"random": unit(torch.randn_like(x)), "mlp_dominant": R[tn], "mlp_other_active": R[other], "attention_write": unit(c.s["ATT"][b][:NT].float().to(DEV)), "pc_top": V[:, -1][None].expand_as(x), "pc_middle": V[:, c.D // 2][None].expand_as(x), "pc_bottom": V[:, 0][None].expand_as(x), "state_direction": unit(x), "embedding": unit(emb)}
    rec = {}
    for nm, U in dirs.items():
        eps = 0.1 * nrm; S1 = run(B, eps * U); g_blk = ((S1["out"] - S0["out"]) * U).sum(1) / eps[:, 0] - 1.0; g_mlp = ((S1["mlp"] - S0["mlp"]) * U).sum(1) / eps[:, 0]
        rec[nm] = dict(block=g_blk[typ].median().item(), mlp=g_mlp[typ].median().item(), attention=(g_blk - g_mlp)[typ].median().item(), iqr=(g_blk[typ].quantile(0.75) - g_blk[typ].quantile(0.25)).item())
    sweep = {}
    for nm in ("random", "mlp_dominant"):
        U = dirs[nm]; sweep[nm] = {}
        for rel in (0.001, 0.01, 0.1, 1.0):
            eps = rel * nrm; S1 = run(B, eps * U); sweep[nm][rel] = (((S1["out"] - S0["out"]) * U).sum(1) / eps[:, 0] - 1.0)[typ].median().item()
    out[b] = dict(gains=rec, sweep=sweep)
    log(f"{tag} block {B} (write block {b}): block gain by direction class: " + " ".join(f"{k}:{v['block']:+.2f}(mlp {v['mlp']:+.2f}, att {v['attention']:+.2f})" for k, v in rec.items()) + " | norm sweep (rel 0.001/0.01/0.1/1): random " + " ".join(f"{sweep['random'][r]:+.2f}" for r in (0.001, 0.01, 0.1, 1.0)) + " dominant " + " ".join(f"{sweep['mlp_dominant'][r]:+.2f}" for r in (0.001, 0.01, 0.1, 1.0)))
import numpy as np
cls = list(out[next(iter(out))]["gains"].keys()); mean_gain = {k: float(np.mean([out[b]["gains"][k]["block"] for b in out])) for k in cls}; spread = max(mean_gain[k] for k in cls if k != "state_direction") - min(mean_gain[k] for k in cls if k != "state_direction")
record(f"e214_lottery_{tag}", dict(model=tag, per_block={str(k): v for k, v in out.items()}, mean_gain=mean_gain, spread_excluding_state=spread), "mean block gain by class: " + " ".join(f"{k}:{v:+.2f}" for k, v in mean_gain.items()) + f" | spread across classes (excluding the state direction) {spread:.2f} | norm sweep random (mean over blocks): " + " ".join(f"{np.mean([out[b]['sweep']['random'][r] for b in out]):+.2f}" for r in (0.001, 0.01, 0.1, 1.0)) + " dominant: " + " ".join(f"{np.mean([out[b]['sweep']['mlp_dominant'][r] for b in out]):+.2f}" for r in (0.001, 0.01, 0.1, 1.0)))
