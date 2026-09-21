"""e215: composition law. Single-block gains g_j (random directions, eps = 0.1 ||x||) for every block, then the
survival of a perturbation injected at the input of block B after k blocks, observed vs the product
prod_{j=B..B+k}(1+g_j). Two injected directions: random and the dominant block-(B-1) MLP write (a real-write-like
perturbation). Kill rule: the composition fails if the prediction misses the observed survival by more than 2x at
k = 3 for random directions."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 6; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; lab = c.d["lab"]; A = c.d["A"]
def rows(b): return A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
def run(B=None, delta=None):
    store = {}; hs = []
    for j in range(NB):
        hs.append(arch.layers[j].register_forward_hook((lambda j_: lambda m, i, o: store.__setitem__(("out", j_), (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(j)))
        hs.append(arch.layers[j].register_forward_pre_hook((lambda j_: lambda m, args, kwargs: store.__setitem__(("in", j_), (args[0] if len(args) > 0 else kwargs["hidden_states"]).detach().float().reshape(-1, c.D)))(j), with_kwargs=True))
    if delta is not None:
        def pre(m, args, kwargs):
            if len(args) > 0: return (args[0] + delta.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + delta.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        hs.append(arch.layers[B].register_forward_pre_hook(pre, with_kwargs=True))
    model(ids_seq); [h.remove() for h in hs]; return store
torch.manual_seed(0); S0 = run(); g = []
for j in range(NB):
    x = S0[("in", j)]; typ = typical_mask(x); U = torch.randn_like(x); U = U / U.norm(dim=1, keepdim=True); eps = 0.1 * x.norm(dim=1, keepdim=True); S1 = run(j, eps * U); g.append((((S1[("out", j)] - S0[("out", j)]) * U).sum(1) / eps[:, 0] - 1.0)[typ].median().item())
log(f"{tag} single-block gains: " + " ".join(f"b{j}:{v:+.2f}" for j, v in enumerate(g)))
res = {}
for B in [bb for bb in (2, 5, 8) if bb + 4 < NB]:
    x = S0[("in", B)]; typ = typical_mask(x); eps = 0.1 * x.norm(dim=1, keepdim=True); led = c.acts[B - 1][:NT].float().to(DEV) * c.d["WN"][B - 1].to(DEV)[None]; tn = led.abs().argmax(1)
    for nm, U in (("random", torch.randn_like(x)), ("dominant_write", rows(B - 1)[tn])):
        U = U / U.norm(dim=1, keepdim=True); S1 = run(B, eps * U); obs = {}; pred = {}; p = 1.0
        for k in range(0, min(8, NB - B)):
            p *= (1 + g[B + k]); obs[k] = (((S1[("out", B + k)] - S0[("out", B + k)]) * U).sum(1) / eps[:, 0])[typ].median().item(); pred[k] = p
        res[f"B{B}_{nm}"] = dict(observed=obs, predicted=pred, log_ratio={k: math.log(max(obs[k], 1e-3) / max(pred[k], 1e-3)) for k in obs})
        log(f"{tag} inject at block {B} ({nm}): survival observed vs product prediction: " + " ".join(f"k{k}: {obs[k]:.2f}/{pred[k]:.2f}" for k in obs))
import numpy as np
lr3 = {nm: float(np.mean([abs(res[k]["log_ratio"][3]) for k in res if k.endswith(nm) and 3 in res[k]["log_ratio"]])) for nm in ("random", "dominant_write")}
lr7 = {nm: float(np.mean([abs(res[k]["log_ratio"][7]) for k in res if k.endswith(nm) and 7 in res[k]["log_ratio"]])) for nm in ("random", "dominant_write")}
record(f"e215_composition_{tag}", dict(model=tag, gains=g, results=res), f"single-block gains b0..: " + " ".join(f"{v:+.2f}" for v in g) + f" | mean |log(observed/predicted)| at k=3: random {lr3['random']:.2f}, dominant write {lr3['dominant_write']:.2f}; at k=7: random {lr7['random']:.2f}, dominant {lr7['dominant_write']:.2f} | " + " | ".join(f"{k}: obs " + " ".join(f"{v['observed'][i]:.2f}" for i in sorted(v['observed'])) + " pred " + " ".join(f"{v['predicted'][i]:.2f}" for i in sorted(v['predicted'])) for k, v in res.items()))
