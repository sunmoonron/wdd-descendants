"""e220: the spectral profile of the contraction. Perturb the input of block B along principal directions of the
state covariance at ranks spanning the spectrum (top 1, 2, 5, 1%, 5%, 10%, 25%, 50%, 75%, 90%, bottom 1%, bottom)
and measure the block gain; Spearman between gain and log eigenvalue across ranks. Kill rule: if the gain is flat
across the spectrum (spread < 0.05), the contraction is not confined to the occupied subspace."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX
def run(B, delta=None):
    store = {}; hs = [arch.layers[B].register_forward_hook(lambda m, i, o: store.__setitem__("out", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D))), arch.layers[B].register_forward_pre_hook(lambda m, args, kwargs: store.__setitem__("in", (args[0] if len(args) > 0 else kwargs["hidden_states"]).detach().float().reshape(-1, c.D)), with_kwargs=True)]
    if delta is not None:
        def pre(m, args, kwargs):
            if len(args) > 0: return (args[0] + delta.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + delta.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        hs.append(arch.layers[B].register_forward_pre_hook(pre, with_kwargs=True))
    model(ids_seq); [h.remove() for h in hs]; return store
def spearman(a, b):
    a, b = torch.tensor(a), torch.tensor(b); ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
out = {}
for B in [bb for bb in (3, 6) if bb < c.NB]:
    S0 = run(B); x = S0["in"]; typ = typical_mask(x); eps = 0.1 * x.norm(dim=1, keepdim=True)
    Xs = c.X(B - 1)[sub(c.NT, 8192, seed=3)]; Sig = Xs.T @ Xs / len(Xs); ev, V = torch.linalg.eigh(Sig); D = c.D
    ranks = {"top1": D - 1, "top2": D - 2, "top5": D - 5, "1%": D - max(1, D // 100), "5%": D - D // 20, "10%": D - D // 10, "25%": D - D // 4, "50%": D // 2, "75%": D // 4, "90%": D // 10, "bottom1%": max(0, D // 100), "bottom": 0}
    rec = {}
    for nm, r in ranks.items():
        U = V[:, r][None].expand_as(x); S1 = run(B, eps * U); rec[nm] = dict(gain=(((S1["out"] - S0["out"]) * U).sum(1) / eps[:, 0] - 1.0)[typ].median().item(), log_eig=math.log(max(ev[r].item(), 1e-12)))
    out[B] = dict(profile=rec, spearman_gain_logeig=spearman([v["gain"] for v in rec.values()], [v["log_eig"] for v in rec.values()]), spread=max(v["gain"] for v in rec.values()) - min(v["gain"] for v in rec.values()))
    log(f"{tag} block {B}: gain by principal rank " + " ".join(f"{k}:{v['gain']:+.2f}" for k, v in rec.items()) + f" | Spearman(gain, log eigenvalue) {out[B]['spearman_gain_logeig']:+.2f} | spread {out[B]['spread']:.2f}")
import numpy as np
record(f"e220_spectral_{tag}", dict(model=tag, per_block={str(k): v for k, v in out.items()}), f"Spearman(gain, log eigenvalue) across 12 ranks: " + " ".join(f"b{B}:{v['spearman_gain_logeig']:+.2f}" for B, v in out.items()) + " | spread: " + " ".join(f"b{B}:{v['spread']:.2f}" for B, v in out.items()) + " | profile (block " + str(min(out)) + "): " + " ".join(f"{k}:{v['gain']:+.2f}" for k, v in out[min(out)]['profile'].items()))
