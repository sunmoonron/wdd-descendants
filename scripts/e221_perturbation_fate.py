"""e221: where does a perturbation go? Inject eps*u (random unit u) at the input of block B = 2 and 5; at k = 0, 1,
3, 7 blocks later measure the total surviving difference ||delta_k|| / eps (not just its projection on u), the
cosine between delta_k and u, and the spectral position of delta_k (fraction of its energy in the top 10% and the
bottom 50% of principal directions of the state at that level, vs the same fractions for u itself). If the
along-u survival floors (e215) because the perturbation rotates into weakly damped low-variance directions, the
total survival should exceed the along-u survival and delta_k should shift toward low-variance directions."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 6; ids_seq = c.s["eval_ids"][:NS].to(DEV); NB = c.NB
def run(B=None, delta=None):
    store = {}; hs = [arch.layers[j].register_forward_hook((lambda j_: lambda m, i, o: store.__setitem__(("out", j_), (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(j)) for j in range(NB)]
    hs.append(arch.layers[B if B is not None else 0].register_forward_pre_hook(lambda m, args, kwargs: store.__setitem__("in", (args[0] if len(args) > 0 else kwargs["hidden_states"]).detach().float().reshape(-1, c.D)), with_kwargs=True))
    if delta is not None:
        def pre(m, args, kwargs):
            if len(args) > 0: return (args[0] + delta.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + delta.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        hs.append(arch.layers[B].register_forward_pre_hook(pre, with_kwargs=True))
    model(ids_seq); [h.remove() for h in hs]; return store
torch.manual_seed(0); out = {}
for B in [bb for bb in (2, 5) if bb + 7 < NB]:
    S0 = run(B); x = S0["in"]; typ = typical_mask(x); U = torch.randn_like(x); U = U / U.norm(dim=1, keepdim=True); eps = 0.1 * x.norm(dim=1, keepdim=True); S1 = run(B, eps * U); rec = {}
    for k in (0, 1, 3, 7):
        lv = B + k; d = (S1[("out", lv)] - S0[("out", lv)]) / eps; along = (d * U).sum(1); total = d.norm(dim=1); cosu = along / total.clamp_min(1e-9)
        Xs = c.X(lv)[sub(c.NT, 8192, seed=3)]; Sig = Xs.T @ Xs / len(Xs); ev, V = torch.linalg.eigh(Sig); P = (d @ V) ** 2; Pu = (U @ V) ** 2; top = c.D - c.D // 10; bot = c.D // 2
        rec[k] = dict(along_u=along[typ].median().item(), total=total[typ].median().item(), cos=cosu[typ].median().item(), frac_top10=(P[:, top:].sum(1) / P.sum(1).clamp_min(1e-12))[typ].median().item(), frac_bottom50=(P[:, :bot].sum(1) / P.sum(1).clamp_min(1e-12))[typ].median().item(), u_frac_top10=(Pu[:, top:].sum(1))[typ].median().item(), u_frac_bottom50=(Pu[:, :bot].sum(1))[typ].median().item())
    out[B] = rec
    log(f"{tag} inject at block {B}: " + " | ".join(f"k{k}: along-u {v['along_u']:.2f}, total {v['total']:.2f}, cos {v['cos']:.2f}, energy in top-10% PCs {v['frac_top10']:.2f} (u: {v['u_frac_top10']:.2f}), bottom-50% {v['frac_bottom50']:.2f} (u: {v['u_frac_bottom50']:.2f})" for k, v in rec.items()))
import numpy as np
record(f"e221_fate_{tag}", dict(model=tag, per_block={str(k): v for k, v in out.items()}), " | ".join(f"B{B}: k7 along-u {v[7]['along_u']:.2f} vs total {v[7]['total']:.2f} (cos {v[7]['cos']:.2f}); top-10%-PC energy k0 {v[0]['frac_top10']:.2f} -> k7 {v[7]['frac_top10']:.2f}; bottom-50% k0 {v[0]['frac_bottom50']:.2f} -> k7 {v[7]['frac_bottom50']:.2f}" for B, v in out.items()))
