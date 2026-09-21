"""e213: the derived formula (gain = normalization gain x tr(J)/D, with normalization gain ~ 1/||x||) predicts that
the per-token generic contraction is inversely proportional to the state norm. Per token: the gain of block b
along a random unit direction (block-input perturbation, eps = 0.1 ||x||) vs log ||x||, vs position, and by norm
quartile; Spearman correlations. Blocks 2, 5, 8."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV)
def run(b, delta=None):
    store = {}; hs = [arch.layers[b].register_forward_hook(lambda m, i, o: store.__setitem__("out", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))]
    hs.append(arch.layers[b].register_forward_pre_hook(lambda m, args, kwargs: store.__setitem__("in", (args[0] if len(args) > 0 else kwargs["hidden_states"]).detach().float().reshape(-1, c.D)), with_kwargs=True))
    if delta is not None:
        def pre(m, args, kwargs):
            if len(args) > 0: return (args[0] + delta.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + delta.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        hs.append(arch.layers[b].register_forward_pre_hook(pre, with_kwargs=True))
    model(ids_seq); [h.remove() for h in hs]; return store
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
torch.manual_seed(0); out = {}; pos = torch.arange(CTX, device=DEV).repeat(NS)
for b in [bb for bb in (2, 5, 8) if bb < c.NB]:
    S0 = run(b); x = S0["in"]; typ = typical_mask(x) & (pos > 0); U = torch.randn_like(x); U = U / U.norm(dim=1, keepdim=True); eps = 0.1 * x.norm(dim=1, keepdim=True); S1 = run(b, eps * U)
    g = ((S1["out"] - S0["out"]) * U).sum(1) / eps[:, 0] - 1.0; nrm = x.norm(dim=1); q = nrm[typ].quantile(torch.tensor([0.25, 0.5, 0.75], device=DEV)); qi = (nrm[:, None] > q[None]).sum(1)
    rec = dict(rho_gain_lognorm=spearman(g[typ], nrm[typ].log()), rho_gain_pos=spearman(g[typ], pos[typ].float()), rho_norm_pos=spearman(nrm[typ], pos[typ].float()), gain_by_norm_quartile=[g[typ & (qi == k)].median().item() for k in range(4)], norm_quartile_med=[nrm[typ & (qi == k)].median().item() for k in range(4)], gain_times_norm_by_quartile=[(g * nrm)[typ & (qi == k)].median().item() for k in range(4)])
    out[b] = rec
    log(f"{tag} block {b}: Spearman(gain, log norm) {rec['rho_gain_lognorm']:+.2f}, (gain, position) {rec['rho_gain_pos']:+.2f}, (norm, position) {rec['rho_norm_pos']:+.2f} | gain by norm quartile " + " ".join(f"{v:+.2f}" for v in rec['gain_by_norm_quartile']) + " (norms " + " ".join(f"{v:.0f}" for v in rec['norm_quartile_med']) + ") | gain x norm by quartile " + " ".join(f"{v:+.1f}" for v in rec['gain_times_norm_by_quartile']))
import numpy as np
record(f"e213_gainnorm_{tag}", dict(model=tag, per_block=out), f"Spearman(per-token gain, log state norm) mean {np.mean([out[b]['rho_gain_lognorm'] for b in out]):+.2f}; (gain, position) {np.mean([out[b]['rho_gain_pos'] for b in out]):+.2f} | gain by norm quartile (mean over blocks): " + " ".join(f"{np.mean([out[b]['gain_by_norm_quartile'][k] for b in out]):+.2f}" for k in range(4)) + " | gain x norm: " + " ".join(f"{np.mean([out[b]['gain_times_norm_by_quartile'][k] for b in out]):+.1f}" for k in range(4)))
