"""e211: does WDD read causal provenance or just the direction? Per token (dominant block-b write, b = 1..3): the
write-dependence of the state's component along its direction at level L (clean minus write-ablated, over clean),
and whether the dual@64 / OMP@64 decoder at level L reads that atom. If reading tracks dependence, identified
atoms are causal contributors; if not, WDD reads the direction whatever its cause. Reports Spearman(dependence,
z-score of the atom), recall in dependence quartiles, and the mean dependence of read vs unread tokens."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; lab = c.d["lab"]
A, labL = c.dictionary(L); mu = c.s["mu"][L + 1].to(DEV); S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
def run(b=None, neuron=None):
    store = {}; hs = [arch.layers[L].register_forward_hook(lambda m, i, o: store.__setitem__("H", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))]
    if b is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    model(ids_seq); [h.remove() for h in hs]; return store["H"]
H0 = run(); typ = typical_mask(H0); X0 = H0 - mu; sel_d = oneshot(X0, A, 64, whiten=Winv)[0]; sel_o = omp(X0, A, 64)[0]; res = {}
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
for b in (1, 2, 3):
    store_a = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: store_a.__setitem__("a", inp[0].detach().float().reshape(-1, c.DFF))); model(ids_seq); h.remove()
    led = store_a["a"] * c.d["WN"][b].to(DEV)[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; row = c.atom_index(L, torch.full_like(tn.cpu(), b), tn.cpu()).to(DEV); d = A[row]; big = typ & (tc.abs() >= tc.abs().quantile(0.5))
    H1 = run(b, tn); p0 = (H0 * d).sum(1); p1 = (H1 * d).sum(1); dep = ((p0 - p1) / p0.abs().clamp_min(1e-3)).clamp(-1, 2); hit_d = (sel_d == row[:, None]).any(1); hit_o = (sel_o == row[:, None]).any(1); prom = (X0 * d).sum(1).abs() / X0.norm(dim=1)
    q = dep[big].quantile(torch.tensor([0.25, 0.5, 0.75], device=DEV)); qi = (dep[:, None] > q[None]).sum(1)
    rec = dict(n=int(big.sum()), rho_dep_hit_dual=spearman(dep[big], hit_d[big].float()), rho_dep_hit_omp=spearman(dep[big], hit_o[big].float()), rho_dep_prom=spearman(dep[big], prom[big]), rho_prom_hit_dual=spearman(prom[big], hit_d[big].float()), dep_read=dep[big & hit_d].median().item(), dep_unread=dep[big & ~hit_d].median().item(), recall_by_dep_quartile=[hit_d[big & (qi == k)].float().mean().item() for k in range(4)], dep_quartile_medians=[dep[big & (qi == k)].median().item() for k in range(4)])
    res[b] = rec
    log(f"{tag} born b{b} (n {rec['n']}): Spearman(dependence, read) dual {rec['rho_dep_hit_dual']:+.2f} omp {rec['rho_dep_hit_omp']:+.2f}; (dependence, prominence) {rec['rho_dep_prom']:+.2f}; (prominence, read) {rec['rho_prom_hit_dual']:+.2f} | dependence when read {rec['dep_read']:.2f} vs unread {rec['dep_unread']:.2f} | dual recall by dependence quartile " + " ".join(f"{v:.2f}" for v in rec['recall_by_dep_quartile']) + " (quartile medians " + " ".join(f"{v:.2f}" for v in rec['dep_quartile_medians']) + ")")
import numpy as np
m = lambda k: float(np.mean([res[b][k] for b in res]))
record(f"e211_readdep_{tag}", dict(model=tag, L=L, per_birth={str(k): v for k, v in res.items()}), f"Spearman(write-dependence at level {L}, atom read): dual {m('rho_dep_hit_dual'):+.2f}, omp {m('rho_dep_hit_omp'):+.2f} | (dependence, prominence) {m('rho_dep_prom'):+.2f} | (prominence, read) {m('rho_prom_hit_dual'):+.2f} | dependence when read {m('dep_read'):.2f} vs unread {m('dep_unread'):.2f} | dual recall by dependence quartile: " + " ".join(f"{np.mean([res[b]['recall_by_dep_quartile'][k] for b in res]):.2f}" for k in range(4)))
