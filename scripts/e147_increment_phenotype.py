"""e147: is a state-level never-neuron also unreadable in its own block's increment? Per neuron dominant on >= 20
typical tokens at the mid layer: state-level recall (cached OMP@64) vs increment-level recall (OMP@16 on the block
increment over the block atoms, and the dual projection@16). Fraction of state-level never-neurons that are
readable at the increment level (cross-block cause) vs unreadable there too (intrinsic)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); tb, tn = tb[:, 0], tn[:, 0]; row = c.atom_index(L, tb, tn).to(DEV); hit_state = (sel == row[:, None]).any(1)
hit_inc = torch.zeros(c.NT, dtype=torch.bool, device=DEV); hit_inc_dual = torch.zeros(c.NT, dtype=torch.bool, device=DEV)
for b in range(L + 1):
    m = (tb == b); idx = torch.nonzero(m)[:, 0]
    if len(idx) == 0: continue
    D_ = c.s["H"][b + 1][idx].float().to(DEV) - c.s["H"][b][idx].float().to(DEV); D_ = D_ - c.s["H"][b + 1].float().mean(0).to(DEV) + c.s["H"][b].float().mean(0).to(DEV)
    Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); mlp_rows = torch.nonzero(labb["type"] == T_MLP)[:, 0].to(DEV); r_b = mlp_rows[tn[idx].to(DEV)]
    Sb = Ab.T @ Ab; evb, Vb = torch.linalg.eigh(Sb); Wb = Vb @ torch.diag(1 / (evb + 1e-2 * evb[-1])) @ Vb.T
    so, _, _ = omp(D_, Ab, 16); sd, _, _ = oneshot(D_, Ab, 16, whiten=Wb); hit_inc[idx.to(DEV)] = (so == r_b[:, None]).any(1); hit_inc_dual[idx.to(DEV)] = (sd == r_b[:, None]).any(1)
key = (tb * c.DFF + tn).to(DEV)[typ]; uk, inv, cnt = key.unique(return_inverse=True, return_counts=True); m = cnt >= 20
agg = lambda v: torch.zeros(len(uk), device=DEV).index_add_(0, inv, v[typ].float()) / cnt
rs, ri, rd = agg(hit_state), agg(hit_inc), agg(hit_inc_dual)
never = m & (rs < 0.1); always = m & (rs > 0.9)
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
res = dict(model=tag, L=L, n=int(m.sum()), recall_state=hit_state[typ].float().mean().item(), recall_increment=hit_inc[typ].float().mean().item(), recall_increment_dual=hit_inc_dual[typ].float().mean().item(),
           spearman_state_vs_increment=spearman(rs[m], ri[m]), n_never=int(never.sum()), never_readable_in_increment=(ri[never] > 0.5).float().mean().item() if never.any() else None, never_increment_recall_mean=ri[never].mean().item() if never.any() else None,
           never_increment_dual_mean=rd[never].mean().item() if never.any() else None, always_increment_recall_mean=ri[always].mean().item() if always.any() else None, icc_increment=None)
p = ri[m]; n_ = cnt[m].float(); pbar = (p * n_).sum() / n_.sum(); vb = p.var().item(); vbin = (pbar * (1 - pbar) / n_).mean().item(); res["icc_increment"] = (vb - vbin) / max(vb, 1e-9)
record(f"e147_incpheno_{tag}", res, f"recall state {res['recall_state']:.3f} increment {res['recall_increment']:.3f} (dual {res['recall_increment_dual']:.3f}) | per-neuron Spearman state vs increment {res['spearman_state_vs_increment']:.2f}; ICC at the increment level {res['icc_increment']:.2f} | state-never neurons (n={res['n_never']}): readable in their increment {res['never_readable_in_increment']} (mean increment recall {res['never_increment_recall_mean']}, dual {res['never_increment_dual_mean']}); always-neurons increment recall {res['always_increment_recall_mean']}")
