"""e101 (meta): do WDD's support statistics recover real co-activation structure? Among the 300 most-selected MLP
atoms at the mid layer: the co-selection matrix (how often two atoms are in the same support) vs the true
co-firing matrix of the same neurons (ledger |c| >= 5% of the token max), as correlation of the off-diagonal
entries, and the same restricted to pairs with different blocks. Also: does co-selection track atom coherence
(geometry) more than co-firing (computation)?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
ismlp = typA[sel] == T_MLP; flat = sel[typ][ismlp[typ]]; uk, cnt = flat.unique(return_counts=True); top = uk[cnt.topk(300).indices]
S = torch.zeros(c.NT, 300, device=DEV); pos = {int(a): i for i, a in enumerate(top.tolist())}
for j in range(sel.shape[1]):
    col = sel[:, j]; m = torch.isin(col, top)
    if m.any(): S[torch.nonzero(m)[:, 0], torch.tensor([pos[int(v)] for v in col[m].tolist()], device=DEV)] = 1
led = c.ledger(L); Cled = torch.cat([led[b] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * Cled.abs().max(1, keepdim=True).values
key = (blkA[top] * c.DFF + idxA[top]); F = (Cled[:, key].abs() >= thr).float()
St, Ft = S[typ], F[typ]; CS = (St.T @ St) / typ.sum(); CF = (Ft.T @ Ft) / typ.sum()
def offdiag_corr(P, Q, mask=None):
    iu = torch.triu_indices(300, 300, 1, device=DEV); p, q = P[iu[0], iu[1]], Q[iu[0], iu[1]]
    if mask is not None: p, q = p[mask], q[mask]
    return torch.corrcoef(torch.stack([p, q]))[0, 1].item()
G = (A[top] @ A[top].T).abs(); iu = torch.triu_indices(300, 300, 1, device=DEV); diffblock = blkA[top][iu[0]] != blkA[top][iu[1]]
res = dict(model=tag, L=L, n_atoms=300, corr_coselect_vs_cofire=offdiag_corr(CS, CF), corr_coselect_vs_cofire_diffblock=offdiag_corr(CS, CF, diffblock), corr_coselect_vs_coherence=offdiag_corr(CS, G),
           corr_cofire_vs_coherence=offdiag_corr(CF, G), select_rate_vs_fire_rate_corr=torch.corrcoef(torch.stack([St.mean(0), Ft.mean(0)]))[0, 1].item(),
           mean_select_rate=St.mean().item(), mean_fire_rate=Ft.mean().item())
record(f"e101_cosel_{tag}", res, f"corr(co-selection, co-firing) {res['corr_coselect_vs_cofire']:.2f} (diff-block pairs {res['corr_coselect_vs_cofire_diffblock']:.2f}) | corr(co-selection, coherence) {res['corr_coselect_vs_coherence']:.2f} | corr(co-firing, coherence) {res['corr_cofire_vs_coherence']:.2f} | select-rate vs fire-rate {res['select_rate_vs_fire_rate_corr']:.2f}")
