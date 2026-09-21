"""e65: the linear readability bound (Wiener 1949 / ridge). For 200 neurons that are dominant on >= 20 tokens,
the best LINEAR estimator of the neuron's true coefficient from the centered state (ridge, fit on odd sequences,
tested on even) gives an out-of-sample R^2: how much of each write is linearly readable at all, vs WDD's
identification rate and the neuron's mean prominence. Which neurons are unreadable by any linear reading?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); hit = (sel == row[:, None]).any(1)
key = (tb[:, 0] * c.DFF + tn[:, 0]).to(DEV)[typ]; uk, inv, cnt = key.unique(return_inverse=True, return_counts=True); rec = torch.zeros(len(uk), device=DEV).index_add_(0, inv, hit[typ].float()) / cnt
g = torch.Generator().manual_seed(2); cand = torch.nonzero(cnt >= 20)[:, 0]; pick = cand[torch.randperm(len(cand), generator=g)[:200].to(DEV)]
seq = (torch.arange(c.NT) // CTX).to(DEV); tr, te = typ & (seq % 2 == 1), typ & (seq % 2 == 0)
led = c.ledger(L); Xt, Xe = X[tr], X[te]; G = Xt.T @ Xt + 1e-1 * Xt.shape[0] * torch.eye(c.D, device=DEV); Ginv = torch.linalg.inv(G)
r2s, recs, proms = [], [], []
for k in pick.tolist():
    b, j = int(uk[k]) // c.DFF, int(uk[k]) % c.DFF; y = led[b][:, j].to(DEV); yt, ye = y[tr], y[te]
    w = Ginv @ (Xt.T @ (yt - yt.mean())); pred = Xe @ w + yt.mean(); r2 = 1 - ((ye - pred) ** 2).sum() / ((ye - ye.mean()) ** 2).sum()
    r2s.append(r2.item()); recs.append(rec[k].item()); m = typ & (key.new_tensor(0) == 0) if False else None
    d = A[c.atom_index(L, torch.tensor([b]), torch.tensor([j])).to(DEV)][0]; proms.append(((X * d).sum(1).abs() / X.norm(dim=1))[typ & (tb[:, 0].to(DEV) == b) & (tn[:, 0].to(DEV) == j)].mean().item())
R2, RC, PR = torch.tensor(r2s), torch.tensor(recs), torch.tensor(proms)
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
res = dict(model=tag, L=L, n=len(pick), r2_median=R2.median().item(), r2_q10=R2.quantile(0.1).item(), r2_q90=R2.quantile(0.9).item(), frac_r2_above_0p5=(R2 > 0.5).float().mean().item(),
           spearman_r2_vs_recall=spearman(R2, RC), spearman_r2_vs_prominence=spearman(R2, PR), r2_of_never=(R2[RC < 0.1].median().item() if (RC < 0.1).any() else None), r2_of_always=(R2[RC > 0.9].median().item() if (RC > 0.9).any() else None))
record(f"e65_wiener_{tag}", res, f"linear R2 of the true coefficient from the state: median {res['r2_median']:.2f} q10 {res['r2_q10']:.2f} q90 {res['r2_q90']:.2f} (>0.5: {res['frac_r2_above_0p5']:.2f}) | Spearman(R2, WDD recall) {res['spearman_r2_vs_recall']:.2f} (R2, prominence) {res['spearman_r2_vs_prominence']:.2f} | R2 of never-neurons {res['r2_of_never']} always {res['r2_of_always']}")
