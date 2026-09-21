"""e12: calibrating the coefficient (Stein 1956 / James-Stein 1961 spirit). The identified dominant write is
over-estimated for small writes because the support absorbs what it leaves out. Fit a correction on half the
sequences, apply to the other half: (a) global multiplicative shrink, (b) log-linear in |c_hat|, (c) per-decile
median ratio; plus alternatives that need no fit: refit on the top-8 support atoms only, and heavier ridge."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c)
X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L); sel, cof, err = get_omp(c, L, A=A, X=X)
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); ct = tc[:, 0].to(DEV)
hit = (sel == row[:, None]); idn = hit.any(1) & typ; chat = (cof * hit).sum(1)
seq = (torch.arange(c.NT) // CTX).to(DEV); half = (seq % 2 == 0)
def relerr(est, m): return ((est[m] - ct[m]).abs() / ct[m].abs()).median().item()
def ratio_dec(est, m):
    q = ct[m].abs(); edges = q.quantile(torch.linspace(0, 1, 11, device=DEV)); r = est[m] / ct[m]
    return [r[(q >= edges[i]) & (q <= edges[i + 1])].median().item() for i in range(10)]
res = dict(model=tag, L=L, n_identified=int(idn.sum()), baseline=dict(relerr_B=relerr(chat, idn & ~half), ratio_deciles_B=ratio_dec(chat, idn & ~half)))
# (a) global shrink fit on A
mA, mB = idn & half, idn & ~half
s = (chat[mA] * ct[mA]).sum() / (chat[mA] ** 2).sum(); res["global_shrink"] = dict(s=s.item(), relerr_B=relerr(chat * s, mB))
# (b) log-linear: log|c| = a + b log|chat| fit on A
x, y = chat[mA].abs().log(), ct[mA].abs().log(); b_ = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum(); a_ = y.mean() - b_ * x.mean()
est = torch.sign(chat) * torch.exp(a_ + b_ * chat.abs().clamp_min(1e-6).log()); res["loglinear"] = dict(a=a_.item(), b=b_.item(), relerr_B=relerr(est, mB), ratio_deciles_B=ratio_dec(est, mB))
# (c) per-decile of |chat| median ratio fit on A, applied to B
q = chat.abs(); edges = q[mA].quantile(torch.linspace(0, 1, 11, device=DEV)); est = chat.clone()
for i in range(10):
    ma = mA & (q >= edges[i]) & (q <= edges[i + 1]); mb = (q >= edges[i]) & (q <= edges[i + 1]) & (~half) if i < 9 else (q >= edges[i]) & (~half)
    if i == 0: mb = (q <= edges[1]) & (~half)
    r = (chat[ma] / ct[ma]).median(); est[mb] = chat[mb] / r
res["decile_calibration"] = dict(relerr_B=relerr(est, mB), ratio_deciles_B=ratio_dec(est, mB))
# no-fit alternatives: prune the support to the top-8 |coef| atoms and refit; heavy ridge refit
top = cof.abs().topk(8, dim=1).indices; sel8 = torch.gather(sel, 1, top); c8, _ = refit(X, A, sel8); h8 = (sel8 == row[:, None]); ch8 = (c8 * h8).sum(1); m8 = h8.any(1) & typ & ~half
res["prune_top8_refit"] = dict(recall_kept=(h8.any(1)[idn]).float().mean().item(), relerr_B=relerr(ch8, m8), ratio_deciles_B=ratio_dec(ch8, m8))
for rg in (1e-2, 1e-1, 1.0):
    cr, _ = refit(X, A, sel, ridge=rg); chr_ = (cr * hit).sum(1); res[f"ridge_{rg}"] = dict(relerr_B=relerr(chr_, mB), ratio_deciles_B=ratio_dec(chr_, mB))
# the oracle: refit on the true top-3 atoms alone (paper: worse); and true top-3 + the OMP support
tb3, tn3, _ = c.top_writes(L, 3); rows3 = torch.stack([c.atom_index(L, tb3[:, j], tn3[:, j]) for j in range(3)], 1).to(DEV)
c3, _ = refit(X, A, rows3); res["oracle_top3_refit"] = dict(relerr_B=relerr(c3[:, 0], typ & ~half), ratio_deciles_B=ratio_dec(c3[:, 0], typ & ~half))
record(f"e12_shrink_{tag}", res, f"relerr B: base {res['baseline']['relerr_B']:.3f} | global s={res['global_shrink']['s']:.2f} -> {res['global_shrink']['relerr_B']:.3f} | loglinear b={res['loglinear']['b']:.2f} -> {res['loglinear']['relerr_B']:.3f} | decile-cal {res['decile_calibration']['relerr_B']:.3f} | top8-refit {res['prune_top8_refit']['relerr_B']:.3f} (kept {res['prune_top8_refit']['recall_kept']:.2f}) | ridge0.1 {res['ridge_0.1']['relerr_B']:.3f} ridge1 {res['ridge_1.0']['relerr_B']:.3f} | oracle-top3 {res['oracle_top3_refit']['relerr_B']:.3f}")
