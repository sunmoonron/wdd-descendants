"""e21: write lifetime. Take the dominant MLP writes born in an early block b0 and follow them: OMP identification
at every later level, raw and centered survival at every later level. The observability half-life of a write."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
b0s = [1, 2, mid(c) // 2] if c.NB > 8 else [1, 2]
res = dict(model=tag, N=N, born={})
for b0 in sorted(set(b0s)):
    led = c.ledger(b0, blocks=[b0])[b0][ids]; tc_, tn_ = led.abs().max(1); tc_ = torch.gather(led, 1, tn_[:, None])[:, 0].to(DEV)
    big = (tc_.abs() >= tc_.abs().quantile(0.5))          # the larger half of block-b0 dominant writes
    curve = []
    for L in range(b0, c.NB):
        X = c.X(L)[ids]; Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw) & big; A, lab = c.dictionary(L)
        row = c.atom_index(L, torch.full_like(tn_, b0), tn_).to(DEV); d = A[row]
        sel, cof, err = omp(X, A, 64); hit = (sel == row[:, None]).any(1)
        raw = ((Xraw * d).sum(1) / tc_); cen = ((X * d).sum(1) / tc_)
        curve.append(dict(L=L, recall=hit[typ].float().mean().item(), raw_surv_med=raw[typ].median().item(), cen_surv_med=cen[typ].median().item(), frac_raw_intact=(raw[typ] > 0.75).float().mean().item()))
    res["born"][b0] = curve; log(f"{tag} born b{b0}: " + " ".join(f"L{r['L']}: rec {r['recall']:.2f} raw {r['raw_surv_med']:.2f} cen {r['cen_surv_med']:.2f}" for r in curve))
record(f"e21_life_{tag}", res, " || ".join(f"b{b0}: " + " ".join(f"L{r['L']}:{r['recall']:.2f}/{r['raw_surv_med']:.2f}" for r in cv) for b0, cv in res["born"].items()))
