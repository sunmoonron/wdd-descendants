"""e120 (audit follow-up): centering on typical tokens only. The cached mean includes the sink states of the
centering slice; recompute the mean from typical evaluation tokens of the other sequences (odd), evaluate on even
sequences: FVU32 weight vs rotated and dominant-write recall, compared with the cached-mean numbers."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); Xraw = c.X(L, center=False); typ = typical_mask(Xraw); seq = (torch.arange(c.NT) // CTX).to(DEV)
mu_typ = Xraw[typ & (seq % 2 == 1)].mean(0); mu_old = c.s["mu"][L + 1].to(DEV); ev = (seq % 2 == 0); ids = torch.nonzero(ev)[:, 0][::2]
A, lab = c.dictionary(L); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids.cpu(), 0], tn[ids.cpu(), 0]).to(DEV); t = typ[ids]; res = dict(model=tag, L=L, mu_shift_over_typical_norm=((mu_typ - mu_old).norm() / (Xraw[typ] - mu_old).norm(dim=1).median()).item())
for nm, mu in (("cached_mean", mu_old), ("typical_mean", mu_typ)):
    X = Xraw[ids] - mu; sel, cof, err = omp(X, A, 64); selr, _, errr = omp(X, rotate(A), 64)
    res[nm] = dict(fvu32=fvu(err[:, 31], X, t), rot32=fvu(errr[:, 31], X, t), gap=fvu(errr[:, 31], X, t) - fvu(err[:, 31], X, t), recall=(sel == row[:, None]).any(1)[t].float().mean().item(), fvu64=fvu(err[:, 63], X, t))
record(f"e120_center_{tag}", res, f"mean shift / typical norm {res['mu_shift_over_typical_norm']:.3f} | cached mean: fvu32 {res['cached_mean']['fvu32']:.3f} rot {res['cached_mean']['rot32']:.3f} gap {res['cached_mean']['gap']:.3f} recall {res['cached_mean']['recall']:.3f} | typical mean: fvu32 {res['typical_mean']['fvu32']:.3f} rot {res['typical_mean']['rot32']:.3f} gap {res['typical_mean']['gap']:.3f} recall {res['typical_mean']['recall']:.3f}")
