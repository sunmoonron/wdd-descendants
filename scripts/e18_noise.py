"""e18: instrument sensitivity. Add isotropic Gaussian noise (SNR sweep) or a foreign state (mixture) to each
state and measure identification and the FVU of the clean state from the noisy support. How robust is WDD to
perturbations of the state?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
g = torch.Generator(device=DEV).manual_seed(0); res = dict(model=tag, L=L, N=N, gaussian={}, mixture={})
xn = X.norm(dim=1, keepdim=True)
for snr_db in (30, 20, 10, 6, 3, 0, -3):
    noise = torch.randn(X.shape, generator=g, device=DEV); noise = noise / noise.norm(dim=1, keepdim=True) * xn * 10 ** (-snr_db / 20)
    sel, cof, err = omp(X + noise, A, 64); hit = (sel == row[:, None]).any(1); cr, e = refit(X, A, sel)
    res["gaussian"][snr_db] = dict(recall=hit[typ].float().mean().item(), fvu_clean_from_noisy_support=fvu(e, X, typ))
for alpha in (0.1, 0.25, 0.5, 1.0):
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(1)).to(DEV); Y = X + alpha * X[perm]
    sel, cof, err = omp(Y, A, 64); hit = (sel == row[:, None]).any(1); hit_other = (sel == row[perm][:, None]).any(1); cr, e = refit(X, A, sel)
    res["mixture"][alpha] = dict(recall_own=hit[typ].float().mean().item(), recall_foreign=hit_other[typ].float().mean().item(), fvu_clean=fvu(e, X, typ))
record(f"e18_noise_{tag}", res, "gauss " + " ".join(f"{k}dB:{v['recall']:.2f}" for k, v in res["gaussian"].items()) + " | mix " + " ".join(f"a{k}: own {v['recall_own']:.2f} foreign {v['recall_foreign']:.2f}" for k, v in res["mixture"].items()))
