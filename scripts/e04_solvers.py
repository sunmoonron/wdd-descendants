"""e04: solver bake-off on identical states and dictionary. OMP (Pati 1993) vs plain matching pursuit
(Mallat-Zhang 1993), one-shot correlation, one-shot with the canonical dual frame S^-1 (Duffin-Schaeffer 1952),
one-shot with Sigma^-1 whitening (Wiener/Mahalanobis), nonnegative OMP (Lawson-Hanson 1974 spirit; the GELU
prior), CoSaMP (Needell-Tropp 2009), LASSO / basis pursuit (Chen-Donoho-Saunders 1998, Tibshirani 1996) and
nonnegative LASSO. Metrics against the ledger at |support| = 64."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); K = 64; N = int(os.environ.get("WDD_N", 4096))
ids = sub(c.NT, N); X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 3); rows3 = torch.stack([c.atom_index(L, tb[:, j], tn[:, j]) for j in range(3)], 1).to(DEV)[ids]; c1 = tc[:, 0].to(DEV)[ids]
def score(name, sel, cof, t):
    hit = sel == rows3[:, :1]; idn = hit.any(1) & typ
    chat = (cof * hit).sum(1)[idn]; ct = c1[idn]
    cr, e = refit(X, A, sel)
    out = dict(recall1=hit.any(1)[typ].float().mean().item(), recall3=(sel[:, :, None] == rows3[:, None, :]).any(1).all(1)[typ].float().mean().item(),
               sign=((chat * ct) > 0).float().mean().item(), med_rel_err=((chat - ct).abs() / ct.abs()).median().item(),
               med_ratio=(chat / ct).median().item(), fvu_refit=fvu(e, X, typ), time=t)
    log(f"{tag} {name}: recall1 {out['recall1']:.3f} recall3 {out['recall3']:.3f} sign {out['sign']:.3f} relerr {out['med_rel_err']:.2f} ratio {out['med_ratio']:.2f} fvu {out['fvu_refit']:.3f} ({t:.0f}s)")
    return out
res = dict(model=tag, L=L, N=N, K=K, solvers={})
t0 = time.time(); sel, cof, err = omp(X, A, K); res["solvers"]["omp"] = score("omp", sel, cof, time.time() - t0)
t0 = time.time(); selm, cofm, _ = mp(X, A, K)
# MP support: unique atoms in order of first appearance, refit coefficients on the unique support (pad with last)
selu = selm.clone()
res["solvers"]["mp"] = score("mp", selu, refit(X, A, selu)[0], time.time() - t0)
res["mp_unique_atoms_mean"] = torch.tensor([len(r.unique()) for r in selm[:512]]).float().mean().item()
t0 = time.time(); s1, c1_, _ = oneshot(X, A, K); res["solvers"]["oneshot"] = score("oneshot", s1, c1_, time.time() - t0)
S = A.T @ A; ev, V = torch.linalg.eigh(S)
for eps_name, eps in (("dual_1e-2", 1e-2), ("dual_1e-1", 1e-1)):
    Winv = V @ torch.diag(1 / (ev + eps * ev[-1])) @ V.T
    t0 = time.time(); s2, c2, _ = oneshot(X, A, K, whiten=Winv); res["solvers"][f"oneshot_{eps_name}"] = score(f"oneshot_{eps_name}", s2, c2, time.time() - t0)
Sig = (X[typ].T @ X[typ]) / typ.sum(); evs, Vs = torch.linalg.eigh(Sig)
for eps_name, eps in (("sigma_1e-2", 1e-2), ("sigma_1e-1", 1e-1)):
    Winv = Vs @ torch.diag(1 / (evs + eps * evs[-1])) @ Vs.T
    t0 = time.time(); s3, c3, _ = oneshot(X, A, K, whiten=Winv); res["solvers"][f"oneshot_{eps_name}"] = score(f"oneshot_{eps_name}", s3, c3, time.time() - t0)
t0 = time.time(); sn, cn, _ = omp(X, A, K, nonneg=True); res["solvers"]["omp_nonneg"] = score("omp_nonneg", sn, cn, time.time() - t0)
t0 = time.time(); sc, cc, _ = cosamp(X, A, K, iters=8); res["solvers"]["cosamp"] = score("cosamp", sc, cc, time.time() - t0)
# LASSO: tune lambda on a 256-state pilot so that the median support (|c| > 1e-3 max) is ~K
Lc = ev[-1].item(); pilot = X[:256]
def nnz(lam, nonneg=False):
    C = fista_l1(pilot, A, lam, iters=200, nonneg=nonneg, L=Lc).float().to(DEV)
    return (C.abs() > 1e-3 * C.abs().max(1, keepdim=True).values).sum(1).float().median().item()
for nonneg in (False, True):
    lo, hi = 1e-3, 10.0
    for _ in range(9):
        m = math.sqrt(lo * hi); n = nnz(m, nonneg)
        if n > K: lo = m
        else: hi = m
    lam = math.sqrt(lo * hi); t0 = time.time()
    C = fista_l1(X, A, lam, iters=300, nonneg=nonneg, L=Lc).float().to(DEV)
    sl = C.abs().topk(K, dim=1).indices; cl = torch.gather(C, 1, sl)
    name = "lasso_nonneg" if nonneg else "lasso"; res["solvers"][name] = score(name, sl, cl, time.time() - t0); res["solvers"][name]["lambda"] = lam
    res["solvers"][name]["nnz_median"] = (C.abs() > 1e-3 * C.abs().max(1, keepdim=True).values).sum(1).float().median().item()
    res["solvers"][name]["fvu_raw"] = fvu(((X - C @ A) ** 2).sum(1), X, typ)
    del C
best = max(res["solvers"], key=lambda k: res["solvers"][k]["recall1"])
record(f"e04_solvers_{tag}", res, " | ".join(f"{k}: r1 {v['recall1']:.2f} r3 {v['recall3']:.2f} err {v['med_rel_err']:.2f}" for k, v in res["solvers"].items()) + f" || best {best}")
