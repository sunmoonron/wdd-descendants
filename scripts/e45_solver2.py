"""e45: solver round 2, built on e04. (a) dual-frame OMP: greedy selection by the whitened correlation
(S^-1 x).a with least-squares refit in the original space; (b) LASSO support + least-squares refit (debiased
basis pursuit); (c) dual one-shot recall vs k (16..256); (d) the estimation calibration of the best identifier.
Question: can one solver have LASSO-level identification and OMP-level reconstruction?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); K = 64; N = int(os.environ.get("WDD_N", 4096))
ids = sub(c.NT, N); X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 3); rows3 = torch.stack([c.atom_index(L, tb[:, j], tn[:, j]) for j in range(3)], 1).to(DEV)[ids]; c1 = tc[:, 0].to(DEV)[ids]
S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
def score(name, sel, cof, t, extra=None):
    hit = sel == rows3[:, :1]; idn = hit.any(1) & typ; chat = (cof * hit).sum(1)[idn]; ct = c1[idn]
    cr, e = refit(X, A, sel)
    out = dict(recall1=hit.any(1)[typ].float().mean().item(), recall3=(sel[:, :, None] == rows3[:, None, :]).any(1).all(1)[typ].float().mean().item(),
               sign=((chat * ct) > 0).float().mean().item(), med_rel_err=((chat - ct).abs() / ct.abs()).median().item(), med_ratio=(chat / ct).median().item(), fvu_refit=fvu(e, X, typ), time=t)
    if extra: out.update(extra)
    log(f"{tag} {name}: r1 {out['recall1']:.3f} r3 {out['recall3']:.3f} relerr {out['med_rel_err']:.2f} ratio {out['med_ratio']:.2f} fvu {out['fvu_refit']:.3f} ({t:.0f}s)"); return out
def dual_omp(X, A, k, batch=1024):
    N_, NA = X.shape[0], A.shape[0]; sel = torch.zeros(N_, k, dtype=torch.long, device=DEV); cof = torch.zeros(N_, k, device=DEV); err = torch.zeros(N_, device=DEV)
    eye = torch.eye(k, device=DEV)
    for s in range(0, N_, batch):
        x = X[s:s + batch]; n = x.shape[0]; r = x.clone(); Sx = torch.zeros(n, 0, dtype=torch.long, device=DEV); taken = torch.zeros(n, NA, dtype=torch.bool, device=DEV)
        for step in range(k):
            pick = ((r @ Winv) @ A.T).abs_().masked_fill_(taken, -1.0).argmax(-1, keepdim=True); taken.scatter_(1, pick, True); Sx = torch.cat([Sx, pick], 1)
            As = A[Sx]; G = As @ As.transpose(1, 2) + 1e-5 * eye[:step + 1, :step + 1]
            cc = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (cc.transpose(1, 2) @ As)[:, 0]
        sel[s:s + n] = Sx; cof[s:s + n] = cc[:, :, 0]; err[s:s + n] = (r ** 2).sum(-1)
    return sel, cof, err
res = dict(model=tag, L=L, N=N, K=K, solvers={})
t0 = time.time(); sel, cof, err = omp(X, A, K); res["solvers"]["omp"] = score("omp", sel, cof, time.time() - t0)
t0 = time.time(); sd, cd, ed = dual_omp(X, A, K); res["solvers"]["dual_omp"] = score("dual_omp", sd, cd, time.time() - t0)
for k in (16, 32, 64, 128, 256):
    t0 = time.time(); s1, c1_, _ = oneshot(X, A, k, whiten=Winv); res["solvers"][f"dual_oneshot_k{k}"] = score(f"dual_oneshot_k{k}", s1, c1_, time.time() - t0)
# LASSO with the e04 lambda if available, else tune quickly
lam = None
try: lam = json.load(open(os.path.join(RESULTS, f"e04_solvers_{tag}.json")))["solvers"]["lasso"]["lambda"]
except Exception: pass
Lc = ev[-1].item()
if lam is None:
    pilot = X[:256]; lo, hi = 1e-3, 10.0
    for _ in range(9):
        m = math.sqrt(lo * hi); C = fista_l1(pilot, A, m, iters=200, L=Lc).float().to(DEV); n_ = (C.abs() > 1e-3 * C.abs().max(1, keepdim=True).values).sum(1).float().median().item()
        if n_ > K: lo = m
        else: hi = m
    lam = math.sqrt(lo * hi)
t0 = time.time(); C = fista_l1(X, A, lam, iters=300, L=Lc).float().to(DEV); sl = C.abs().topk(K, dim=1).indices; del C
cl, _ = refit(X, A, sl); res["solvers"]["lasso_refit"] = score("lasso_refit", sl, cl, time.time() - t0, dict(**{"lambda": lam}))
# two-stage: dual one-shot at 128 to screen, then OMP restricted to the screened atoms (k=64)
t0 = time.time(); s128, _, _ = oneshot(X, A, 128, whiten=Winv); sel2 = torch.zeros(N, K, dtype=torch.long, device=DEV); cof2 = torch.zeros(N, K, device=DEV)
for i in range(0, N, 512):
    cand = s128[i:i + 512]; Ac = A[cand]                                                         # [n,128,D]
    x = X[i:i + 512]; n = x.shape[0]; r = x.clone(); Sx = []; taken = torch.zeros(n, 128, dtype=torch.bool, device=DEV)
    for step in range(K):
        pick = torch.einsum("nd,nkd->nk", r, Ac).abs().masked_fill(taken, -1).argmax(1); taken[torch.arange(n), pick] = True; Sx.append(pick)
        Ssel = torch.stack(Sx, 1); As = torch.gather(Ac, 1, Ssel[:, :, None].expand(-1, -1, c.D)); G = As @ As.transpose(1, 2) + 1e-5 * torch.eye(step + 1, device=DEV)
        cc = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (cc.transpose(1, 2) @ As)[:, 0]
    sel2[i:i + n] = torch.gather(cand, 1, Ssel); cof2[i:i + n] = cc[:, :, 0]
res["solvers"]["screen128_then_omp"] = score("screen128_then_omp", sel2, cof2, time.time() - t0)
best = max(res["solvers"], key=lambda k: res["solvers"][k]["recall1"])
record(f"e45_solver2_{tag}", res, " | ".join(f"{k}: r1 {v['recall1']:.2f} r3 {v['recall3']:.2f} err {v['med_rel_err']:.2f} fvu {v['fvu_refit']:.2f}" for k, v in res["solvers"].items()) + f" || best {best}")
