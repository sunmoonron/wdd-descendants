"""e93: a third solver family, iterative hard thresholding (Blumensath-Davies 2008) and hard thresholding pursuit
(Foucart 2011), on the same states: support = top-64 after gradient steps with hard thresholding; HTP refits on the
support each step. Compared with OMP / one-shot / dual on identification and reconstruction."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); K = 64; N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 3); rows3 = torch.stack([c.atom_index(L, tb[:, j], tn[:, j]) for j in range(3)], 1).to(DEV)[ids]; c1 = tc[:, 0].to(DEV)[ids]
ev = torch.linalg.eigvalsh(A.T @ A); Lc = ev[-1].item()
def iht(X, A, k, iters=100, htp=False, batch=512):
    N_, NA = X.shape[0], A.shape[0]; sel = torch.zeros(N_, k, dtype=torch.long, device=DEV); cof = torch.zeros(N_, k, device=DEV)
    for s in range(0, N_, batch):
        x = X[s:s + batch]; n = x.shape[0]; cc = torch.zeros(n, NA, device=DEV)
        for it in range(iters):
            g = (x - cc @ A) @ A.T; z = cc + g / Lc; top = z.abs().topk(k, dim=1).indices; cc = torch.zeros_like(cc); cc.scatter_(1, top, torch.gather(z, 1, top))
            if htp: cr, _ = refit(x, A, top); cc = torch.zeros_like(cc); cc.scatter_(1, top, cr)
        sel[s:s + n] = top; cof[s:s + n] = torch.gather(cc, 1, top)
    return sel, cof
def score(name, sel, cof):
    hit = sel == rows3[:, :1]; idn = hit.any(1) & typ; chat = (cof * hit).sum(1)[idn]; ct = c1[idn]; cr, e = refit(X, A, sel)
    r = dict(recall1=hit.any(1)[typ].float().mean().item(), recall3=(sel[:, :, None] == rows3[:, None, :]).any(1).all(1)[typ].float().mean().item(), med_rel_err=((chat - ct).abs() / ct.abs()).median().item(), fvu_refit=fvu(e, X, typ))
    log(f"{tag} {name}: r1 {r['recall1']:.3f} r3 {r['recall3']:.3f} err {r['med_rel_err']:.2f} fvu {r['fvu_refit']:.3f}"); return r
res = dict(model=tag, L=L, N=N, solvers={})
sel, cof, _ = omp(X, A, K); res["solvers"]["omp"] = score("omp", sel, cof)
t0 = time.time(); s_, c_ = iht(X, A, K, iters=100); res["solvers"]["iht100"] = score("iht100", s_, c_); res["solvers"]["iht100"]["time"] = time.time() - t0
t0 = time.time(); s_, c_ = iht(X, A, K, iters=30, htp=True); res["solvers"]["htp30"] = score("htp30", s_, c_); res["solvers"]["htp30"]["time"] = time.time() - t0
record(f"e93_iht_{tag}", res, " | ".join(f"{k}: r1 {v['recall1']:.2f} r3 {v['recall3']:.2f} err {v['med_rel_err']:.2f} fvu {v['fvu_refit']:.2f}" for k, v in res["solvers"].items()))
