"""e154: reweighted l1 (Candes-Wakin-Boyd 2008, an ARD/sparse-Bayesian proxy): three rounds of FISTA with weights
1/(|c|+eps) from the previous round; support = top-64; identification, calibration (refit), FVU. Does iterative
reweighting fix LASSO's shrinkage without losing its identification advantage?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); K = 64; N = 2048; ids = sub(c.NT, N); X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 3); rows3 = torch.stack([c.atom_index(L, tb[:, j], tn[:, j]) for j in range(3)], 1).to(DEV)[ids]; c1 = tc[:, 0].to(DEV)[ids]
ev = torch.linalg.eigvalsh(A.T @ A); Lc = ev[-1].item()
lam = 0.3
try: lam = json.load(open(os.path.join(RESULTS, f"e04_solvers_{tag}.json")))["solvers"]["lasso"]["lambda"]
except Exception: pass
def fista_w(X, A, lam, w, iters=250, batch=512):
    N_, NA = X.shape[0], A.shape[0]; out = torch.zeros(N_, NA, dtype=torch.float16)
    for s in range(0, N_, batch):
        x = X[s:s + batch]; n = x.shape[0]; cc = torch.zeros(n, NA, device=DEV); y = cc.clone(); t = 1.0; ww = w[s:s + batch]
        for it in range(iters):
            grad = (y @ A - x) @ A.T; z = y - grad / Lc; z = torch.sign(z) * (z.abs() - lam * ww / Lc).clamp_min(0); t2 = (1 + math.sqrt(1 + 4 * t * t)) / 2; y = z + ((t - 1) / t2) * (z - cc); cc, t = z, t2
        out[s:s + n] = cc.half().cpu()
    return out
def score(name, sel, cof):
    hit = sel == rows3[:, :1]; idn = hit.any(1) & typ; chat = (cof * hit).sum(1)[idn]; ct = c1[idn]; cr, e = refit(X, A, sel)
    r = dict(recall1=hit.any(1)[typ].float().mean().item(), recall3=(sel[:, :, None] == rows3[:, None, :]).any(1).all(1)[typ].float().mean().item(), med_rel_err_raw=((chat - ct).abs() / ct.abs()).median().item(), med_ratio_raw=(chat / ct).median().item(), fvu_refit=fvu(e, X, typ))
    chr_ = (cr * hit).sum(1)[idn]; r["med_rel_err_refit"] = ((chr_ - ct).abs() / ct.abs()).median().item(); log(f"{tag} {name}: r1 {r['recall1']:.3f} r3 {r['recall3']:.3f} raw err {r['med_rel_err_raw']:.2f} ratio {r['med_ratio_raw']:.2f} refit err {r['med_rel_err_refit']:.2f} fvu {r['fvu_refit']:.3f}"); return r
res = dict(model=tag, L=L, N=N, rounds={}); w = torch.ones(N, A.shape[0], device=DEV)
for r in range(3):
    C = fista_w(X, A, lam, w); Cd = C.float().to(DEV); sl = Cd.abs().topk(K, dim=1).indices; cl = torch.gather(Cd, 1, sl); res["rounds"][r] = score(f"round{r}", sl, cl)
    w = 1.0 / (Cd.abs() + 0.05 * Cd.abs().max(1, keepdim=True).values.clamp_min(1e-6)); w = w / w.mean(1, keepdim=True); del Cd, C
so, co, _ = omp(X, A, K); res["omp"] = score("omp", so, co)
record(f"e154_reweighted_{tag}", res, " | ".join(f"round {r}: r1 {v['recall1']:.2f} r3 {v['recall3']:.2f} raw-ratio {v['med_ratio_raw']:.2f} refit-err {v['med_rel_err_refit']:.2f} fvu {v['fvu_refit']:.2f}" for r, v in res["rounds"].items()) + f" | omp r1 {res['omp']['recall1']:.2f} err {res['omp']['med_rel_err_refit']:.2f} fvu {res['omp']['fvu_refit']:.2f}")
