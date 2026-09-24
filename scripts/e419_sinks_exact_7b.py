"""e419: keep the huge directions exact, then describe by function. At 7B (e418) the states' top-8 principal directions
hold 97% of the variance and 0.3% of the Fisher trace, and a Fisher-metric pursuit made the native description worse
(0.79 to 0.55 at k 16): chosen by local function, words leak into directions that matter only at large displacement.
Here, as e404's condition B: the state's part in M (top-8 principal subspace, from other text) is kept exactly and only
the rest is described, with words projected off M, under the Euclidean or the Fisher metric (restricted to the
complement). Qwen2.5-7B, middle depth, 4 sequences; native words, their rotation (by rotating states), and Gaussian words
with the complement's covariance. Pre-registered: with M exact, the Fisher pursuit is at least as good as the Euclidean
one for the native words, and the native words beat their rotation under both."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
from sd_common import Level, fisher_gram, metric_sqrt, gauss_like, unitr
name = sys.argv[1] if len(sys.argv) > 1 else "qwen7"; KS = [4, 8, 16, 32, 64]
model, tok, fam = load_bf16(name); arch = Arch(model, fam); L = arch.NB // 2
E = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"]; ev = E[:4].to(DEV); fit = E[4:8].to(DEV)
A, blk, typ, ends = lean_dictionary(arch, L); A = A[:ends[L]]
lv = Level(model, arch, ev, L); fl = Level(model, arch, fit, L); SX = (fl.Xc.T @ fl.Xc) / fl.Xc.shape[0]
evX, UX = torch.linalg.eigh(SX.double()); UM = UX[:, -8:].float(); PM = UM @ UM.T; Pp = torch.eye(arch.D, device=DEV) - PM
XM = lv.Xc @ PM; Xp = lv.Xc - XM; SXp = Pp @ SX @ Pp; del fl
GF = fisher_gram(model, arch, fit, L); SFp = metric_sqrt(Pp @ GF @ Pp + 1e-6 * torch.eye(arch.D, device=DEV) * GF.trace() / arch.D)
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
# words projected off M, renormalised, built in place in chunks (no second dictionary): A <- unit(A - A PM)
for s in range(0, A.shape[0], 65536):
    w = A[s:s + 65536]; w = w - (w @ UM) @ UM.T; A[s:s + 65536] = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
res = dict(model=name, level=L, k=KS, gap=lv.gap, cells={})
tot = lv.Xc.pow(2).sum().item()
def run(V, S=None, rot=False):
    """OMP of the complement part over V under the Euclidean metric (S None) or x -> x S; with rot, the rotated
    vocabulary V R is emulated by rotating the targets and the metric"""
    Xt = Xp @ R.T if rot else Xp; St = None if S is None else (R @ S @ R.T if rot else S); out = {}
    if St is None: sel, _, _ = omp(Xt, V, max(KS), batch=128, record_err=False)
    else:
        M = St @ St.T; nrm = torch.cat([(V[s:s + 65536] @ St).norm(dim=-1) for s in range(0, V.shape[0], 65536)]).clamp_min(1e-8)
        sel = torch.zeros(Xt.shape[0], max(KS), dtype=torch.long, device=DEV)
        for s0 in range(0, Xt.shape[0], 128):
            x = Xt[s0:s0 + 128]; r = x.clone(); chosen = []
            for step in range(max(KS)):
                corr = ((r @ M) @ V.T) / nrm[None]
                if chosen: corr.scatter_(1, torch.stack(chosen, 1), 0.0)
                chosen.append(corr.abs().argmax(1)); idx = torch.stack(chosen, 1)
                As = V[idx] @ St; Gm = As @ As.transpose(1, 2) + 1e-5 * torch.eye(len(chosen), device=DEV)
                c = torch.cholesky_solve(As @ (x @ St)[:, :, None], torch.linalg.cholesky(Gm))[:, :, 0]; r = x - torch.einsum("nk,nkd->nd", c, V[idx])
            sel[s0:s0 + 128] = torch.stack(chosen, 1)
    for k in KS:
        s = sel[:, :k]
        if St is None: cof, _ = refit(Xt, V, s)
        else:
            As = V[s] @ St; Gm = As @ As.transpose(1, 2) + 1e-5 * torch.eye(k, device=DEV)
            cof = torch.cholesky_solve(As @ (Xt @ St)[:, :, None], torch.linalg.cholesky(Gm))[:, :, 0]
        Xh = torch.einsum("nk,nkd->nd", cof, V[s]); Xh = Xh @ R if rot else Xh
        Xh = Xh - (Xh @ UM) @ UM.T                                             # stay off M: M is given exactly
        r_, lo, hi = lv.recovered(lv.splice(lv.mu + XM + Xh)); out[str(k)] = dict(rec=r_, lo=lo, hi=hi, fvu=(lv.Xc - XM - Xh).pow(2).sum().item() / tot)
    return out
base = lv.recovered(lv.splice(lv.mu + XM))[0]; res["M_only"] = base
cX = gauss_like(A.shape[0] // 2, SXp + 1e-6 * torch.eye(arch.D, device=DEV) * SXp.trace() / arch.D, seed=2)
for mt, S in [("euc", None), ("fisher", SFp)]:
    res["cells"][f"own:{mt}"] = run(A, S); res["cells"][f"rot:{mt}"] = run(A, S, rot=True); res["cells"][f"covX:{mt}"] = run(cX, S)
    for v in ("own", "rot", "covX"):
        c_ = res["cells"][f"{v}:{mt}"]; log(f"{name} M-exact {v} {mt}: rec k4..64 " + " ".join(f"{c_[str(k)]['rec']:.2f}" for k in KS))
g_ = lambda n, k=16: res["cells"][n][str(k)]["rec"]
summ = (f"{name} L{L} M exact (M alone recovers {base:.2f}) k16 euc/fisher: own {g_('own:euc'):.2f}/{g_('own:fisher'):.2f} rot {g_('rot:euc'):.2f}/{g_('rot:fisher'):.2f} covX {g_('covX:euc'):.2f}/{g_('covX:fisher'):.2f} | "
        f"k4 own {g_('own:euc', 4):.2f}/{g_('own:fisher', 4):.2f}")
log(summ); record(f"e419_sinksexact_{name}", res, summ)
