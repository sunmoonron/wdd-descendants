"""e128: the honest depth profile. Per level: identification of the largest physically surviving write (raw
survival > 0.75) under OMP@64 and the dual-frame projection@64, next to the ledger-top-1 numbers."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = 4096; ids = sub(c.NT, N); rows = []
for L in range(c.NB):
    Xraw = c.X(L, center=False)[ids]; X = c.X(L)[ids]; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
    led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); mlp_rows = torch.cat([c.atom_index(L, torch.full((c.DFF,), b), torch.arange(c.DFF)) for b in range(L + 1)]).to(DEV)
    big = C.abs() >= 0.05 * C.abs().max(1, keepdim=True).values; surv = (Xraw @ A[mlp_rows].T) / C.where(C != 0, torch.ones_like(C)); intact = big & (surv > 0.75); has = intact.any(1)
    t_int = mlp_rows[C.abs().masked_fill(~intact, -1).argmax(1)]; t_top = mlp_rows[C.abs().argmax(1)]
    S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
    so, _, _ = omp(X, A, 64); sd, _, _ = oneshot(X, A, 64, whiten=Winv); m = typ & has
    rows.append(dict(L=L, frac_has_intact=has[typ].float().mean().item(), intact_is_top1=(t_int == t_top)[m].float().mean().item(), omp_intact=(so == t_int[:, None]).any(1)[m].float().mean().item(), dual_intact=(sd == t_int[:, None]).any(1)[m].float().mean().item(), omp_top1=(so == t_top[:, None]).any(1)[typ].float().mean().item(), dual_top1=(sd == t_top[:, None]).any(1)[typ].float().mean().item()))
    log(f"{tag} L{L}: intact write exists {rows[-1]['frac_has_intact']:.2f} (= top-1 {rows[-1]['intact_is_top1']:.2f}) | intact: omp {rows[-1]['omp_intact']:.2f} dual {rows[-1]['dual_intact']:.2f} | top-1: omp {rows[-1]['omp_top1']:.2f} dual {rows[-1]['dual_top1']:.2f}")
record(f"e128_depth_{tag}", dict(model=tag, rows=rows), " | ".join(f"L{r['L']}: intact omp {r['omp_intact']:.2f} dual {r['dual_intact']:.2f} (top1 {r['omp_top1']:.2f}/{r['dual_top1']:.2f})" for r in rows))
