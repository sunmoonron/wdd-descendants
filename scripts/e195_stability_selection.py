"""e195: stability selection (Meinshausen-Buhlmann 2010) as a provenance certificate. e189 showed the OMP support
is chaotic under perturbation. Here: OMP@64 (and dual@64) on x and on R=4 noisy copies x + sigma * ||x|| * z
(sigma = 0.1, 0.3); an atom is 'stable' if it is selected in all runs. Reports the precision of stable vs unstable
atoms (real = ledger |coef| >= 5% of the token's max), the recall of dominant writes among stable atoms, the
stable fraction, and the function of the stable vs unstable part (reconstruction energy)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; A, lab = c.dictionary(L); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * C.abs().max(1, keepdim=True).values
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
def is_real(sel):
    ismlp = typA[sel] == T_MLP; key = torch.where(ismlp, blkA[sel] * c.DFF + idxA[sel], torch.zeros_like(sel)); return ismlp & (C.gather(1, key).abs() >= thr)
out = {}
for dec in ("omp", "dual"):
    solve = (lambda Z: omp(Z, A, 64)[0]) if dec == "omp" else (lambda Z: oneshot(Z, A, 64, whiten=Winv)[0])
    sel0 = solve(X); real0 = is_real(sel0)
    for sigma in (0.1, 0.3):
        torch.manual_seed(1); count = torch.zeros_like(sel0, dtype=torch.float)
        for r in range(4):
            Z = torch.randn_like(X); Z = Z / Z.norm(dim=1, keepdim=True) * X.norm(dim=1, keepdim=True) * sigma; s = solve(X + Z); count += (sel0[..., None] == s[:, None, :]).any(-1).float()
        stable = count == 4; st, un = stable[typ], ~stable[typ]; rr = real0[typ]
        prec_stable = rr[st].float().mean().item(); prec_unstable = rr[un].float().mean().item(); frac_stable = st.float().mean().item()
        dom_in = (sel0 == row[:, None]); dom_stable = (dom_in & stable).any(1)[typ].float().mean().item(); dom_any = dom_in.any(1)[typ].float().mean().item()
        # stability of the real subset vs the spurious subset
        st_real = stable[real0 & typ[:, None]].float().mean().item(); st_spur = stable[~real0 & typ[:, None]].float().mean().item()
        out[f"{dec}_s{sigma}"] = dict(frac_stable=frac_stable, precision_stable=prec_stable, precision_unstable=prec_unstable, precision_base=rr.float().mean().item(), dominant_recall_any=dom_any, dominant_recall_stable=dom_stable, stable_rate_real=st_real, stable_rate_spurious=st_spur)
        log(f"{tag} {dec} sigma {sigma}: stable fraction {frac_stable:.2f} | precision stable {prec_stable:.2f} vs unstable {prec_unstable:.2f} (base {rr.float().mean().item():.2f}) | real atoms stable {st_real:.2f} vs spurious {st_spur:.2f} | dominant write in support {dom_any:.2f}, and stable {dom_stable:.2f}")
record(f"e195_stability_{tag}", dict(model=tag, L=L, results=out), " | ".join(f"{k}: stable {v['frac_stable']:.2f}, precision stable {v['precision_stable']:.2f} vs unstable {v['precision_unstable']:.2f} (base {v['precision_base']:.2f}), real-atom stability {v['stable_rate_real']:.2f} vs spurious {v['stable_rate_spurious']:.2f}, dominant recall {v['dominant_recall_any']:.2f} -> stable {v['dominant_recall_stable']:.2f}" for k, v in out.items()))
