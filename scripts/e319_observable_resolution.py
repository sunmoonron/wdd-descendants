"""e319: does the resolution of the observable set the resolution of the quotient? The logit footprint is coarsened to
its top-r principal directions, r in 1, 2, 4, 8, 16, 32, 64, 128, 256; for each, the quotient dimension needed for
90% of the full score in predicting that coarsened footprint, and the overlap of the r-quotient with the 256-quotient."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, Bv = logit_scores(nat["dl"], tr); dims = [1, 2, 4, 8, 16, 32, 64, 128, 256]; Q256 = pls(Fc[tr], Z[tr], 16); out = {}
for r in dims:
    Zr = Z[:, :r]; Qr = pls(Fc[tr], Zr[tr], min(16, S.D)); curve = {q: knn_cos(Fc @ Qr[:, :q], Zr, tr, te) if r > 1 else knn_spear(Fc @ Qr[:, :q], Zr[:, 0], tr, te) for q in (1, 2, 4, 8, 16)}; full = curve[16]; need = next((q for q in (1, 2, 4, 8, 16) if curve[q] >= 0.9 * full), 16); out[r] = dict(quotient_dim=need, full=full, overlap_with_256=inside(Qr[:, :min(16, r)], Q256), curve={str(k): v for k, v in curve.items()})
log(f"{tag} (K {S.K}): observable resolution r -> quotient dimension for 90% (full score; overlap with the 256-quotient): " + " ; ".join(f"{r}: {v['quotient_dim']} ({v['full']:.2f}; {v['overlap_with_256']:.2f})" for r, v in out.items()))
record(f"e319_resolution_{tag}", dict(model=tag, L=L, K=S.K, per_r={str(k): v for k, v in out.items()}), " ".join(f"r{r}:{v['quotient_dim']}/{v['overlap_with_256']:.2f}" for r, v in out.items()))
