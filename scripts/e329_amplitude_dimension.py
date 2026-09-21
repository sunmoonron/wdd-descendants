"""e329: where the quotient breaks. A family of 30 random directions injected at the block-3 input at 0.25, 0.5, 1,
2, 4, 8 and 16 times the natural median amplitude: at each amplitude the quotient dimensions for 90% of the full
score (identity, function, future), the full scores, the physical rank of the images, the linearity of the images
(cosine of the image at this amplitude to the image at 0.25x, per direction), and the overlap of the function
quotient with the one at 1x. Whether the causal dimension expands as the response leaves the linear range."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S0_ = Setup(tag, levels=[0]); L = S0_.L; NB = S0_.NB; S = Setup(tag, levels=[L, min(L + 2, NB - 2)]); lf = min(L + 2, NB - 2); Kf = 30; d = 16; torch.manual_seed(0); V = unit(torch.randn(Kf, S.D, device=DEV)); dims = [dd for dd in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, S.D) if dd <= S.D]; out = {}; ref = None; Q1 = None
for al in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
    r = S.inject_family(V, S.b + 1, amp=al * S.s_inj, seed=1); F = r["F"][L] / al; Ff = unit(r["F"][lf]); dln = unit(r["dl"]); a = r["a"]; tr, te = S.halves(len(r["pos"]), seed=2); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(r["dl"], tr); cent = torch.stack([F[a == k].mean(0) for k in range(Kf)])
    if ref is None: ref = cent
    U = {"identity": scatter_basis(Fc[tr], a[tr], Kf, S.D).T if False else scatter_basis(Fc[tr], a[tr], Kf, min(64, S.D)).T, "function": pls(Fc[tr], Z[tr], min(64, S.D)).T, "future": pls(Fc[tr], Ff[tr], min(64, S.D)).T}
    def score(Ub, rr, obs):
        P = Ub[:min(rr, Ub.shape[0])].T; X = Fc @ P
        if obs == "identity": return accuracy(X[te], centroids(X[tr], a[tr], Kf), a[te])
        if obs == "function": return knn_cos(X, dln, tr, te)
        return knn_cos(X, Ff, tr, te)
    rec = {}
    for obs in U:
        curve = {rr: score(U[obs], rr, obs) for rr in dims if rr <= 64}; full = score(Fc if obs != "identity" else Fc, S.D, obs) if False else curve[64 if 64 in curve else max(curve)]; ch = 1 / Kf if obs == "identity" else 0.0; rec[obs] = dict(dim=next((rr for rr in curve if curve[rr] - ch >= 0.9 * (full - ch)), max(curve)), full=full)
    Qf = pls(Fc[tr], Z[tr], d); rec["quotient_overlap_with_1x"] = inside(Qf, Q1) if Q1 is not None else 1.0
    if al == 1.0: Q1 = Qf
    rec["physical_prank"] = prank(F); rec["linearity_cos"] = ((unit(cent) * unit(ref)).sum(1)).median().item(); rec["gain"] = (F.norm(dim=1) / S.s_inj).median().item(); out[al] = rec
log(f"{tag}: amplitude -> identity-dim/function-dim/future-dim (full scores) | physical rank | linearity to 0.25x | overlap of the function quotient with 1x :: " + " ; ".join(f"{al}x: {v['identity']['dim']}/{v['function']['dim']}/{v['future']['dim']} ({v['identity']['full']:.2f}/{v['function']['full']:.2f}/{v['future']['full']:.2f}) | {v['physical_prank']:.0f} | {v['linearity_cos']:.2f} | {v['quotient_overlap_with_1x']:.2f}" for al, v in out.items()))
record(f"e329_ampdim_{tag}", dict(model=tag, L=L, per_amplitude={str(k): v for k, v in out.items()}), " ; ".join(f"{al}x: {v['identity']['dim']}/{v['function']['dim']}/{v['future']['dim']} lin {v['linearity_cos']:.2f} ov {v['quotient_overlap_with_1x']:.2f}" for al, v in out.items()))
