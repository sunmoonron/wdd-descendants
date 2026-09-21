"""e331: the quotient along the massive channels. Families of 30 directions concentrated on the model's three largest-
magnitude channels at the injection level (plus small random remainder) against ordinary random directions, at 1x
and 4x amplitude: quotient dimensions, full scores, gain, evenness of the response (cosine of the +1x and -1x images)
and the overlap of their function quotient with the ordinary one."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S0_ = Setup(tag, levels=[0]); L = S0_.L; NB = S0_.NB; S = Setup(tag, levels=[S0_.b, L, min(L + 2, NB - 2)]); lf = min(L + 2, NB - 2); Kf = 30; d = 16; ch = S.S0[S.b][S.typ].mean(0).abs().argsort(descending=True)[:3]; torch.manual_seed(0)
massive = torch.randn(Kf, S.D, device=DEV) * 0.1; massive[:, ch] += torch.randn(Kf, 3, device=DEV) * 3; fams = {"massive_channels": unit(massive), "random": unit(torch.randn(Kf, S.D, device=DEV))}; out = {}; Qref = {}
for nm, V in fams.items():
    for al in (1.0, 4.0):
        r = S.inject_family(V, S.b + 1, amp=al * S.s_inj, seed=1); rm = S.inject_family(-V, S.b + 1, amp=al * S.s_inj, seed=1); F = r["F"][L]; a = r["a"]; tr, te = S.halves(len(r["pos"]), seed=2); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(r["dl"], tr); dln = unit(r["dl"]); Ff = unit(r["F"][lf])
        cent = torch.stack([F[a == k].mean(0) for k in range(Kf)]); centm = torch.stack([rm["F"][L][rm["a"] == k].mean(0) for k in range(Kf)]); even = ((unit(cent) * unit(centm)).sum(1)).median().item()
        Ufn = pls(Fc[tr], Z[tr], 64).T; Uid = scatter_basis(Fc[tr], a[tr], Kf, 64).T
        def score(Ub, rr, obs):
            X = Fc @ Ub[:rr].T
            if obs == "identity": return accuracy(X[te], centroids(X[tr], a[tr], Kf), a[te])
            return knn_cos(X, dln if obs == "function" else Ff, tr, te)
        rec = {}
        for obs, Ub in (("identity", Uid), ("function", Ufn), ("future", pls(Fc[tr], Ff[tr], 64).T)):
            curve = {rr: score(Ub, rr, obs) for rr in (1, 2, 4, 8, 16, 32, 64)}; full = curve[64]; c0 = 1 / Kf if obs == "identity" else 0.0; rec[obs] = dict(dim=next((rr for rr in curve if curve[rr] - c0 >= 0.9 * (full - c0)), 64), full=full)
        Q = pls(Fc[tr], Z[tr], d); Qref[(nm, al)] = Q; rec.update(gain=(F.norm(dim=1) / (al * S.s_inj)).median().item(), evenness=even, logit_response=r["dl"].norm(dim=1).median().item()); out[f"{nm}_{al}x"] = rec
ov = {f"{nm}_{al}x_vs_random_1x": inside(Qref[(nm, al)], Qref[("random", 1.0)]) for nm in fams for al in (1.0, 4.0)}
log(f"{tag} (massive channels {ch.tolist()}): family_amplitude -> identity/function/future dims (full) | gain | evenness cos(+,-) | logit response :: " + " ; ".join(f"{k}: {v['identity']['dim']}/{v['function']['dim']}/{v['future']['dim']} ({v['identity']['full']:.2f}/{v['function']['full']:.2f}/{v['future']['full']:.2f}) | {v['gain']:.2f} | {v['evenness']:+.2f} | {v['logit_response']:.1f}" for k, v in out.items()) + " | function-quotient overlaps with random 1x: " + " ".join(f"{k} {v:.2f}" for k, v in ov.items()))
record(f"e331_massivequot_{tag}", dict(model=tag, L=L, channels=ch.tolist(), per_family=out, overlaps=ov), " ; ".join(f"{k}: {v['identity']['dim']}/{v['function']['dim']}/{v['future']['dim']} gain {v['gain']:.2f} even {v['evenness']:+.2f}" for k, v in out.items()) + " | " + " ".join(f"{k} {v:.2f}" for k, v in ov.items()))
