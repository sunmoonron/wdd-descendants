"""e342: try to destroy the quotient. Sixteen perturbation families of 30 vectors at the block-3 input (random,
sparse, one-hot, Rademacher, heavy-tailed, low-rank, rank-one, covariance-whitened, covariance-amplified, orthogonal
to the activation cloud, parallel to the residual mean, orthogonal to it, top and bottom transport singular
directions, anti-natural (negated cov-matched), massive-channel) at 1x and 4x amplitude. For each: the logit footprint
decoded from the family's own 16-dim quotient against a random 16-projection (the ratio is the quotient's
advantage), the quotient dimension, the overlap with the WDD-write quotient, and the gain. The families with the
smallest advantage are the failure candidates."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; b = S.b; S = Setup(tag, levels=[b, L]); Kf = 30; d = 16; torch.manual_seed(0); D = S.D; Sc = S.S0[b][S.typ] - S.S0[b][S.typ].mean(0, keepdim=True); U = torch.linalg.svd(Sc, full_matrices=False); mean_dir = unit(S.S0[b][S.typ].mean(0)[None])[0]; T = S.fit_T(b + 1, L, seed=3); Ut = torch.linalg.svd(T)[0]; ch = S.S0[b][S.typ].mean(0).abs().argsort(descending=True)[:3]
def cov(k=Kf): return unit((torch.randn(k, len(Sc), device=DEV) / len(Sc) ** 0.5) @ Sc)
def whiten(V): return unit((V @ U[2].T / U[1].clamp_min(1e-3)[None]) @ U[2])
oh = torch.zeros(Kf, D, device=DEV); oh[torch.arange(Kf), torch.randperm(D, device=DEV)[:Kf]] = 1; sp = torch.zeros(Kf, D, device=DEV); sp[torch.arange(Kf, device=DEV)[:, None], torch.randint(0, D, (Kf, 5), device=DEV)] = torch.randn(Kf, 5, device=DEV); massive = torch.randn(Kf, D, device=DEV) * 0.1; massive[:, ch] += torch.randn(Kf, 3, device=DEV) * 3; Q4 = torch.linalg.qr(torch.randn(D, 4, device=DEV))[0]
fams = {"wdd": S.W2[torch.randperm(S.DFF, device=DEV)[:Kf]], "random": unit(torch.randn(Kf, D, device=DEV)), "sparse5": unit(sp), "onehot": oh, "rademacher": unit(torch.sign(torch.randn(Kf, D, device=DEV))), "heavy_tailed": unit(torch.distributions.StudentT(1.5).sample((Kf, D)).to(DEV)), "lowrank4": unit(torch.randn(Kf, 4, device=DEV) @ Q4.T), "rank1_scaled": unit(torch.randn(Kf, 1, device=DEV) * torch.randn(1, D, device=DEV)), "cov_whitened": whiten(cov()), "cov_amplified": unit((cov() @ U[2].T * U[1][None]) @ U[2]), "orthogonal_to_cloud": unit(torch.randn(Kf, D, device=DEV) - (torch.randn(Kf, D, device=DEV) @ U[2][:64].T) @ U[2][:64]), "parallel_mean": unit(mean_dir[None] + 0.1 * torch.randn(Kf, D, device=DEV)), "orthogonal_mean": unit(torch.randn(Kf, D, device=DEV) - (torch.randn(Kf, D, device=DEV) @ mean_dir)[:, None] * mean_dir), "transport_top": unit(Ut[:, :Kf].T), "transport_bottom": unit(Ut[:, -Kf:].T), "anti_natural": -cov(), "massive_channels": unit(massive)}
out = {}; Qw = None
for nm, V in fams.items():
    for al in (1.0, 4.0):
        r = S.inject_family(V, b + 1, amp=al * S.s_inj, seed=1); F = r["F"][L]; a = r["a"]; tr, te = S.halves(len(r["pos"]), seed=2); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(r["dl"], tr); dln = unit(r["dl"]); Q = pls(Fc[tr], Z[tr], 64); own = knn_cos(Fc @ Q[:, :d], dln, tr, te); rnd = knn_cos(Fc @ torch.linalg.qr(torch.randn(D, d, device=DEV))[0], dln, tr, te); curve = {q: knn_cos(Fc @ Q[:, :q], dln, tr, te) for q in (1, 2, 4, 8, 16, 32, 64)}; full = curve[64]; need = next((q for q in curve if curve[q] >= 0.9 * full), 64)
        if nm == "wdd" and al == 1.0: Qw = Q[:, :d]
        out[f"{nm}_{al}x"] = dict(own=own, random=rnd, advantage=own / max(rnd, 1e-6), dim=need, overlap_with_wdd=inside(Q[:, :d], Qw) if Qw is not None else 1.0, gain=(F.norm(dim=1) / (al * S.s_inj)).median().item(), identity=accuracy((Fc @ Q[:, :d])[te], centroids((Fc @ Q[:, :d])[tr], a[tr], Kf), a[te]))
worst = sorted(out.items(), key=lambda kv: kv[1]["advantage"])[:4]
log(f"{tag}: family_amplitude -> own-quotient score / random-16 score (advantage) | dim | overlap with WDD quotient | identity | gain :: " + " ; ".join(f"{k}: {v['own']:.2f}/{v['random']:.2f} ({v['advantage']:.2f}) | {v['dim']} | {v['overlap_with_wdd']:.2f} | {v['identity']:.2f} | {v['gain']:.2f}" for k, v in out.items()) + " || weakest advantage: " + ", ".join(f"{k} {v['advantage']:.2f}" for k, v in worst))
record(f"e342_destroy_{tag}", dict(model=tag, L=L, per_family=out), "weakest: " + ", ".join(f"{k} {v['advantage']:.2f} (own {v['own']:.2f})" for k, v in worst) + " | wdd 1x " + f"{out['wdd_1.0x']['own']:.2f}/{out['wdd_1.0x']['random']:.2f}")
