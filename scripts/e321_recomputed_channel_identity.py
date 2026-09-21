"""e321: channel identity with bases recomputed at every level (not a fixed basis). The logit quotient is refitted at
each level from the level's own descendants; consecutive-level coordinates are matched by the optimal assignment on
absolute correlation; reported: the fraction of coordinates that keep their index, the strength of the matched
correlation, the fraction of energy the two bases share, and how much of the identity survives over the full depth
by composing the matchings."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
try:
    from scipy.optimize import linear_sum_assignment
except Exception: linear_sum_assignment = None
tag = sys.argv[1]; S0_ = Setup(tag, levels=[0]); L = S0_.L; NB = S0_.NB; b = 2; S = Setup(tag, levels=list(range(b + 1, NB))); d = 16; nat = S.natural(); tr, te = S.halves(len(S.idx)); Z, _ = logit_scores(nat["dl"], tr); Ps = {}; zs = {}
for lv in S.levels:
    F = nat["F"][lv]; Fc = F - F[tr].mean(0, keepdim=True); P = pls(Fc[tr], Z[tr], d); s = torch.sign((Fc[tr] @ P).sum(0)).clamp_min(0) * 2 - 1; Ps[lv] = P; zs[lv] = Fc @ P
keep_idx = []; strength = []; shared = []; perm = torch.arange(d)
for i in range(len(S.levels) - 1):
    a, b_ = S.levels[i], S.levels[i + 1]; C = torch.corrcoef(torch.cat([zs[a], zs[b_]], 1).T)[:d, d:].abs().nan_to_num()
    if linear_sum_assignment is not None: ri, ci = linear_sum_assignment(-C.cpu().numpy()); ci = torch.tensor(ci)
    else: ci = C.argmax(1).cpu()
    keep_idx.append(float((ci == torch.arange(d)).float().mean())); strength.append(C[torch.arange(d), ci.to(DEV)].mean().item()); shared.append(inside(Ps[a], Ps[b_])); perm = perm[ci]
res = dict(K=S.K, index_kept_mean=sum(keep_idx) / len(keep_idx), index_kept_min=min(keep_idx), matched_corr_mean=sum(strength) / len(strength), matched_corr_min=min(strength), basis_shared_mean=sum(shared) / len(shared), basis_shared_min=min(shared), composed_identity=float((perm == torch.arange(d)).float().mean()), per_step_kept=keep_idx, per_step_shared=shared)
log(f"{tag} (K {S.K}, chance {d / S.D:.3f}): with bases refitted at every level, consecutive coordinates keep their index in {res['index_kept_mean']:.2f} of cases (min {res['index_kept_min']:.2f}), matched |corr| {res['matched_corr_mean']:.2f} (min {res['matched_corr_min']:.2f}), consecutive bases share {res['basis_shared_mean']:.2f} of their energy (min {res['basis_shared_min']:.2f}); composing the matchings over the whole depth returns {res['composed_identity']:.2f} of coordinates to themselves | per step kept: " + " ".join(f"{v:.2f}" for v in keep_idx))
record(f"e321_recomputed_{tag}", dict(model=tag, L=L, **res), f"index kept {res['index_kept_mean']:.2f} (min {res['index_kept_min']:.2f}) matched corr {res['matched_corr_mean']:.2f} basis shared {res['basis_shared_mean']:.2f} composed {res['composed_identity']:.2f}")
