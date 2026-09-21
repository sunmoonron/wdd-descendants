"""e335: an observable-independent quotient, and a universal response basis. (1) Bases fitted to nothing downstream:
the descendant cloud's top-16 PCs, its top-16 whitened directions, a covariance-matched random 16-subspace, a
random 16-subspace; each tested on every observable (logits, KL, entropy, future state, identity) against the
observable's own quotient. (2) The universal basis: the top directions of the concatenated standardised targets
(logits, future, KL, entropy, identity) by PLS; the dimension at which every observable reaches 90% of its own
score at once, and the overlap of the universal basis with each single-observable quotient."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S0_ = Setup(tag, levels=[0]); L = S0_.L; NB = S0_.NB; S = Setup(tag, levels=[L, min(L + 2, NB - 2)]); lf = min(L + 2, NB - 2); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); dln = unit(nat["dl"]); Ff = unit(nat["F"][lf]); Y1 = torch.nn.functional.one_hot(S.lab_i, S.K).float()
targets = {"logits": Z, "future": Ff, "kl": nat["kl"][:, None], "entropy": nat["entropy"][:, None], "identity": Y1}
def own_q(key):
    Y = targets[key]
    if Y.shape[1] == 1: yc = Y[tr] - Y[tr].mean(); return torch.linalg.eigh(Fc[tr].T @ (yc * Fc[tr]))[1].flip(1)[:, :d]
    return pls(Fc[tr], Y[tr], d)
def score(P, key, dd=None):
    X = Fc @ (P if dd is None else P[:, :dd]); Y = targets[key]
    if key == "identity": return accuracy(X[te], centroids(X[tr], S.lab_i[tr], S.K), S.lab_i[te])
    if key == "logits": return knn_cos(X, dln, tr, te)
    if key == "future": return knn_cos(X, Ff, tr, te)
    return knn_spear(X, Y[:, 0], tr, te)
Qown = {k: own_q(k) for k in targets}; own = {k: score(Qown[k], k) for k in targets}; torch.manual_seed(0); Sc = Fc[tr]; U = torch.linalg.svd(Sc, full_matrices=False); pca = U[2][:d].T; white = U[2][:d].T; cov_r = torch.linalg.qr(((torch.randn(d, len(tr), device=DEV) / len(tr) ** 0.5) @ Sc).T)[0]; rnd = torch.linalg.qr(torch.randn(S.D, d, device=DEV))[0]
blind = {"pca16": pca, "cov_random16": cov_r, "random16": rnd}; blind_sc = {nm: {k: score(P, k) for k in targets} for nm, P in blind.items()}
def std(Y): Yc = Y - Y[tr].mean(0, keepdim=True); return Yc / Yc[tr].pow(2).sum().sqrt()
Yall = torch.cat([std(targets[k]) for k in targets], 1); Uni = pls(Fc[tr], Yall[tr], 64); curves = {k: {dd: score(Uni, k, dd) for dd in (1, 2, 4, 8, 16, 32, 64)} for k in targets}; need_all = next((dd for dd in (1, 2, 4, 8, 16, 32, 64) if all(curves[k][dd] - (1 / S.K if k == "identity" else 0) >= 0.9 * (own[k] - (1 / S.K if k == "identity" else 0)) for k in targets)), 64); ovu = {k: inside(Qown[k], Uni[:, :d]) for k in targets}
log(f"{tag} (K {S.K}, chance {d / S.D:.3f}): own-quotient scores " + " ".join(f"{k} {own[k]:.2f}" for k in targets) + " | blind bases (pca16 / cov-random16 / random16): " + " ".join(f"{k} {blind_sc['pca16'][k]:.2f}/{blind_sc['cov_random16'][k]:.2f}/{blind_sc['random16'][k]:.2f}" for k in targets) + f" | universal basis: dimension at which every observable reaches 90% of its own score {need_all}; universal-16 scores " + " ".join(f"{k} {curves[k][16]:.2f}" for k in targets) + "; overlap with own quotients " + " ".join(f"{k} {ovu[k]:.2f}" for k in targets))
record(f"e335_universal_{tag}", dict(model=tag, L=L, K=S.K, own=own, blind=blind_sc, universal_dim=need_all, universal_curves={k: {str(a): b_ for a, b_ in v.items()} for k, v in curves.items()}, overlap_universal=ovu), f"universal dim {need_all}; own " + " ".join(f"{k} {own[k]:.2f}" for k in targets) + " | pca16 " + " ".join(f"{blind_sc['pca16'][k]:.2f}" for k in targets) + " | uni16 " + " ".join(f"{curves[k][16]:.2f}" for k in targets))
