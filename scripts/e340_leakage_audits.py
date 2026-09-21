"""e340: leakage audits and a permutation test of the quotient pipeline. Baselines that use no descendant geometry:
norm only, token position only, source-neuron identity only, coefficient only, the original write only, a random
matched-dimension subspace, a random matched-covariance subspace; each decodes the logit footprint, identity and
KL. Permutation test: the pipeline rerun with the logit footprints shuffled across tokens (the quotient fitted to
nonsense) and with the descendants shuffled; the resulting held-out scores are the null."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); dln = unit(nat["dl"]); kl = nat["kl"]; P = pls(Fc[tr], Z[tr], d)
def scores(X, lab=S.lab_i, target=dln, klv=kl): return dict(function=knn_cos(X, target, tr, te), identity=accuracy(X[te], centroids(X[tr], lab[tr], S.K), lab[te]), kl=knn_spear(X, klv, tr, te))
torch.manual_seed(0); Sc = Fc[tr]; inputs = {"quotient16": Fc @ P, "norm_only": F.norm(dim=1, keepdim=True), "position_only": (S.idx % CTX).float()[:, None], "source_identity_only": torch.nn.functional.one_hot(S.lab_i, S.K).float(), "coefficient_only": S.tc[S.idx][:, None], "original_write": S.W2[S.tn[S.idx]] * S.tc[S.idx][:, None], "random16": Fc @ torch.linalg.qr(torch.randn(S.D, d, device=DEV))[0], "cov_random16": Fc @ torch.linalg.qr(((torch.randn(d, len(tr), device=DEV) / len(tr) ** 0.5) @ Sc).T)[0]}
out = {k: scores(X) for k, X in inputs.items()}
perm = torch.randperm(len(S.idx), device=DEV); Zp = Z[perm]; Pp = pls(Fc[tr], Zp[tr], d); out["permuted_targets"] = scores(Fc @ Pp); Fp = Fc[perm]; Pq = pls(Fp[tr], Z[tr], d); out["permuted_descendants"] = dict(function=knn_cos(Fp @ Pq, dln, tr, te), identity=accuracy((Fp @ Pq)[te], centroids((Fp @ Pq)[tr], S.lab_i[tr], S.K), S.lab_i[te]), kl=knn_spear(Fp @ Pq, kl, tr, te))
log(f"{tag} (K {S.K}): function / identity / KL scores: " + " | ".join(f"{k} {v['function']:.2f}/{v['identity']:.2f}/{v['kl']:.2f}" for k, v in out.items()))
record(f"e340_leakage_{tag}", dict(model=tag, L=L, K=S.K, scores=out), " | ".join(f"{k} {v['function']:.2f}/{v['identity']:.2f}/{v['kl']:.2f}" for k, v in out.items()))
