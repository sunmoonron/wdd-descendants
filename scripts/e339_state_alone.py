"""e339: is the causal coordinate a coordinate of the perturbation or of the state? The natural descendant's core
coordinate at L is predicted on held-out tokens from: the residual state at the token (top-64 PCs, ridge), the
source neuron's identity (class mean), the WDD coefficient alone, the coefficient times the write's projection on the
core (the atom's coordinate, no context), coefficient plus state (ridge on both), the downstream state at L+2
without the perturbation, and the descendant itself (ceiling). R2 in the core."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S0_ = Setup(tag, levels=[0]); L = S0_.L; NB = S0_.NB; S = Setup(tag, NS=16, levels=[S0_.b, L, min(L + 2, NB - 2)]); lf = min(L + 2, NB - 2); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); P = pls(Fc[tr], Z[tr], d); z = Fc @ P
def pcs(X, k=64): Xc = X - X[tr].mean(0, keepdim=True); U = torch.linalg.svd(Xc[tr], full_matrices=False)[2][:k]; return Xc @ U.T
def r2(pred): return 1 - ((z[te] - pred[te]) ** 2).sum().item() / ((z[te] - z[tr].mean(0, keepdim=True)) ** 2).sum().item()
def ridge_pred(X): Xtr = X[tr]; mx, my = Xtr.mean(0, keepdim=True), z[tr].mean(0, keepdim=True); W = ridge(Xtr - mx, z[tr] - my, 1e-1); return (X - mx) @ W + my
coef = S.tc[S.idx][:, None]; wproj = (S.W2[S.tn[S.idx]] @ P) * coef; cls_mean = torch.stack([z[tr][S.lab_i[tr] == k].mean(0) if (S.lab_i[tr] == k).any() else z[tr].mean(0) for k in range(S.K)])[S.lab_i]
preds = {"state_at_L": ridge_pred(pcs(S.S0[L][S.idx])), "state_at_birth": ridge_pred(pcs(S.S0[S.b][S.idx])), "downstream_state_L2": ridge_pred(pcs(S.S0[lf][S.idx])), "source_identity": cls_mean, "coefficient_only": ridge_pred(coef), "atom_coordinate_x_coefficient": ridge_pred(wproj), "coefficient_plus_state": ridge_pred(torch.cat([pcs(S.S0[L][S.idx]), coef, wproj], 1)), "identity_plus_state": ridge_pred(torch.cat([pcs(S.S0[L][S.idx]), cls_mean], 1)), "descendant_pca64": ridge_pred(pcs(F))}
out = {k: r2(v) for k, v in preds.items()}
log(f"{tag} (K {S.K}): held-out R2 of the core coordinate from " + ", ".join(f"{k} {v:+.2f}" for k, v in out.items()))
record(f"e339_statealone_{tag}", dict(model=tag, L=L, K=S.K, r2=out), " ".join(f"{k} {v:+.2f}" for k, v in out.items()))
