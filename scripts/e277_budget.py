"""e277: the energy budget of a per-token descendant across depth, and what provenance needs of it. At each level
after the write: the coherent fraction (energy along the class centroid, centroids from the other half of the
tokens), the fraction of the incoherent remainder that a ridge map from the top-64 principal components of the
state predicts on held-out tokens (from the birth-level state = state-mediated transport, and from the same-level
state), the held-out nearest-centroid identification, and the transport gain ||F||/|coef|. Provenance survives on
the coherent fraction; a decomposition would have to explain the remainder."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; levels = [lv for lv in sorted({b, b + 1, b + 2, b + 4, L, (L + NB - 1) // 2, NB - 2}) if lv < NB]
run = make_runner(model, arch, c, ids_seq, levels, NT); led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); S1 = run(b, tn)
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5
def pcs(S, d=64):
    Sc = S - S.mean(0, keepdim=True); U = torch.linalg.svd(Sc, full_matrices=False)[2][:d]; return Sc @ U.T
def ridge_r2(Xf, Y):
    Xtr, Ytr, Xte, Yte = Xf[split], Y[split], Xf[~split], Y[~split]; mx, my = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True); Xc, Yc = Xtr - mx, Ytr - my; G = Xc.T @ Xc; W = torch.linalg.solve(G + 1e-1 * G.diagonal().mean() * torch.eye(G.shape[0], device=DEV), Xc.T @ Yc); pred = (Xte - mx) @ W + my; return 1 - ((Yte - pred) ** 2).sum().item() / ((Yte - my) ** 2).sum().item()
Xb = pcs(S0[b][idx]); out = {}
for lv in levels:
    if lv <= b: continue
    F = (S0[lv] - S1[lv])[idx]; Cd = centroids(F[split], lab_i[split], K); proj = (F * Cd[lab_i]).sum(1); coh = proj ** 2 / (F ** 2).sum(1).clamp_min(1e-9); r = F - proj[:, None] * Cd[lab_i]
    rec = dict(coherent_fraction=coh[~split].median().item(), coherent_fraction_mean=coh[~split].mean().item(), r2_from_birth_state=ridge_r2(Xb, r), r2_from_same_level_state=ridge_r2(pcs(S0[lv][idx]), r), identification_heldout=accuracy(F[~split], Cd, lab_i[~split]), gain=(F.norm(dim=1) / tc[idx].abs()).median().item(), chance=1 / K)
    out[lv] = rec; log(f"{tag} level {lv} (K {K}, chance {1 / K:.2f}): coherent fraction {rec['coherent_fraction']:.2f} (mean {rec['coherent_fraction_mean']:.2f}), incoherent remainder predicted from birth state R2 {rec['r2_from_birth_state']:+.2f}, from same-level state R2 {rec['r2_from_same_level_state']:+.2f}, identification {rec['identification_heldout']:.2f}, gain {rec['gain']:.2f}")
record(f"e277_budget_{tag}", dict(model=tag, b=b, L=L, K=K, per_level={str(k): v for k, v in out.items()}), " | ".join(f"lv{lv}: coherent {v['coherent_fraction']:.2f}, R2 birth {v['r2_from_birth_state']:+.2f}, R2 same {v['r2_from_same_level_state']:+.2f}, id {v['identification_heldout']:.2f}, gain {v['gain']:.2f}" for lv, v in out.items()))
