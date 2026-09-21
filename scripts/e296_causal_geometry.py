"""e296: causal geometry. (A) A causal metric M = W W^T from the ridge map W (descendant -> logit-footprint scores, fitted
on half the tokens): does the M-cosine of descendant centroids recover the functional geometry (held-out footprint
cosines) better than the Euclidean cosine, and is the cloud lower-rank under M than under the Euclidean metric
(participation ratio)? (K) Principal-angle energies between the top-16 subspaces of the activation covariance at L,
the descendant covariance, and the sensitivity metric M. (C) Per-coordinate cross-layer alignment in the fixed
16-dimensional function basis: correlation of each coordinate with itself two blocks later against the best other
coordinate."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lf = L + 2 if L + 2 < NB - 1 else NB - 2; d = 16
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L, lf], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); F = (S0i[L] - S1i[L])[idx]; Ff = (S0i[lf] - S1i[lf])[idx]
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]; Fc = F - F[tr].mean(0, keepdim=True)
G = dl[tr] @ dl[tr].T; evg, Vg = torch.linalg.eigh(G); Bv = dl[tr].T @ (Vg.flip(1)[:, :256] / evg.flip(0)[:256].clamp_min(1e-6).sqrt()[None]); Zs = dl @ Bv; Zc = Zs - Zs[tr].mean(0, keepdim=True)
Gx = Fc[tr].T @ Fc[tr]; W = torch.linalg.solve(Gx + 1e-1 * Gx.diagonal().mean() * torch.eye(D, device=DEV), Fc[tr].T @ Zc[tr]); M = W @ W.T
Cd = centroids(F[te], lab_i[te], K); Cf = centroids(dl[te], lab_i[te], K); iu = torch.triu_indices(K, K, 1, device=DEV)
def spearman(a_, b_):
    ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
GE = (Cd @ Cd.T)[iu[0], iu[1]]; CM = Cd @ M; GM = ((CM @ Cd.T) / torch.sqrt(torch.outer((CM * Cd).sum(1), (CM * Cd).sum(1))).clamp_min(1e-9))[iu[0], iu[1]]; GF = (Cf @ Cf.T)[iu[0], iu[1]]
def prank(X):
    ev = torch.linalg.eigvalsh(X.T @ X / len(X)).clamp_min(0); return (ev.sum() ** 2 / (ev ** 2).sum()).item()
evM, VM = torch.linalg.eigh(M); Mh = VM @ torch.diag(evM.clamp_min(0).sqrt()) @ VM.T; geom = dict(spearman_euclid_vs_functional=spearman(GE, GF), spearman_causal_metric_vs_functional=spearman(GM, GF), participation_rank_euclid=prank(Fc[te]), participation_rank_causal=prank(Fc[te] @ Mh))
Sact = torch.linalg.svd(S0[L][typ] - S0[L][typ].mean(0, keepdim=True), full_matrices=False)[2][:d].T; Sdesc = torch.linalg.svd(Fc[tr], full_matrices=False)[2][:d].T; Ssens = VM.flip(1)[:, :d]; inside = lambda A, B: ((B.T @ A) ** 2).sum().item() / A.shape[1]; angles = dict(activation_vs_descendant=inside(Sact, Sdesc), activation_vs_sensitivity=inside(Sact, Ssens), descendant_vs_sensitivity=inside(Sdesc, Ssens), chance=d / D)
U_fn = torch.linalg.svd(Fc[tr].T @ Zc[tr], full_matrices=False)[0][:, :d]; zL = (Fc @ U_fn)[te]; zf = ((Ff - Ff[tr].mean(0, keepdim=True)) @ U_fn)[te]; C = torch.corrcoef(torch.cat([zL, zf], 1).T)[:d, d:]; diag = C.diagonal().abs(); best_other = (C.abs() - torch.diag(diag)).max(1).values; align = dict(self_correlation_median=diag.median().item(), best_other_median=best_other.median().item(), self_wins=(diag > best_other).float().mean().item())
log(f"{tag} (K {K}): (A) Spearman of centroid geometry with the held-out functional geometry: Euclidean {geom['spearman_euclid_vs_functional']:.2f}, causal metric {geom['spearman_causal_metric_vs_functional']:.2f}; participation rank of the cloud Euclidean {geom['participation_rank_euclid']:.1f} vs under the causal metric {geom['participation_rank_causal']:.1f} | (K) top-16 subspace energies: activation-descendant {angles['activation_vs_descendant']:.2f}, activation-sensitivity {angles['activation_vs_sensitivity']:.2f}, descendant-sensitivity {angles['descendant_vs_sensitivity']:.2f} (chance {d / D:.3f}) | (C) fixed-basis coordinates two blocks later: self-correlation {align['self_correlation_median']:.2f} vs best other coordinate {align['best_other_median']:.2f}, self wins {align['self_wins']:.2f}")
record(f"e296_geometry_{tag}", dict(model=tag, b=b, L=L, K=K, geometry=geom, angles=angles, alignment=align), f"Spearman Euclid {geom['spearman_euclid_vs_functional']:.2f} causal {geom['spearman_causal_metric_vs_functional']:.2f}; prank E {geom['participation_rank_euclid']:.1f} M {geom['participation_rank_causal']:.1f} | angles act-desc {angles['activation_vs_descendant']:.2f} act-sens {angles['activation_vs_sensitivity']:.2f} desc-sens {angles['descendant_vs_sensitivity']:.2f} | coord self {align['self_correlation_median']:.2f} vs other {align['best_other_median']:.2f} wins {align['self_wins']:.2f}")
