"""e275: are the massive neurons the tail of the transport law or a separate mechanism? Fit the linear transport
operator T (ridge on random injections at the block-3 input read at L, as in e248); for every candidate neuron the
residual 1 - cos(d_k, T w_k) between its natural descendant centroid and its predicted transport; the largest-
coefficient neuron's residual as a rank and z-score among candidates; Spearman of the residual with median
|coefficient|, with natural identification, and with the along-direction survival; and the same residual for the
zero-shot transplant image (e242) where available."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); run = make_runner(model, arch, c, ids_seq, [L], NT)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(b, tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); F = S0[L] - S1[L]; Cd = unit(centroids(F[idx], lab_i, K)); med = torch.stack([tc[idx][lab_i == k].median().abs() for k in range(K)]); d_w = R[tn]; along = torch.stack([((F * d_w).sum(1) / tc)[idx][lab_i == k].median() for k in range(K)]); ident = torch.stack([accuracy(F[idx][lab_i == k], Cd, lab_i[lab_i == k]) if (lab_i == k).any() else torch.tensor(0.0) for k in range(K)]) if False else torch.tensor([((unit(F[idx][lab_i == k]) @ Cd.T).argmax(1) == k).float().mean().item() for k in range(K)], device=DEV)
pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); torch.manual_seed(0); Vr = unit(torch.randn(1024, D, device=DEV)); X = []; Y = []
for p in range(2):
    a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=b + 1); X.append(Vr[a]); Y.append((S2[L] - S0[L])[pool] / s_inj)
X, Y = torch.cat(X), torch.cat(Y); G = X.T @ X; T = torch.linalg.solve(G + 1e-2 * G.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y); pred = unit(R[keep] @ T); resid = 1 - (pred * Cd).sum(1)
M = int(med.argmax()); z = ((resid[M] - resid.mean()) / resid.std().clamp_min(1e-9)).item(); rank = int((resid > resid[M]).sum().item()) + 1
def spearman(a_, b_):
    ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
res = dict(K=K, massive_neuron=int(keep[M]), massive_coef=med[M].item(), median_coef=med.median().item(), massive_residual=resid[M].item(), residual_median=resid.median().item(), residual_max_other=resid[torch.arange(K, device=DEV) != M].max().item(), massive_z=z, massive_rank=rank, rho_resid_coef=spearman(resid, med), rho_resid_identification=spearman(resid, ident), rho_resid_along=spearman(resid, along))
log(f"{tag} (K {K}): transport residual 1 - cos(descendant, T w): median {res['residual_median']:.2f}, largest-coefficient neuron {int(keep[M])} (|coef| {med[M]:.1f} vs median {med.median():.1f}) has residual {res['massive_residual']:.2f} = rank {rank} of {K}, z {z:+.1f} (largest other {res['residual_max_other']:.2f}) | Spearman(residual, |coef|) {res['rho_resid_coef']:+.2f}, (residual, natural identification) {res['rho_resid_identification']:+.2f}, (residual, along-survival) {res['rho_resid_along']:+.2f}")
record(f"e275_tresid_{tag}", dict(model=tag, b=b, L=L, **res), f"K {K}: residual median {res['residual_median']:.2f}; massive neuron residual {res['massive_residual']:.2f} rank {rank}/{K} z {z:+.1f}; rho(resid, coef) {res['rho_resid_coef']:+.2f}, (resid, identification) {res['rho_resid_identification']:+.2f}, (resid, along) {res['rho_resid_along']:+.2f}")
