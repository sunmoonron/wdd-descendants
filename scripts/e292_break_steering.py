"""e292: try to break e289. A ridge map from descendants to logit-footprint scores is fitted on half the candidate
neurons' tokens; it predicts the ENTIRE logit change of an injection (through the footprint PCA basis), not only its
projection on eight paired directions. Injected at foreign tokens at the natural footprint norm: the eight PLS
coordinates, the centroids of the held-out neurons, the centroids of the training neurons, covariance-matched random
directions (in the cloud), plain random directions (out of the cloud) and the cloud's top PCs. Reported: the median
cosine between the actual logit change and the predicted full vector per class, and for the PLS coordinates the
fraction of the actual response inside the span of the 32 paired logit directions against random 32-dimensional
subspaces of the same footprint PCA space."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); F = (S0i[L] - S1i[L])[idx]; fnorm = F.norm(dim=1).median()
torch.manual_seed(0); cls_tr = torch.rand(K, device=DEV) < 0.5; tr = torch.nonzero(cls_tr[lab_i])[:, 0]; Fmu = F[tr].mean(0, keepdim=True); Fc = F - Fmu
Ud, Sd, Vd = torch.linalg.svd(dl, full_matrices=False); Zs = Ud[:, :256] * Sd[:256][None]; B = Vd[:256].T; zmu = Zs[tr].mean(0, keepdim=True); Zc = Zs - zmu
G = Fc[tr].T @ Fc[tr]; W = torch.linalg.solve(G + 1e-1 * G.diagonal().mean() * torch.eye(D, device=DEV), Fc[tr].T @ Zc[tr]); Up, Sp, Wt = torch.linalg.svd(Fc[tr].T @ Zc[tr], full_matrices=False); u = Up[:, :8].T; lpair = unit((B @ Wt[:32].T).T)
Cd = centroids(F, lab_i, K); cov_dirs = unit((torch.randn(8, len(tr), device=DEV) / len(tr) ** 0.5) @ Fc[tr]); pcs = torch.linalg.svd(Fc[tr], full_matrices=False)[2][:8]
kinds = {"pls_coordinates": u, "heldout_centroids": Cd[~cls_tr][:8], "training_centroids": Cd[cls_tr][:8], "cov_matched_random": cov_dirs, "plain_random": unit(torch.randn(8, D, device=DEV)), "top_pcs": pcs}
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0][:1536]; NF = len(foreign); base = run(positions=foreign); lg0 = base["lg"]; out = {}
def respond(v):
    inj = torch.zeros(NT, D, device=DEV); inj[foreign] = fnorm * v[None]; r = run(positions=foreign, inject=inj, inject_block=L + 1); dlr = r["lg"] - lg0; return dlr - dlr.mean(1, keepdim=True)
Prand = [torch.linalg.qr(torch.randn(256, 32, device=DEV))[0] for _ in range(4)]
for nm, dirs in kinds.items():
    cs = []; inspan = []; inrand = []
    for k in range(dirs.shape[0]):
        E = respond(dirs[k]); pred = ((fnorm * dirs[k][None]) @ W) @ B.T; cs.append(((unit(E) * unit(pred)).sum(1)).median().item()); Ez = E @ B; inpca = (Ez ** 2).sum(1) / (E ** 2).sum(1).clamp_min(1e-9)
        if nm == "pls_coordinates": inspan.append((((E @ lpair.T) ** 2).sum(1) / (E ** 2).sum(1).clamp_min(1e-9)).median().item()); inrand.append(float(sum((((Ez @ P) ** 2).sum(1) / (E ** 2).sum(1).clamp_min(1e-9)).median().item() for P in Prand) / 4))
    out[nm] = dict(cos_actual_vs_predicted_full=sum(cs) / len(cs), per_direction=cs, in_pca256=inpca.median().item())
    if inspan: out[nm]["in_paired_span32"] = sum(inspan) / len(inspan); out[nm]["in_random_span32"] = sum(inrand) / len(inrand)
log(f"{tag} (K {K}): cosine of the actual logit change with the predicted full vector: " + ", ".join(f"{nm} {v['cos_actual_vs_predicted_full']:.2f}" for nm, v in out.items()) + f" | PLS coordinates: response energy inside the 32 paired logit directions {out['pls_coordinates']['in_paired_span32']:.2f} vs random 32-subspaces of the footprint PCA {out['pls_coordinates']['in_random_span32']:.2f} (inside the 256-PCA {out['pls_coordinates']['in_pca256']:.2f})")
record(f"e292_break_{tag}", dict(model=tag, b=b, L=L, K=K, per_kind=out), "cos actual vs predicted full: " + ", ".join(f"{nm} {v['cos_actual_vs_predicted_full']:.2f}" for nm, v in out.items()) + f" | paired-span32 {out['pls_coordinates']['in_paired_span32']:.2f} vs random32 {out['pls_coordinates']['in_random_span32']:.2f}")
