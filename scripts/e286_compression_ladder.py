"""e286: the compression test. Every downstream observable is decoded from the same object, the per-token descendant
at the middle level, through a projection onto r directions chosen for that observable (supervised: between-class
scatter for identity, partial least squares for the logit footprint and for the future descendant, the footprint's
subspace for the removal KL; and unsupervised PCA for all), with one decoder family (nearest centroid for identity,
5-NN regression for the rest), fitted on half the tokens and scored on the other half. Reported: the smallest r at
which each observable reaches 90% of its full-dimension score. The theory's ladder: reconstruction > future
descendant > identity > function >= behaviour."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lf = L + 2 if L + 2 < NB - 1 else NB - 2
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L, lf], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0 = run(positions=idx); S1 = run(tn, positions=idx); dl = S0["lg"] - S1["lg"]; lp0, lp1 = torch.log_softmax(S0["lg"], -1), torch.log_softmax(S1["lg"], -1); kl = (lp0.exp() * (lp0 - lp1)).sum(1); dl = dl - dl.mean(1, keepdim=True); dln = unit(dl); F = (S0[L] - S1[L])[idx]; Ff = unit((S0[lf] - S1[lf])[idx])
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]; mu = F[tr].mean(0, keepdim=True); Fc = F - mu
G = dl[tr] @ dl[tr].T; evg, Vg = torch.linalg.eigh(G); Zt = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Zt = Zt - Zt.mean(0, keepdim=True)
U_pca = torch.linalg.svd(Fc[tr], full_matrices=False)[2]
cents = torch.stack([Fc[tr][lab_i[tr] == k].mean(0) for k in range(K)]); w = torch.bincount(lab_i[tr], minlength=K).float(); Sb = (cents * w[:, None]).T @ cents / w.sum(); U_id = torch.linalg.eigh(Sb)[1].flip(1).T
U_fn = torch.linalg.svd(Fc[tr].T @ Zt, full_matrices=False)[0].T; Fft = Ff[tr] - Ff[tr].mean(0, keepdim=True); U_fu = torch.linalg.svd(Fc[tr].T @ Fft, full_matrices=False)[0].T
def knn(Ptr, Pte, target, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return target[tr][nn].mean(1)
def score(U, r, obs):
    P = U[:r].T; Ptr, Pte = Fc[tr] @ P, Fc[te] @ P
    if obs == "reconstruction": return 1 - ((Fc[te] - Pte @ P.T) ** 2).sum().item() / (Fc[te] ** 2).sum().item()
    if obs == "identity": return accuracy(Pte, centroids(Ptr, lab_i[tr], K), lab_i[te])
    if obs == "function": return ((unit(knn(Ptr, Pte, dln)) * dln[te]).sum(1)).median().item()
    if obs == "future_descendant": return ((unit(knn(Ptr, Pte, Ff)) * Ff[te]).sum(1)).median().item()
    if obs == "behaviour":
        a_ = knn(Ptr, Pte, kl); ra = a_.argsort().argsort().float(); rb = kl[te].argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
dims = [dd for dd in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, D) if dd <= D]; subspaces = {"reconstruction": {"pca": U_pca}, "future_descendant": {"pca": U_pca, "supervised": U_fu}, "identity": {"pca": U_pca, "supervised": U_id}, "function": {"pca": U_pca, "supervised": U_fn}, "behaviour": {"pca": U_pca, "supervised": U_fn}}
out = {}
for obs, subs in subspaces.items():
    out[obs] = {}
    for nm, U in subs.items():
        curve = {r: score(U, min(r, U.shape[0]), obs) for r in dims}; full = curve[dims[-1]]; chance = 1 / K if obs == "identity" else 0.0; need = next((r for r in dims if curve[r] - chance >= 0.9 * (full - chance)), dims[-1]); out[obs][nm] = dict(required_dim=need, full_score=full, curve={str(r): v for r, v in curve.items()})
log(f"{tag} (K {K}, D {D}, L {L}, future {lf}): dimensions for 90% of the full score, PCA / supervised: " + " | ".join(f"{obs} {v['pca']['required_dim']}" + (f" / {v['supervised']['required_dim']}" if 'supervised' in v else "") + f" (full {v['pca']['full_score']:.2f})" for obs, v in out.items()))
record(f"e286_compression_{tag}", dict(model=tag, b=b, L=L, future=lf, K=K, D=D, per_observable=out), "PCA/supervised dims: " + " | ".join(f"{obs} {v['pca']['required_dim']}" + (f"/{v['supervised']['required_dim']}" if 'supervised' in v else "") + f" (full {v['pca']['full_score']:.2f})" for obs, v in out.items()))
