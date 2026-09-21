"""e263: how many descendant dimensions predict FUNCTION, vs identity? Natural footprints F at L and same-position
logit footprints at the candidate tokens. For d = 1..D: the descendant projected (random Gaussian, and top-d PCs of F)
to d dimensions; a held-out token's logit footprint predicted by the mean logit footprint of its 5 nearest training
tokens in the projected space (kNN regression), scored by cosine; and source identification at the same d. Baselines:
the neuron's own mean logit footprint (identity oracle), the global mean logit footprint, random projections of the
state at the same d."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0 = run(positions=idx); S1 = run(tn, positions=idx); F = (S0[L] - S1[L])[idx]; X = (S0[L] - c.s["mu"][L + 1].to(DEV))[idx]; dl = S0["lg"] - S1["lg"]; dl = dl - dl.mean(1, keepdim=True); dln = unit(dl)
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]
def knn_predict(Ptr, Pte, k=5):
    d2 = torch.cdist(Pte, Ptr); nn = d2.topk(k, dim=1, largest=False).indices; pred = unit(dln[tr][nn].mean(1)); return ((pred * dln[te]).sum(1)).median().item()
def ident(Ptr, Pte): return accuracy(Pte, centroids(Ptr, lab_i[tr], K), lab_i[te])
oracle = ((unit(centroids(dl[tr], lab_i[tr], K))[lab_i[te]] * dln[te]).sum(1)).median().item(); gmean = ((unit(dl[tr].mean(0, keepdim=True)) * dln[te]).sum(1)).median().item()
Fc = F[tr] - F[tr].mean(0, keepdim=True); Uf = torch.linalg.svd(Fc, full_matrices=False)[2]; out = {}
for d in [dd for dd in (1, 2, 4, 8, 16, 32, 64, 128, 256, D) if dd <= D]:
    torch.manual_seed(d); P = torch.randn(D, d, device=DEV) / math.sqrt(d); Pp = Uf[:d].T
    out[d] = dict(func_desc_random=knn_predict(F[tr] @ P, F[te] @ P), func_desc_pca=knn_predict(F[tr] @ Pp, F[te] @ Pp), func_state_random=knn_predict(X[tr] @ P, X[te] @ P), id_desc_random=ident(F[tr] @ P, F[te] @ P), id_desc_pca=ident(F[tr] @ Pp, F[te] @ Pp))
    log(f"{tag} d={d}: function prediction cos via descendant random/pca {out[d]['func_desc_random']:.2f}/{out[d]['func_desc_pca']:.2f}, via state random {out[d]['func_state_random']:.2f} | identity via descendant random/pca {out[d]['id_desc_random']:.2f}/{out[d]['id_desc_pca']:.2f}")
def first_d(key, frac=0.9):
    full = out[max(out)][key]; base = gmean if key.startswith("func") else 0.0
    for d in sorted(out):
        if out[d][key] - base >= frac * (full - base): return d
    return max(out)
log(f"{tag} baselines: identity oracle {oracle:.2f}, global mean {gmean:.2f} | dims for 90% of the full gain: function (descendant pca) {first_d('func_desc_pca')} vs identity (descendant pca) {first_d('id_desc_pca')}; random projections: function {first_d('func_desc_random')} vs identity {first_d('id_desc_random')}")
record(f"e263_funcdim_{tag}", dict(model=tag, b=b, L=L, K=K, oracle=oracle, global_mean=gmean, per_d={str(k): v for k, v in out.items()}), f"K {K}; identity oracle {oracle:.2f}, global mean {gmean:.2f}; function by d (desc pca): " + " ".join(f"{d}:{v['func_desc_pca']:.2f}" for d, v in out.items()) + " | identity by d (desc pca): " + " ".join(f"{d}:{v['id_desc_pca']:.2f}" for d, v in out.items()) + f" | dims for 90% of full gain: function {first_d('func_desc_pca')} vs identity {first_d('id_desc_pca')}")
