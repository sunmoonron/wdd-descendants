"""e255: does descendant geometry organise neurons by function? K block-2 candidates with natural footprints at L and
same-position logit footprints. (1) kNN test: for each neuron, its 3 nearest neighbours in write space and in
descendant space; the mean logit-footprint cosine with those neighbours (and with 3 random neurons). (2) Held-out
functional prediction: neurons split in halves; a held-out neuron's logit footprint predicted as the mean logit
footprint of its 3 nearest train neurons in descendant space vs write space; cos(predicted, actual). (3) Conditional:
functional similarity of pairs with dissimilar write vectors but similar descendants, and the reverse. (4) Excluding
the largest-coefficient neuron. Split: descendants organise by function (descendant neighbours more functionally
similar than write neighbours, held-out prediction transfers) vs not."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0 = run(positions=idx); S1 = run(tn, positions=idx); F = S0[L] - S1[L]; dl = S0["lg"] - S1["lg"]; dl = dl - dl.mean(1, keepdim=True)
Cd = unit(centroids(F[idx], lab_i, K)); Cl = unit(centroids(dl, lab_i, K)); Rw = unit(R[keep]); Gd, Gl, Gw = Cd @ Cd.T, Cl @ Cl.T, Rw @ Rw.T; med = torch.stack([tc[idx][lab_i == k].median().abs() for k in range(K)]); M = med.argmax()
def knn_func(G, k=3, exclude=None):
    G2 = G.clone(); G2.fill_diagonal_(-2); nn = G2.topk(k, dim=1).indices; vals = Gl[torch.arange(K, device=DEV)[:, None], nn].mean(1)
    if exclude is not None: vals = vals[torch.arange(K, device=DEV) != exclude]
    return vals.mean().item()
torch.manual_seed(0); rand_nn = torch.stack([torch.randperm(K, device=DEV)[:3] for _ in range(K)]); rand_func = Gl[torch.arange(K, device=DEV)[:, None], rand_nn].mean().item()
res = dict(K=K, knn_functional_similarity=dict(descendant=knn_func(Gd), write=knn_func(Gw), random=rand_func, descendant_excl_massive=knn_func(Gd, exclude=M), write_excl_massive=knn_func(Gw, exclude=M)), gram_spearman=dict(desc_logit=gram_spearman(Gd, Gl), write_logit=gram_spearman(Gw, Gl)))
# held-out prediction by kNN transfer
torch.manual_seed(1); perm = torch.randperm(K, device=DEV); half = perm[:K // 2]; rest = perm[K // 2:]; pred = {}
for nm, G in (("descendant", Gd), ("write", Gw)):
    sims = G[rest][:, half]; nn = sims.topk(3, dim=1).indices; predicted = unit(Cl[half][nn].mean(1)); pred[nm] = ((predicted * Cl[rest]).sum(1)).median().item()
pred["random"] = ((unit(Cl[half][torch.randint(0, len(half), (len(rest), 3), device=DEV)].mean(1)) * Cl[rest]).sum(1)).median().item(); res["heldout_functional_prediction_cos"] = pred
# conditional: pairs
iu = torch.triu_indices(K, K, 1, device=DEV); gw, gd, gl = Gw[iu[0], iu[1]], Gd[iu[0], iu[1]], Gl[iu[0], iu[1]]; lo_w = gw < gw.quantile(0.5); hi_d = gd > gd.quantile(0.75); lo_d = gd < gd.quantile(0.25); hi_w = gw > gw.quantile(0.75)
res["conditional"] = dict(func_sim_dissimilar_writes_similar_descendants=gl[lo_w & hi_d].mean().item() if (lo_w & hi_d).any() else float("nan"), func_sim_dissimilar_writes_dissimilar_descendants=gl[lo_w & lo_d].mean().item() if (lo_w & lo_d).any() else float("nan"), func_sim_similar_writes=gl[hi_w].mean().item(), func_sim_all=gl.mean().item(), n_pairs_lo_w_hi_d=int((lo_w & hi_d).sum()))
log(f"{tag} (K {K}): kNN functional similarity via descendant neighbours {res['knn_functional_similarity']['descendant']:.3f} vs write neighbours {res['knn_functional_similarity']['write']:.3f} vs random {res['knn_functional_similarity']['random']:.3f} (excluding the largest neuron: {res['knn_functional_similarity']['descendant_excl_massive']:.3f} vs {res['knn_functional_similarity']['write_excl_massive']:.3f}) | held-out logit-footprint prediction cos: descendant kNN {pred['descendant']:.2f}, write kNN {pred['write']:.2f}, random {pred['random']:.2f} | pairs with dissimilar writes: functional similarity {res['conditional']['func_sim_dissimilar_writes_similar_descendants']:.3f} when descendants are similar vs {res['conditional']['func_sim_dissimilar_writes_dissimilar_descendants']:.3f} when dissimilar (all pairs {res['conditional']['func_sim_all']:.3f}, similar writes {res['conditional']['func_sim_similar_writes']:.3f})")
record(f"e255_funcnn_{tag}", dict(model=tag, b=b, L=L, **res), f"K {K}: kNN functional similarity descendant {res['knn_functional_similarity']['descendant']:.3f} vs write {res['knn_functional_similarity']['write']:.3f} vs random {res['knn_functional_similarity']['random']:.3f} | held-out prediction cos descendant {pred['descendant']:.2f} vs write {pred['write']:.2f} vs random {pred['random']:.2f} | dissimilar writes: similar descendants {res['conditional']['func_sim_dissimilar_writes_similar_descendants']:.3f} vs dissimilar {res['conditional']['func_sim_dissimilar_writes_dissimilar_descendants']:.3f}")
