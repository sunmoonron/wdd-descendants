"""e268: (1) SUFFICIENCY: predicting a held-out token's logit footprint by 5-nearest-neighbour regression from the
write vector alone (W: the neuron's atom), the descendant alone (D, at +2 and L), and both (D concatenated with W at
matched scale); kill if adding W improves D by more than 0.05. (2) QUOTIENT: for high-D / low-W pairs, the normalised
distances in W, D and F vs all pairs. (3) SIX-SPACE MATRIX per level: Spearman between the pairwise Grams of W, the
natural state centroids S, the descendant centroids D and the logit footprints F. (4) BOTTLENECK per level:
descendant dimensions (top principal components) for 90% of the full gain in identity, function (logit footprint
cosine) and behaviour (Spearman with the removal KL)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); levels = [lv for lv in sorted({b + 1, b + 2, b + 4, L, (L + NB - 1) // 2, NB - 2}) if lv < NB]
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, levels, NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0 = run(positions=idx); S1 = run(tn, positions=idx); dl = S0["lg"] - S1["lg"]; kl = (torch.log_softmax(S0["lg"], -1).exp() * (torch.log_softmax(S0["lg"], -1) - torch.log_softmax(S1["lg"], -1))).sum(1); dl = dl - dl.mean(1, keepdim=True); dln = unit(dl); torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]
def knn_cos(Ptr, Pte, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return ((unit(dln[tr][nn].mean(1)) * dln[te]).sum(1)).median().item()
def knn_kl(Ptr, Pte, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; a_, b_ = kl[tr][nn].mean(1), kl[te]; ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
Wtok = R[tn[idx]]; Cl = unit(centroids(dl, lab_i, K)); Gl = Cl @ Cl.T; Rw = unit(R[keep]); Gw = Rw @ Rw.T; out = {}
for lv in levels:
    F = (S0[lv] - S1[lv])[idx]; Fu = unit(F); Xs = (S0[lv] - c.s["mu"][lv + 1].to(DEV))[idx]
    suff = dict(W=knn_cos(Wtok[tr], Wtok[te]), D=knn_cos(Fu[tr], Fu[te]), WD=knn_cos(torch.cat([Fu[tr], Wtok[tr]], 1), torch.cat([Fu[te], Wtok[te]], 1)))
    Cd = unit(centroids(F, lab_i, K)); Cs = unit(centroids(Xs, lab_i, K)); Gd, Gs = Cd @ Cd.T, Cs @ Cs.T; six = dict(W_S=gram_spearman(Gw, Gs), W_D=gram_spearman(Gw, Gd), W_F=gram_spearman(Gw, Gl), S_D=gram_spearman(Gs, Gd), S_F=gram_spearman(Gs, Gl), D_F=gram_spearman(Gd, Gl))
    iu = torch.triu_indices(K, K, 1, device=DEV); gw, gd, gl = Gw[iu[0], iu[1]], Gd[iu[0], iu[1]], Gl[iu[0], iu[1]]; dist = lambda g: (2 - 2 * g).clamp_min(0).sqrt(); sel = (gd > gd.quantile(0.75)) & (gw < gw.median())
    quot = dict(n=int(sel.sum()), dW=dist(gw[sel]).mean().item() / dist(gw).mean().item(), dD=dist(gd[sel]).mean().item() / dist(gd).mean().item(), dF=dist(gl[sel]).mean().item() / dist(gl).mean().item())
    Fc = F[tr] - F[tr].mean(0, keepdim=True); Uf = torch.linalg.svd(Fc, full_matrices=False)[2]; dims = [dd for dd in (1, 2, 4, 8, 16, 32, 64, 128, 256, D) if dd <= D]; curves = {}
    for d in dims: P = Uf[:d].T; curves[d] = dict(identity=accuracy(F[te] @ P, centroids(F[tr] @ P, lab_i[tr], K), lab_i[te]), function=knn_cos(F[tr] @ P, F[te] @ P), behaviour=knn_kl(F[tr] @ P, F[te] @ P))
    def first_d(key):
        full = curves[dims[-1]][key]; base = min(curves[d][key] for d in dims)
        for d in dims:
            if curves[d][key] - base >= 0.9 * (full - base): return d
        return dims[-1]
    out[lv] = dict(sufficiency=suff, six=six, quotient=quot, dims_90=dict(identity=first_d("identity"), function=first_d("function"), behaviour=first_d("behaviour")), full=curves[dims[-1]])
    log(f"{tag} level {lv} (K {K}): sufficiency W {suff['W']:.2f} D {suff['D']:.2f} W+D {suff['WD']:.2f} | six-space Spearman W-S {six['W_S']:+.2f} W-D {six['W_D']:+.2f} W-F {six['W_F']:+.2f} S-D {six['S_D']:+.2f} S-F {six['S_F']:+.2f} D-F {six['D_F']:+.2f} | quotient (hiD/loW pairs, n {quot['n']}): relative distance W {quot['dW']:.2f} D {quot['dD']:.2f} F {quot['dF']:.2f} | dims for 90% gain: identity {out[lv]['dims_90']['identity']}, function {out[lv]['dims_90']['function']}, behaviour {out[lv]['dims_90']['behaviour']} (full {curves[dims[-1]]['identity']:.2f} / {curves[dims[-1]]['function']:.2f} / {curves[dims[-1]]['behaviour']:.2f})")
record(f"e268_sufficiency_{tag}", dict(model=tag, b=b, L=L, K=K, per_level={str(k): v for k, v in out.items()}), " | ".join(f"lv{lv}: W {v['sufficiency']['W']:.2f} D {v['sufficiency']['D']:.2f} W+D {v['sufficiency']['WD']:.2f}; D-F {v['six']['D_F']:+.2f} S-F {v['six']['S_F']:+.2f} W-F {v['six']['W_F']:+.2f}; quotient dW/dD/dF {v['quotient']['dW']:.2f}/{v['quotient']['dD']:.2f}/{v['quotient']['dF']:.2f}; dims id/func/beh {v['dims_90']['identity']}/{v['dims_90']['function']}/{v['dims_90']['behaviour']}" for lv, v in out.items()))
