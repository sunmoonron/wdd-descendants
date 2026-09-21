"""e254: does descendant geometry predict functional geometry? For the K candidate block-2 neurons: the Gram (cosine)
matrix of their native write vectors, of their descendant centroids at +2 and L, and of their logit footprints (the
per-neuron mean change of the same-position logits when the write is removed). Spearman between the logit Gram and
the other Grams; K-way identification of the source from the per-token logit footprint (functional identity) and its
agreement with the descendant identification; and per token, whether the descendant's nearest neuron predicts the
logit footprint's nearest neuron better than the native atom does."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; levels = sorted({b + 2, L}); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
def run(neuron=None, positions=None):
    st = {}; hs = [arch.layers[lv].register_forward_hook((lambda lv_: lambda m, i, o: st.__setitem__(lv_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, D)))(lv)) for lv in levels]
    if neuron is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    out = model(ids_seq); [h.remove() for h in hs]; lg = out.logits.reshape(NT, -1).float(); st["lg"] = lg[positions] if positions is not None else None; del out; return st
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); S0 = run(positions=idx); S1 = run(tn, positions=idx); dl = S0["lg"] - S1["lg"]; dl = dl - dl.mean(1, keepdim=True)
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]; ltr, lte = lab_i[tr], lab_i[te]
def gram_spearman(G1, G2):
    iu = torch.triu_indices(K, K, 1, device=DEV); a_, b_ = G1[iu[0], iu[1]], G2[iu[0], iu[1]]; ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
Cl = unit(centroids(dl[tr], ltr, K)); Gl = Cl @ Cl.T; Rw = unit(R[keep]); Gw = Rw @ Rw.T; res = dict(K=K, spearman_logit_vs_write=gram_spearman(Gl, Gw), functional_id=accuracy(dl[te], Cl, lte), chance=1.0 / K)
func_pred = (unit(dl[te]) @ Cl.T).argmax(1)
for lv in levels:
    F = S0[lv] - S1[lv]; Cd = centroids(F[idx][tr], ltr, K); Gd = Cd @ Cd.T; desc_pred = (unit(F[idx][te]) @ Cd.T).argmax(1); nat_pred = (unit(F[idx][te]) @ Rw.T).argmax(1)
    res[f"lv{lv}"] = dict(spearman_logit_vs_descendant=gram_spearman(Gl, Gd), spearman_write_vs_descendant=gram_spearman(Gw, Gd), descendant_id=(desc_pred == lte).float().mean().item(), agreement_desc_vs_functional=(desc_pred == func_pred).float().mean().item(), agreement_native_vs_functional=(nat_pred == func_pred).float().mean().item())
    log(f"{tag} level {lv} (K {K}): Gram Spearman logit-vs-descendant {res[f'lv{lv}']['spearman_logit_vs_descendant']:+.2f}, logit-vs-write {res['spearman_logit_vs_write']:+.2f}, write-vs-descendant {res[f'lv{lv}']['spearman_write_vs_descendant']:+.2f} | source from the logit footprint {res['functional_id']:.2f}, from the descendant {res[f'lv{lv}']['descendant_id']:.2f} (chance {1 / K:.2f}) | per-token agreement with the functional identity: descendant {res[f'lv{lv}']['agreement_desc_vs_functional']:.2f} vs native atom {res[f'lv{lv}']['agreement_native_vs_functional']:.2f}")
record(f"e254_bridge_{tag}", dict(model=tag, b=b, L=L, **res), f"K {K}: source from logit footprint {res['functional_id']:.2f} (chance {res['chance']:.2f}); Gram Spearman logit-vs-write {res['spearman_logit_vs_write']:+.2f}, logit-vs-descendant " + " ".join(f"lv{lv}:{res[f'lv{lv}']['spearman_logit_vs_descendant']:+.2f}" for lv in levels) + " | agreement with functional identity: descendant " + " ".join(f"lv{lv}:{res[f'lv{lv}']['agreement_desc_vs_functional']:.2f}" for lv in levels) + " vs native " + " ".join(f"lv{lv}:{res[f'lv{lv}']['agreement_native_vs_functional']:.2f}" for lv in levels))
