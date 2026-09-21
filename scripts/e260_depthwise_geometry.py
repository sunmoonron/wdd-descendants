"""e260: where does functional geometry move from write space into descendant space? For every level from the write
to the last block: Spearman between the pairwise Gram of the K neurons' descendant centroids at that level and (a)
the logit-footprint Gram, (b) the write-vector Gram; the kNN functional similarity of descendant neighbours at that
level; and, for reference, the write-logit Spearman (constant). One natural ablation run with logits."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); levels = list(range(b, NB))
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, levels, NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0 = run(positions=idx); S1 = run(tn, positions=idx); dl = S0["lg"] - S1["lg"]; dl = dl - dl.mean(1, keepdim=True); Cl = unit(centroids(dl, lab_i, K)); Gl = Cl @ Cl.T; Rw = unit(R[keep]); Gw = Rw @ Rw.T; wl = gram_spearman(Gw, Gl)
def knn_func(G, k=3):
    G2 = G.clone(); G2.fill_diagonal_(-2); nn = G2.topk(k, dim=1).indices; return Gl[torch.arange(K, device=DEV)[:, None], nn].mean().item()
out = {}
for lv in levels:
    Fl = S0[lv] - S1[lv]; Cd = unit(centroids(Fl[idx], lab_i, K)); Gd = Cd @ Cd.T; out[lv] = dict(desc_logit=gram_spearman(Gd, Gl), desc_write=gram_spearman(Gd, Gw), knn_func_desc=knn_func(Gd))
    log(f"{tag} level {lv}: Spearman descendant-logit {out[lv]['desc_logit']:+.2f}, descendant-write {out[lv]['desc_write']:+.2f}, kNN functional similarity via descendant neighbours {out[lv]['knn_func_desc']:.3f}")
record(f"e260_depthgeo_{tag}", dict(model=tag, b=b, L=L, K=K, write_logit=wl, knn_func_write=knn_func(Gw), per_level={str(k): v for k, v in out.items()}), f"K {K}; write-logit {wl:+.2f}; descendant-logit by level " + " ".join(f"{lv}:{v['desc_logit']:+.2f}" for lv, v in out.items()) + " | descendant-write by level " + " ".join(f"{lv}:{v['desc_write']:+.2f}" for lv, v in out.items()))
