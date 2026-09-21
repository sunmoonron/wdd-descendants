"""e258: when does descendant geometry start predicting functional geometry in training? Pythia-410m at a revision
(argv[1]): the checkpoint's own dominant block-2 writes, K candidates (>= 20 tokens), natural ablation with same-
position logit footprints. Reported: Gram Spearman of the logit-footprint geometry with the descendant geometry at L
and with the write-vector geometry; kNN functional similarity via descendant vs write neighbours; source
identification from the footprint and from the logit footprint."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
rev = sys.argv[1]; c = Cache("pythia410"); model, tok, fam = load_model(c.name, revision=None if rev == "final" else rev); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; L = 12; D = arch.D; W = arch.wdir(b).to(DEV); WN = W.norm(dim=1); R = W / WN[:, None].clamp_min(1e-9)
st = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st.__setitem__("a", inp[0].detach().float().reshape(-1, arch.DFF))); model(ids_seq); h.remove(); led = st["a"] * WN[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]
run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0 = run(positions=idx); S1 = run(tn, positions=idx); F = S0[L] - S1[L]; dl = S0["lg"] - S1["lg"]; dl = dl - dl.mean(1, keepdim=True); torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5
Cd = unit(centroids(F[idx], lab_i, K)); Cl = unit(centroids(dl, lab_i, K)); Rw = unit(R[keep]); Gd, Gl, Gw = Cd @ Cd.T, Cl @ Cl.T, Rw @ Rw.T
def knn_func(G, k=3):
    G2 = G.clone(); G2.fill_diagonal_(-2); nn = G2.topk(k, dim=1).indices; return Gl[torch.arange(K, device=DEV)[:, None], nn].mean().item()
torch.manual_seed(1); rand_func = Gl[torch.arange(K, device=DEV)[:, None], torch.stack([torch.randperm(K, device=DEV)[:3] for _ in range(K)])].mean().item()
res = dict(revision=rev, K=K, spearman_desc_logit=gram_spearman(Gd, Gl), spearman_write_logit=gram_spearman(Gw, Gl), spearman_write_desc=gram_spearman(Gw, Gd), knn_desc=knn_func(Gd), knn_write=knn_func(Gw), knn_random=rand_func, footprint_id=accuracy(F[idx][~split], unit(centroids(F[idx][split], lab_i[split], K)), lab_i[~split]), logit_id=accuracy(dl[~split], unit(centroids(dl[split], lab_i[split], K)), lab_i[~split]), chance=1.0 / K)
log(f"{rev} (K {K}): Gram Spearman logit-vs-descendant {res['spearman_desc_logit']:+.2f}, logit-vs-write {res['spearman_write_logit']:+.2f}, write-vs-descendant {res['spearman_write_desc']:+.2f} | kNN functional similarity descendant {res['knn_desc']:.3f} vs write {res['knn_write']:.3f} vs random {res['knn_random']:.3f} | source from footprint {res['footprint_id']:.2f}, from logit footprint {res['logit_id']:.2f} (chance {1 / K:.2f})")
record(f"e258_coupling_{rev}", res, f"{rev}: Spearman desc-logit {res['spearman_desc_logit']:+.2f} vs write-logit {res['spearman_write_logit']:+.2f}; kNN func desc {res['knn_desc']:.3f} vs write {res['knn_write']:.3f} vs random {res['knn_random']:.3f}; footprint id {res['footprint_id']:.2f}, logit id {res['logit_id']:.2f}")
