"""e334: do functional equivalence classes of atoms mean anything in vocabulary space? For 1024 block-2 atoms with
exact coordinates: classes by k-means on unit coordinates (16 classes); the vocabulary projections of the atoms
(logit lens of the write through the unembedding, top-20 tokens) compared within class and across classes (Jaccard
of top-20 sets and cosine of the vocabulary vectors), and the same for the atoms' direct-logit projections against
their functional coordinates: is coordinate similarity predicted by vocabulary similarity?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); P = pls(Fc[tr], Z[tr], d)
torch.manual_seed(0); atoms = torch.randperm(S.DFF, device=DEV)[:1024]; V = S.W2[atoms]; img = torch.zeros(1024, S.D, device=DEV); cnt = torch.zeros(1024, device=DEV)
for p in range(6):
    r = S.inject_family(V, S.b + 1, seed=p); img.index_add_(0, r["a"], r["F"][L]); cnt.index_add_(0, r["a"], torch.ones(len(r["a"]), device=DEV))
img = img / cnt[:, None].clamp_min(1); z = unit(img @ P); WU = S.arch.model.get_output_embeddings().weight.detach().float().to(DEV); voc = V @ WU.T; voc = voc - voc.mean(1, keepdim=True); vocn = unit(voc); top = voc.topk(20, dim=1).indices
torch.manual_seed(2); cent = z[torch.randperm(1024, device=DEV)[:16]]
for it in range(20):
    assign = (z @ cent.T).argmax(1); cent = unit(torch.stack([z[assign == k].mean(0) if (assign == k).any() else cent[k] for k in range(16)]))
iu = torch.triu_indices(1024, 1024, 1, device=DEV); same = assign[iu[0]] == assign[iu[1]]; vc = (vocn @ vocn.T)[iu[0], iu[1]]; zc = (z @ z.T)[iu[0], iu[1]]; wc = (V @ V.T)[iu[0], iu[1]]
def jac(i, j): a_, b_ = set(top[i].tolist()), set(top[j].tolist()); return len(a_ & b_) / len(a_ | b_)
torch.manual_seed(3); sidx = torch.nonzero(same)[:, 0]; sidx = sidx[torch.randperm(len(sidx), device=DEV)[:300]]; didx = torch.nonzero(~same)[:, 0]; didx = didx[torch.randperm(len(didx), device=DEV)[:300]]; j_same = float(torch.tensor([jac(int(iu[0][i]), int(iu[1][i])) for i in sidx.tolist()]).mean()); j_diff = float(torch.tensor([jac(int(iu[0][i]), int(iu[1][i])) for i in didx.tolist()]).mean())
res = dict(vocab_cos_same_class=vc[same].mean().item(), vocab_cos_other_class=vc[~same].mean().item(), write_cos_same_class=wc[same].abs().mean().item(), write_cos_other_class=wc[~same].abs().mean().item(), jaccard_top20_same=j_same, jaccard_top20_other=j_diff, spearman_vocab_vs_coordinate=spearman(vc, zc), spearman_write_vs_coordinate=spearman(wc.abs(), zc.abs()))
log(f"{tag} (1024 atoms, 16 classes): vocabulary-projection cosine within class {res['vocab_cos_same_class']:.3f} vs across {res['vocab_cos_other_class']:.3f}; top-20 token Jaccard within {res['jaccard_top20_same']:.3f} vs across {res['jaccard_top20_other']:.3f}; write cosine within {res['write_cos_same_class']:.3f} vs across {res['write_cos_other_class']:.3f}; Spearman(vocabulary similarity, coordinate similarity) {res['spearman_vocab_vs_coordinate']:+.2f}; Spearman(|write cos|, |coordinate cos|) {res['spearman_write_vs_coordinate']:+.2f}")
record(f"e334_semantics_{tag}", dict(model=tag, L=L, **res), f"vocab within/across {res['vocab_cos_same_class']:.3f}/{res['vocab_cos_other_class']:.3f} jaccard {res['jaccard_top20_same']:.3f}/{res['jaccard_top20_other']:.3f} rho vocab-coord {res['spearman_vocab_vs_coordinate']:+.2f} rho write-coord {res['spearman_write_vs_coordinate']:+.2f}")
