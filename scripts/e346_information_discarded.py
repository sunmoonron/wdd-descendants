"""e346: what the discarded dimensions know. The descendant split into the 16-dim core and its orthogonal complement;
each decodes on held-out tokens: source identity (nearest centroid), the logit footprint (kNN cosine), the future
state (kNN cosine), the token identity (nearest centroid over the 50 most frequent tokens), the token position
(kNN Spearman), and the sequence identity (nearest centroid); the complement's top-64 PCs are used so the comparison
is at matched dimension; the core's information is reported as well, and a random 64-dim projection as baseline."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S0_ = Setup(tag, levels=[0]); L = S0_.L; NB = S0_.NB; S = Setup(tag, NS=16, levels=[L, min(L + 2, NB - 2)]); lf = min(L + 2, NB - 2); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); dln = unit(nat["dl"]); Ff = unit(nat["F"][lf]); P = pls(Fc[tr], Z[tr], d); core = Fc @ P; rem = Fc - core @ P.T; U64 = torch.linalg.svd(rem[tr], full_matrices=False)[2][:64].T; rem64 = rem @ U64; torch.manual_seed(0); rnd64 = Fc @ torch.linalg.qr(torch.randn(S.D, 64, device=DEV))[0]
toks = S.ids_seq.reshape(-1)[S.idx]; freq = torch.bincount(toks); top = freq.argsort(descending=True)[:50]; m = torch.isin(toks, top); tok_lab = torch.searchsorted(top.sort().values, toks[m]); pos = (S.idx % CTX).float(); seq = S.idx // CTX
def cent_acc(X, lab, mask=None):
    if mask is not None: X = X[mask]; lab = lab
    n = len(X); t_ = torch.arange(n, device=DEV); a = t_[::2]; b_ = t_[1::2]; K_ = int(lab.max().item()) + 1; return accuracy(X[b_], centroids(X[a], lab[a], K_), lab[b_]), 1 / K_
def sc(X): tacc, tch = cent_acc(X, tok_lab, m); sacc, sch = cent_acc(X, seq); return dict(identity=accuracy(X[te], centroids(X[tr], S.lab_i[tr], S.K), S.lab_i[te]), function=knn_cos(X, dln, tr, te), future=knn_cos(X, Ff, tr, te), token_identity=tacc, token_chance=tch, position=knn_spear(X, pos, tr, te), sequence=sacc, sequence_chance=sch)
out = {"core16": sc(core), "complement_pca64": sc(rem64), "random64": sc(rnd64), "full": sc(Fc)}
log(f"{tag} (K {S.K}): identity / function / future / token identity (chance {out['core16']['token_chance']:.2f}) / position / sequence (chance {out['core16']['sequence_chance']:.2f}): " + " | ".join(f"{k}: {v['identity']:.2f} / {v['function']:.2f} / {v['future']:.2f} / {v['token_identity']:.2f} / {v['position']:.2f} / {v['sequence']:.2f}" for k, v in out.items()))
record(f"e346_discarded_{tag}", dict(model=tag, L=L, K=S.K, per_input=out), " | ".join(f"{k} {v['identity']:.2f}/{v['function']:.2f}/{v['future']:.2f}/{v['token_identity']:.2f}/{v['position']:.2f}/{v['sequence']:.2f}" for k, v in out.items()))
