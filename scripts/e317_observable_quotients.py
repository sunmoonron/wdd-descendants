"""e317: is the quotient universal or observable-relative? From one descendant cloud at L, a 16-dimensional quotient
for each of nine observables (logit footprint, removal KL, entropy change, top-1 probability change, next-token
log-probability change, future state at L+2, future state at L+4, last-level state, source identity). Pairwise
energy overlaps against chance and against each observable's split-half reliability, and the transfer matrix:
each observable decoded from every other observable's quotient, as a fraction of its own-quotient score."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=None if False else [0]); L = S.L; S = Setup(tag, levels=[L, L + 2, L + 4, S.NB - 2]); L = S.L; d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); nxt = S.ids_seq.reshape(-1)[(S.idx + 1).clamp_max(S.NT - 1)]; ar = torch.arange(len(S.idx), device=DEV)
targets = {"logits": Z, "kl": nat["kl"][:, None], "entropy": nat["entropy"][:, None], "top1": nat["top1"][:, None], "next_logprob": (nat["lp0"][ar, nxt] - nat["lp1"][ar, nxt])[:, None], "future2": unit(nat["F"][L + 2]) if L + 2 in nat["F"] else None, "future4": unit(nat["F"][L + 4]) if L + 4 in nat["F"] else None, "last_state": unit(nat["F"][S.NB - 2]), "identity": torch.nn.functional.one_hot(S.lab_i, S.K).float()}
targets = {k: v for k, v in targets.items() if v is not None}
def quot(rows, key):
    Y = targets[key]; Xc = F[rows] - F[rows].mean(0, keepdim=True)
    if Y.shape[1] == 1: yc = Y[rows] - Y[rows].mean(); return torch.linalg.eigh(Xc.T @ (yc * Xc))[1].flip(1)[:, :d]
    return pls(Xc, Y[rows], d)
Q = {k: quot(tr, k) for k in targets}; ha, hb = S.halves(len(S.idx), seed=3); rel = {k: inside(quot(ha, k), quot(hb, k)) for k in targets}; names = list(targets); ov = {a: {b_: inside(Q[a], Q[b_]) for b_ in names} for a in names}
def score(P, key):
    Y = targets[key]; X = Fc @ P
    if key == "identity": return accuracy(X[te], centroids(X[tr], S.lab_i[tr], S.K), S.lab_i[te])
    if Y.shape[1] == 1: return knn_spear(X, Y[:, 0], tr, te)
    return knn_cos(X, Y, tr, te)
own = {k: score(Q[k], k) for k in names}; transfer = {src: {k: score(Q[src], k) for k in names} for src in names}; torch.manual_seed(1); rnd = torch.linalg.qr(torch.randn(S.D, d, device=DEV))[0]; rand_sc = {k: score(rnd, k) for k in names}
offs = [ov[a][b_] for i, a in enumerate(names) for b_ in names[i + 1:]]; rel_mean = sum(rel.values()) / len(rel); tr_rel = [transfer[src][k] / own[k] if abs(own[k]) > 1e-6 else float("nan") for src in names for k in names if src != k]
res = dict(K=S.K, d=d, chance=d / S.D, overlap=ov, reliability=rel, own=own, transfer=transfer, random=rand_sc, mean_cross_overlap=sum(offs) / len(offs), mean_reliability=rel_mean, mean_transfer_ratio=float(torch.tensor([x for x in tr_rel if x == x]).mean()))
log(f"{tag} (K {S.K}, chance {d / S.D:.3f}): mean pairwise overlap of the nine quotients {res['mean_cross_overlap']:.2f} vs mean split-half reliability {rel_mean:.2f}; logits-vs-others: " + " ".join(f"{k} {ov['logits'][k]:.2f}" for k in names if k != 'logits') + " | own scores: " + " ".join(f"{k} {own[k]:.2f}" for k in names) + " | random-16 scores: " + " ".join(f"{k} {rand_sc[k]:.2f}" for k in names) + f" | mean transfer ratio (other quotient / own) {res['mean_transfer_ratio']:.2f}; from the logit quotient: " + " ".join(f"{k} {transfer['logits'][k]:.2f}" for k in names))
record(f"e317_obsquot_{tag}", dict(model=tag, L=L, **res), f"cross overlap {res['mean_cross_overlap']:.2f} vs reliability {rel_mean:.2f}; logits vs kl {ov['logits']['kl']:.2f} entropy {ov['logits']['entropy']:.2f} future2 {ov['logits'].get('future2', float('nan')):.2f} identity {ov['logits']['identity']:.2f}; transfer ratio {res['mean_transfer_ratio']:.2f}")
