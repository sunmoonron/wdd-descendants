"""e146: token identity vs context. For token ids with >= 20 occurrences among typical tokens at the mid layer:
the variance of the dominant write's prominence explained by the token id (between/total), the conditional
entropy of the dominant NEURON given the token id (how often the same token has the same dominant neuron), and
the same at level 1 and at the last level."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); tok = c.s["eval_ids"].reshape(-1).to(DEV); out = {}
for L in sorted(set([1, mid(c), c.NB - 1])):
    X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV)
    prom = (X * A[row]).sum(1).abs() / X.norm(dim=1); key = (tb[:, 0] * c.DFF + tn[:, 0]).to(DEV)
    u, inv, cnt = tok[typ].unique(return_inverse=True, return_counts=True); m = cnt >= 20; keep = m[inv]
    p = prom[typ][keep]; inv2 = inv[keep]; means = torch.zeros(len(u), device=DEV).index_add_(0, inv2, p) / cnt.clamp_min(1); between = ((means[inv2] - p.mean()) ** 2).sum(); total = ((p - p.mean()) ** 2).sum()
    # dominant neuron consistency: fraction of a token's occurrences sharing its modal dominant neuron
    kk = key[typ][keep]; frac = []
    for i in torch.nonzero(m)[:, 0].tolist():
        kks = kk[inv2 == i]; _, cc = kks.unique(return_counts=True); frac.append((cc.max() / len(kks)).item())
    # same for hit/miss consistency
    sel, cof, err = get_omp(c, L, A=A, X=X) if L == mid(c) else (None, None, None)
    out[L] = dict(n_token_types=int(m.sum()), n_tokens=int(keep.sum()), prominence_R2_by_token=(between / total).item(), modal_neuron_share_median=float(np.median(frac)), modal_neuron_share_mean=float(np.mean(frac)))
    log(f"{tag} L{L}: prominence R2 by token id {out[L]['prominence_R2_by_token']:.2f}; same token -> same dominant neuron {out[L]['modal_neuron_share_median']:.2f} (median share)")
record(f"e146_tokctx_{tag}", dict(model=tag, levels=out), " | ".join(f"L{L}: prom R2(token) {v['prominence_R2_by_token']:.2f}, modal-neuron share {v['modal_neuron_share_median']:.2f} (n types {v['n_token_types']})" for L, v in out.items()))
