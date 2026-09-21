"""e324: does the quotient bend? The descendant cloud is split into eight regions by k-means; the logit quotient is
refitted inside each region and compared with the global quotient (energy overlap) and between regions; the logit
footprint is decoded inside each region from its local quotient and from the global one. A curved quotient shows as
region-specific bases that decode better locally."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, NS=16, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); dln = unit(nat["dl"]); Qg = pls(Fc[tr], Z[tr], d)
torch.manual_seed(0); U = unit(Fc); cent = U[torch.randperm(len(U), device=DEV)[:8]]
for it in range(20):
    reg = (U @ cent.T).argmax(1); cent = unit(torch.stack([U[reg == k].mean(0) if (reg == k).any() else cent[k] for k in range(8)]))
locals_ = {}; ov_glob = []; loc_scores = []; glob_scores = []; sizes = []
for k in range(8):
    rows = torch.nonzero(reg == k)[:, 0]; rtr = rows[torch.isin(rows, tr)]; rte = rows[torch.isin(rows, te)]
    if len(rtr) < 40 or len(rte) < 20: continue
    Ql = pls(Fc[rtr] - Fc[rtr].mean(0, keepdim=True), Z[rtr], d); locals_[k] = Ql; ov_glob.append(inside(Ql, Qg)); loc_scores.append(knn_cos(Fc @ Ql, dln, rtr, rte)); glob_scores.append(knn_cos(Fc @ Qg, dln, rtr, rte)); sizes.append(int(len(rows)))
ks = list(locals_); ov_pairs = [inside(locals_[a], locals_[b_]) for i, a in enumerate(ks) for b_ in ks[i + 1:]]
res = dict(K=S.K, regions=len(ks), region_sizes=sizes, local_vs_global_overlap=sum(ov_glob) / len(ov_glob), local_vs_local_overlap=sum(ov_pairs) / max(len(ov_pairs), 1), local_decode=sum(loc_scores) / len(loc_scores), global_decode_in_region=sum(glob_scores) / len(glob_scores), chance=d / S.D)
log(f"{tag} (K {S.K}, {len(ks)} regions of {min(sizes)}-{max(sizes)} tokens, chance {d / S.D:.3f}): local quotient vs global overlap {res['local_vs_global_overlap']:.2f}, local vs local {res['local_vs_local_overlap']:.2f}; logit decoded inside regions from the local quotient {res['local_decode']:.2f} vs from the global quotient {res['global_decode_in_region']:.2f}")
record(f"e324_local_{tag}", dict(model=tag, L=L, **res), f"local-global {res['local_vs_global_overlap']:.2f} local-local {res['local_vs_local_overlap']:.2f} decode local {res['local_decode']:.2f} global {res['global_decode_in_region']:.2f}")
