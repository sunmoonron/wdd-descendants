"""e320: persistence of channel identity over long horizons. The 16-dimensional logit quotient fixed at an early level
(b+2) and at L; in that fixed basis the map from the coordinate at the base level to the coordinate k blocks later,
for every k to the last block: R2, diagonal energy fraction, geometric-mean gain, and the identity of the optimal
assignment. Whether the channels survive the whole depth or only locally."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
try:
    from scipy.optimize import linear_sum_assignment
except Exception: linear_sum_assignment = None
tag = sys.argv[1]; S0_ = Setup(tag, levels=[0]); L = S0_.L; NB = S0_.NB; b = 2; S = Setup(tag, levels=list(range(b + 1, NB))); d = 16; nat = S.natural(); tr, te = S.halves(len(S.idx)); Z, _ = logit_scores(nat["dl"], tr); out = {}
for base in (b + 2, L):
    F0 = nat["F"][base]; F0c = F0 - F0[tr].mean(0, keepdim=True); P = pls(F0c[tr], Z[tr], d); z0 = F0c @ P; rec = {}
    for lv in range(base + 1, NB):
        Fk = nat["F"][lv]; zk = (Fk - Fk[tr].mean(0, keepdim=True)) @ P; A = torch.linalg.lstsq(z0[tr], zk[tr]).solution; pred = z0[te] @ A; r2 = 1 - ((zk[te] - pred) ** 2).sum().item() / ((zk[te] - zk[tr].mean(0, keepdim=True)) ** 2).sum().item(); diag = ((A.diagonal() ** 2).sum() / (A ** 2).sum()).item(); C = torch.corrcoef(torch.cat([z0, zk], 1).T)[:d, d:].abs().nan_to_num()
        ident = float((linear_sum_assignment(-C.cpu().numpy())[1] == range(d)).mean()) if linear_sum_assignment is not None else float((C.argmax(1) == torch.arange(d, device=DEV)).float().mean().item()); rec[lv] = dict(r2=r2, diag=diag, gain=torch.linalg.svdvals(A).log().mean().exp().item(), identity=ident)
    out[base] = rec
log(f"{tag} (K {S.K}): " + " || ".join(f"basis at {base}: k -> R2/diag/gain/identity :: " + " ".join(f"+{lv - base}:{v['r2']:.2f}/{v['diag']:.2f}/{v['gain']:.2f}/{v['identity']:.2f}" for lv, v in rec.items()) for base, rec in out.items()))
record(f"e320_longhorizon_{tag}", dict(model=tag, L=L, K=S.K, per_base={str(k): {str(kk): vv for kk, vv in v.items()} for k, v in out.items()}), " || ".join(f"basis {base}: last R2 {list(rec.values())[-1]['r2']:.2f} diag {list(rec.values())[-1]['diag']:.2f} identity {list(rec.values())[-1]['identity']:.2f}; min R2 {min(v['r2'] for v in rec.values()):.2f}" for base, rec in out.items()))
