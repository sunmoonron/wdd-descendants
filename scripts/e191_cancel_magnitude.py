"""e191: is cancellation linear damping or feature-specific? For dominant block-b writes, the relative direct MLP
contribution of the main canceller block (within for GPT-2, next otherwise) and of all later blocks, by quartile of
the write's |coefficient| within the birth block. Constant relative cancellation across quartiles = gain-like
damping; increasing = saturation of large writes; decreasing = cleaning of small writes."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 8192; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 1); tb, tn, ct = tb[ids, 0], tn[ids, 0], tc[ids, 0].to(DEV); row = c.atom_index(L, tb, tn).to(DEV); d = A[row]; tbd = tb.to(DEV); within = tag.startswith("gpt2")
mlp = {}
for bp in range(L + 1):
    w = c.acts[bp][ids].float().to(DEV) @ c.wdir_cpu(bp).to(DEV)
    if c.d["mlp_bias"][bp] is not None: w = w + c.d["mlp_bias"][bp].to(DEV)
    mlp[bp] = (w * d).sum(1) / ct
out = {}
for b in range(1, min(L, 8)):
    bp = b if within else b + 1; m = typ & (tbd == b)
    if m.sum() < 120: continue
    q = ct[m].abs(); edges = q.quantile(torch.tensor([0.25, 0.5, 0.75], device=DEV)); qi = (q[:, None] > edges[None]).sum(1); rec = []
    for k in range(4):
        mk = qi == k; canc = mlp[bp][m][mk] - (1.0 if bp == b else 0.0); later = sum(mlp[x][m][mk] for x in range(b + 1, L + 1)); own = mlp[b][m][mk] - 1.0
        rec.append(dict(q=k, n=int(mk.sum()), coef_med=q[mk].median().item(), canceller=canc.mean().item(), later_net=later.mean().item(), own_others=own.mean().item(), abs_canc=(canc * q[mk]).mean().item()))
    out[b] = rec
    log(f"{tag} born b{b} (canceller b{bp}): relative canceller contribution by |coef| quartile: " + " ".join(f"q{r['q']}(|c|~{r['coef_med']:.1f}): {r['canceller']:+.2f}" for r in rec) + " | later-net: " + " ".join(f"{r['later_net']:+.2f}" for r in rec) + " | absolute canceller: " + " ".join(f"{r['abs_canc']:+.2f}" for r in rec))
import numpy as np
qm = lambda k, key: float(np.mean([out[b][k][key] for b in out]))
record(f"e191_magnitude_{tag}", dict(model=tag, L=L, within=within, per_birth=out), "relative canceller contribution q1..q4 (mean over birth blocks): " + " ".join(f"{qm(k, 'canceller'):+.2f}" for k in range(4)) + " | later-net q1..q4: " + " ".join(f"{qm(k, 'later_net'):+.2f}" for k in range(4)) + " | absolute canceller q1..q4: " + " ".join(f"{qm(k, 'abs_canc'):+.2f}" for k in range(4)))
