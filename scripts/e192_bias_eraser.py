"""e192: is the cancellation a constant (bias) or activation-driven? For models with an MLP output bias (GPT-2,
Pythia), split each block's direct contribution to the survival of dominant block-b writes into the bias part
(bias_bp . d / coef) and the activation part; report the share of the strongest canceller's contribution carried
by the bias, and the bias-only erasure row (b -> bp) for the dedicated canceller blocks."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 8192; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 1); tb, tn, ct = tb[ids, 0], tn[ids, 0], tc[ids, 0].to(DEV); row = c.atom_index(L, tb, tn).to(DEV); d = A[row]; tbd = tb.to(DEV)
if c.d["mlp_bias"][0] is None: record(f"e192_bias_{tag}", dict(model=tag, note="no MLP output bias"), "no MLP output bias in this model: cancellation is activation-driven by construction"); sys.exit()
Mb = torch.zeros(L + 1, L + 1); Ma = torch.zeros(L + 1, L + 1); cnt = torch.zeros(L + 1)
for bp in range(L + 1):
    bias = c.d["mlp_bias"][bp].to(DEV); pb = (bias[None] * d).sum(1) / ct; act = c.acts[bp][ids].float().to(DEV) @ c.wdir_cpu(bp).to(DEV); pa = (act * d).sum(1) / ct
    for b in range(L + 1):
        m = typ & (tbd == b)
        if m.sum() < 30: continue
        Mb[b, bp] = pb[m].mean().item(); Ma[b, bp] = pa[m].mean().item() - (1.0 if bp == b else 0.0); cnt[b] = m.sum()
tot = Mb + Ma; out = {}
for b in range(L + 1):
    if cnt[b] == 0: continue
    bp = int(tot[b].argmin()); out[b] = dict(canceller=bp, total=tot[b, bp].item(), bias_part=Mb[b, bp].item(), act_part=Ma[b, bp].item(), bias_row_sum=Mb[b].sum().item(), act_row_sum=Ma[b].sum().item())
    log(f"{tag} born b{b}: strongest canceller b{bp} {tot[b, bp]:+.2f} = bias {Mb[b, bp]:+.2f} + activations {Ma[b, bp]:+.2f} | all-block bias sum {Mb[b].sum():+.2f} vs activation sum {Ma[b].sum():+.2f}")
import numpy as np
bs = float(np.mean([abs(out[b]['bias_part']) / max(abs(out[b]['total']), 1e-6) for b in out if out[b]['total'] < 0]))
record(f"e192_bias_{tag}", dict(model=tag, L=L, bias=Mb.tolist(), act=Ma.tolist(), per_birth=out), f"share of the strongest canceller's contribution carried by the MLP output bias: {bs:.2f} (mean over birth blocks) | bias row sums: " + " ".join(f"b{b}:{out[b]['bias_row_sum']:+.2f}" for b in out) + " | activation row sums: " + " ".join(f"b{b}:{out[b]['act_row_sum']:+.2f}" for b in out))
