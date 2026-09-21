"""e219: how much write activity leaves no state evidence? Per block: the coherence ratio ||sum_i c_i w_i||^2 /
sum_i c_i^2 of the MLP increment (1 = no cancellation, < 1 destructive, > 1 constructive), median over typical
tokens; across blocks 0..L: the energy of the summed MLP contribution vs the sum of block energies (cross-block
cancellation) and vs the total per-neuron write energy (the fraction of all MLP write energy that survives as
state). Descriptive, no kill rule."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 8192; ids = sub(c.NT, N); NBa = len(c.acts); out = {}; T = None; sumE = 0.0; sumP = 0.0
for b in range(min(NBa, c.NB)):
    acts = c.acts[b][ids].float().to(DEV); C = acts * c.d["WN"][b].to(DEV)[None]; inc = acts @ c.wdir_cpu(b).to(DEV); Ep = (C ** 2).sum(1); Es = (inc ** 2).sum(1); typ = typical_mask(c.X(b, center=False)[ids])
    out[b] = dict(coherence_ratio=(Es / Ep.clamp_min(1e-9))[typ].median().item(), parts_energy=Ep[typ].median().item(), sum_energy=Es[typ].median().item())
    if b <= L: T = inc if T is None else T + inc; sumE = sumE + Es; sumP = sumP + Ep
typL = typical_mask(c.X(L, center=False)[ids]); cross = ((T ** 2).sum(1) / sumE.clamp_min(1e-9))[typL].median().item(); total = ((T ** 2).sum(1) / sumP.clamp_min(1e-9))[typL].median().item()
log(f"{tag}: within-block coherence ratio by block: " + " ".join(f"b{b}:{v['coherence_ratio']:.2f}" for b, v in out.items()) + f" | cross-block (blocks 0..{L}): summed MLP energy / sum of block energies {cross:.2f} | surviving fraction of all per-neuron write energy {total:.3f}")
import numpy as np
record(f"e219_zerosum_{tag}", dict(model=tag, L=L, per_block=out, cross_block_ratio=cross, total_survival=total), f"within-block coherence ratio median over blocks {np.median([v['coherence_ratio'] for v in out.values()]):.2f} (min {min(v['coherence_ratio'] for v in out.values()):.2f} at b{min(out, key=lambda b: out[b]['coherence_ratio'])}, max {max(v['coherence_ratio'] for v in out.values()):.2f}) | cross-block ratio {cross:.2f} | fraction of all MLP write energy (blocks 0..{L}) surviving as state {total:.3f} | by block: " + " ".join(f"b{b}:{v['coherence_ratio']:.2f}" for b, v in out.items()))
