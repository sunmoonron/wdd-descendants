"""e187: is the cancellation topology visible in the weights alone (hook-free)? For GELU models (activations
essentially nonnegative) a canceller row must be anti-aligned with the write direction, so the most negative
signed cosine between block-b rows and rows of block b (within), b+1 (next) and b+3 (control) vs random rows is a
static eraser signature. For all models the largest |cos| (aliasing) is reported the same way."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); lab = c.d["lab"]; A = c.d["A"]
def rows(b): return A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
torch.manual_seed(0); R = torch.randn(c.DFF, c.D, device=DEV); R = R / R.norm(dim=1, keepdim=True); out = {}
for b in range(0, L):
    Rb = rows(b); rec = {}
    for nm, bp in (("within", b), ("next", b + 1), ("plus3", b + 3), ("random", None)):
        if bp is not None and bp > L: continue
        G = Rb @ (R if bp is None else rows(bp)).T
        if bp == b: G.fill_diagonal_(0)
        mn, mx = G.min(1).values, G.max(1).values; ab = G.abs().max(1).values
        rec[nm] = dict(min_med=mn.median().item(), frac_min_below_0p3=(mn < -0.3).float().mean().item(), frac_min_below_0p5=(mn < -0.5).float().mean().item(), max_med=mx.median().item(), abs_med=ab.median().item(), frac_abs_above_0p5=(ab > 0.5).float().mean().item())
    out[b] = rec
    log(f"{tag} b{b}: most-negative cos median within {rec['within']['min_med']:+.2f} next {rec['next']['min_med']:+.2f} plus3 {rec.get('plus3', {}).get('min_med', float('nan')):+.2f} random {rec['random']['min_med']:+.2f} | frac rows with an anti-aligned partner (<-0.5): within {rec['within']['frac_min_below_0p5']:.3f} next {rec['next']['frac_min_below_0p5']:.3f} random {rec['random']['frac_min_below_0p5']:.3f} | |cos|>0.5 partner: within {rec['within']['frac_abs_above_0p5']:.3f} next {rec['next']['frac_abs_above_0p5']:.3f}")
import numpy as np
agg = lambda nm, key: float(np.mean([out[b][nm][key] for b in out if nm in out[b]]))
record(f"e187_static_{tag}", dict(model=tag, L=L, per_block=out), f"most-negative cosine (median over rows, mean over blocks): within {agg('within', 'min_med'):+.3f} next {agg('next', 'min_med'):+.3f} plus3 {agg('plus3', 'min_med'):+.3f} random {agg('random', 'min_med'):+.3f} | frac rows with anti-aligned partner < -0.5: within {agg('within', 'frac_min_below_0p5'):.3f} next {agg('next', 'frac_min_below_0p5'):.3f} plus3 {agg('plus3', 'frac_min_below_0p5'):.3f} random {agg('random', 'frac_min_below_0p5'):.3f} | |cos| > 0.5 partner: within {agg('within', 'frac_abs_above_0p5'):.3f} next {agg('next', 'frac_abs_above_0p5'):.3f} plus3 {agg('plus3', 'frac_abs_above_0p5'):.3f}")
