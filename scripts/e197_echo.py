"""e197: is the eraser readable as a negative echo? The damping model predicts that block b+1's increment contains
-g * coef * d (a copy of block b's dominant write with a negative coefficient). Decompose block b+1's increment
(state after b+1 minus state after b) with OMP@32 over the dictionary of blocks b and b+1 (+ embeddings) and ask
whether the block-b dominant atom is selected, with which sign, and whether its selection follows the prominence
law (recall by prominence bin). Same for the MLP-only increment of block b+1."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 6144; ids = sub(c.NT, N); lab = c.d["lab"]; A = c.d["A"]; births = (6, 7, 8) if tag.startswith("pythia") else (2, 3, 4)
def rows(b): return A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
out = {}
for b in births:
    if b + 2 > len(c.s["H"]) - 1: continue
    led = c.acts[b][ids].float() * c.d["WN"][b][None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0].to(DEV); d = rows(b)[tn.to(DEV)]
    H1 = c.s["H"][b + 1][ids].float().to(DEV); H2 = c.s["H"][b + 2][ids].float().to(DEV); typ = typical_mask(H2) & typical_mask(H1); big = typ & (tc.abs() >= tc.abs().quantile(0.5))
    A2, lab2 = c.dictionary(b + 1, blocks=(b, b + 1)); mask = (lab2["type"] == T_MLP) & (lab2["block"] == b); lut = torch.full((c.DFF,), -1, dtype=torch.long); lut[lab2["index"][mask]] = torch.nonzero(mask)[:, 0]; row2 = lut[tn].to(DEV)
    mlp = c.acts[b + 1][ids].float().to(DEV) @ c.wdir_cpu(b + 1).to(DEV)
    if c.d["mlp_bias"][b + 1] is not None: mlp = mlp + c.d["mlp_bias"][b + 1].to(DEV)
    rec = {}
    for nm, inc in (("block_increment", H2 - H1), ("mlp_increment", mlp)):
        Z = inc - inc[typ].mean(0); direct = ((inc * d).sum(1) / tc); prom = (Z * d).sum(1).abs() / Z.norm(dim=1).clamp_min(1e-6)
        sel, cof, _ = omp(Z, A2, 32); hit = (sel == row2[:, None]); found = hit.any(1); coef_echo = (cof * hit.float()).sum(1)
        neg = (coef_echo < 0) & found; pb = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.5, 1.01], device=DEV); bins = (prom[:, None] >= pb[None, :-1]).sum(1) - 1
        law = {f"{pb[i].item():.1f}-{pb[i + 1].item():.1f}": (found[big & (bins == i)].float().mean().item() if (big & (bins == i)).sum() > 30 else None) for i in range(5)}
        rec[nm] = dict(direct_med=direct[big].median().item(), prom_med=prom[big].median().item(), echo_recall=found[big].float().mean().item(), echo_negative_share=(neg[big].float().sum() / found[big].float().sum().clamp_min(1)).item(), coef_ratio_med=(coef_echo / tc)[big & found].median().item() if (big & found).sum() > 0 else float("nan"), law=law, n=int(big.sum()))
        log(f"{tag} born b{b} {nm}: direct (inc.d/coef) {rec[nm]['direct_med']:+.2f} | echo prominence {rec[nm]['prom_med']:.2f} | block-b dominant atom selected in {rec[nm]['echo_recall']:.2f} of tokens, negative sign in {rec[nm]['echo_negative_share']:.2f} of those, coef/coef_b median {rec[nm]['coef_ratio_med']:+.2f} | recall by prominence bin: " + " ".join(f"{k}:{(v if v is not None else float('nan')):.2f}" for k, v in law.items()))
    out[b] = rec
import numpy as np
m = lambda nm, key: float(np.mean([out[b][nm][key] for b in out]))
record(f"e197_echo_{tag}", dict(model=tag, per_birth=out), f"block increment of b+1: direct {m('block_increment', 'direct_med'):+.2f}, echo atom selected {m('block_increment', 'echo_recall'):.2f} (negative sign {m('block_increment', 'echo_negative_share'):.2f}, coef ratio {m('block_increment', 'coef_ratio_med'):+.2f}) | MLP-only increment: direct {m('mlp_increment', 'direct_med'):+.2f}, echo selected {m('mlp_increment', 'echo_recall'):.2f} (negative {m('mlp_increment', 'echo_negative_share'):.2f}, ratio {m('mlp_increment', 'coef_ratio_med'):+.2f})")
