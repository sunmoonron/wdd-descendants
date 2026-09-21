"""e193 (= e188 on CENTERED states, the states WDD actually decomposes): shape of provenance decay with depth. For dominant block-b writes (b = 0..4, typical states), the raw
survival (state . d / coef) at every cached level from the pre-write state H[b] to the last level: one-step
erasure (drop concentrated at the next block, then flat) vs memoryless exponential decay (constant ratio of
successive drops). Reports the survival profile and the drop ratios."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); NL = len(c.s["H"]); N = 8192; ids = sub(c.NT, N); lab = c.d["lab"]; A = c.d["A"]; out = {}
def rows(b): return A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
for b in range(0, min(5, NL - 2)):
    led = c.acts[b][ids].float() * c.d["WN"][b][None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0].to(DEV); d = rows(b)[tn.to(DEV)]
    big = tc.abs() >= tc.abs().quantile(0.5); prof = {}
    for lv in range(b, NL):                                                                        # H[b] = pre-write state, H[b+1] = after block b
        H = c.s["H"][lv][ids].float().to(DEV); typ = typical_mask(H); m = typ & big; H = H - c.s["mu"][lv].to(DEV)
        prof[lv - b - 1] = ((H * d).sum(1) / tc)[m].median().item()                                # key = distance after the write (-1 = pre-write)
    ks = sorted(prof); s = [prof[k] for k in ks]
    drops = [s[i] - s[i + 1] for i in range(1, len(s) - 1)]                                         # drops after the write level
    ratio = [drops[i + 1] / drops[i] if abs(drops[i]) > 0.02 else float("nan") for i in range(len(drops) - 1)]
    out[b] = dict(profile=prof, drops=drops, drop_ratios=ratio)
    log(f"{tag} born b{b}: pre {prof[-1]:+.2f} | " + " ".join(f"+{k}:{prof[k]:.2f}" for k in ks if k >= 0) + " | drops " + " ".join(f"{x:+.2f}" for x in drops[:5]) + " | ratio2/1 " + (f"{ratio[0]:.2f}" if ratio and ratio[0] == ratio[0] else "nan"))
import numpy as np
first = [out[b]["drops"][0] for b in out if out[b]["drops"]]; second = [out[b]["drops"][1] for b in out if len(out[b]["drops"]) > 1]; later = [float(np.mean(out[b]["drops"][2:])) for b in out if len(out[b]["drops"]) > 2]
record(f"e193_csurvdist_{tag}", dict(model=tag, per_birth=out), f"median drop at +1: {np.median(first):+.2f}, at +2: {np.median(second):+.2f}, mean later drops: {np.median(later):+.3f} | profiles: " + " || ".join(f"b{b}: pre {out[b]['profile'][-1]:+.2f} " + " ".join(f"{out[b]['profile'][k]:.2f}" for k in sorted(out[b]['profile']) if k >= 0) for b in out))
