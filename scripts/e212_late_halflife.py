"""e212: the contraction model predicts that writes born late, where the generic gain is near 0, keep their causal
footprint much longer than early writes. Same intervention as e208 (zero the token's dominant neuron of block b)
for b near the top (NB-7, NB-6, NB-5), dependence of the state's component on the write at +1, +2, +3 and the last
level; compared with the early-block values of e208."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; lab = c.d["lab"]; A = c.d["A"]; NB = c.NB
def rows(b): return A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
def run(b=None, neuron=None):
    store = {}; hs = [arch.layers[lv].register_forward_hook((lambda lv_: lambda m, i, o: store.__setitem__(lv_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(lv)) for lv in range(NB)]
    if b is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    model(ids_seq); [h.remove() for h in hs]; return store
S0 = run(); res = {}
for b in (NB - 7, NB - 6, NB - 5):
    store_a = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: store_a.__setitem__("a", inp[0].detach().float().reshape(-1, c.DFF))); model(ids_seq); h.remove()
    led = store_a["a"] * c.d["WN"][b].to(DEV)[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; d = rows(b)[tn]; big = tc.abs() >= tc.abs().quantile(0.5)
    S1 = run(b, tn); prof = {}
    for lv in sorted({b, b + 1, b + 2, b + 3, NB - 1}):
        typ = typical_mask(S0[lv]) & big; p0 = (S0[lv] * d).sum(1) / tc; p1 = (S1[lv] * d).sum(1) / tc; dep = ((p0 - p1) / p0.abs().clamp_min(0.05))[typ]
        prof[lv - b] = dict(level=lv, clean=p0[typ].median().item(), ablated=p1[typ].median().item(), dependence_med=dep.median().item(), removed_abs=(p0 - p1)[typ].median().item())
    res[b] = prof
    log(f"{tag} born b{b} (of {NB}): projection on d clean -> ablated, dependence: " + " ".join(f"+{k}: {v['clean']:.2f}->{v['ablated']:.2f} dep {v['dependence_med']:.2f}" for k, v in sorted(prof.items())))
import numpy as np
d1 = float(np.mean([res[b][1]["dependence_med"] for b in res])); d2 = float(np.mean([res[b][2]["dependence_med"] for b in res])); d3 = float(np.mean([res[b][3]["dependence_med"] for b in res if 3 in res[b]])); dl = float(np.mean([res[b][max(res[b])]["dependence_med"] for b in res]))
record(f"e212_latehalf_{tag}", dict(model=tag, NB=NB, per_birth={str(k): v for k, v in res.items()}), f"late-born writes (blocks {NB - 7}-{NB - 5}): dependence at +1 {d1:.2f}, +2 {d2:.2f}, +3 {d3:.2f}, last level {dl:.2f} | removed at +2: " + " ".join(f"b{b}:{res[b][2]['removed_abs']:.2f} of {res[b][2]['clean']:.2f}" for b in res))
