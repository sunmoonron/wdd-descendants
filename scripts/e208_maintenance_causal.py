"""e208: is the later presence of a write's direction caused by the write? Zero the token's dominant block-b neuron
(b = 1, 2, 3) in the forward pass and measure, at levels b+1 .. b+4 and L, the projection of the state on the
write direction relative to the clean run. dependence = (clean - ablated) / clean: 1 = the later component exists
only because of the write (it is read and re-written, or simply persists), 0 = later blocks write it anyway
(redundant writing, provenance intrinsically ambiguous). Also the passive-contraction expectation and the
next-token loss change for calibration."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; lab = c.d["lab"]; A = c.d["A"]
def rows(b): return A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
def run(b=None, neuron=None):
    store = {}; hs = [arch.layers[lv].register_forward_hook((lambda lv_: lambda m, i, o: store.__setitem__(lv_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(lv)) for lv in range(c.NB)]
    if b is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); store["act"] = flat[torch.arange(NT, device=DEV), neuron].detach().clone(); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    else:
        hs.append(arch.mlp_lin(0).register_forward_pre_hook(lambda m, inp: None))
    out = model(ids_seq); [h.remove() for h in hs]; lg = out.logits[:, :-1].reshape(-1, out.logits.shape[-1]).float(); ce = torch.nn.functional.cross_entropy(lg, ids_seq[:, 1:].reshape(-1), reduction="none"); return store, ce
S0, ce0 = run(); res = {}
for b in (1, 2, 3):
    acts_b = None
    # dominant neuron per token from a hooked clean pass of block b's activations
    store_a = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: store_a.__setitem__("a", inp[0].detach().float().reshape(-1, c.DFF))); model(ids_seq); h.remove()
    led = store_a["a"] * c.d["WN"][b].to(DEV)[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; d = rows(b)[tn]; big = tc.abs() >= tc.abs().quantile(0.5)
    S1, ce1 = run(b, tn); prof = {}
    for lv in sorted({b, b + 1, b + 2, b + 3, b + 4, L}):
        if lv >= c.NB: continue
        typ = typical_mask(S0[lv]) & big; p0 = (S0[lv] * d).sum(1) / tc; p1 = (S1[lv] * d).sum(1) / tc; dep = ((p0 - p1) / p0.abs().clamp_min(0.05))[typ]
        prof[lv - b] = dict(level=lv, clean=p0[typ].median().item(), ablated=p1[typ].median().item(), dependence_med=dep.median().item(), removed_abs=(p0 - p1)[typ].median().item())
    # position-shifted dCE: the write at token t affects the prediction at t (same position) and later ones; report same-position
    dce = (ce1 - ce0).reshape(NS, CTX - 1); bigm = big.reshape(NS, CTX)[:, :-1]; res[b] = dict(profile=prof, dce_same_position=dce[bigm].mean().item())
    log(f"{tag} born b{b}: projection on d (clean -> write ablated), dependence: " + " ".join(f"+{k}: {v['clean']:.2f}->{v['ablated']:.2f} dep {v['dependence_med']:.2f}" for k, v in sorted(prof.items())) + f" | dCE at the token {res[b]['dce_same_position']:+.3f}")
import numpy as np
dep2 = float(np.mean([res[b]["profile"][2]["dependence_med"] for b in res if 2 in res[b]["profile"]])); dep4 = float(np.mean([res[b]["profile"][4]["dependence_med"] for b in res if 4 in res[b]["profile"]])); depL = float(np.mean([res[b]["profile"][max(res[b]["profile"])]["dependence_med"] for b in res]))
record(f"e208_maint_{tag}", dict(model=tag, L=L, per_birth={str(k): v for k, v in res.items()}), f"dependence of the later component on the write (1 = only because of the write, 0 = written anyway): +2 blocks {dep2:.2f}, +4 blocks {dep4:.2f}, at level {L} {depL:.2f} | removed component at +2: " + " ".join(f"b{b}:{res[b]['profile'][2]['removed_abs']:.2f} of {res[b]['profile'][2]['clean']:.2f}" for b in res if 2 in res[b]['profile']))
