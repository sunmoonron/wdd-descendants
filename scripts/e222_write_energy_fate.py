"""e222: the direct test of 'provenance fades by decorrelation, not loss'. Zero the token's dominant block-b write
(b = 1, 2, 3) and, at levels b, b+1, b+2, b+4, b+6 and the mid layer L, measure (i) the readability of its atom
(dual@64 at that level with that level's dictionary), (ii) the along-direction causal footprint (clean minus ablated
projection on d, over the write's coefficient), (iii) the TOTAL causal footprint ||H_clean - H_ablated|| over the
write's magnitude (the write's consequences in any direction), and (iv) the same-position loss change.
Prediction: (iii) stays near or above 1 while (i) and (ii) decay."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 10; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; lab = c.d["lab"]; A = c.d["A"]
def rows(b): return A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
def run(b=None, neuron=None):
    store = {}; hs = [arch.layers[lv].register_forward_hook((lambda lv_: lambda m, i, o: store.__setitem__(lv_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(lv)) for lv in range(c.NB)]
    if b is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    out = model(ids_seq); [h.remove() for h in hs]; lg = out.logits[:, :-1].reshape(-1, out.logits.shape[-1]).float(); ce = torch.nn.functional.cross_entropy(lg, ids_seq[:, 1:].reshape(-1), reduction="none"); return store, ce
S0, ce0 = run(); res = {}; Wcache = {}
for b in (1, 2, 3):
    st = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st.__setitem__("a", inp[0].detach().float().reshape(-1, c.DFF))); model(ids_seq); h.remove()
    led = st["a"] * c.d["WN"][b].to(DEV)[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; d = rows(b)[tn]; big = tc.abs() >= tc.abs().quantile(0.5); S1, ce1 = run(b, tn); prof = {}
    for lv in sorted({b, b + 1, b + 2, b + 4, b + 6, L}):
        if lv >= c.NB: continue
        typ = typical_mask(S0[lv]) & big; diff = S0[lv] - S1[lv]; along = ((diff * d).sum(1) / tc); total = diff.norm(dim=1) / tc.abs()
        Alv, lablv = c.dictionary(lv); row = c.atom_index(lv, torch.full_like(tn.cpu(), b), tn.cpu()).to(DEV); X = S0[lv] - c.s["mu"][lv + 1].to(DEV)
        if lv not in Wcache: S = Alv.T @ Alv; ev, V = torch.linalg.eigh(S); Wcache[lv] = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
        sel = oneshot(X, Alv, 64, whiten=Wcache[lv])[0]; read = (sel == row[:, None]).any(1)
        prof[lv - b] = dict(level=lv, readability=read[typ].float().mean().item(), along_footprint=along[typ].median().item(), total_footprint=total[typ].median().item(), total_over_state=(diff.norm(dim=1) / S0[lv].norm(dim=1))[typ].median().item())
    dce = (ce1 - ce0).reshape(NS, CTX - 1); bigm = big.reshape(NS, CTX)[:, :-1]; res[b] = dict(profile=prof, dce=dce[bigm].mean().item())
    log(f"{tag} born b{b}: " + " | ".join(f"+{k}: read {v['readability']:.2f}, along {v['along_footprint']:.2f}, total {v['total_footprint']:.2f}" for k, v in sorted(prof.items())) + f" | dCE {res[b]['dce']:+.3f}")
import numpy as np
ks = sorted(set(k for b in res for k in res[b]["profile"])); agg = {k: {q: float(np.mean([res[b]["profile"][k][q] for b in res if k in res[b]["profile"]])) for q in ("readability", "along_footprint", "total_footprint")} for k in ks}
record(f"e222_energyfate_{tag}", dict(model=tag, L=L, per_birth={str(k): v for k, v in res.items()}, aggregate={str(k): v for k, v in agg.items()}), "distance after the write -> (readability, along-direction footprint, total footprint): " + " ".join(f"+{k}: ({v['readability']:.2f}, {v['along_footprint']:.2f}, {v['total_footprint']:.2f})" for k, v in agg.items()))
