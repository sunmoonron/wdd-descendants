"""e224: do different writes converge (provenance collision) or stay distinct, in residual space and in function
space? For each token take its three largest block-b writes (distinct neurons; b = 2) and two random directions of
the same size injected at the same point. Footprints delta^(a)(lv) from separate ablation / injection runs; pairwise
cosines between footprints at levels b+1, b+2, b+4, L (vs the cosine of the original directions), and pairwise
cosines of the logit changes at the token. Merging: residual cosine rises with depth for real writes but not random
directions. Laundering: residual cosine stays low while logit cosine is high."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 6; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; lab = c.d["lab"]; A = c.d["A"]; b = 2
def rows(bb): return A[(lab["type"] == T_MLP) & (lab["block"] == bb)].float().to(DEV)
def run(neuron=None, inject=None):
    st = {}; hs = [arch.layers[j].register_forward_hook((lambda j_: lambda m, i, o: st.__setitem__(j_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(j)) for j in range(NB)]
    if neuron is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    if inject is not None:
        def pre2(m, args, kwargs):
            if len(args) > 0: return (args[0] + inject.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + inject.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        hs.append(arch.layers[b + 1].register_forward_pre_hook(pre2, with_kwargs=True))
    out = model(ids_seq); [h.remove() for h in hs]; return st, out.logits[:, :-1].reshape(-1, out.logits.shape[-1]).float()
st_a = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st_a.__setitem__("a", inp[0].detach().float().reshape(-1, c.DFF))); model(ids_seq); h.remove()
led = st_a["a"] * c.d["WN"][b].to(DEV)[None]; top3 = led.abs().topk(3, dim=1).indices; coefs = torch.gather(led, 1, top3); R = rows(b); big = coefs[:, 0].abs() >= coefs[:, 0].abs().quantile(0.5)
S0, lg0 = run(); runs = {}
for k in range(3): runs[f"w{k}"] = run(neuron=top3[:, k])
torch.manual_seed(0)
for k in range(2):
    U = torch.randn(NT, c.D, device=DEV); U = U / U.norm(dim=1, keepdim=True); runs[f"r{k}"] = run(inject=-coefs[:, 0:1].abs() * U)
def cosv(x, y): return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1)).clamp_min(1e-9)
names = list(runs); pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]; kind = lambda p: "real-real" if p[0][0] == "w" and p[1][0] == "w" else ("random-random" if p[0][0] == "r" and p[1][0] == "r" else "real-random")
out = {}
dirs = {f"w{k}": R[top3[:, k]] for k in range(3)}
for lv in sorted({b, b + 1, b + 2, b + 4, L}):
    if lv >= NB: continue
    typ = typical_mask(S0[lv]) & big; rec = {"real-real": [], "random-random": [], "real-random": []}; orig = []
    for p in pairs:
        da = runs[p[0]][0][lv] - S0[lv]; db = runs[p[1]][0][lv] - S0[lv]; rec[kind(p)].append(cosv(da, db)[typ].median().item())
        if kind(p) == "real-real": orig.append(cosv(dirs[p[0]], dirs[p[1]])[typ].median().item())
    out[lv] = {k: float(sum(v) / len(v)) for k, v in rec.items()}; out[lv]["orig_real_real"] = float(sum(orig) / len(orig))
    log(f"{tag} level {lv}: median pairwise footprint cosine real-real {out[lv]['real-real']:+.2f} (their write directions {out[lv]['orig_real_real']:+.2f}), real-random {out[lv]['real-random']:+.2f}, random-random {out[lv]['random-random']:+.2f}")
# logit space
bigm = big.reshape(NS, CTX)[:, :-1].reshape(-1); recl = {"real-real": [], "random-random": [], "real-random": []}
for p in pairs:
    da = runs[p[0]][1] - lg0; db = runs[p[1]][1] - lg0; recl[kind(p)].append(cosv(da, db)[bigm].median().item())
lgt = {k: float(sum(v) / len(v)) for k, v in recl.items()}
log(f"{tag} logit-space footprint cosine: real-real {lgt['real-real']:+.2f}, real-random {lgt['real-random']:+.2f}, random-random {lgt['random-random']:+.2f}")
record(f"e224_collision_{tag}", dict(model=tag, b=b, L=L, per_level={str(k): v for k, v in out.items()}, logit=lgt), "residual-space real-real footprint cosine by level " + " ".join(f"{lv}:{v['real-real']:+.2f}" for lv, v in out.items()) + f" (write directions {out[min(out)]['orig_real_real']:+.2f}) | random-random " + " ".join(f"{lv}:{v['random-random']:+.2f}" for lv, v in out.items()) + f" | logit-space: real-real {lgt['real-real']:+.2f}, random-random {lgt['random-random']:+.2f}, real-random {lgt['real-random']:+.2f}")
