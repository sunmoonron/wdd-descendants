"""e194: is the next-block cancellation a local linear gain? Perturb the residual stream entering block b+1 by
eps * d (d = the token's dominant block-b write direction, eps = a fraction of that write's coefficient) and
measure the response of block b+1's MLP write and of the whole block along d: gain = (delta . d) / eps. Compare
with random unit directions (specific vs generic damping) and across eps (linearity). e173 predicts a gain of
about -0.3 along d for the chain models if the crowd cancellation is a feedback response to what is present."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX
births = (6, 7, 8) if tag.startswith("pythia") else (2, 3, 4); WN = [c.d["WN"][b].to(DEV) for b in range(L + 1)]; wd = [c.wdir_cpu(b).to(DEV) for b in range(L + 1)]; mb = [None if c.d["mlp_bias"][b] is None else c.d["mlp_bias"][b].to(DEV) for b in range(L + 1)]
lab = c.d["lab"]; A = c.d["A"]
def rows(b): return A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
def run(b1=None, delta=None):
    store = {}; hs = []
    for b in range(L + 1): hs.append(arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, inp: store.__setitem__(("acts", b_), inp[0].detach().float().reshape(-1, c.DFF)))(b)))
    for b in range(L + 1): hs.append(arch.layers[b].register_forward_hook((lambda b_: lambda m, i, o: store.__setitem__(("out", b_), (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(b)))
    if delta is not None:
        def pre(m, args, kwargs):
            if len(args) > 0: return (args[0] + delta.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + delta.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        hs.append(arch.layers[b1].register_forward_pre_hook(pre, with_kwargs=True))
    model(ids_seq); [h.remove() for h in hs]; return store
S0 = run(); typ = typical_mask(S0[("out", L)]); out = {}
torch.manual_seed(0)
for b in births:
    if b + 1 > L: continue
    led = S0[("acts", b)] * WN[b][None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; d = rows(b)[tn]; big = typ & (tc.abs() >= tc.abs().quantile(0.5))
    R = torch.randn_like(d); R = R / R.norm(dim=1, keepdim=True); rec = {}
    w0 = S0[("acts", b + 1)] @ wd[b + 1] + (mb[b + 1] if mb[b + 1] is not None else 0); o0 = S0[("out", b + 1)]
    for frac in (0.25, 0.5, 1.0):
        eps = frac * tc.abs()
        for nm, dirn in (("d", d), ("random", R)):
            S1 = run(b + 1, eps[:, None] * dirn); w1 = S1[("acts", b + 1)] @ wd[b + 1] + (mb[b + 1] if mb[b + 1] is not None else 0); o1 = S1[("out", b + 1)]
            g_mlp = ((w1 - w0) * dirn).sum(1) / eps; g_blk = ((o1 - o0) * dirn).sum(1) / eps - 1.0; g_att = g_blk - g_mlp
            # response of the NEXT-next block too (b+2), from the same perturbed run: total gain seen at its output
            o2_0, o2_1 = S0[("out", b + 2)], S1[("out", b + 2)]; g_two = ((o2_1 - o2_0) * dirn).sum(1) / eps - 1.0
            rec[(frac, nm)] = dict(mlp=g_mlp[big].median().item(), block=g_blk[big].median().item(), attention=g_att[big].median().item(), two_blocks=g_two[big].median().item(), mlp_mean=g_mlp[big].mean().item())
    out[b] = {f"{k[0]}_{k[1]}": v for k, v in rec.items()}
    log(f"{tag} born b{b}, perturb input of block {b + 1}: gain along d (mlp/att/block, eps=0.25,0.5,1.0 x coef): " + " ".join(f"{rec[(f, 'd')]['mlp']:+.2f}/{rec[(f, 'd')]['attention']:+.2f}/{rec[(f, 'd')]['block']:+.2f}" for f in (0.25, 0.5, 1.0)) + " | random dir: " + " ".join(f"{rec[(f, 'random')]['mlp']:+.2f}/{rec[(f, 'random')]['attention']:+.2f}/{rec[(f, 'random')]['block']:+.2f}" for f in (0.25, 0.5, 1.0)) + f" | after two blocks (eps=0.5): d {rec[(0.5, 'd')]['two_blocks']:+.2f} random {rec[(0.5, 'random')]['two_blocks']:+.2f}")
import numpy as np
m = lambda key, nm, f: float(np.mean([out[b][f"{f}_{nm}"][key] for b in out]))
record(f"e194_linresp_{tag}", dict(model=tag, L=L, per_birth=out), f"local gain of block b+1 along the block-b write direction d: MLP {m('mlp', 'd', 0.5):+.2f} attention {m('attention', 'd', 0.5):+.2f} whole block {m('block', 'd', 0.5):+.2f} (eps = 0.5 x coef; at 0.25x: {m('block', 'd', 0.25):+.2f}, 1x: {m('block', 'd', 1.0):+.2f}) | random directions: MLP {m('mlp', 'random', 0.5):+.2f} attention {m('attention', 'random', 0.5):+.2f} block {m('block', 'random', 0.5):+.2f} | gain after two blocks: d {m('two_blocks', 'd', 0.5):+.2f} random {m('two_blocks', 'random', 0.5):+.2f}")
