"""e217: when is the contraction born? For a Pythia-410m training revision (argv[1] = step name): the read/write
alignment per neuron (weights only), the exact Jacobian trace per block from hooked pre-activations with the
predicted generic MLP gain s * tr(J)/D, and one directly measured generic gain (block 5, random directions). The
revision's snapshot is deleted afterwards to save disk. The tokens come from the pythia410 cache."""
import sys, os, glob, shutil; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
rev = sys.argv[1]; c = Cache("pythia410"); hub = os.path.join(os.environ.get("HF_HOME", "/workspace/.hf_home"), "hub", "models--EleutherAI--pythia-410m"); before = set(glob.glob(os.path.join(hub, "snapshots", "*")))
model, tok, fam = load_model(c.name, revision=rev); arch = Arch(model, fam); NS = 4; ids_seq = c.s["eval_ids"][:NS].to(DEV)
def norm_module(b): return arch.layers[b].post_attention_layernorm
store = {}; hs = [norm_module(b).register_forward_hook((lambda b_: lambda m, i, o: store.__setitem__(("y", b_), o.detach().float().reshape(-1, arch.D)) or store.__setitem__(("xin", b_), i[0].detach().float().reshape(-1, arch.D)))(b)) for b in range(arch.NB)]
model(ids_seq); [h.remove() for h in hs]; per = {}
for b in range(arch.NB):
    W = arch.wdir(b).to(DEV); R = arch.rdir(b).to(DEV); cosu = (W * R).sum(1) / (W.norm(dim=1) * R.norm(dim=1)).clamp_min(1e-9)
    y = store[("y", b)]; xin = store[("xin", b)]; s = (y.norm(dim=1) / xin.norm(dim=1)).median().item(); bias = arch.layers[b].mlp.dense_h_to_4h.bias.detach().float(); z = y @ R.T + bias; dg = 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi); tr = (dg * (R * W).sum(1)[None]).sum(1) / arch.D
    per[b] = dict(cos_mean=cosu.mean().item(), trace=tr.median().item(), norm_gain=s, predicted_gain=s * tr.median().item())
# direct measurement at block 5
def run(B, delta=None):
    st = {}; hs = [arch.layers[B].register_forward_hook(lambda m, i, o: st.__setitem__("out", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, arch.D))), arch.layers[B].register_forward_pre_hook(lambda m, args, kwargs: st.__setitem__("in", (args[0] if len(args) > 0 else kwargs["hidden_states"]).detach().float().reshape(-1, arch.D)), with_kwargs=True)]
    if delta is not None:
        def pre(m, args, kwargs):
            if len(args) > 0: return (args[0] + delta.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + delta.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        hs.append(arch.layers[B].register_forward_pre_hook(pre, with_kwargs=True))
    model(ids_seq); [h.remove() for h in hs]; return st
torch.manual_seed(0); S0 = run(5); x = S0["in"]; typ = typical_mask(x); U = torch.randn_like(x); U = U / U.norm(dim=1, keepdim=True); eps = 0.1 * x.norm(dim=1, keepdim=True); S1 = run(5, eps * U); g5 = (((S1["out"] - S0["out"]) * U).sum(1) / eps[:, 0] - 1.0)[typ].median().item()
import numpy as np
mid = [per[b]["cos_mean"] for b in range(1, 13)]; mpred = [per[b]["predicted_gain"] for b in range(1, 13)]
record(f"e217_birth_{rev}", dict(revision=rev, per_block=per, measured_gain_block5=g5), f"{rev}: cos(read, write) mean over blocks 1-12 {np.mean(mid):+.3f} (block 5 {per[5]['cos_mean']:+.3f}) | predicted generic MLP gain mean over blocks 1-12 {np.mean(mpred):+.3f} (block 5 {per[5]['predicted_gain']:+.3f}) | measured block-5 gain {g5:+.3f} | block 0 cos {per[0]['cos_mean']:+.3f}")
del model; torch.cuda.empty_cache()
for p in set(glob.glob(os.path.join(hub, "snapshots", "*"))) - before: shutil.rmtree(p, ignore_errors=True)
rf = os.path.join(hub, "refs", rev)
if os.path.exists(rf): os.remove(rf)
