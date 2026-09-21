"""e207: where does the generic contraction come from, and when does it appear? For blocks (2, 5, 8): the gain of
the whole block, of the MLP write, and of the attention write along random unit directions when the BLOCK INPUT
is perturbed (through LayerNorm), and the gain of the MLP write when its POST-LayerNorm input is perturbed
directly (the weights' Jacobian alone). Run on trained models, on random initialisation, and on Pythia training
checkpoints (via the cache's revision)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name, revision=getattr(c, "revision", None), random_init=bool(getattr(c, "random_init", False))); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV)
def mlp_module(b):
    lin = arch.mlp_lin(b)
    for name, mod in arch.layers[b].named_modules():
        if any(ch is lin for ch in mod.children()): return mod
    raise RuntimeError("mlp module not found")
def run(b, where=None, delta=None):
    store = {}; hs = [arch.layers[b].register_forward_hook(lambda m, i, o: store.__setitem__("out", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D))), mlp_module(b).register_forward_hook(lambda m, i, o: store.__setitem__("mlp", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D))), mlp_module(b).register_forward_pre_hook(lambda m, inp: store.__setitem__("mlp_in", inp[0].detach().float().reshape(-1, c.D)))]
    if where is not None:
        def pre(m, args, kwargs):
            if len(args) > 0: return (args[0] + delta.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + delta.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        target = arch.layers[b] if where == "block" else mlp_module(b); hs.append(target.register_forward_pre_hook(pre, with_kwargs=True))
    model(ids_seq); [h.remove() for h in hs]; return store
torch.manual_seed(0); out = {}
BL = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (2, 5, 8); SUF = ("_" + sys.argv[2].replace(",", "_")) if len(sys.argv) > 2 else ""
for b in [bb for bb in BL if bb < c.NB]:
    S0 = run(b); x0 = S0["out"]; typ = typical_mask(x0); U = torch.randn_like(x0); U = U / U.norm(dim=1, keepdim=True)
    # block-input perturbation, relative size 0.1 of the pre-block state norm proxy (use output norm)
    eps = 0.1 * x0.norm(dim=1, keepdim=True); S1 = run(b, "block", eps * U); g_blk = (((S1["out"] - x0) * U).sum(1) / eps[:, 0] - 1.0); g_mlp = (((S1["mlp"] - S0["mlp"]) * U).sum(1) / eps[:, 0]); g_att = g_blk - g_mlp
    # post-LN perturbation of the MLP input, relative size 0.1 of the LN output norm
    eps2 = 0.1 * S0["mlp_in"].norm(dim=1, keepdim=True); S2 = run(b, "mlp", eps2 * U); g_post = (((S2["mlp"] - S0["mlp"]) * U).sum(1) / eps2[:, 0])
    out[b] = dict(block=g_blk[typ].median().item(), mlp=g_mlp[typ].median().item(), attention=g_att[typ].median().item(), mlp_post_ln=g_post[typ].median().item(), ln_scale=(S0["mlp_in"].norm(dim=1) / x0.norm(dim=1))[typ].median().item())
    log(f"{tag} block {b}: gain along random directions: whole block {out[b]['block']:+.2f} = MLP {out[b]['mlp']:+.2f} + attention {out[b]['attention']:+.2f} | MLP gain for a post-LN perturbation {out[b]['mlp_post_ln']:+.2f} | LN output / state norm {out[b]['ln_scale']:.2f}")
import numpy as np
record(f"e207_contraction_{tag}{SUF}", dict(model=tag, revision=getattr(c, "revision", None), random_init=bool(getattr(c, "random_init", False)), per_block=out), "generic gain (whole block / MLP / attention / MLP post-LN): " + " ".join(f"b{b}: {v['block']:+.2f}/{v['mlp']:+.2f}/{v['attention']:+.2f}/{v['mlp_post_ln']:+.2f}" for b, v in out.items()) + f" | mean block gain {np.mean([v['block'] for v in out.values()]):+.2f}, mean post-LN MLP gain {np.mean([v['mlp_post_ln'] for v in out.values()]):+.2f}")
