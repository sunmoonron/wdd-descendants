"""e244: how context-dependent is each MLP's linearisation? Per block (2, 5, 8) the per-neuron factor of the first-
order response, gelu'(z) for GELU MLPs and silu(g) (up path) for SwiGLU MLPs, over typical tokens: the relative
deviation of the factor vector from its token mean, ||f_t - f_mean|| / ||f_mean|| (median over tokens), the cosine
of f_t with f_mean, and the fraction of neurons whose factor changes sign across tokens. A candidate mechanism for
the GELU-vs-gated split in how context shapes descendants (e231, e238)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV)
def norm_module(b): return arch.layers[b].ln_2 if fam == "gpt2" else arch.layers[b].post_attention_layernorm
store = {}; hs = [norm_module(b).register_forward_hook((lambda b_: lambda m, i, o: store.__setitem__(b_, o.detach().float().reshape(-1, arch.D)))(b)) for b in (2, 5, 8) if b < arch.NB]; hs.append(arch.layers[min(8, arch.NB - 1)].register_forward_hook(lambda m, i, o: store.__setitem__("H", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, arch.D)))); model(ids_seq); [h.remove() for h in hs]; typ = typical_mask(store["H"]); out = {}
for b in [bb for bb in (2, 5, 8) if bb < arch.NB]:
    y = store[b][typ]
    if fam == "llama": g = y @ arch.layers[b].mlp.gate_proj.weight.detach().float().T; f = g * torch.sigmoid(g)
    else:
        R = arch.rdir(b).to(DEV); bias = (arch.layers[b].mlp.c_fc.bias if fam == "gpt2" else arch.layers[b].mlp.dense_h_to_4h.bias).detach().float(); z = y @ R.T + bias; f = 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi)
    fm = f.mean(0, keepdim=True); rel = ((f - fm).norm(dim=1) / fm.norm().clamp_min(1e-9)); cosm = (f @ fm.T)[:, 0] / (f.norm(dim=1) * fm.norm()).clamp_min(1e-9); sign_flip = ((f > 0).float().mean(0) * (f < 0).float().mean(0) > 0).float().mean().item()
    out[b] = dict(relative_deviation=rel.median().item(), cos_with_mean=cosm.median().item(), sign_flip_fraction=sign_flip, mean_factor=fm.mean().item())
    log(f"{tag} block {b} ({'silu(gate)' if fam == 'llama' else 'gelu-prime'}): relative deviation of the linearisation factor from its mean {out[b]['relative_deviation']:.2f}, cos with mean {out[b]['cos_with_mean']:.2f}, neurons whose factor changes sign {out[b]['sign_flip_fraction']:.2f}, mean factor {out[b]['mean_factor']:.3f}")
import numpy as np
record(f"e244_linvar_{tag}", dict(model=tag, family=fam, per_block={str(k): v for k, v in out.items()}), f"{'SwiGLU' if fam == 'llama' else 'GELU'}: relative deviation mean over blocks {np.mean([v['relative_deviation'] for v in out.values()]):.2f}, cos with mean {np.mean([v['cos_with_mean'] for v in out.values()]):.2f}, sign-flipping neurons {np.mean([v['sign_flip_fraction'] for v in out.values()]):.2f}")
