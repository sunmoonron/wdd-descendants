"""e209: the static weight signature of the learned contraction. To first order the MLP Jacobian is
W_out^T diag(act') W_in, so a negative generic gain requires the neurons' read and write vectors to be anti-
aligned on average. Per block: the distribution of cos(read_i, write_i) over neurons (up-projection row vs
down-projection row; for gated MLPs also the gate row), the trace proxy sum_i read_i . write_i / D, and the same
for random initialisation and Pythia checkpoints (weights only, no forward pass)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name, revision=getattr(c, "revision", None), random_init=bool(getattr(c, "random_init", False))); arch = Arch(model, fam); out = {}
for b in range(arch.NB):
    W = arch.wdir(b).to(DEV); R = arch.rdir(b, "up").to(DEV); cosu = (W * R).sum(1) / (W.norm(dim=1) * R.norm(dim=1)).clamp_min(1e-9); tr = (W * R).sum().item() / arch.D
    rec = dict(cos_mean=cosu.mean().item(), cos_med=cosu.median().item(), frac_neg_0p1=(cosu < -0.1).float().mean().item(), frac_pos_0p1=(cosu > 0.1).float().mean().item(), trace_proxy=tr)
    if fam == "llama":
        G = arch.rdir(b, "gate").to(DEV); cosg = (W * G).sum(1) / (W.norm(dim=1) * G.norm(dim=1)).clamp_min(1e-9); rec.update(gate_cos_mean=cosg.mean().item(), gate_frac_neg_0p1=(cosg < -0.1).float().mean().item(), gate_frac_pos_0p1=(cosg > 0.1).float().mean().item())
    out[b] = rec
    if b in (0, 2, 5, 8, arch.NB - 1): log(f"{tag} block {b}: cos(read, write) mean {rec['cos_mean']:+.3f} median {rec['cos_med']:+.3f}, < -0.1: {rec['frac_neg_0p1']:.2f}, > +0.1: {rec['frac_pos_0p1']:.2f}, trace proxy {tr:+.3f}" + (f" | gate: mean {rec['gate_cos_mean']:+.3f} < -0.1: {rec['gate_frac_neg_0p1']:.2f} > 0.1: {rec['gate_frac_pos_0p1']:.2f}" if fam == "llama" else ""))
import numpy as np
mean_cos = float(np.mean([out[b]["cos_mean"] for b in out])); mean_tr = float(np.mean([out[b]["trace_proxy"] for b in out])); neg = float(np.mean([out[b]["frac_neg_0p1"] for b in out])); pos = float(np.mean([out[b]["frac_pos_0p1"] for b in out]))
record(f"e209_inout_{tag}", dict(model=tag, revision=getattr(c, "revision", None), random_init=bool(getattr(c, "random_init", False)), per_block=out), f"cos(read, write) per neuron: mean {mean_cos:+.3f} over blocks; fraction < -0.1: {neg:.2f}, > +0.1: {pos:.2f}; trace proxy {mean_tr:+.3f} | by block: " + " ".join(f"b{b}:{out[b]['cos_mean']:+.3f}" for b in out) + (" | gate cos mean " + f"{np.mean([out[b]['gate_cos_mean'] for b in out]):+.3f}" if fam == "llama" else ""))
