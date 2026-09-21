"""e210: first-principles contraction. The expected gain of an MLP along a random direction is tr(J)/D with
J = W_down diag(act') W_in (GELU) or W_down [diag(silu(g)) W_up + diag(silu'(g) u) W_gate] (SwiGLU). Compute the
trace per token from hooked pre-activations, split by term (GELU: single term; gated: up-path vs gate-path), and
compare s * tr(J)/D (s = normalization gain = ||norm output|| / ||norm input||) with the measured generic gain of
e207. Also the fraction of neurons contributing negatively, and the per-block profile."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name, revision=getattr(c, "revision", None), random_init=bool(getattr(c, "random_init", False))); arch = Arch(model, fam); NS = 6; ids_seq = c.s["eval_ids"][:NS].to(DEV)
def norm_module(b):
    l = arch.layers[b]; return l.ln_2 if fam == "gpt2" else l.post_attention_layernorm
store = {}; hs = []
for b in range(arch.NB):
    hs.append(norm_module(b).register_forward_hook((lambda b_: lambda m, i, o: store.__setitem__(("y", b_), o.detach().float().reshape(-1, arch.D)) or store.__setitem__(("xin", b_), i[0].detach().float().reshape(-1, arch.D)))(b)))
model(ids_seq); [h.remove() for h in hs]; out = {}; BL = [b for b in (0, 2, 5, 8, 12, 16, 20, arch.NB - 1) if b < arch.NB]
for b in BL:
    y = store[("y", b)]; xin = store[("xin", b)]; W = arch.wdir(b).to(DEV); s = (y.norm(dim=1) / xin.norm(dim=1)).median().item(); rec = dict(norm_gain=s)
    if fam == "llama":
        l = arch.layers[b]; Wg = l.mlp.gate_proj.weight.detach().float(); Wu = l.mlp.up_proj.weight.detach().float(); g = y @ Wg.T; u = y @ Wu.T; sg = torch.sigmoid(g); silu = g * sg; dsilu = sg * (1 + g * (1 - sg))
        du = (Wu * W).sum(1); dg = (Wg * W).sum(1); t1 = (silu * du[None]).sum(1) / arch.D; t2 = (dsilu * u * dg[None]).sum(1) / arch.D; tr = t1 + t2
        rec.update(trace=tr.median().item(), up_term=t1.median().item(), gate_term=t2.median().item(), frac_neg_neurons=((silu * du[None] + dsilu * u * dg[None]) < 0).float().mean().item())
    else:
        R = arch.rdir(b).to(DEV); bias = (arch.layers[b].mlp.c_fc.bias if fam == "gpt2" else arch.layers[b].mlp.dense_h_to_4h.bias).detach().float(); z = y @ R.T + bias; dg = 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi)
        din = (R * W).sum(1); tr = (dg * din[None]).sum(1) / arch.D; rec.update(trace=tr.median().item(), up_term=tr.median().item(), gate_term=0.0, frac_neg_neurons=((dg * din[None]) < 0).float().mean().item(), static_trace=din.sum().item() / arch.D)
    rec["predicted_mlp_gain"] = rec["norm_gain"] * rec["trace"]; out[b] = rec
    log(f"{tag} block {b}: tr(J)/D {rec['trace']:+.4f} (up/in path {rec['up_term']:+.4f}, gate path {rec['gate_term']:+.4f}), norm gain {s:.2f} -> predicted generic MLP gain {rec['predicted_mlp_gain']:+.3f} | neurons contributing negatively {rec['frac_neg_neurons']:.2f}")
record(f"e210_jtrace_{tag}", dict(model=tag, per_block=out), "predicted generic MLP gain s*tr(J)/D by block: " + " ".join(f"b{b}:{v['predicted_mlp_gain']:+.2f}" for b, v in out.items()) + " | trace split (up-or-in path / gate path): " + " ".join(f"b{b}:{v['up_term']:+.3f}/{v['gate_term']:+.3f}" for b, v in out.items()))
