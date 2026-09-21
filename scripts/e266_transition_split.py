"""e266: which transition builds the functional organisation of descendants, and is it attention or the MLP? Natural
ablation with the state read at every block's input, after its attention residual add, and at its output (parallel
model: the middle point is the attention-only partial sum); logits at the candidate positions. Per block: the Gram
Spearman between descendant-centroid geometry and logit-footprint geometry at the three points; the increment due to
attention (mid - in) and to the MLP (out - mid); totals over blocks and the block with the largest increment."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; blocks = list(range(b + 1, NB))
def run(neuron=None, positions=None):
    st = {"att": {}}; hs = [arch.layers[j].register_forward_hook((lambda j_: lambda m, i, o: st.__setitem__(j_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, D)))(j)) for j in range(b, NB)]
    for j in blocks: hs.append(arch.attn_lin(j).register_forward_hook((lambda j_: lambda m, i, o: st["att"].__setitem__(j_, o.detach().float().reshape(-1, D)))(j)))
    if neuron is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    out = model(ids_seq); [h.remove() for h in hs]; st["lg"] = out.logits.reshape(NT, -1).float()[positions] if positions is not None else None; del out; return st
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); S0 = run(positions=idx); S1 = run(tn, positions=idx); dl = S0["lg"] - S1["lg"]; dl = dl - dl.mean(1, keepdim=True); Cl = unit(centroids(dl, lab_i, K)); Gl = Cl @ Cl.T
def sp(H0, H1):
    F = (H0 - H1)[idx]; Cd = unit(centroids(F, lab_i, K)); return gram_spearman(Cd @ Cd.T, Gl)
out = {}; att_total = 0.0; mlp_total = 0.0
for j in blocks:
    s_in = sp(S0[j - 1], S1[j - 1]); s_mid = sp(S0[j - 1] + S0["att"][j], S1[j - 1] + S1["att"][j]); s_out = sp(S0[j], S1[j]); out[j] = dict(s_in=s_in, s_mid=s_mid, s_out=s_out, att_increment=s_mid - s_in, mlp_increment=s_out - s_mid); att_total += s_mid - s_in; mlp_total += s_out - s_mid
    log(f"{tag} block {j}: descendant-logit Spearman in {s_in:+.2f} -> after attention {s_mid:+.2f} -> out {s_out:+.2f} (attention {s_mid - s_in:+.2f}, MLP {s_out - s_mid:+.2f})")
best = max(out, key=lambda j: out[j]["att_increment"] + out[j]["mlp_increment"])
record(f"e266_transition_{tag}", dict(model=tag, b=b, L=L, K=K, parallel=(fam == "neox"), per_block={str(k): v for k, v in out.items()}, att_total=att_total, mlp_total=mlp_total, largest_block=best), f"K {K}: total increment of descendant-logit correspondence from attention {att_total:+.2f} vs MLP {mlp_total:+.2f}; largest single-block increment at block {best} ({out[best]['att_increment'] + out[best]['mlp_increment']:+.2f}) | per block att/mlp " + " ".join(f"{j}:{v['att_increment']:+.2f}/{v['mlp_increment']:+.2f}" for j, v in out.items()))
