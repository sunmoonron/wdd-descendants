"""e261: do the neurons that compensate for a removed write live near it in write space or in descendant space? The
dominant block-2 write is removed at every token; the delta-ledger of later blocks (3..L) gives each later neuron's
response. For each candidate neuron k: the 20 later neurons with the largest mean |delta coefficient| over k's tokens
(the compensators). Reported: |cos| of the compensators' write directions with k's write vector (W-space) and with k's
descendant centroid at L (D-space), vs random later neurons; and the share of the compensating energy carried by
neurons aligned (|cos| > 0.1) with the descendant vs with the write."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; DFF = c.DFF
def rows(bb): return A[(lab["type"] == T_MLP) & (lab["block"] == bb)].float().to(DEV)
later = list(range(b + 1, L + 1)); WN = {j: c.d["WN"][j].to(DEV) for j in later}; Rl = torch.cat([rows(j) for j in later]); blk_of = torch.cat([torch.full((DFF,), j, device=DEV) for j in later])
def run(neuron=None):
    st = {"a": {}}; hs = [arch.layers[L].register_forward_hook(lambda m, i, o: st.__setitem__("H", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, D)))]
    for j in later: hs.append(arch.mlp_lin(j).register_forward_pre_hook((lambda j_: lambda m, inp: st["a"].__setitem__(j_, inp[0].detach().float().reshape(-1, DFF)))(j)))
    if neuron is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    model(ids_seq); [h.remove() for h in hs]; return st
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0["H"]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); F = S0["H"] - S1["H"]; Cd = unit(centroids(F[idx], lab_i, K)); R2 = rows(b); torch.manual_seed(0)
dC = torch.cat([(S1["a"][j] - S0["a"][j]) * WN[j][None] for j in later], 1)                                   # [NT, n_later] delta-ledger
res_k = []
for k in range(K):
    toks = idx[lab_i == k]; mean_abs = dC[toks].abs().mean(0); top = mean_abs.topk(20).indices; rnd = torch.randperm(len(mean_abs), device=DEV)[:20]; wk = R2[keep[k]]; dk = Cd[k]
    cw, cd = (Rl[top] @ wk).abs(), (Rl[top] @ dk).abs(); rw, rd = (Rl[rnd] @ wk).abs(), (Rl[rnd] @ dk).abs(); energy = (dC[toks] ** 2).mean(0); aligned_w = (Rl @ wk).abs() > 0.1; aligned_d = (Rl @ dk).abs() > 0.1
    res_k.append(dict(top_cos_w=cw.mean().item(), top_cos_d=cd.mean().item(), rnd_cos_w=rw.mean().item(), rnd_cos_d=rd.mean().item(), energy_share_aligned_w=(energy[aligned_w].sum() / energy.sum()).item(), energy_share_aligned_d=(energy[aligned_d].sum() / energy.sum()).item(), frac_aligned_w=aligned_w.float().mean().item(), frac_aligned_d=aligned_d.float().mean().item(), top_blocks=[int(x) for x in blk_of[top].tolist()[:5]]))
import numpy as np
agg = {k_: float(np.mean([r[k_] for r in res_k])) for k_ in res_k[0] if k_ != "top_blocks"}
log(f"{tag} (K {K}, later blocks {later[0]}-{later[-1]}): compensators' |cos| with the removed WRITE {agg['top_cos_w']:.3f} (random later neurons {agg['rnd_cos_w']:.3f}) vs with its DESCENDANT {agg['top_cos_d']:.3f} (random {agg['rnd_cos_d']:.3f}) | compensating energy carried by neurons aligned with the write {agg['energy_share_aligned_w']:.2f} (such neurons are {agg['frac_aligned_w']:.2f} of all) vs with the descendant {agg['energy_share_aligned_d']:.2f} ({agg['frac_aligned_d']:.2f} of all)")
record(f"e261_compnb_{tag}", dict(model=tag, b=b, L=L, K=K, aggregate=agg, per_neuron=res_k), f"K {K}: compensators |cos| with write {agg['top_cos_w']:.3f} (random {agg['rnd_cos_w']:.3f}) vs with descendant {agg['top_cos_d']:.3f} (random {agg['rnd_cos_d']:.3f}); energy share aligned with write {agg['energy_share_aligned_w']:.2f} ({agg['frac_aligned_w']:.2f} of neurons) vs descendant {agg['energy_share_aligned_d']:.2f} ({agg['frac_aligned_d']:.2f})")
