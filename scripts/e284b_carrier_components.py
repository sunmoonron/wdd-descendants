"""e284b: e284 at component resolution and at centroid level. For the largest-coefficient candidate and a typical one,
the natural footprint at L is recomputed with one component (the attention output or the MLP output of one block)
held at its unablated value at the neuron's own tokens, for every block from b+1 to L. Reported per (component,
block): the cosine between the frozen footprint centroid and the transplant image centroid (context-free
reference), and the change relative to the unfrozen natural footprint centroid. The component whose freeze moves
the centroid most toward the transplant is the carrier of the context-bound reaction."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); run = make_runner(model, arch, c, ids_seq, [L], NT)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(b, tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); F = S0[L] - S1[L]; smed = torch.stack([tc[idx][lab_i == k].median() for k in range(K)]); med = smed.abs(); sgn = torch.sign(smed); order = med.argsort(); picks = {"largest": int(med.argmax()), "typical": int(order[len(order) // 2])}
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0]; torch.manual_seed(0); assign = torch.randint(0, 2, (len(foreign),), device=DEV); ks = list(picks.values()); vec = unit(R[keep[ks]]); amp = (sgn[ks] * med[ks])[assign]; inj = torch.zeros(NT, D, device=DEV); inj[foreign] = amp[:, None] * vec[assign]; S2 = run(inject=inj, inject_block=b + 1); timg = {nm: unit((S2[L] - S0[L])[foreign[assign == j]].mean(0, keepdim=True))[0] for j, nm in enumerate(picks)}
mods = {("attn", lv): arch.attn_lin(lv) for lv in range(b + 1, L + 1)}; mods.update({("mlp", lv): arch.mlp_lin(lv) for lv in range(b + 1, L + 1)}); cache = {}
hs = [m.register_forward_hook((lambda key: lambda mod, i, o: cache.__setitem__(key, o.detach().clone()))(key)) for key, m in mods.items()]; model(ids_seq); [h.remove() for h in hs]
def frozen_centroid(rows, keys):
    def mk(key):
        def hook(mod, i, o):
            o2 = o.clone(); flat = o2.reshape(-1, o2.shape[-1]); flat[rows] = cache[key].reshape(-1, o2.shape[-1])[rows]; return o2
        return hook
    hs = [mods[key].register_forward_hook(mk(key)) for key in keys]; S1f = run(b, tn); [h.remove() for h in hs]; return unit((S0[L] - S1f[L])[rows].mean(0, keepdim=True))[0]
out = {}
for nm, k in picks.items():
    rows = idx[lab_i == k]; ref = timg[nm]; nat = unit(F[rows].mean(0, keepdim=True))[0]; base = (nat @ ref).item(); rec = dict(neuron=int(keep[k]), coef=med[k].item(), natural_centroid_vs_transplant=base, all_attention=(frozen_centroid(rows, [key for key in mods if key[0] == "attn"]) @ ref).item(), all_mlp=(frozen_centroid(rows, [key for key in mods if key[0] == "mlp"]) @ ref).item(), scan={})
    for lv in range(b + 1, L + 1):
        for comp in ("attn", "mlp"):
            rec["scan"][f"{comp}{lv}"] = (frozen_centroid(rows, [(comp, lv)]) @ ref).item()
    out[nm] = rec; top = sorted(rec["scan"].items(), key=lambda kv: -(kv[1] - base))[:3]
    log(f"{tag} {nm} neuron {int(keep[k])} (|coef| {med[k]:.1f}): natural centroid vs transplant {base:+.2f}; all attention frozen {rec['all_attention']:+.2f}, all MLP frozen {rec['all_mlp']:+.2f}; single-component freezes with the largest move toward the transplant: " + ", ".join(f"{kk} {vv:+.2f}" for kk, vv in top) + " | full scan: " + " ".join(f"{kk}:{vv:+.2f}" for kk, vv in rec["scan"].items()))
record(f"e284b_components_{tag}", dict(model=tag, b=b, L=L, K=K, per_neuron=out), " | ".join(f"{nm} ({v['coef']:.1f}): natural {v['natural_centroid_vs_transplant']:+.2f}, all attn {v['all_attention']:+.2f}, all mlp {v['all_mlp']:+.2f}; top moves " + ", ".join(f"{kk} {vv:+.2f}" for kk, vv in sorted(v['scan'].items(), key=lambda kv: -(kv[1] - v['natural_centroid_vs_transplant']))[:3]) for nm, v in out.items()))
