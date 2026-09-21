"""e284: what operation carries the context-bound reaction? For the largest-coefficient candidate (the massive neuron
in SmolLM2) and a typical candidate, the natural footprint at L is recomputed with downstream components frozen at
the neuron's own tokens: every attention output in blocks b+1..L held at its unablated value, every MLP output held,
both held (sanity: the footprint must then be the write itself), and a block-by-block scan holding one block's
attention and MLP. Each footprint is compared with the context-free reference (the transplant image centroid) and
with the linear transport sign*T w. The freeze that flips the massive neuron's footprint from anti-parallel to
aligned names the carrier of the reaction."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); run = make_runner(model, arch, c, ids_seq, [L], NT)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(b, tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); F = S0[L] - S1[L]; Cd = unit(centroids(F[idx], lab_i, K)); smed = torch.stack([tc[idx][lab_i == k].median() for k in range(K)]); med = smed.abs(); sgn = torch.sign(smed); order = med.argsort(); picks = {"largest": int(med.argmax()), "typical": int(order[len(order) // 2])}
pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); torch.manual_seed(0); Vr = unit(torch.randn(1024, D, device=DEV)); X = []; Y = []
for p in range(2):
    a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=b + 1); X.append(Vr[a]); Y.append((S2[L] - S0[L])[pool] / s_inj)
X, Y = torch.cat(X), torch.cat(Y); G = X.T @ X; T = torch.linalg.solve(G + 1e-2 * G.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y)
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0]; assign = torch.randint(0, 2, (len(foreign),), device=DEV); ks = list(picks.values()); vec = unit(R[keep[ks]]); amp = (sgn[ks] * med[ks])[assign]; inj = torch.zeros(NT, D, device=DEV); inj[foreign] = amp[:, None] * vec[assign]; S2 = run(inject=inj, inject_block=b + 1); timg = {nm: unit((S2[L] - S0[L])[foreign[assign == j]].mean(0, keepdim=True))[0] for j, nm in enumerate(picks)}
mods = {("attn", lv): arch.attn_lin(lv) for lv in range(b + 1, L + 1)}; mods.update({("mlp", lv): arch.mlp_lin(lv) for lv in range(b + 1, L + 1)}); cache = {}
hs = [m.register_forward_hook((lambda key: lambda mod, i, o: cache.__setitem__(key, o.detach().clone()))(key)) for key, m in mods.items()]; model(ids_seq); [h.remove() for h in hs]
def frozen_footprint(rows, keys):
    def mk(key):
        def hook(mod, i, o):
            o2 = o.clone(); flat = o2.reshape(-1, o2.shape[-1]); flat[rows] = cache[key].reshape(-1, o2.shape[-1])[rows]; return o2
        return hook
    hs = [mods[key].register_forward_hook(mk(key)) for key in keys]; S1f = run(b, tn); [h.remove() for h in hs]; return (S0[L] - S1f[L])[rows]
out = {}
for nm, k in picks.items():
    rows = idx[lab_i == k]; ref = timg[nm]; tw = unit((sgn[k] * (R[keep[k]] @ T))[None])[0]; w = unit(R[keep[k]][None])[0] * sgn[k]; nat = F[rows]; cs = lambda Fp, v: ((unit(Fp) * v).sum(1)).median().item(); rec = dict(neuron=int(keep[k]), coef=med[k].item(), natural_vs_transplant=cs(nat, ref), natural_vs_linear=cs(nat, tw), natural_vs_write=cs(nat, w), transplant_vs_linear=(ref @ tw).item())
    variants = {"freeze_attention": [key for key in mods if key[0] == "attn"], "freeze_mlp": [key for key in mods if key[0] == "mlp"], "freeze_both": list(mods.keys())}
    for vn, keys in variants.items():
        Fp = frozen_footprint(rows, keys); rec[vn] = dict(vs_transplant=cs(Fp, ref), vs_linear=cs(Fp, tw), vs_write=cs(Fp, w), norm_per_coef=(Fp.norm(dim=1).median() / med[k]).item())
    scan = {}
    for lv in range(b + 1, L + 1):
        Fp = frozen_footprint(rows, [("attn", lv), ("mlp", lv)]); scan[lv] = dict(vs_transplant=cs(Fp, ref), vs_natural=cs(Fp, unit(nat.mean(0, keepdim=True))[0]))
    rec["scan"] = {str(lv): v for lv, v in scan.items()}; out[nm] = rec
    log(f"{tag} {nm} neuron {int(keep[k])} (|coef| {med[k]:.1f}): natural footprint vs transplant {rec['natural_vs_transplant']:+.2f}, vs linear {rec['natural_vs_linear']:+.2f}, vs write {rec['natural_vs_write']:+.2f} (transplant vs linear {rec['transplant_vs_linear']:+.2f}) | attention frozen: vs transplant {rec['freeze_attention']['vs_transplant']:+.2f}, vs linear {rec['freeze_attention']['vs_linear']:+.2f} | MLP frozen: vs transplant {rec['freeze_mlp']['vs_transplant']:+.2f}, vs linear {rec['freeze_mlp']['vs_linear']:+.2f} | both frozen: vs write {rec['freeze_both']['vs_write']:+.2f} (sanity) | block scan (one block frozen), vs transplant: " + " ".join(f"{lv}:{v['vs_transplant']:+.2f}" for lv, v in scan.items()))
record(f"e284_carrier_{tag}", dict(model=tag, b=b, L=L, K=K, per_neuron=out), " | ".join(f"{nm} ({v['coef']:.1f}): natural vs transplant {v['natural_vs_transplant']:+.2f}; attention frozen {v['freeze_attention']['vs_transplant']:+.2f}; MLP frozen {v['freeze_mlp']['vs_transplant']:+.2f}; both frozen vs write {v['freeze_both']['vs_write']:+.2f}; scan " + " ".join(f"{lv}:{s['vs_transplant']:+.2f}" for lv, s in v['scan'].items()) for nm, v in out.items()))
