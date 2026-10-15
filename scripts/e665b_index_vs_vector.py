"""e665b (session 147; Gemini 4's co-rotation hypothesis for e665): across heavy continued pretraining, does an attribute
edit transfer by neuron index or by residual-space direction? e665 found the base model's rows, negated at the same
(block, row) indices, edit Qwen2.5-Coder-0.5B and its instruct model as well as their own chains, although their write
rows moved to median cosine 0.63 and 0.44. Gemini's reading: a neuron's slot co-rotates with the readers that consume
it, so index transfer survives where a fixed residual-space vector from the base would point off-target. The
alternative: the attribute's residual direction barely moved even though the rows did. Here, with no OMP (e664b's
direct-contribution ranking), per attribute: index transfer (the target's own columns negated at the base model's top
k direct-contribution indices, k = 8, 16, 32) against the target's own top k; vector transfer as directional ablation
(the base model's mean pole-difference direction removed from every block output 0..L at every position, Arditi et al.
2024 style) against the target's own direction and a random one; and as additive steering toward the unmarked pole at
the most consistent level, base direction against own at equal norm; plus the cosine between base and target mean
differences per level. Run on Qwen2.5-0.5B first (it saves the source), then on -0.5B-Instruct, Coder-0.5B and
Coder-0.5B-Instruct. Reuses e661's setup. Arguments: model [--smoke]."""
import sys
src = open("e661_markedness.py").read().replace('"qwen15": "Qwen/Qwen2.5-1.5B", ', '"qwen15": "Qwen/Qwen2.5-1.5B", "coder05": "Qwen/Qwen2.5-Coder-0.5B", "coder05i": "Qwen/Qwen2.5-Coder-0.5B-Instruct", ')
exec(src[src.index("import sys\n"): src.index("# ==== SETUP END ====")], globals())
ATT = ("gender", "number", "polarity", "generation"); KS = [8, 32] if SMOKE else [8, 16, 32]; SRC = f"results/e665b_source{'_smoke' if SMOKE else ''}.pt"
WN = [layers[b].mlp.down_proj.weight.detach().float().cpu() for b in range(L + 1)]
def own_stats(a):
    its = [it for it in ITEMS[a] if it["disc"]]; A_ = torch.stack([it["A"] for it in its]); s_ = torch.tensor([it["side"] for it in its]); da = A_[s_ == 1].mean(0) - A_[s_ == 0].mean(0)
    md = torch.stack([DIFF[a][:, b].mean(0) for b in range(L + 1)]); dh = md / md.norm(dim=1, keepdim=True).clamp_min(1e-12)
    loc = torch.stack([da[b] * ((WN[b].T @ dh[b]) / WN[b].norm(dim=0).clamp_min(1e-12)) for b in range(L + 1)])
    dc = [divmod(int(i_), DFF) for i_ in loc.reshape(-1).argsort(descending=True)[:max(KS)].tolist()]
    consist = [float(torch.nn.functional.cosine_similarity(DIFF[a][:, b], md[b][None], dim=1).mean()) for b in range(L + 1)]
    return dict(dc=dc, md=md, level=int(np.argmax(consist[1:])) + 1, consist=consist)
OWN = {a: own_stats(a) for a in ATT}
if name == "qwen05": torch.save({a: dict(dc=OWN[a]["dc"], md=OWN[a]["md"], level=OWN[a]["level"]) for a in ATT}, SRC)
S = torch.load(SRC)
class Ablate:
    def __init__(self, dirs): self.dirs = dirs; self.h = []
    def __enter__(self):
        for b, d in self.dirs.items():
            dd = (d / d.norm().clamp_min(1e-12)).to(DEV)
            def hk(m, a_, o, dd=dd):
                x = o[0] if isinstance(o, tuple) else o; x = x - (x @ dd)[..., None] * dd
                return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
            self.h.append(layers[b].register_forward_hook(hk))
        return self
    def __exit__(self, *a_): [h.remove() for h in self.h]
class Steer:
    def __init__(self, b, v): self.b, self.v = b, v.to(DEV)
    def __enter__(self):
        def hk(m, a_, o): return (o[0] + self.v,) + tuple(o[1:]) if isinstance(o, tuple) else o + self.v
        self.h = layers[self.b].register_forward_hook(hk); return self
    def __exit__(self, *a_): self.h.remove()
base_cols = None
if name != "qwen05":
    bm_ = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32); base_cols = [l.mlp.down_proj.weight.detach().float() for l in bm_.model.layers]; del bm_
g_ = torch.Generator().manual_seed(665)
RES = dict(model=name, ks=KS, attrs={})
for a in ATT:
    o, s = OWN[a], S[a]; lev = s["level"]
    cosl = [float(torch.nn.functional.cosine_similarity(o["md"][b], s["md"][b], dim=0)) for b in range(L + 1)]
    rowcos = [float(torch.nn.functional.cosine_similarity(base_cols[b][:, r], WN[b][:, r], dim=0)) for b, r in s["dc"][:max(KS)]] if base_cols is not None else [1.0] * max(KS)
    out = dict(own_level=o["level"], src_level=lev, md_cos_by_level=cosl, md_cos_at_level=cosl[lev], dc_row_cos_mean=float(np.mean(rowcos)), dc_overlap_top32=len(set(map(tuple, s["dc"])) & set(o["dc"])) / max(KS), res={"none": by_pole(a)})
    for k in KS:
        with Negate([tuple(x) for x in s["dc"][:k]]): out["res"][f"index_base_{k}"] = by_pole(a) | {"kl": kl_off()}
        with Negate(o["dc"][:k]): out["res"][f"index_own_{k}"] = by_pole(a) | {"kl": kl_off()}
    with Ablate({b: s["md"][b] for b in range(L + 1)}): out["res"]["ablate_base"] = by_pole(a) | {"kl": kl_off()}
    with Ablate({b: o["md"][b] for b in range(L + 1)}): out["res"]["ablate_own"] = by_pole(a) | {"kl": kl_off()}
    with Ablate({b: torch.randn(D, generator=g_) for b in range(L + 1)}): out["res"]["ablate_random"] = by_pole(a) | {"kl": kl_off()}
    nrm = float(o["md"][lev].norm())
    for al in ((1.0,) if SMOKE else (0.5, 1.0, 2.0)):
        vb = -s["md"][lev] / s["md"][lev].norm() * nrm * al; vo = -o["md"][lev] / o["md"][lev].norm() * nrm * al
        with Steer(lev, vb): out["res"][f"steer_base_{al}"] = by_pole(a) | {"kl": kl_off()}
        with Steer(lev, vo): out["res"][f"steer_own_{al}"] = by_pole(a) | {"kl": kl_off()}
    RES["attrs"][a] = out; r_ = out["res"]
    log(f"{a}: mean-difference cosine base vs own at level {lev} {out['md_cos_at_level']:.3f} (levels {min(cosl):.2f} to {max(cosl):.2f}); base direct-contribution rows' cosine {out['dc_row_cos_mean']:.3f}, top-32 overlap {out['dc_overlap_top32']:.2f}; flips from pole1 (pole0): "
        + "; ".join(f"{k} {v['flip_from_1']:.2f} ({v['flip_from_0']:.2f})" for k, v in r_.items()))
f1 = lambda a, k: RES["attrs"][a]["res"][k]["flip_from_1"]; kk = str(max(KS))
GN = ("gender", "number")
RES["verdicts"] = dict(V1=all(f1(a, "ablate_base") < 0.5 * f1(a, "ablate_own") and f1(a, f"index_base_{kk}") >= 0.75 * f1(a, f"index_own_{kk}") for a in GN),
                       V2=all(RES["attrs"][a]["md_cos_at_level"] >= 0.8 for a in GN), V3=all(f1(a, "ablate_base") >= f1(a, "ablate_own") - 0.1 for a in GN),
                       V4=all(f1(a, f"index_base_{kk}") >= f1(a, f"index_own_{kk}") - 0.1 for a in GN))
summ = (f"[{name}] index vs vector transfer from Qwen2.5-0.5B: " + "; ".join(f"{a} md cos {RES['attrs'][a]['md_cos_at_level']:.2f}, flips from the marked pole index base/own {f1(a, 'index_base_' + kk):.2f}/{f1(a, 'index_own_' + kk):.2f}, ablation base/own/random {f1(a, 'ablate_base'):.2f}/{f1(a, 'ablate_own'):.2f}/{f1(a, 'ablate_random'):.2f}" for a in GN)
        + f"; verdicts {RES['verdicts']} ({time.time() - t0:.0f}s)")
RES["summary"] = summ; RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"results/e665b_index_vs_vector_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o_: o_.item() if hasattr(o_, "item") else str(o_)); print(summ, flush=True); print("done", flush=True)
