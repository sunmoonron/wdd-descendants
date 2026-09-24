"""e401: does "training creates self-describability" replicate beyond one Pythia run at one depth?
Mode "init" (argument: model): the five architectures at a random initialisation (from their configurations) and, for
contrast, trained, at a quarter, half and three quarters of the depth, 4 sequences: FVU of k-sparse descriptions in the
own vocabulary against three rotations of it (at initialisation the next-token loss barely depends on the state, so only
FVU is meaningful there; the clean-to-mean gap is recorded). Pre-registered: at initialisation own and rotated FVU agree
within 0.01 at every k, depth and architecture; trained, own is below rotated.
Mode "olmo" (arguments: revisions): OLMo-1B checkpoints (a second family: different data, tokenizer, norm and MLP), middle
depth, 4 sequences: loss recovered and FVU, own against three rotations. Pre-registered: own above the rotations at every
trained checkpoint (the gap at initialisation is covered by mode init)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
mode = sys.argv[1]; KS = [4, 16, 64]
def own_vs_rot(lv, A, tag):
    cells = {"own": describe(lv, A, KS)}
    for sd in [7, 11, 13]: cells[f"rot{sd}"] = describe(lv, rotate(A, seed=sd), KS)
    ro = lambda k, key: [cells[f"rot{s}"][str(k)][key] for s in (7, 11, 13)]
    log(f"{tag} gap {lv.gap:.3f} | FVU own " + " ".join(f"{cells['own'][str(k)]['fvu']:.3f}" for k in KS) + " rot " + " ".join(f"{min(ro(k, 'fvu')):.3f}-{max(ro(k, 'fvu')):.3f}" for k in KS)
        + " | rec own " + " ".join(f"{cells['own'][str(k)]['rec']:.2f}" for k in KS) + " rot " + " ".join(f"{min(ro(k, 'rec')):.2f}-{max(ro(k, 'rec')):.2f}" for k in KS))
    return dict(gap=lv.gap, cells=cells)
if mode == "init":
    name = sys.argv[2]; c = Cache(name); ev = c.s["eval_ids"][:4].to(DEV); res = dict(model=name, k=KS, init={}, trained={})
    for state, ri in [("init", True), ("trained", False)]:
        model, tok, fam = load_model(name, random_init=ri); arch = Arch(model, fam)
        for L in [arch.NB // 4, arch.NB // 2, (3 * arch.NB) // 4]:
            A, _ = build_dictionary(arch, blocks=list(range(L + 1))); lv = Level(model, arch, ev, L)
            res[state][str(L)] = own_vs_rot(lv, A, f"{name} {state} L{L}"); del A, lv
        del model, arch
    d = lambda st: max(abs(v["cells"]["own"][str(k)]["fvu"] - sum(v["cells"][f"rot{s}"][str(k)]["fvu"] for s in (7, 11, 13)) / 3) for v in res[st].values() for k in KS)
    t = lambda st: min(sum(v["cells"][f"rot{s}"]["16"]["fvu"] for s in (7, 11, 13)) / 3 - v["cells"]["own"]["16"]["fvu"] for v in res[st].values())
    summ = f"{name}: init max |FVU own - rot| {d('init'):.3f} over depths and k; trained min (rot - own) FVU at k16 {t('trained'):.3f}"
    log(summ); record(f"e401_init_{name}", res, summ)
else:
    revs = sys.argv[2:]; c = Cache("olmo1b"); ev = c.s["eval_ids"][:4].to(DEV); res = dict(model="olmo1b", k=KS, revs=revs, cells={})
    for r in revs:
        model, tok, fam = load_model("olmo1b", revision=r); arch = Arch(model, fam); L = arch.NB // 2
        A, _ = build_dictionary(arch, blocks=list(range(L + 1))); lv = Level(model, arch, ev, L)
        res["cells"][r] = own_vs_rot(lv, A, f"olmo1b {r} L{L}"); del model, arch, A, lv
    g = lambda r, n: res["cells"][r]["cells"][n]["16"]["rec"]
    summ = "olmo1b k16 own/rot-mean: " + " | ".join(f"{r.split('-')[0]} {g(r, 'own'):.2f}/{sum(g(r, f'rot{s}') for s in (7, 11, 13)) / 3:.2f}" for r in revs)
    log(summ); record(f"e401_olmo_{'_'.join(r.split('-')[0] for r in revs)}", res, summ)
