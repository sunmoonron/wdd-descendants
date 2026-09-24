"""e406: do false friends exist in a second family? The e402 intelligibility map with nulls for OLMo-1B (different data,
tokenizer, norm and MLP): vocabularies of steps 1000, 4000, 16000, 64000, 256000 and 1454000 describing the middle-depth
states of the target checkpoints given as arguments (4 sequences, k 8 and 16), each target with three rotations of its
own vocabulary as the null; also the share of the target states' variance in their top-8 principal directions (Pythia's
false friends were mediated by such huge directions, e404). Pre-registered: later vocabularies read earlier states at
least as well as the earlier model's own words (accretion); false-friend cells (below every rotation) appear only if the
target states have a few huge directions (top-8 variance share above one half)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
revs = ["step1000-tokens2B", "step4000-tokens8B", "step16000-tokens33B", "step64000-tokens134B", "step256000-tokens536B", "step1454000-tokens3048B"]
tg = sys.argv[1:]; c = Cache("olmo1b"); ev = c.s["eval_ids"][:4].to(DEV); KS = [8, 16]; V = {}
for r in revs:
    m, tok, fam = load_model("olmo1b", revision=r); a = Arch(m, fam); L = a.NB // 2; V[r], _ = build_dictionary(a, blocks=list(range(L + 1))); del m, a
sh = lambda r: r.split("-")[0].replace("step", "")
res = dict(revs=revs, targets=tg, level=L, k=KS, cells={}, null={}, gap={}, top8={})
for j in tg:
    model, tok, fam = load_model("olmo1b", revision=j); arch = Arch(model, fam); lv = Level(model, arch, ev, L); res["gap"][j] = lv.gap
    evx = torch.linalg.eigvalsh(((lv.Xc.T @ lv.Xc) / lv.Xc.shape[0]).double()); res["top8"][j] = (evx[-8:].sum() / evx.sum()).item()
    for sd in [7, 11, 13]: res["null"][f"{j}:{sd}"] = describe(lv, rotate(V[j], seed=sd), KS)
    for i in revs: res["cells"][f"{i}->{j}"] = describe(lv, V[i], KS)
    nl = [res["null"][f"{j}:{sd}"]["16"]["rec"] for sd in (7, 11, 13)]
    log(f"states {sh(j)} (gap {lv.gap:.2f}, top-8 variance share {res['top8'][j]:.2f}) k16 null {min(nl):.2f}-{max(nl):.2f}: " + " ".join(f"{sh(i)}:{res['cells'][f'{i}->{j}']['16']['rec']:.2f}" for i in revs))
    del model, arch, lv
ff = []
for j in tg:
    for k in KS:
        lo = min(res["null"][f"{j}:{sd}"][str(k)]["rec"] for sd in (7, 11, 13))
        for i in revs:
            if res["cells"][f"{i}->{j}"][str(k)]["rec"] < lo: ff.append(f"k{k} {sh(i)}->{sh(j)} ({res['cells'][f'{i}->{j}'][str(k)]['rec']:.2f} < {lo:.2f})")
res["false_friend_cells"] = ff
summ = f"olmo1b targets {' '.join(sh(j) for j in tg)}: {len(ff)} false-friend cells: " + "; ".join(ff[:20]) + " | top-8 variance shares " + " ".join(f"{sh(j)} {res['top8'][j]:.2f}" for j in tg)
log(summ); record(f"e406_olmomap_{'_'.join(sh(j) for j in tg)}", res, summ)
