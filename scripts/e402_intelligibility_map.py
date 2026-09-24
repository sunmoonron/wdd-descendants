"""e402: the intelligibility map with its null. e396's 6x6 matrix had no per-target null, so "worse than random" rested on
one rotation of one vocabulary. Here: ten Pythia-410m checkpoints (steps 256, 512, 1000, 2000, 4000, 8000, 16000, 33000,
63000, 143000), every vocabulary describing every checkpoint's middle-depth states (4 sequences, k 8 and 16, loss
recovered and FVU), and for every target three rotations of its own vocabulary as the null (mean and spread).
Reported: the matrix, the matrix minus the target's null mean, and the cells below the lowest rotation (false-friend
cells). Questions: when are false friends born and whose false friends are they (only step 4000 for the final model, or
a window of stages, and for which later targets)? Does the accretion asymmetry hold on the finer grid? Pre-registered:
false-friend cells exist only for source stages between the induction transition (step 1000) and step 16000 describing
later targets."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
revs = ["step256", "step512", "step1000", "step2000", "step4000", "step8000", "step16000", "step33000", "step63000", "step143000"]
c = Cache("pythia410"); ev = c.s["eval_ids"][:4].to(DEV); KS = [8, 16]; V = {}
for r in revs:
    m, tok, fam = load_model("pythia410", revision=r); a = Arch(m, fam); L = a.NB // 2; V[r], _ = build_dictionary(a, blocks=list(range(L + 1))); del m, a
res = dict(revs=revs, level=L, k=KS, cells={}, null={}, gap={})
for j in revs:
    model, tok, fam = load_model("pythia410", revision=j); arch = Arch(model, fam); lv = Level(model, arch, ev, L); res["gap"][j] = lv.gap
    for sd in [7, 11, 13]: res["null"][f"{j}:{sd}"] = describe(lv, rotate(V[j], seed=sd), KS)
    for i in revs: res["cells"][f"{i}->{j}"] = describe(lv, V[i], KS)
    nl = [res["null"][f"{j}:{sd}"]["16"]["rec"] for sd in (7, 11, 13)]
    log(f"states {j} (gap {lv.gap:.2f}) k16 null {min(nl):.2f}-{max(nl):.2f}: " + " ".join(f"{i.replace('step', '')}:{res['cells'][f'{i}->{j}']['16']['rec']:.2f}" for i in revs))
    del model, arch, lv
ff = []
for j in revs:
    for k in KS:
        lo = min(res["null"][f"{j}:{sd}"][str(k)]["rec"] for sd in (7, 11, 13))
        for i in revs:
            if res["cells"][f"{i}->{j}"][str(k)]["rec"] < lo: ff.append(f"k{k} {i.replace('step', '')}->{j.replace('step', '')} ({res['cells'][f'{i}->{j}'][str(k)]['rec']:.2f} < {lo:.2f})")
res["false_friend_cells"] = ff
summ = f"{len(ff)} false-friend cells: " + "; ".join(ff[:24])
log(summ); record("e402_intelmap_pythia410", res, summ)
