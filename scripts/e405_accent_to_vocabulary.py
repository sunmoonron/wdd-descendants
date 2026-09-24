"""e405: when does the own vocabulary's advantage turn from second-order to word-level? e399 found that at step 1000 most
of the advantage over a rotation is carried by the vocabulary's second moment (covA) and span (mix8: signed sums of 8
own words), while at steps 16000 and 143000 both sit at the rotation level and only the individual words carry it. Here
the same four vocabularies (own, rotation, covA, mix8) at steps 256, 512, 2000, 4000, 8000 and 63000 (arguments), 8
sequences, middle depth. Reported per checkpoint: the shares of the own-minus-rotation gap carried by covA and by mix8
at k 4 and 16. Pre-registered: the shares fall from above one half (step 1000 and before) to near zero by step 16000,
and the fall happens between steps 2000 and 8000, the window of the false friends (e402)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
c = Cache("pythia410"); ev = c.s["eval_ids"][:8].to(DEV); KS = [4, 8, 16, 32, 64]; res = dict(k=KS, revs=sys.argv[1:], cells={})
for rev in sys.argv[1:]:
    model, tok, fam = load_model("pythia410", revision=rev); arch = Arch(model, fam); L = arch.NB // 2
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); lv = Level(model, arch, ev, L); cells = dict(gap=lv.gap)
    for nm, V in [("own", A), ("rot7", rotate(A, seed=7)), ("covA", gauss_like(A.shape[0], (A.T @ A) / A.shape[0], seed=1)),
                  ("mix8", mixtures(A, [torch.nonzero(typ == t)[:, 0] for t in (T_TOK, T_MLP, T_ATT, T_BIAS) if (typ == t).any()], m=8, seed=4))]:
        cells[nm] = describe(lv, V, KS)
    sh = {f"{nm}_k{k}": (cells[nm][str(k)]["rec"] - cells["rot7"][str(k)]["rec"]) / max(cells["own"][str(k)]["rec"] - cells["rot7"][str(k)]["rec"], 1e-9) for nm in ("covA", "mix8") for k in (4, 16)}
    cells["shares"] = sh; res["cells"][rev] = cells
    log(f"{rev} (gap {lv.gap:.2f}) k4/k16: own {cells['own']['4']['rec']:.2f}/{cells['own']['16']['rec']:.2f} rot {cells['rot7']['4']['rec']:.2f}/{cells['rot7']['16']['rec']:.2f} "
        f"covA {cells['covA']['4']['rec']:.2f}/{cells['covA']['16']['rec']:.2f} mix8 {cells['mix8']['4']['rec']:.2f}/{cells['mix8']['16']['rec']:.2f} | shares covA {sh['covA_k4']:.2f}/{sh['covA_k16']:.2f} mix8 {sh['mix8_k4']:.2f}/{sh['mix8_k16']:.2f}")
    del model, arch, A, lv
summ = " | ".join(f"{r.replace('step', '')}: covA {res['cells'][r]['shares']['covA_k16']:.2f} mix8 {res['cells'][r]['shares']['mix8_k16']:.2f}" for r in res["revs"])
log("k16 second-order shares: " + summ); record(f"e405_accent_{'_'.join(r.replace('step', '') for r in res['revs'])}", res, "k16 second-order shares of the own advantage: " + summ)
