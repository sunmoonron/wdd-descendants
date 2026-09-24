"""e409: is Pythia-410m's false-friend window a property of the family or of one size, and does the accent turn into a
vocabulary at every size? Pythia-70m and Pythia-160m (argument; same data, data order and tokenizer as 410m) at steps
1000, 4000, 16000 and 143000, middle depth, 4 sequences: the 4x4 intelligibility map with three rotations of each
target's own vocabulary as the null (k 4, 8, 16); at each checkpoint the covA and mix8 shares of the own advantage over
rotation (k 4 and 16) and the target states' top-8 variance share. Pre-registered: if the window is a family
property, the step-4000 words are below every rotation on the final states at both sizes; the covA and mix8 shares
fall from step 1000 to the end at both sizes."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
MODELS["pythia70"] = ("EleutherAI/pythia-70m", "neox"); MODELS["pythia160"] = ("EleutherAI/pythia-160m", "neox")
name = sys.argv[1]; revs = ["step1000", "step4000", "step16000", "step143000"]
c = Cache("pythia410"); ev = c.s["eval_ids"][:4].to(DEV); KS = [4, 8, 16]; V = {}
for r in revs:
    m, tok, fam = load_model(name, revision=r); a = Arch(m, fam); L = a.NB // 2; V[r], lab = build_dictionary(a, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); del m, a
groups = [torch.nonzero(typ == t)[:, 0] for t in (T_TOK, T_MLP, T_ATT, T_BIAS) if (typ == t).any()]
res = dict(model=name, revs=revs, level=L, k=KS, cells={}, null={}, gap={}, top8={}, own_ctl={})
for j in revs:
    model, tok, fam = load_model(name, revision=j); arch = Arch(model, fam); lv = Level(model, arch, ev, L); res["gap"][j] = lv.gap
    evx = torch.linalg.eigvalsh(((lv.Xc.T @ lv.Xc) / lv.Xc.shape[0]).double()); res["top8"][j] = (evx[-8:].sum() / evx.sum()).item()
    for sd in [7, 11, 13]: res["null"][f"{j}:{sd}"] = describe(lv, rotate(V[j], seed=sd), KS)
    for i in revs: res["cells"][f"{i}->{j}"] = describe(lv, V[i], KS)
    res["own_ctl"][j] = dict(covA=describe(lv, gauss_like(V[j].shape[0], (V[j].T @ V[j]) / V[j].shape[0], seed=1), KS), mix8=describe(lv, mixtures(V[j], groups, m=8, seed=4), KS))
    own = lambda k: res["cells"][f"{j}->{j}"][str(k)]["rec"]; rot = lambda k: sum(res["null"][f"{j}:{sd}"][str(k)]["rec"] for sd in (7, 11, 13)) / 3
    res["own_ctl"][j]["shares"] = {f"{ctl}_k{k}": (res["own_ctl"][j][ctl][str(k)]["rec"] - rot(k)) / max(own(k) - rot(k), 1e-9) for ctl in ("covA", "mix8") for k in (4, 16)}
    nl = [res["null"][f"{j}:{sd}"]["16"]["rec"] for sd in (7, 11, 13)]
    log(f"{name} states {j} (gap {lv.gap:.2f}, top-8 var {res['top8'][j]:.2f}) k16 null {min(nl):.2f}-{max(nl):.2f}: " + " ".join(f"{i.replace('step', '')}:{res['cells'][f'{i}->{j}']['16']['rec']:.2f}" for i in revs)
        + " | shares k16 covA {covA_k16:.2f} mix8 {mix8_k16:.2f}".format(**res["own_ctl"][j]["shares"]))
    del model, arch, lv
ff = []
for j in revs:
    for k in KS:
        lo = min(res["null"][f"{j}:{sd}"][str(k)]["rec"] for sd in (7, 11, 13))
        for i in revs:
            if res["cells"][f"{i}->{j}"][str(k)]["rec"] < lo: ff.append(f"k{k} {i.replace('step', '')}->{j.replace('step', '')} ({res['cells'][f'{i}->{j}'][str(k)]['rec']:.2f} < {lo:.2f})")
res["false_friend_cells"] = ff
summ = (f"{name}: {len(ff)} false-friend cells: " + "; ".join(ff[:12]) + " | k16 shares covA/mix8 " + " ".join(f"{j.replace('step', '')} {res['own_ctl'][j]['shares']['covA_k16']:.2f}/{res['own_ctl'][j]['shares']['mix8_k16']:.2f}" for j in revs)
        + " | top-8 var " + " ".join(f"{j.replace('step', '')} {res['top8'][j]:.2f}" for j in revs))
log(summ); record(f"e409_ladder_{name}", res, summ)
