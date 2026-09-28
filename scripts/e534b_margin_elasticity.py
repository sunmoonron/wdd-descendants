"""e534b: retention as a function of the margin (an analysis over e534's per-word records; no run). e534 found that
within pooled quartiles of S native and random-dictionary words survive a random 20-degree rotation alike. Here the
curve itself: the survival of a word under a random rotation of 20 and 30 degrees as a function of its margin over the
floor S, in pooled quintiles of S for native and random words apart; the slope of survival on S (least squares, the
elasticity of retention with respect to margin) within the range of S both dictionaries populate, 0.9-1.25; and the
mean difference of native and random survival within the common bins.
Pre-registered (honest guesses): the slopes of native and random words agree within a factor of 1.5 (0.5); within
the common bins native survival is within 0.10 of random (0.5).
Arguments: none (reads results/e534_basin_anatomy_pythia410.json)."""
import json, os, time
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"; r = json.load(open(root + "results/e534_basin_anatomy_pythia410.json")); LO, HI = 0.9, 1.25
def slope(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n; sxx = sum((a - mx) ** 2 for a in x); return (sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx) if sxx > 0 and n >= 10 else None
res = dict(common_range=[LO, HI], by_checkpoint={})
for n, o in r["by_checkpoint"].items():
    rec = dict(slope_common_range={}, survival_common_range={}, n_common={}, bins=[], matched_gap={})
    for d in ("native", "random"):
        pw = o[d]["per_word"]; idx = [j for j, s in enumerate(pw["S"]) if LO <= s <= HI]; rec["n_common"][d] = len(idx); rec["slope_common_range"][d] = {}; rec["survival_common_range"][d] = {}
        for th in ("20", "30"):
            x = [pw["S"][j] for j in idx]; y = [pw[f"random_{th}_static"][j] for j in idx]; rec["slope_common_range"][d][th] = slope(x, y); rec["survival_common_range"][d][th] = (sum(y) / len(y)) if y else None
    allS = sorted(o["native"]["per_word"]["S"] + o["random"]["per_word"]["S"]); edges = [allS[int(q * (len(allS) - 1))] for q in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]; row = dict(S_range=[lo, hi])
        for d in ("native", "random"):
            pw = o[d]["per_word"]; idx = [j for j, s in enumerate(pw["S"]) if lo <= s <= hi]; row[d] = dict(n=len(idx), survival_20=(sum(pw["random_20_static"][j] for j in idx) / len(idx)) if idx else None, survival_30=(sum(pw["random_30_static"][j] for j in idx) / len(idx)) if idx else None)
        rec["bins"].append(row)
    for th in ("20", "30"):
        gaps = [b_["native"][f"survival_{th}"] - b_["random"][f"survival_{th}"] for b_ in rec["bins"] if b_["native"]["n"] >= 20 and b_["random"]["n"] >= 20]; rec["matched_gap"][th] = (sum(gaps) / len(gaps)) if gaps else None
    res["by_checkpoint"][n] = rec; fm = lambda x: "n/a" if x is None else f"{x:.2f}"
    print(f"step {n}: slope of survival on S within S {LO}-{HI} (n native {rec['n_common']['native']}, random {rec['n_common']['random']}), 20/30 degrees: native {fm(rec['slope_common_range']['native']['20'])}/{fm(rec['slope_common_range']['native']['30'])}, random {fm(rec['slope_common_range']['random']['20'])}/{fm(rec['slope_common_range']['random']['30'])}; survival in the common range native {fm(rec['survival_common_range']['native']['20'])}/{fm(rec['survival_common_range']['native']['30'])}, random {fm(rec['survival_common_range']['random']['20'])}/{fm(rec['survival_common_range']['random']['30'])}; native minus random in the populated bins {fm(rec['matched_gap']['20'])}/{fm(rec['matched_gap']['30'])}; pooled quintiles (S range: native n, survival 20/30 | random n, survival 20/30): " + " | ".join(f"{b_['S_range'][0]:.2f}-{b_['S_range'][1]:.2f}: {b_['native']['n']}, {fm(b_['native']['survival_20'])}/{fm(b_['native']['survival_30'])} | {b_['random']['n']}, {fm(b_['random']['survival_20'])}/{fm(b_['random']['survival_30'])}" for b_ in rec["bins"]))
sl = {d: {th: [res["by_checkpoint"][n]["slope_common_range"][d][th] for n in res["by_checkpoint"]] for th in ("20", "30")} for d in ("native", "random")}
ratio = [a / b for a, b in zip(sl["native"]["20"], sl["random"]["20"]) if a is not None and b]; gaps = [res["by_checkpoint"][n]["matched_gap"]["20"] for n in res["by_checkpoint"] if res["by_checkpoint"][n]["matched_gap"]["20"] is not None]
res["checks"] = dict(slopes_within_factor_1_5=bool(ratio) and all(1 / 1.5 <= x <= 1.5 for x in ratio), matched_within_0_10=bool(gaps) and all(abs(g) <= 0.10 for g in gaps)); res["_exp"] = "e534b_margin_elasticity_pythia410"; res["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(res, open(root + "results/e534b_margin_elasticity_pythia410.json", "w"), indent=1)
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = "pythia410 block 12: slope of survival on S within S 0.9-1.25 at 20/30 degrees, native " + "/".join(fm(x) for x in sl["native"]["20"]) + " and " + "/".join(fm(x) for x in sl["native"]["30"]) + ", random " + "/".join(fm(x) for x in sl["random"]["20"]) + " and " + "/".join(fm(x) for x in sl["random"]["30"]) + " (checkpoints 4000/8000/12000); native minus random survival in the populated bins at 20 degrees " + "/".join(fm(res["by_checkpoint"][n]["matched_gap"]["20"]) for n in res["by_checkpoint"]) + f" | checks {json.dumps(res['checks'])}"
print(summ); L = open(root + "results/FINDINGS_box5.log").read().splitlines(); L = [l for l in L if "e534b_margin_elasticity" not in l]; open(root + "results/FINDINGS_box5.log", "w").write("\n".join(L) + f"\n{res['_time']} e534b_margin_elasticity_pythia410: {summ}\n")
