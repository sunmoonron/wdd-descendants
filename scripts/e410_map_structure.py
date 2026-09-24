"""e410 (local, from e402 and e406): the structure of the intelligibility maps. With D_ij = M_ij - null_j (vocabulary of
stage i describing the states of stage j, minus the target's rotation null, k 16): accretion (later words on earlier
states against earlier words on later states, D_ji - D_ij for i < j), drift (D against the log-step distance, forward
and backward), and symmetry (correlation of D_ij with D_ji over pairs)."""
import json, math, os, numpy as np
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); out = {}
def load_pythia():
    d = json.load(open(os.path.join(root, "results", "e402_intelmap_pythia410.json"))); revs = d["revs"]
    M = {(i, j): d["cells"][f"{i}->{j}"]["16"]["rec"] for i in revs for j in revs}; nl = {j: np.mean([d["null"][f"{j}:{s}"]["16"]["rec"] for s in (7, 11, 13)]) for j in revs}
    return revs, M, nl, lambda r: int(r.replace("step", ""))
def load_olmo():
    cells, null = {}, {}; revs = None
    for f in ["e406_olmomap_1000_16000_256000.json", "e406_olmomap_4000_64000_1454000.json"]:
        d = json.load(open(os.path.join(root, "results", f))); revs = d["revs"]; cells.update(d["cells"]); null.update(d["null"])
    M = {(i, j): cells[f"{i}->{j}"]["16"]["rec"] for i in revs for j in revs}; nl = {j: np.mean([null[f"{j}:{s}"]["16"]["rec"] for s in (7, 11, 13)]) for j in revs}
    return revs, M, nl, lambda r: int(r.split("-")[0].replace("step", ""))
for name, fn in [("pythia410", load_pythia), ("olmo1b", load_olmo)]:
    revs, M, nl, st = fn(); D = {(i, j): M[(i, j)] - nl[j] for i in revs for j in revs}
    pairs = [(i, j) for a, i in enumerate(revs) for b, j in enumerate(revs) if a < b]
    acc = [D[(j, i)] - D[(i, j)] for i, j in pairs]; fw = [(math.log10(st(j) / st(i)), D[(i, j)]) for i, j in pairs]; bw = [(math.log10(st(j) / st(i)), D[(j, i)]) for i, j in pairs]
    sym = np.corrcoef([D[(i, j)] for i, j in pairs], [D[(j, i)] for i, j in pairs])[0, 1]
    slope = lambda xy: np.polyfit([x for x, _ in xy], [y for _, y in xy], 1)[0]
    out[name] = dict(mean_accretion=float(np.mean(acc)), frac_accreting=float(np.mean([a > 0 for a in acc])), symmetry_corr=float(sym),
                     forward_slope_per_decade=float(slope(fw)), backward_slope_per_decade=float(slope(bw)), n_pairs=len(pairs),
                     diag_minus_null=[float(D[(r, r)]) for r in revs])
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items() if k != "diag_minus_null"}, "diag-null", [round(x, 2) for x in out[name]["diag_minus_null"]])
json.dump(out, open(os.path.join(root, "results", "e410_mapstructure.json"), "w"), indent=1)
