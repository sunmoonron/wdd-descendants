"""e729: the dose-scaled landmine predictor scored offline against e728's measured rows, on Qwen3.5-4B. No model runs: reads e728's results.
P = 0.5 (0.8 MED_L)^2 R_dir is the second-order KL of the zoo's injection (0.8 times the layer's median residual norm along the row's
direction), with R_dir the sampled-Fisher read energy per unit norm from e728. P is scored against every row e728 measured at the zoo's
0.8 dose: the 24 stage-1 rows (top-8, bottom-8, random-8 by R_dir), the 40 stage-3 and 3b rows, and the 8 stage-2 specs against their
same-layer, same-sign random controls. The rows file (results/e728_qwen35_rows.npz, 10.6 MB, kept outside the public repository) is
optional: without it the global percentiles of P (D6) and the per-layer medians are skipped. Pre-registered in results/e729_prereg.json.
Usage: python e729_dose_scaled.py [--json results/e728_qwen35.json] [--npz results/e728_qwen35_rows.npz] [--out results/e729_dose_scaled.json]"""
import argparse, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
def first(*cands):
    for c in cands:
        if c and os.path.exists(c): return c
    return cands[0]
ap = argparse.ArgumentParser()
ap.add_argument("--json", default=first(UP + "/results/e728_qwen35.json", UP + "/results/e728/e728_qwen35.json"))
ap.add_argument("--npz", default=first(UP + "/results/e728_qwen35_rows.npz", UP + "/results/e728/e728_qwen35_rows.npz"))
ap.add_argument("--out", default=None)
A = ap.parse_args()
if A.out is None: A.out = (UP + "/results/e729/e729_dose_scaled.json") if "/e728/" in A.json else (UP + "/results/e729_dose_scaled.json")
t0 = time.time()
d = json.load(open(A.json)); MED = np.array(d["median_residual"], float); DOSE = 0.8

def rankdata(a):
    a = np.asarray(a, float); n = len(a); order = np.argsort(a, kind="mergesort"); sa = a[order]; r = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return None
    return round(float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]), 3)
def P_of(row): return float(0.5 * (DOSE * MED[row["layer"]]) ** 2 * row["R_dir"])
def ent(group, row, m):
    return {"group": group, "layer": row["layer"], "neuron": row["neuron"], "sign": row["sign"], "R_dir": row["R_dir"], "R_nat": row["R_nat"], "R_abl": row["R_abl"],
            "q999": row["q999"], "dose2": float((DOSE * MED[row["layer"]]) ** 2), "P": P_of(row), "nll_cost": m["nll_cost"], "kl_answers": m["kl_answers"]}
PRED = ("P", "R_dir", "R_nat", "R_abl", "layer", "dose2")
def table(rows, target):
    return {p: spearman([r[p] for r in rows], [r[target] for r in rows]) for p in PRED}
med = lambda v: float(np.median(v))

# stage 1: 24 rows, held-out NLL cost and KL on the answers at the zoo's 0.8
S1 = [ent(g, e["row"], e) for g in ("top_R_dir", "bottom_R_dir", "random") for e in d["stage1"][g]]
assert len(S1) == 24
groups = {g: [r for r in S1 if r["group"] == g] for g in ("top_R_dir", "bottom_R_dir", "random")}
s1 = {"n": 24, "spearman_nll": table(S1, "nll_cost"), "spearman_kl": table(S1, "kl_answers"),
      "median_P": {g: med([r["P"] for r in v]) for g, v in groups.items()}, "median_nll": {g: med([r["nll_cost"] for r in v]) for g, v in groups.items()},
      "median_kl": {g: med([r["kl_answers"] for r in v]) for g, v in groups.items()}, "layers": {g: sorted(r["layer"] for r in v) for g, v in groups.items()}}
ratio = [r["P"] / max(r["kl_answers"], 1e-9) for r in S1]
s1["calibration"] = {"n_within_3x_of_kl": int(sum(1 / 3 <= x <= 3 for x in ratio)), "median_P_over_kl": med(ratio),
                     "median_P_over_kl_by_group": {g: med([r["P"] / max(r["kl_answers"], 1e-9) for r in v]) for g, v in groups.items()},
                     "n_within_3x_of_nll": int(sum(1 / 3 <= r["P"] / max(r["nll_cost"], 1e-9) <= 3 for r in S1))}

# stages 3 and 3b: 40 rows at the zoo's 0.8, paired by index (same layer)
S3 = []; pairs3 = []
for st, tag in (("stage3", "s3"), ("stage3b", "s3b")):
    W, Rn = d[st]["wdd"], d[st]["random"]; assert len(W) == len(Rn) == 10
    for w, r in zip(W, Rn):
        assert w["row"]["layer"] == r["row"]["layer"], (st, w["row"]["layer"], r["row"]["layer"])
        ew, er = ent(tag + "_wdd", w["row"], w["zoo0.8"]), ent(tag + "_random", r["row"], r["zoo0.8"]); S3 += [ew, er]
        pairs3.append({"stage": tag, "layer": ew["layer"], "P_wdd": ew["P"], "P_random": er["P"], "kl_wdd": ew["kl_answers"], "kl_random": er["kl_answers"],
                       "P_order_matches_kl_order": bool((ew["P"] > er["P"]) == (ew["kl_answers"] > er["kl_answers"]))})
assert len(S3) == 40
s3 = {"n": 40, "spearman_kl": table(S3, "kl_answers"), "spearman_nll": table(S3, "nll_cost"),
      "spearman_kl_stage3_only": table([r for r in S3 if r["group"].startswith("s3_")], "kl_answers"),
      "spearman_kl_stage3b_only": table([r for r in S3 if r["group"].startswith("s3b_")], "kl_answers"),
      "pairs_P_order_matches_kl_order": int(sum(p["P_order_matches_kl_order"] for p in pairs3)), "pairs_n": len(pairs3),
      "pairs_wdd_P_higher": int(sum(p["P_wdd"] > p["P_random"] for p in pairs3)), "pairs_wdd_kl_higher": int(sum(p["kl_wdd"] > p["kl_random"] for p in pairs3)),
      "median_P": {g: med([r["P"] for r in S3 if r["group"] == g]) for g in ("s3_wdd", "s3_random", "s3b_wdd", "s3b_random")},
      "median_kl": {g: med([r["kl_answers"] for r in S3 if r["group"] == g]) for g in ("s3_wdd", "s3_random", "s3b_wdd", "s3b_random")}, "pairs": pairs3}

# stage 2: the 8 specs against their same-layer, same-sign random controls (P summed over a spec's rows)
specs = sorted({k.split(" | ")[0] for k in d["stage2"]}); s2rows = []; S2single = []
for sp in specs:
    z, r = d["stage2"][sp + " | zoo"], d["stage2"][sp + " | random"]
    Pz, Pr = sum(P_of(x) for x in z["rows"]), sum(P_of(x) for x in r["rows"])
    kz, kr = z["doses"]["zoo0.8"]["kl_answers"], r["doses"]["zoo0.8"]["kl_answers"]
    s2rows.append({"spec": sp, "n_rows": len(z["rows"]), "layers": [x["layer"] for x in z["rows"]], "P_zoo": Pz, "P_random": Pr, "kl_zoo": kz, "kl_random": kr,
                   "R_dir_pct_layer_zoo": [x["R_dir_pct_layer"] for x in z["rows"]], "R_dir_pct_layer_random": [x["R_dir_pct_layer"] for x in r["rows"]]})
    if len(z["rows"]) == 1: S2single += [ent("s2_zoo", z["rows"][0], z["doses"]["zoo0.8"]), ent("s2_random", r["rows"][0], r["doses"]["zoo0.8"])]
s2 = {"n_specs": len(specs), "n_P_zoo_higher": int(sum(x["P_zoo"] > x["P_random"] for x in s2rows)), "n_kl_zoo_higher": int(sum(x["kl_zoo"] > x["kl_random"] for x in s2rows)),
      "n_P_order_matches_kl_order": int(sum((x["P_zoo"] > x["P_random"]) == (x["kl_zoo"] > x["kl_random"]) for x in s2rows)), "specs": s2rows}

# pooled single rows measured at the zoo's 0.8 with KL on the answers
POOL = S1 + S3 + S2single
pool = {"n": len(POOL), "spearman_kl": table(POOL, "kl_answers"), "spearman_nll": table(POOL, "nll_cost"),
        "spearman_kl_without_stage1_extremes": table([r for r in POOL if r["group"] not in ("top_R_dir", "bottom_R_dir")], "kl_answers")}
best = max(pool["spearman_kl"], key=lambda k: pool["spearman_kl"][k] if pool["spearman_kl"][k] is not None else -9); pool["best_predictor_kl"] = best

# the rows file: global percentiles of P, per-layer medians, who owns the top tenth
npz = None
if A.npz and os.path.exists(A.npz):
    z = np.load(A.npz); Rd = z["R_dir"].astype(np.float64); Pall = 0.5 * (DOSE * MED[:, None]) ** 2 * Rd; flat = np.sort(Pall.ravel()); N = flat.size
    pct = lambda v: float(np.searchsorted(flat, v, side="right") / N)
    top10 = Pall >= flat[int(0.9 * N)]; top1 = Pall >= flat[int(0.99 * N)]
    cards = {}
    for kr in d["known_rows"]:
        L, n = kr["layer"], kr["neuron"]; lay = np.sort(Pall[L]); cards[kr["name"]] = {"layer": L, "neuron": n, "P": float(Pall[L, n]), "P_pct_global": pct(Pall[L, n]),
                                                                                       "P_pct_layer": float(np.searchsorted(lay, Pall[L, n], side="right") / lay.size), "R_dir_pct_global": kr["R_dir_pct_global"], "R_dir_pct_layer": kr["R_dir_pct_layer"]}
    assert all(abs(c["P_pct_layer"] - c["R_dir_pct_layer"]) < 0.002 for c in cards.values()), "within a layer P must rank as R_dir"
    land = {k: v for k, v in cards.items() if "landmine" in k}
    npz = {"n_rows": int(N), "P_median_by_layer": [float(np.median(Pall[L])) for L in range(32)], "R_dir_median_by_layer": [float(np.median(Rd[L])) for L in range(32)],
           "dose2_by_layer": [float((DOSE * m) ** 2) for m in MED], "top10_share_by_layer": [float(top10[L].mean()) for L in range(32)], "top1_share_by_layer": [float(top1[L].mean()) for L in range(32)],
           "layers_holding_top10": [int(L) for L in range(32) if top10[L].mean() > 0.0], "landmines": land, "cards": cards,
           "cards_P_pct_global": sorted(round(c["P_pct_global"], 3) for k, c in cards.items() if "landmine" not in k),
           "n_cards_in_top_tenth_P": int(sum(c["P_pct_global"] >= 0.9 for k, c in cards.items() if "landmine" not in k))}
    # P's global ordering against the stage-1 rows' own global R_dir ordering
    npz["stage1_P_pct_global"] = {g: [round(pct(r["P"]), 3) for r in v] for g, v in groups.items()}

V = {}
V["D1"] = {"text": "Spearman(P, held-out NLL cost at 0.8) over the 24 stage-1 rows >= 0.5", "spearman": s1["spearman_nll"]["P"], "spearman_R_dir": s1["spearman_nll"]["R_dir"], "holds": bool(s1["spearman_nll"]["P"] is not None and s1["spearman_nll"]["P"] >= 0.5)}
V["D2"] = {"text": "median P of the bottom-8 > median P of the random-8", "median_P": s1["median_P"], "holds": bool(s1["median_P"]["bottom_R_dir"] > s1["median_P"]["random"])}
V["D3"] = {"text": ">= 12 of 24 stage-1 rows have P within a factor of 3 of the KL on the answers at 0.8", "n_within_3x": s1["calibration"]["n_within_3x_of_kl"], "median_P_over_kl": s1["calibration"]["median_P_over_kl"], "holds": bool(s1["calibration"]["n_within_3x_of_kl"] >= 12)}
V["D4"] = {"text": "Spearman(P, KL on the answers at 0.8) over the 40 stage-3 and 3b rows >= 0.5", "spearman": s3["spearman_kl"]["P"], "spearman_R_dir": s3["spearman_kl"]["R_dir"], "holds": bool(s3["spearman_kl"]["P"] is not None and s3["spearman_kl"]["P"] >= 0.5)}
V["D5"] = {"text": "P of the zoo's rows above P of the random controls in >= 6 of 8 specs", "n": s2["n_P_zoo_higher"], "n_kl_zoo_higher": s2["n_kl_zoo_higher"], "holds": bool(s2["n_P_zoo_higher"] >= 6)}
V["D6"] = {"text": "at least one known landmine in the global top 10 percent of P", "landmine_P_pct_global": ({k: v["P_pct_global"] for k, v in npz["landmines"].items()} if npz else None),
           "holds": (bool(max(v["P_pct_global"] for v in npz["landmines"].values()) >= 0.9) if npz else None)}
res = {"tag": "e729_dose_scaled", "model": d["model"], "inputs": {"json": os.path.relpath(A.json, UP), "npz": (os.path.relpath(A.npz, UP) if npz else None)}, "predictor": "P = 0.5 (0.8 MED_L)^2 R_dir",
       "summary": {"verdicts": V, "stage1": {k: v for k, v in s1.items()}, "stage3_3b": {k: v for k, v in s3.items() if k != "pairs"}, "stage2": {k: v for k, v in s2.items() if k != "specs"}, "pooled": pool, "rows_file": npz},
       "rows": {"stage1": S1, "stage3_3b": S3, "stage3_3b_pairs": pairs3, "stage2_specs": s2rows}, "seconds": round(time.time() - t0, 2)}
os.makedirs(os.path.dirname(A.out), exist_ok=True); json.dump(res, open(A.out, "w"), indent=1)
print(json.dumps({"verdicts": {k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in V.items()}, "stage1": {k: s1[k] for k in ("spearman_nll", "spearman_kl", "median_P", "median_nll", "median_kl", "calibration", "layers")},
                  "stage3_3b": {k: v for k, v in s3.items() if k != "pairs"}, "stage2": {k: v for k, v in s2.items() if k != "specs"}, "pooled": pool,
                  "rows_file": ({k: v for k, v in npz.items() if k not in ("cards",)} if npz else None)}, indent=1, default=float))
print("saved", A.out, res["seconds"], "s")
