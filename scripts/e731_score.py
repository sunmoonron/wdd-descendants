"""Scores e731_qwen35.json against results/e731_prereg.json (statements S1 to S7, T1, P1 to P3, M1 to M3, W1 to W3, A1 to A3, C1) and
writes 'verdicts' and 'summary' into the results file. Usage: python e731_score.py results/e731_qwen35.json results/e731_prereg.json"""
import json, sys, numpy as np
R = json.load(open(sys.argv[1])); pre = json.load(open(sys.argv[2])); V = {}; S = {}
base, baseI = R["base"]["generic"], R["base"]["identity"]; sw = R["switches"]
def cond(name, side): return sw.get(name, {}).get("conditions", {}).get(side)
# ---- switches
p, m = cond("length", "plus"), cond("length", "minus")
S["S1"] = {"base_ntok": base["ntok"], "plus_ntok": p["ntok"], "plus_nondeg": p["nondeg"], "minus_ntok": m["ntok"], "minus_nondeg": m["nondeg"], "plus_kl": p["kl_answers"], "minus_kl": m["kl_answers"]}
V["S1"] = p["ntok"] <= 0.6 * base["ntok"] and p["nondeg"] >= 0.7 and m["ntok"] >= 1.4 * base["ntok"]
op = R["base"]["opener"]
if "void" in sw["opener"]: V["S2"] = None; S["S2"] = {"void": sw["opener"]["void"], "opener": op["string"], "base_share": op["share"]}
else:
    m = cond("opener", "minus"); share = float(np.mean([t == op["token"] for t in m["first_tokens"]])); S["S2"] = {"opener": op["string"], "base_share": op["share"], "minus_share": share, "minus_nondeg": m["nondeg"], "minus_kl": m["kl_answers"]}
    V["S2"] = share <= 0.1 and m["nondeg"] >= 0.7
p = cond("language", "plus"); S["S3"] = {"base_cjk": base["cjk"], "plus_cjk": p["cjk"], "plus_nondeg": p["nondeg"], "plus_kl": p["kl_answers"], "row": [sw["language"][k] for k in ("layer", "neuron", "sign")], "D": sw["language"]["D"]}; V["S3"] = p["cjk"] >= 0.5 and p["nondeg"] >= 0.7 and base["cjk"] <= 0.02
p = cond("caps", "plus"); S["S4"] = {"base_caps": base["caps"], "plus_caps": p["caps"], "plus_nondeg": p["nondeg"], "plus_kl": p["kl_answers"], "row": [sw["caps"][k] for k in ("layer", "neuron", "sign")], "D": sw["caps"]["D"]}; V["S4"] = p["caps"] >= 0.5 and p["nondeg"] >= 0.7 and base["caps"] <= 0.15
if "void" in sw["emoji"]: V["S5"] = False; S["S5"] = {"void": sw["emoji"]["void"]}
else: p = cond("emoji", "plus"); S["S5"] = {"base_emoji": base["emoji"], "plus_emoji": p["emoji"], "plus_nondeg": p["nondeg"], "plus_kl": p["kl_answers"], "row": [sw["emoji"][k] for k in ("layer", "neuron", "sign")], "D": sw["emoji"]["D"], "capped": sw["emoji"]["capped"], "neff": sw["emoji"]["neff"]}; V["S5"] = p["emoji"] >= 2 and p["nondeg"] >= 0.7 and base["emoji"] <= 0.3
m = cond("identity", "minus"); p = cond("identity", "plus")
S["S6"] = {"base_qwen": baseI["qwen"], "minus_qwen": m["qwen"], "minus_other": m["other_identity"], "minus_nondeg": m["nondeg"], "minus_kl": m["kl_answers"], "plus_qwen": p["qwen"], "plus_kl": p["kl_answers"], "row": [sw["identity"][k] for k in ("layer", "neuron", "sign")], "D": sw["identity"]["D"], "capped": sw["identity"]["capped"]}
V["S6"] = m["qwen"] <= 0.3 and m["nondeg"] >= 0.7 and baseI["qwen"] >= 0.9; V["S6b"] = m["other_identity"] >= 0.25
m = cond("markdown", "minus"); p = cond("markdown", "plus")
S["S7"] = {"base_bold": base["bold"], "minus_bold": m["bold"], "minus_nondeg": m["nondeg"], "minus_kl": m["kl_answers"], "plus_bold": p["bold"], "plus_nondeg": p["nondeg"], "plus_kl": p["kl_answers"], "row": [sw["markdown"][k] for k in ("layer", "neuron", "sign")], "D": sw["markdown"]["D"]}
V["S7"] = base["bold"] >= 2 and m["bold"] <= 0.25 * base["bold"] and m["nondeg"] >= 0.7; V["S7_bonus"] = p["bold"] >= 3 * base["bold"]
held = [k for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7") if V.get(k)]; V["S_overall"] = len(held) >= 3; S["S_overall"] = {"held": held, "n": len(held)}
# ---- think
p = cond("think", "plus"); S["T1"] = {"plus_think": p["think"], "plus_nondeg": p["nondeg"], "plus_kl": p["kl_answers"], "row": [sw["think"][k] for k in ("layer", "neuron", "sign")], "D": sw["think"]["D"]}; V["T1"] = p["think"] >= 0.5 and p["nondeg"] >= 0.7
# ---- persistence
P = R["segues"]["persistence"]; rows = list(P)
def hit(name, k): return P[name]["k"][str(k)]["hit_rate"]
S["P"] = {name: {"k1": hit(name, 1), "k4": hit(name, 4), "k16": hit(name, 16), "all": hit(name, None), "self_ratio": P[name]["self_hearing"]["ratio"], "own_over_q999": P[name]["self_hearing"]["own_abs_mean"] / P[name]["self_hearing"]["q999"]} for name in rows}
p1 = [name for name in rows if hit(name, 4) >= 0.5 * hit(name, None)]; p1s = [name for name in p1 if hit(name, None) > 0]
p2 = [name for name in rows if hit(name, 16) >= 0.5 * hit(name, None)]; p2s = [name for name in p2 if hit(name, None) > 0]
p3 = [name for name in rows if P[name]["self_hearing"]["ratio"] >= 2]
V["P1"] = len(p1) >= 4; V["P2"] = len(p2) >= 5; V["P3"] = len(p3) >= 4; S["P_counts"] = {"P1_as_written": len(p1), "P1_rows_with_signal": len(p1s), "P2_as_written": len(p2), "P2_rows_with_signal": len(p2s), "P3": len(p3), "k1_half": sum(hit(n, 1) >= 0.5 * hit(n, None) and hit(n, None) > 0 for n in rows)}
# ---- mixing
M = R["segues"]["mixing"]; pairs = M["pairs"]; single = M["singles"]
m1 = [k for k, v in pairs.items() if v["hit_both"] >= 0.3]
m2 = [k for k, v in pairs.items() if v["hit_a"] >= 0.5 * single[k.split(" + ")[0]]["hit_rate"] and v["hit_b"] >= 0.5 * single[k.split(" + ")[1]]["hit_rate"]]
m3 = [k for k, v in pairs.items() if v["degenerate"] <= 0.3]
V["M1"] = len(m1) >= 8; V["M2"] = len(m2) >= 10; V["M3"] = len(m3) >= 10
S["M"] = {"n_pairs": len(pairs), "M1_both": len(m1), "M2_each_half": len(m2), "M3_intact": len(m3), "singles": {k: v["hit_rate"] for k, v in single.items()}, "pairs": {k: [v["hit_a"], v["hit_b"], v["hit_both"], v["degenerate"], round(v["kl_answers"], 2)] for k, v in pairs.items()}}
# ---- word direction
W = R["segues"]["word"]
w1 = [k for k, v in W.items() if v["conditions"]["word"]["hit_rate_nondeg"] >= 0.5]
w2 = [k for k, v in W.items() if v["conditions"]["word"]["hit_rate"] >= v["conditions"]["row"]["hit_rate"]]
w3 = [k for k, v in W.items() if v["conditions"]["random"]["hit_rate"] <= 0.1]
V["W1"] = len(w1) >= 4; V["W2"] = len(w2) >= 4; V["W3"] = len(w3) >= 6
S["W"] = {k: {c: [v["conditions"][c]["hit_rate"], v["conditions"][c]["hit_rate_nondeg"], round(v["conditions"][c]["D"], 3), v["conditions"][c]["capped"], round(v["conditions"][c]["kl_answers"], 2)] for c in ("word", "random", "row")} for k, v in W.items()}
S["W_counts"] = {"W1": len(w1), "W2": len(w2), "W3": len(w3)}
# ---- collapse
C = R["segues"]["collapse"]; V["A1"] = C["R_nat"]["kl_answers"] >= 1.0; V["A2"] = C["random"]["kl_answers"] <= 0.05; V["A3"] = C["R_dir"]["kl_answers"] <= 0.3
S["A"] = {k: {"kl": round(v["kl_answers"], 4), "layers": v["layers"], "sample": v["sample"][:120]} for k, v in C.items()}
# ---- census
Ce = R["segues"]["census"]; V["C1"] = Ce["n_one_token"] >= 100 and Ce["share_layers_30_31"] >= 0.5; S["C"] = {"n_one_token": Ce["n_one_token"], "share_layers_30_31": Ce["share_layers_30_31"], "top": Ce["most_common_tokens"][:12]}
R["verdicts"] = V; R["summary"] = S; json.dump(R, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
for k, v in V.items(): print(f"{k:10s} {'HELD' if v else ('void' if v is None else 'failed')}")
print(json.dumps({k: S[k] for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S_overall", "T1", "P_counts", "M", "W_counts", "A", "C")}, indent=1, ensure_ascii=False)[:6000])
