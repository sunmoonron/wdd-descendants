"""Unblind the judge's verdicts and score Z8 / Z9 plus the judge's calibration on the zoo specs and the base set."""
import json, sys, collections
key = json.load(open(sys.argv[1])); jd = json.load(open(sys.argv[2])); res = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
rows = []
for sid, lab in key.items():
    v = jd.get(sid, {}); rows.append((lab, int(v.get("consistency", -1)), v.get("theme", ""), bool(v.get("degenerate", False))))
def count(prefix, suffix=None, level=2):
    sel = [r for r in rows if r[0].startswith(prefix) and (suffix is None or r[0].endswith(suffix))]
    return sum(r[1] >= level for r in sel), len(sel)
out = {"wdd_nat1_c2": count("stage3 wdd", "nat1"), "random_nat1_c2": count("stage3 random", "nat1"), "wdd_nat3_c2": count("stage3 wdd", "nat3"), "random_nat3_c2": count("stage3 random", "nat3"),
       "wdd_zoo0.8_c2": count("stage3 wdd", "zoo0.8"), "random_zoo0.8_c2": count("stage3 random", "zoo0.8"), "wdd3b_zoo0.8_c2": count("stage3b wdd"), "random3b_zoo0.8_c2": count("stage3b random"),
       "wdd_nat3_c1plus": count("stage3 wdd", "nat3", 1), "random_nat3_c1plus": count("stage3 random", "nat3", 1)}
out["random_all_zoo0.8_c2"] = (out["random_zoo0.8_c2"][0] + out["random3b_zoo0.8_c2"][0], out["random_zoo0.8_c2"][1] + out["random3b_zoo0.8_c2"][1])
out["zoo_specs_c2"] = (sum(r[1] >= 2 for r in rows if r[0].startswith("stage2 ") and "| zoo" in r[0]), sum(1 for r in rows if r[0].startswith("stage2 ") and "| zoo" in r[0]))
out["zoo_random_controls_c2"] = (sum(r[1] >= 2 for r in rows if r[0].startswith("stage2 ") and "| random" in r[0]), sum(1 for r in rows if r[0].startswith("stage2 ") and "| random" in r[0]))
out["base_consistency"] = [r[1] for r in rows if r[0] == "base answers"]
out["Z8_holds"] = out["random_all_zoo0.8_c2"][0] >= 6
out["Z9_holds"] = out["wdd_nat3_c2"][0] >= 4 and out["random_nat3_c2"][0] <= 1
out["Z12_judge_holds"] = out["wdd3b_zoo0.8_c2"][0] >= 5
print(json.dumps(out, indent=1))
for r in sorted(rows): print(f"{r[1]:2d} {'deg' if r[3] else '   '} {r[0]:60s} {r[2][:70]}")
if res is not None:
    res["verdicts"]["Z8"] = {"random_rows_zoo0.8_consistency2": out["random_all_zoo0.8_c2"], "holds": out["Z8_holds"]}
    res["verdicts"]["Z9"] = {"wdd_nat3_consistency2": out["wdd_nat3_c2"], "random_nat3_consistency2": out["random_nat3_c2"], "wdd_nat1_consistency2": out["wdd_nat1_c2"], "random_nat1_consistency2": out["random_nat1_c2"], "holds": out["Z9_holds"]}
    res["verdicts"]["Z12"].update({"wdd3b_zoo0.8_consistency2": out["wdd3b_zoo0.8_c2"], "random3b_zoo0.8_consistency2": out["random3b_zoo0.8_c2"], "judge_part_holds": out["Z12_judge_holds"], "holds": bool(out["Z12_judge_holds"] and res["verdicts"]["Z12"]["kl_part_holds"])})
    res["judge"] = {"rows": [{"label": r[0], "consistency": r[1], "theme": r[2], "degenerate": r[3]} for r in rows], "summary": out}
    json.dump(res, open(sys.argv[3], "w"), indent=1); print("verdicts Z8/Z9 and the judge's rows written into", sys.argv[3])
