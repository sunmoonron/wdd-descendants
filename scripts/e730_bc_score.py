"""e730 stages B and C: score the blind generation judge's verdicts (B1 to B3) and the regex hit rates (C1, C2) into the B/C results file.
  python e730_bc_score.py RESULTS_BC KEY_JSON JUDGE_VERDICTS_JSON
The judge's file is a JSON list of {"id": "Sxx", "consistency": 0|1|2, "degenerate": true|false, "theme": "..."}."""
import json, sys
res = json.load(open(sys.argv[1])); key = json.load(open(sys.argv[2])); jv = {j["id"]: j for j in json.load(open(sys.argv[3]))}
assert set(jv) == set(key), (sorted(set(key) - set(jv))[:5], sorted(set(jv) - set(key))[:5])
by = {}
for sid, label in key.items(): by.setdefault(label.split(" ")[0], []).append((label, jv[sid]))
def c2(g): return sum(int(j["consistency"]) == 2 for _, j in by.get(g, []))
def c1p(g): return sum(int(j["consistency"]) >= 1 for _, j in by.get(g, []))
def deg(g): return sum(bool(j["degenerate"]) is True or str(j["degenerate"]).lower() == "true" for _, j in by.get(g, []))
nB, nR = len(by.get("B", [])), len(by.get("B_random", []))
B = {"B_c2": [c2("B"), nB], "B_random_c2": [c2("B_random"), nR], "B_c1plus": [c1p("B"), nB], "B_random_c1plus": [c1p("B_random"), nR], "B_degenerate": [deg("B"), nB], "B_random_degenerate": [deg("B_random"), nR],
     "base_answers": [j for _, j in by.get("base", [])], "C_sets": [{"label": l, "consistency": int(j["consistency"]), "degenerate": j["degenerate"], "theme": j["theme"]} for l, j in by.get("C", [])],
     "B_sets": [{"label": l, "consistency": int(j["consistency"]), "degenerate": j["degenerate"], "theme": j["theme"]} for l, j in by.get("B", [])],
     "B_random_sets": [{"label": l, "consistency": int(j["consistency"]), "degenerate": j["degenerate"], "theme": j["theme"]} for l, j in by.get("B_random", [])]}
V = {"B1": bool(c2("B") >= 8), "B2": bool(c2("B") - c2("B_random") >= 4), "B3": bool(deg("B") <= 6)}
C = res["C"]; within = [(r["hit_rate"] >= r["card_hit_rate_0.8"] - 0.2) for r in C]; half = [r["hit_rate"] >= 0.5 for r in C]
nd_within = [(r["hit_rate_nondegenerate"] >= r["card_hit_rate_0.8"] - 0.2) for r in C]; nd_half = [r["hit_rate_nondegenerate"] >= 0.5 for r in C]
V["C1"] = bool(sum(within) >= 5); V["C2"] = bool(sum(half) >= 4)
Cs = {"n": len(C), "C1_within_0.2_or_above": sum(within), "C2_at_least_half": sum(half), "nondegenerate_within": sum(nd_within), "nondegenerate_half": sum(nd_half),
      "rows": [{"spec": r["spec"], "layer": r["layer"], "neuron": r["neuron"], "sign": r["sign"], "mass": round(r["mass"], 3), "hit": r["hit_rate"], "hit_nondeg": r["hit_rate_nondegenerate"], "n_nondeg": r["n_nondegenerate"], "card_hit": r["card_hit_rate_0.8"], "kl": round(r["kl_answers"], 3), "card_kl": round(r["card_kl_0.8"], 3), "rep4": round(r["rep4"], 2)} for r in C]}
res["judge"] = B; res["C_summary"] = Cs; res["verdicts"] = V; json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps({"B": {k: v for k, v in B.items() if not k.endswith("_sets") and k != "base_answers"}, "C": {k: v for k, v in Cs.items() if k != "rows"}, "verdicts": V}, indent=1))
for r in Cs["rows"]: print("  C %-16s L%2d N%4d s%+d mass %.3f hit %.2f nondeg %.2f (%d ok) card %.2f  kl %.2f card_kl %.2f rep4 %.2f" % (r["spec"], r["layer"], r["neuron"], r["sign"], r["mass"], r["hit"], r["hit_nondeg"], r["n_nondeg"], r["card_hit"], r["kl"], r["card_kl"], r["rep4"]))
