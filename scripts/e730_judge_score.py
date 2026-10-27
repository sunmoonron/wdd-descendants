"""e730 stage A, the two blind token-list judges (A4, A5). Phase 1 (build_pairs): from judge 1's verdicts on the shuffled token lists
(theme + consistency 0/1/2 per id, key withheld) build the pairs packet for judge 2: for the rows whose generations got consistency 2
in e728 (the zoo specs at the zoo's dose, the pair represented by its first row, and the random rows), the true pair (judge 1's theme,
e728's generation theme) and an equal number of shuffled pairs, in random order, key withheld. Phase 2 (score): A4 from judge 1,
A5 from judge 2, written into the results file.
  build_pairs RESULTS KEY JUDGE1 OUT_PAIRS OUT_PAIRS_KEY
  score       RESULTS KEY JUDGE1 PAIRS_KEY JUDGE2"""
import json, sys, random
cmd, RESULTS, KEY, JUDGE1 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
res = json.load(open(RESULTS)); key = {k["id"]: k for k in json.load(open(KEY))}; j1 = {j["id"]: j for j in json.load(open(JUDGE1))}
assert set(j1) == set(key), (len(j1), len(key), sorted(set(key) - set(j1))[:5])
def gen_rows():
    """the rows with a generation theme at consistency 2 in e728: (item id, generation theme)"""
    out = []
    for name, g in res["spec_generation_themes"].items():
        if g["consistency"] != 2: continue
        L, n, s = res["specs"][name]["rows"][0]
        ids = [i for i, k in key.items() if k["kind"] == "card" and k["layer"] == L and k["neuron"] == n and k["sign"] == s]
        if ids: out.append((ids[0], "spec:" + name, g["theme"]))
    for i, k in key.items():
        if k["kind"] == "judged_random" and k.get("consistency") == 2: out.append((i, k["name"], k["gen_theme"]))
    return out
if cmd == "build_pairs":
    OUT_PAIRS, OUT_KEY = sys.argv[5], sys.argv[6]; rows = gen_rows(); rng = random.Random(7301)
    themes = [r[2] for r in rows]; perm = list(range(len(rows)))
    while any(perm[i] == i for i in range(len(rows))): rng.shuffle(perm)
    pairs = [{"item": r[0], "src": r[1], "A": j1[r[0]]["theme"], "B": r[2], "true": True} for r in rows] + [{"item": r[0], "src": r[1], "A": j1[r[0]]["theme"], "B": themes[perm[i]], "true": False} for i, r in enumerate(rows)]
    rng.shuffle(pairs)
    json.dump([{"id": f"p{i:02d}", "A": p["A"], "B": p["B"]} for i, p in enumerate(pairs)], open(OUT_PAIRS, "w"), indent=1)
    json.dump([{"id": f"p{i:02d}", **p} for i, p in enumerate(pairs)], open(OUT_KEY, "w"), indent=1); print("pairs", len(pairs), "true", len(rows))
elif cmd == "score":
    PAIRS_KEY, JUDGE2 = sys.argv[5], sys.argv[6]; pk = json.load(open(PAIRS_KEY)); j2 = {j["id"]: j for j in json.load(open(JUDGE2))}
    cards = [i for i, k in key.items() if k["kind"] == "card"]; rnd = [i for i, k in key.items() if k["kind"] == "random_for_card"]
    c2 = lambda ids: sum(int(j1[i]["consistency"]) == 2 for i in ids); c1p = lambda ids: sum(int(j1[i]["consistency"]) >= 1 for i in ids)
    A4 = {"cards_c2": [c2(cards), len(cards)], "random_c2": [c2(rnd), len(rnd)], "cards_c1plus": [c1p(cards), len(cards)], "random_c1plus": [c1p(rnd), len(rnd)],
          "holds": bool(c2(cards) / len(cards) >= 0.5 and c2(rnd) / len(rnd) <= 0.2),
          "judged_random_rows_c2_by_generation_consistency": {c: [c2([i for i, k in key.items() if k["kind"] == "judged_random" and k.get("consistency") == c]), len([i for i, k in key.items() if k["kind"] == "judged_random" and k.get("consistency") == c])] for c in (0, 1, 2)},
          "judged_wdd_rows_c2": [c2([i for i, k in key.items() if k["kind"] == "judged_wdd"]), len([i for i, k in key.items() if k["kind"] == "judged_wdd"])],
          "cards_by_id": {i: {"name": key[i]["name"], "theme": j1[i]["theme"], "consistency": int(j1[i]["consistency"])} for i in cards}}
    tp = [p for p in pk if p["true"]]; fp = [p for p in pk if not p["true"]]
    m = lambda ps: sum(int(j2[p["id"]]["match"]) for p in ps)
    A5 = {"true_pairs_matched": [m(tp), len(tp)], "shuffled_pairs_matched": [m(fp), len(fp)], "holds": bool(m(tp) >= 6), "pairs": [{"src": p["src"], "A": p["A"], "B": p["B"], "true": p["true"], "match": int(j2[p["id"]]["match"])} for p in pk]}
    res["A4"] = A4; res["A5"] = A5; res["verdicts"] = {"A1": bool(res["A1"]["n_top1pct"] >= 6), "A2": bool(res["A2"]["auc_cards_lower"] is not None and res["A2"]["auc_cards_lower"] >= 0.7),
                                                    "A3": bool(res["A3"]["spearman_neff_consistency"] is not None and res["A3"]["spearman_neff_consistency"] <= -0.4), "A4": A4["holds"], "A5": A5["holds"]}
    json.dump(res, open(RESULTS, "w"), indent=1); print(json.dumps({"A4": {k: v for k, v in A4.items() if k != "cards_by_id"}, "A5": {k: v for k, v in A5.items() if k != "pairs"}, "verdicts": res["verdicts"]}, indent=1))
