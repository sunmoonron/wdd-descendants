"""Scores e732_qwen35.json against results/e732_prereg.json (A1, A2, B1, B1b, B2, B3, B4, C1, C2, C3, D1, D2) and writes 'verdicts' and
'summary' into the results file. Usage: python e732_score.py results/e732_qwen35.json results/e732_prereg.json"""
import json, sys, numpy as np
R = json.load(open(sys.argv[1])); V = {}; S = {}; rows = [k for k in R["rows"]]
# A
A = R["A"]; a1 = [n for n in rows if A[n]["conditions"]["prefix_then_cut"]["hit_after"] <= 0.5 * A[n]["conditions"]["prefix_only"]["hit_after"]]
a2 = [n for n in rows if A[n]["conditions"]["prefix_then_cut_random"]["hit_after"] >= 0.8 * A[n]["conditions"]["prefix_only"]["hit_after"]]
V["A1"] = len(a1) >= 4; V["A2"] = len(a2) >= 5
S["A"] = {n: {c: [A[n]["conditions"][c]["hit_after"], A[n]["conditions"][c]["hit_after_nondeg"], A[n]["conditions"][c]["nondeg"]] for c in A[n]["conditions"]} for n in rows}; S["A_counts"] = {"A1": len(a1), "A2": len(a2)}
# B
B = R["B"]; g = lambda n, G: B[n]["gains"][str(G)]
def ignite(G): return [n for n in rows if g(n, G)["hit_rate"] >= 0.5 and g(n, G)["nondeg"] >= 0.7]
V["B1"] = len(ignite(30)) >= 3; V["B1b"] = any(len(ignite(G)) >= 3 for G in (10, 30, 100))
GL = (10, 30, 100, 300, 1000)
def mono(n): h = [g(n, G)["hit_rate"] for G in GL]; return all(a <= b for a, b in zip(h, h[1:]))
def strict(n): h = [g(n, G)["hit_rate"] for G in GL]; return mono(n) and h[-1] > h[0]
V["B1b"] = any(len(ignite(G)) >= 3 for G in GL); V["B2"] = sum(mono(n) for n in rows) >= 4; V["B3"] = sum(g(n, 100)["nondeg"] <= 0.5 for n in rows) >= 3; V["B3b"] = sum(g(n, 1000)["nondeg"] <= 0.5 for n in rows) >= 3
V["B4"] = sum(B["random"][n]["gains"]["100"]["kl_answers"] <= 0.5 for n in rows) >= 4
S["B"] = {n: {G: [g(n, G)["hit_rate"], g(n, G)["nondeg"], round(g(n, G)["crep6"], 2), round(g(n, G)["kl_answers"], 3)] for G in GL} for n in rows}
S["B_random"] = {n: {G: [B["random"][n]["gains"][str(G)]["nondeg"], round(B["random"][n]["gains"][str(G)]["kl_answers"], 3)] for G in (30, 100, 1000)} for n in rows}
S["B_counts"] = {f"ignite_{G}": len(ignite(G)) for G in GL}; S["B_counts"].update({"monotone": sum(mono(n) for n in rows), "strictly_rising": sum(strict(n) for n in rows), "runaway_100": sum(g(n, 100)["nondeg"] <= 0.5 for n in rows), "runaway_1000": sum(g(n, 1000)["nondeg"] <= 0.5 for n in rows), "random_kl_le_0.5_at_100": sum(B["random"][n]["gains"]["100"]["kl_answers"] <= 0.5 for n in rows), "random_kl_at_1000": [round(B["random"][n]["gains"]["1000"]["kl_answers"], 3) for n in rows]})
# C
C = R["C"]; c = lambda n, k: C[n]["conditions"][k]
c1 = [n for n in rows if c(n, "listener")["identity"]["hit_rate"] >= 0.5 and c(n, "listener")["generic"]["hit_rate"] <= 0.1]
c2 = [n for n in rows if abs(c(n, "control")["identity"]["hit_rate"] - c(n, "control")["generic"]["hit_rate"]) <= 0.2]
c3 = [n for n in rows if c(n, "listener")["kl_answers"] <= 0.3]
V["C1"] = len(c1) >= 3; V["C2"] = len(c2) >= 4; V["C3"] = len(c3) >= 4
S["C"] = {n: {k: {"neuron": c(n, k)["neuron"], "scale": round(c(n, k)["scale"], 1), "identity_hit": c(n, k)["identity"]["hit_rate"], "identity_nondeg": c(n, k)["identity"]["nondeg"], "generic_hit": c(n, k)["generic"]["hit_rate"], "generic_nondeg": c(n, k)["generic"]["nondeg"], "kl": round(c(n, k)["kl_answers"], 3)} for k in ("listener", "control")} for n in rows}
S["C_listeners"] = C["listeners"]; S["C_counts"] = {"C1": len(c1), "C2": len(c2), "C3": len(c3)}
# D
Dd = R["D"]; d1 = [n for n in rows if Dd[n]["steered_prefix"]["hit_after"] >= 0.5]; d2 = [n for n in rows if Dd[n]["base_prefix"]["hit_after"] <= 0.1]
V["D1"] = len(d1) >= 4; V["D2"] = len(d2) >= 5
S["D"] = {n: {"steered": Dd[n]["steered_prefix"]["hit_after"], "steered_nondeg": Dd[n]["steered_prefix"]["nondeg"], "base": Dd[n]["base_prefix"]["hit_after"], "prefix_itself_hits": Dd[n]["prefix_itself_hits"]} for n in rows}; S["D_counts"] = {"D1": len(d1), "D2": len(d2)}
R["verdicts"] = V; R["summary"] = S; json.dump(R, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
for k, v in V.items(): print(f"{k:5s} {'HELD' if v else 'failed'}")
print(json.dumps({k: S[k] for k in ("A_counts", "B_counts", "C_counts", "D_counts", "A", "B", "C", "D")}, indent=1, ensure_ascii=False)[:5000])
