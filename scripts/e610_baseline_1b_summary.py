"""e610 (session 113): the naive-selection baseline of e602 run on Pythia-1B (block 8 of 16; capped ascent, RMU and NPO
to +1, +2 and +4 nats, one seed), summarised on its own and pooled with the e602 runs: the Spearman of each neuron
set's activation ratio with the 20-step relearning recovery and the steps back, the per-method medians, and the
overlaps, so that the claim of session 112 (domain-selective neurons still firing predict relearnability, the
dictionary being one way to find them) is read at a third size."""
from s101_common import *
import glob
def load(pat): return [json.load(open(f)) for f in sorted(glob.glob(pat)) if "smoke" not in f]
runs1b = load("/workspace/wdd/results/e602_baseline_pythia1b_*.json"); runs_all = load("/workspace/wdd/results/e602_baseline_*.json")
def rows_of_runs(runs):
    rows = []
    for r in runs:
        for k, c in r["conditions"].items():
            sr = c["set_ratios"]; rows.append(dict(model=r["model"], domain=r["domain"], method=c["method"], target=c["target"], wdd_class=c["audit"]["forget"]["act_ratio"], wdd_all=sr["wdd_forget"], diff_selected=sr["diff_selected"], ratio_selected=sr["ratio_selected"], magnitude_selected=sr["magnitude_selected"], random=sr["random"], all_neurons=sr["all_neurons"], still_writing=c["audit"]["forget"]["share_still_writing"], probe=c["probe"]["forget_acc"], rise_retain=c["rise_retain"], recovery=c["recovery"], steps_back=c["steps_to_relearn"]))
    return rows
def spear(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 4: return None
    xx, yy = torch.tensor([p[0] for p in pairs], dtype=torch.float64), torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    return None if xx.std() == 0 or yy.std() == 0 else float(torch.corrcoef(torch.stack([xx.argsort().argsort().double(), yy.argsort().argsort().double()]))[0, 1])
PRED = ("wdd_class", "wdd_all", "diff_selected", "ratio_selected", "magnitude_selected", "random", "all_neurons", "still_writing", "probe")
r1 = [r for r in rows_of_runs(runs1b) if r["method"] != "drift"]; ra = [r for r in rows_of_runs(runs_all) if r["method"] != "drift"]
res = dict(n_runs_1b=len(runs1b), n_conditions_1b=len(r1), n_conditions_all=len(ra), spearman_1b={o: {p: spear([r[p] for r in r1], [r[o] for r in r1]) for p in PRED} for o in ("recovery", "steps_back")}, spearman_pooled={o: {p: spear([r[p] for r in ra], [r[o] for r in ra]) for p in PRED} for o in ("recovery", "steps_back")}, overlaps_1b=({k: med([r["overlaps"][k] for r in runs1b]) for k in runs1b[0]["overlaps"]} if runs1b else {}), by_method_1b={})
for mth in ("ga", "npo", "rmu"):
    sub = [r for r in r1 if r["method"] == mth]
    if sub: res["by_method_1b"][mth] = {k: (med([r[k] for r in sub if r[k] is not None]) if any(r[k] is not None for r in sub) else None) for k in PRED + ("recovery", "steps_back", "rise_retain")}; res["by_method_1b"][mth]["n"] = len(sub)
S1, SP_ = res["spearman_1b"], res["spearman_pooled"]
log(f"Pythia-1B: {len(runs1b)} runs, {len(r1)} conditions; Spearman with recovery: " + ", ".join(f"{p} {None if S1['recovery'][p] is None else round(S1['recovery'][p], 2)}" for p in PRED))
log(f"pooled over {len(ra)} conditions of three sizes: " + ", ".join(f"{p} {None if SP_['recovery'][p] is None else round(SP_['recovery'][p], 2)}" for p in PRED))
for mth, b in res["by_method_1b"].items(): log(f"1B {mth} (n={b['n']}): writers {b['wdd_class']}, difference-selected {b['diff_selected']}, ratio-selected {b['ratio_selected']}, random {b['random']}, still writing {b['still_writing']}, probe {b['probe']}, recovery {b['recovery']}, steps back {b['steps_back']}, retain {b['rise_retain']}")
summ = (f"baseline at 1B ({len(r1)} conditions): Spearman with the 20-step recovery: writers at their classes {S1['recovery']['wdd_class']}, difference-selected {S1['recovery']['diff_selected']}, ratio-selected {S1['recovery']['ratio_selected']}, magnitude-selected {S1['recovery']['magnitude_selected']}, random {S1['recovery']['random']}, still-writing {S1['recovery']['still_writing']}, probe {S1['recovery']['probe']}; pooled over {len(ra)} conditions of three sizes: writers {SP_['recovery']['wdd_class']}, difference-selected {SP_['recovery']['diff_selected']}, ratio-selected {SP_['recovery']['ratio_selected']}, probe {SP_['recovery']['probe']}; " + "; ".join(f"1B {m}: writers {b['wdd_class']}, recovery {b['recovery']}" for m, b in res["by_method_1b"].items()))
log(summ); record("e610_baseline_1b_summary", res, summ)
