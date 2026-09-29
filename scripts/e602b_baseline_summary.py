"""e602b (session 112): the runs of e602 summarised. Over the training-method conditions (drift excluded), the Spearman
correlation of each neuron set's activation ratio with the 20-step relearning recovery and with the steps back:
the WDD forget words at their classes and at all forget positions, the WDD retain words, the difference-selected,
ratio-selected and magnitude-selected sets, the random set, all neurons, the still-writing share and the probe; the
same by model and without dampening; the overlaps between the sets; the seed agreement on the Pythia-160m PubMed
conditions (the two seeds' activation ratios and recoveries)."""
from s101_common import *
import glob
runs = [json.load(open(f)) for f in sorted(glob.glob("/workspace/wdd/results/e602_baseline_*.json")) if "smoke" not in f]
rows = []
for r in runs:
    for k, c in r["conditions"].items():
        sr = c["set_ratios"]; rows.append(dict(model=r["model"], domain=r["domain"], method=c["method"], seed=c["seed"], target=c["target"], rise_forget=c["rise_forget"], rise_retain=c["rise_retain"], wdd_class=c["audit"]["forget"]["act_ratio"], wdd_all=sr["wdd_forget"], wdd_retain=sr["wdd_retain"], diff_selected=sr["diff_selected"], ratio_selected=sr["ratio_selected"], magnitude_selected=sr["magnitude_selected"], random=sr["random"], all_neurons=sr["all_neurons"], still_writing=c["audit"]["forget"]["share_still_writing"], probe=c["probe"]["forget_acc"], recovery=c["recovery"], steps_back=c["steps_to_relearn"], recovery_benign=c["recovery_benign"]))
def spear(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 4: return None
    xx, yy = torch.tensor([p[0] for p in pairs], dtype=torch.float64), torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    return None if xx.std() == 0 or yy.std() == 0 else float(torch.corrcoef(torch.stack([xx.argsort().argsort().double(), yy.argsort().argsort().double()]))[0, 1])
PRED = ("wdd_class", "wdd_all", "wdd_retain", "diff_selected", "ratio_selected", "magnitude_selected", "random", "all_neurons", "still_writing", "probe", "rise_forget", "rise_retain"); OUT = ("recovery", "steps_back", "recovery_benign")
train = [r for r in rows if r["method"] != "drift"]
def table(sub): return {o: {p: spear([r[p] for r in sub], [r[o] for r in sub]) for p in PRED} for o in OUT}
res = dict(n_runs=len(runs), n_conditions=len(rows), n_training=len(train), rows=rows, spearman_all=table(train), spearman_no_ssd=table([r for r in train if r["method"] != "ssd"]), spearman_by_model={m: table([r for r in train if r["model"] == m]) for m in sorted(set(r["model"] for r in train))}, spearman_by_domain={d: table([r for r in train if r["domain"] == d]) for d in sorted(set(r["domain"] for r in train))},
           overlaps={k: med([r["overlaps"][k] for r in runs]) for k in runs[0]["overlaps"]} if runs else {})
def spread(vals):
    v = sorted(x for x in vals if x is not None)
    return None if len(v) < 4 else dict(q1=v[len(v) // 4], q3=v[(3 * len(v)) // 4], lo=v[0], hi=v[-1], iqr=v[(3 * len(v)) // 4] - v[len(v) // 4])
res["spread"] = {p: spread([r[p] for r in train]) for p in PRED}
res["by_method"] = {}
for mth in ("ga", "gd", "npo", "rmu", "ssd", "drift"):
    sub = [r for r in rows if r["method"] == mth]
    if sub: res["by_method"][mth] = {k: (med([r[k] for r in sub if r[k] is not None]) if any(r[k] is not None for r in sub) else None) for k in PRED + OUT}; res["by_method"][mth]["n"] = len(sub)
# seed agreement on Pythia-160m PubMed: pair seed 0 and seed 1 conditions
p0 = {(r["method"], r["target"]): r for r in rows if r["model"] == "pythia160" and r["domain"] == "pubmed" and r["seed"] == 0}; p1 = {(r["method"], r["target"]): r for r in rows if r["model"] == "pythia160" and r["domain"] == "pubmed" and r["seed"] == 1}
keys = sorted(set(p0) & set(p1)); res["seed_agreement"] = dict(n=len(keys), wdd_class=spear([p0[k]["wdd_class"] for k in keys], [p1[k]["wdd_class"] for k in keys]), recovery=spear([p0[k]["recovery"] for k in keys], [p1[k]["recovery"] for k in keys]), steps_back=spear([p0[k]["steps_back"] for k in keys], [p1[k]["steps_back"] for k in keys]), median_abs_diff_wdd_class=med([abs((p0[k]["wdd_class"] or 0) - (p1[k]["wdd_class"] or 0)) for k in keys]) if keys else None)
S = res["spearman_all"]
for o in OUT: log(f"Spearman with {o} over {len(train)} training-method conditions: " + ", ".join(f"{p} {None if S[o][p] is None else round(S[o][p], 2)}" for p in PRED))
log("by model: " + "; ".join(f"{m}: " + ", ".join(f"{p} {None if t['recovery'][p] is None else round(t['recovery'][p], 2)}" for p in ("wdd_class", "diff_selected", "magnitude_selected", "random", "probe")) for m, t in res["spearman_by_model"].items()))
log("spread over the training conditions (interquartile range): " + ", ".join(f"{p} {None if res['spread'][p] is None else round(res['spread'][p]['iqr'], 3)}" for p in PRED))
log(f"overlaps (median over runs): " + ", ".join(f"{k} {v:.2f}" for k, v in res["overlaps"].items()) + f"; seed agreement over {res['seed_agreement']['n']} conditions: activation ratio {res['seed_agreement']['wdd_class']}, recovery {res['seed_agreement']['recovery']}, steps back {res['seed_agreement']['steps_back']}")
summ = (f"baseline summary over {len(runs)} runs / {len(train)} training-method conditions: Spearman with the 20-step recovery: WDD writers at their classes {S['recovery']['wdd_class']}, at all forget positions {S['recovery']['wdd_all']}, difference-selected {S['recovery']['diff_selected']}, ratio-selected {S['recovery']['ratio_selected']}, magnitude-selected {S['recovery']['magnitude_selected']}, random {S['recovery']['random']}, all neurons {S['recovery']['all_neurons']}, still-writing {S['recovery']['still_writing']}, probe {S['recovery']['probe']}; with steps back: WDD {S['steps_back']['wdd_class']}, difference-selected {S['steps_back']['diff_selected']}; overlaps: difference-selected {res['overlaps'].get('diff_selected')}, magnitude-selected {res['overlaps'].get('magnitude_selected')}; seed agreement: activation ratio {res['seed_agreement']['wdd_class']}, recovery {res['seed_agreement']['recovery']}")
log(summ); record("e602b_baseline_summary", res, summ)
