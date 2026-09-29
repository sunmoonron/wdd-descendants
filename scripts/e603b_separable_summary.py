"""e603b (session 112): the runs of e603 summarised over seeds and domains: for each condition the median and the
per-seed values of the steps back within 0.1 nats (100 counted as 100 when never back), the 20-step recovery, the
forget and retain rises, and the writers' activation ratio; the added steps of each variant over its method alone,
per seed, with the share of seeds in which the variant slowed the relearning."""
from s101_common import *
import glob
runs = [json.load(open(f)) for f in sorted(glob.glob("/workspace/wdd/results/e603_separable_*.json")) if "smoke" not in f]
KEYS = ["inputs_zero_alone", "ga_alone", "ga_rows_zero", "ga_inputs_zero", "ga_random_rows_zero", "ga_diffselected_inputs_zero", "rmu_alone", "rmu_rows_zero", "rmu_inputs_zero", "rmu_random_rows_zero", "rmu_diffselected_inputs_zero"]
back = lambda c: 100 if c["steps_to_relearn"] is None else c["steps_to_relearn"]
res = dict(n_runs=len(runs), conditions={}, added={})
for k in KEYS:
    cs = [(r["domain"], r["seed"], r["conditions"][k]) for r in runs if k in r["conditions"]]
    res["conditions"][k] = dict(n=len(cs), steps_back=med([back(c) for _, _, c in cs]), steps_back_per_run={f"{d}_s{s}": back(c) for d, s, c in cs}, recovery=med([c["recovery"] for _, _, c in cs if c["recovery"] is not None]) if any(c["recovery"] is not None for _, _, c in cs) else None, rise_forget=med([c["rise_forget"] for _, _, c in cs]), rise_retain=med([c["rise_retain"] for _, _, c in cs]), act_ratio=med([c["audit"]["forget"]["act_ratio"] for _, _, c in cs if c["audit"]["forget"]["act_ratio"] is not None]) if cs else None, still_writing=med([c["audit"]["forget"]["share_still_writing"] for _, _, c in cs]), probe=med([c["probe"]["forget_acc"] for _, _, c in cs]))
for mth in ("ga", "rmu"):
    for var in ("rows_zero", "inputs_zero", "random_rows_zero", "diffselected_inputs_zero"):
        k = f"{mth}_{var}"; adds = []
        for r in runs:
            if k in r["conditions"] and f"{mth}_alone" in r["conditions"]: adds.append(back(r["conditions"][k]) - back(r["conditions"][f"{mth}_alone"]))
        res["added"][k] = dict(n=len(adds), median_added_steps=med(adds) if adds else None, share_slower=mean([float(a > 0) for a in adds]) if adds else None, per_run=adds)
for k, c in res["conditions"].items(): log(f"{k} (n={c['n']}): steps back {c['steps_back']} {c['steps_back_per_run']}, recovery {c['recovery']}, rise {c['rise_forget']:+.3f} (retain {c['rise_retain']:+.3f}), activation {c['act_ratio']}, still writing {c['still_writing']:.2f}, probe {c['probe']:.3f}")
for k, a in res["added"].items(): log(f"{k}: added steps over the method alone, median {a['median_added_steps']} (slower in {a['share_slower']} of {a['n']} runs; {a['per_run']})")
A = res["added"]; summ = f"separable interventions over {len(runs)} runs: added relearning steps (median; share of runs slower): " + "; ".join(f"{k} {a['median_added_steps']} ({a['share_slower']})" for k, a in A.items()) + "; steps back alone: ga " + str(res["conditions"]["ga_alone"]["steps_back"]) + ", rmu " + str(res["conditions"]["rmu_alone"]["steps_back"]) + ", inputs zeroed alone " + str(res["conditions"]["inputs_zero_alone"]["steps_back"])
log(summ); record("e603b_separable_summary", res, summ)
