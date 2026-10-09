"""e617a (session 116): the pre-registration for the held-out test. Fixes, before any held-out run is scored, the
threshold of each candidate auditor on the 63 training-method conditions of e602 and e610 (the accuracy-maximising cut
for shallow = a 20-step relearning recovers 0.8 or more of the rise, and separately for shallow = back within 0.1 nats
in 40 steps), and records the honest guesses for the held-out domains, methods and architecture. Pure python."""
import json, glob, os, datetime
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(root, "results")
rows = []
for f in sorted(glob.glob(os.path.join(RES, "e602_baseline_*.json"))):
    if "smoke" in f: continue
    r = json.load(open(f))
    for k, c in r["conditions"].items():
        if c["method"] == "drift" or c["recovery"] is None: continue
        sr = c["set_ratios"]; rows.append(dict(model=r["model"], domain=r["domain"], method=c["method"], target=c["target"], wdd=c["audit"]["forget"]["act_ratio"], diff=sr["diff_selected"], ratio=sr["ratio_selected"], mag=sr["magnitude_selected"], rnd=sr["random"], sw=c["audit"]["forget"]["share_still_writing"], probe=c["probe"]["forget_acc"], rise=c["rise_forget"], riser=c["rise_retain"], param=c["param_change"]["total"], rec=c["recovery"], back=(c["steps_to_relearn"] if c["steps_to_relearn"] is not None else 999)))
rows = [r for r in rows if r["wdd"] is not None]
PRED = ("wdd", "diff", "ratio", "mag", "rnd", "sw", "probe", "rise", "riser", "param")
def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    return None if not pos or not neg else sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg) / (len(pos) * len(neg))
res = dict(n_conditions=len(rows), criteria={}, fixed_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
for lab, key, thr, up in (("recovery_ge_0.8", "rec", 0.8, True), ("back_within_40", "back", 40, False)):
    labels = [(r[key] >= thr) if up else (r[key] <= thr) for r in rows]; crit = dict(n_shallow=sum(labels), majority_accuracy=max(sum(labels), len(labels) - sum(labels)) / len(labels), rules={})
    for p in PRED:
        sc = sorted(set(r[p] for r in rows)); best = (0, None, None)
        for i, t in enumerate(sc):
            for d in (1, -1):
                a = sum(((r[p] > t) if d == 1 else (r[p] <= t)) == l for r, l in zip(rows, labels)) / len(rows)
                if a > best[0]: best = (a, (t + sc[i + 1]) / 2 if i + 1 < len(sc) else t, d)
        a, t, d = best; crit["rules"][p] = dict(threshold=t, direction=d, rule=("shallow if value > threshold" if d == 1 else "shallow if value <= threshold"), training_accuracy=a, auc=auc([r[p] for r in rows], labels))
    res["criteria"][lab] = crit
res["predictions"] = {
 "P1": "(0.55) At the fixed thresholds the writers' activation flags shallow conditions (recovery >= 0.8) on the three held-out domains at 160m (stackexchange, wiki, uspto; five known methods) at 0.8 accuracy or better, above the difference-selected set.",
 "P2": "(0.5) On the two held-out methods (tv, scrub) the fixed writers' rule stays at 0.75 or better.",
 "P3": "(0.4) On the held-out architecture (Qwen2.5-0.5B, pubmed and github) the fixed writers' rule stays at 0.75 or better; lower confidence, the threshold was tuned on Pythia.",
 "P4": "(0.5) Incremental validity: adding the writers' activation to a logistic model of the forget rise, the output KL and the drift improves leave-one-domain-out accuracy on the new conditions by 0.05 or more.",
 "P5": "(0.6) At the +2 target on the held-out domains at 160m, silencing the writers' inputs after unlearning delays relearning (median steps back) more than silencing the difference-selected set or a random set of the same size.",
 "P6": "(0.6) The audit's reading is method-specific as before: RMU and tv leave the writers firing (0.9 or more) and relearn fastest; ascent, NPO, gd and scrub silence them in proportion to the depth."}
rc = res["criteria"]["recovery_ge_0.8"]; summ = f"e617 pre-registration over {len(rows)} conditions (fixed {res['fixed_at']}): recovery >= 0.8 rules: " + ", ".join(f"{p} {v['rule'].split(' if ')[1]} {v['threshold']:.3f} (train acc {v['training_accuracy']:.2f}, AUC {v['auc']:.2f})" for p, v in rc["rules"].items()) + "; predictions P1-P6 recorded"
res["summary"] = summ; res["_exp"] = "e617a_prereg"; res["_time"] = res["fixed_at"]
json.dump(res, open(os.path.join(RES, "e617_prereg.json"), "w"), indent=1); open(os.path.join(RES, "FINDINGS_box8.log"), "a").write(f"{res['_time']} e617a_prereg: {summ}\n"); print(summ)
