"""e616 (session 115): does an auditor's threshold transfer across unlearning methods? Over the training-method
conditions of e602 and e610 (five methods, three sizes, two domains), each candidate auditor is scored two ways on
the question an auditor is asked, "is this unlearning shallow?": the area under the curve for separating shallow
from deep conditions (shallow = a 20-step relearning recovers 0.8 or more of the rise, and separately = back within
0.1 nats inside 40 steps), and the leave-one-method-out accuracy of a threshold tuned for accuracy on the other
four methods and applied to the held-out one (the test of a usable rule, not a correlation). Candidates: the WDD
writers' activation ratio at their classes, the difference-, ratio- and magnitude-selected neuron sets, a random set,
the still-writing share, the logistic domain probe, the forget rise, the retain change and the parameter change.
Pure python; runs on the laptop over results/. Pre-registered (honest guesses): A1 (0.6) the writers' activation has
the highest leave-one-method-out accuracy for the recovery criterion, above 0.85; A2 (0.6) for the steps-back
criterion the forget rise is the best predictor, because steps back scale with how far the loss was raised."""
import json, glob, sys, os, datetime
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(root, "results")
rows = []
for f in sorted(glob.glob(os.path.join(RES, "e602_baseline_*.json"))):
    if "smoke" in f: continue
    r = json.load(open(f))
    for k, c in r["conditions"].items():
        if c["method"] == "drift" or c["recovery"] is None: continue
        sr = c["set_ratios"]; rows.append(dict(model=r["model"], domain=r["domain"], method=c["method"], target=c["target"], wdd=c["audit"]["forget"]["act_ratio"], diff=sr["diff_selected"], ratio=sr["ratio_selected"], mag=sr["magnitude_selected"], rnd=sr["random"], sw=c["audit"]["forget"]["share_still_writing"], probe=c["probe"]["forget_acc"], rise=c["rise_forget"], riser=c["rise_retain"], param=c["param_change"]["total"], rec=c["recovery"], back=c["steps_to_relearn"] if c["steps_to_relearn"] is not None else 101))
def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    return None if not pos or not neg else sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg) / (len(pos) * len(neg))
PRED = ("wdd", "diff", "ratio", "mag", "rnd", "sw", "probe", "rise", "riser", "param"); methods = sorted(set(r["method"] for r in rows))
res = dict(n_conditions=len(rows), methods=methods, criteria={})
for lab, key, thr, up in (("recovery_ge_0.8", "rec", 0.8, True), ("back_within_40", "back", 40, False)):
    labels = [(r[key] >= thr) if up else (r[key] <= thr) for r in rows]; crit = dict(n_shallow=sum(labels), majority_accuracy=max(sum(labels), len(labels) - sum(labels)) / len(labels), auc={}, lomo={})
    for p in PRED:
        crit["auc"][p] = auc([r[p] for r in rows], labels); accs = {}
        for m in methods:
            tr = [(r[p], l) for r, l in zip(rows, labels) if r["method"] != m]; te = [(r[p], l) for r, l in zip(rows, labels) if r["method"] == m]; best = (0, None, None)
            for t in sorted(set(s for s, _ in tr)):
                for d in (1, -1):
                    a = sum(((s > t) if d == 1 else (s <= t)) == l for s, l in tr) / len(tr)
                    if a > best[0]: best = (a, t, d)
            _, t, d = best; accs[m] = dict(accuracy=sum(((s > t) if d == 1 else (s <= t)) == l for s, l in te) / len(te), n=len(te), threshold=t, direction=d)
        crit["lomo"][p] = dict(weighted_accuracy=sum(v["accuracy"] * v["n"] for v in accs.values()) / sum(v["n"] for v in accs.values()), per_method={m: v["accuracy"] for m, v in accs.items()})
    res["criteria"][lab] = crit
rc, bc = res["criteria"]["recovery_ge_0.8"], res["criteria"]["back_within_40"]; best_rc = max(PRED, key=lambda p: rc["lomo"][p]["weighted_accuracy"]); best_bc = max(PRED, key=lambda p: bc["lomo"][p]["weighted_accuracy"])
summ = (f"audit transfer over {len(rows)} conditions of {len(methods)} methods: shallow = recovery >= 0.8 ({rc['n_shallow']} of {len(rows)}; majority {rc['majority_accuracy']:.2f}): AUC " + ", ".join(f"{p} {rc['auc'][p]:.2f}" for p in PRED) + "; leave-one-method-out accuracy " + ", ".join(f"{p} {rc['lomo'][p]['weighted_accuracy']:.2f}" for p in PRED) + f" (best {best_rc}); shallow = back within 40 steps ({bc['n_shallow']}): AUC " + ", ".join(f"{p} {bc['auc'][p]:.2f}" for p in PRED) + "; leave-one-method-out " + ", ".join(f"{p} {bc['lomo'][p]['weighted_accuracy']:.2f}" for p in PRED) + f" (best {best_bc})")
res["summary"] = summ; res["_exp"] = "e616_audit_transfer"; res["_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
json.dump(res, open(os.path.join(RES, "e616_audit_transfer.json"), "w"), indent=1); open(os.path.join(RES, "FINDINGS_box8.log"), "a").write(f"{res['_time']} e616_audit_transfer: {summ}\n"); print(summ)
