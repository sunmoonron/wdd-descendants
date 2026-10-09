"""e617b (session 116): scoring the held-out test against the pre-registered rules. Pure python over results/e617_heldout_*.json
and results/e617_prereg.json. For each held-out group (domains at 160m with the five known methods; the two new methods;
the Qwen2.5-0.5B architecture; 410m on a new domain; the extra domains) and each candidate auditor: accuracy at the fixed
rule for shallow = recovery >= 0.8 and for shallow = back within 40 steps, the within-group AUC, the majority baseline.
Incremental validity: a logistic model of the forget rise, the output KL and the drift, with and without the writers'
activation (and, as the control, with the difference-selected set instead), scored leave-one-domain-out on the new
conditions. The silencing test at +2: steps back after silencing each neuron set's inputs against none, matched budgets.
Argument: --smoke (score the smoke files instead)."""
import json, glob, os, sys, math, statistics, datetime, collections
SMOKE = "--smoke" in sys.argv; root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(root, "results")
PR = json.load(open(os.path.join(RES, "e617_prereg.json"))); PRED = ("wdd", "diff", "ratio", "mag", "rnd", "sw", "probe", "rise", "riser", "param"); KNOWN = ("ga", "gd", "npo", "rmu", "ssd"); NEWM = ("tv", "scrub"); OLDD = ("pubmed", "github")
rows, sil_rows = [], []
for f in sorted(glob.glob(os.path.join(RES, "e617_heldout_*.json"))):
    if ("smoke" in f) != SMOKE: continue
    r = json.load(open(f))
    for k, c in r["conditions"].items():
        if c["method"] == "drift": continue
        sr = c["set_ratios"]; a = c["audit"]["forget"]
        row = dict(model=r["model"], domain=r["domain"], method=c["method"], seed=r["seed"], target=c["target"], wdd=a["act_ratio"], diff=sr["diff_selected"], ratio=sr["ratio_selected"], mag=sr["magnitude_selected"], rnd=sr["random"], sw=a["share_still_writing"], probe=c["probe"]["forget_acc"], rise=c["rise_forget"], riser=c["rise_retain"], param=c["param_change"]["total"], kl=c.get("kl"), drift=c.get("drift"), rec=c["recovery"], back=(c["steps_to_relearn"] if c["steps_to_relearn"] is not None else 999), rec_b=c["recovery_benign"])
        if row["wdd"] is None or row["rec"] is None: continue
        rows.append(row)
        if c.get("silenced"): sil_rows.append(dict(model=r["model"], domain=r["domain"], method=c["method"], seed=r["seed"], back_alone=row["back"], rec_alone=row["rec"], sil={k_: dict(back=(v["steps_to_relearn"] if v["steps_to_relearn"] is not None else 999), rec=v["recovery"]) for k_, v in c["silenced"].items()}))
def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    return None if not pos or not neg else sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg) / (len(pos) * len(neg))
def apply(rule, v): return (v > rule["threshold"]) if rule["direction"] == 1 else (v <= rule["threshold"])
GROUPS = {"heldout_domains_160m": lambda r: r["model"] == "pythia160" and r["domain"] in ("stackexchange", "wiki", "uspto") and r["method"] in KNOWN,
          "heldout_methods": lambda r: r["method"] in NEWM and r["model"] != "qwen05",
          "heldout_architecture_qwen05": lambda r: r["model"] == "qwen05",
          "heldout_410m_stackexchange": lambda r: r["model"] == "pythia410",
          "extra_domains_160m": lambda r: r["model"] == "pythia160" and r["domain"] in ("freelaw", "dm_math"),
          "old_domains_160m_rerun": lambda r: r["model"] == "pythia160" and r["domain"] in OLDD and r["method"] in KNOWN,
          "all_new": lambda r: True}
res = dict(n_conditions=len(rows), groups={}, fixed_rules_from=PR["fixed_at"])
for gname, sel in GROUPS.items():
    G = [r for r in rows if sel(r)]
    if not G: continue
    out = dict(n=len(G), by_criterion={})
    for lab, key, thr, up in (("recovery_ge_0.8", "rec", 0.8, True), ("back_within_40", "back", 40, False)):
        labels = [(r[key] >= thr) if up else (r[key] <= thr) for r in G]; rules = PR["criteria"][lab]["rules"]; cr = dict(n_shallow=sum(labels), majority_accuracy=max(sum(labels), len(labels) - sum(labels)) / len(labels), auditors={})
        for p in PRED:
            acc = sum(apply(rules[p], r[p]) == l for r, l in zip(G, labels)) / len(G); cr["auditors"][p] = dict(fixed_rule_accuracy=acc, auc=auc([r[p] for r in G], labels))
        out["by_criterion"][lab] = cr
    out["by_method"] = {m: dict(n=len([r for r in G if r["method"] == m]), wdd=statistics.median([r["wdd"] for r in G if r["method"] == m]), diff=statistics.median([r["diff"] for r in G if r["method"] == m]), recovery=statistics.median([r["rec"] for r in G if r["method"] == m]), back=statistics.median([r["back"] for r in G if r["method"] == m])) for m in sorted(set(r["method"] for r in G))}
    res["groups"][gname] = out
# ---- incremental validity: logistic regression, leave-one-domain-out over the new conditions with kl and drift
def logreg(X, y, l2=0.1, iters=3000, lr=0.1):
    d = len(X[0]); w = [0.0] * d; b = 0.0
    for _ in range(iters):
        gw = [0.0] * d; gb = 0.0
        for x, t in zip(X, y):
            z = b + sum(wi * xi for wi, xi in zip(w, x)); p = 1 / (1 + math.exp(-max(min(z, 30), -30))); e = p - t; gb += e
            for j in range(d): gw[j] += e * x[j]
        n = len(X); w = [wi - lr * (g / n + l2 * wi / n) for wi, g in zip(w, gw)]; b -= lr * gb / n
    return w, b
def predict(w, b, x): z = b + sum(wi * xi for wi, xi in zip(w, x)); return 1 / (1 + math.exp(-max(min(z, 30), -30)))
def standardise(cols, X):
    mu = [statistics.mean(c) for c in zip(*X)]; sd = [max(statistics.pstdev(c), 1e-6) for c in zip(*X)]; return [[(v - m) / s for v, m, s in zip(x, mu, sd)] for x in X], (mu, sd)
IV = [r for r in rows if r["kl"] is not None and r["drift"] is not None]; res["incremental"] = {}
if len(IV) >= 20 and len(set(r["domain"] for r in IV)) >= 2:
    labels = [r["rec"] >= 0.8 for r in IV]
    for fname, feats in (("base_rise_kl_drift", ("rise", "kl", "drift")), ("plus_wdd", ("rise", "kl", "drift", "wdd")), ("plus_diff", ("rise", "kl", "drift", "diff")), ("plus_probe", ("rise", "kl", "drift", "probe")), ("wdd_alone", ("wdd",))):
        correct = 0; ll = 0.0
        for dom in sorted(set(r["domain"] for r in IV)):
            tr = [(i, r) for i, r in enumerate(IV) if r["domain"] != dom]; te = [(i, r) for i, r in enumerate(IV) if r["domain"] == dom]
            Xtr = [[r[f] for f in feats] for _, r in tr]; Xs, (mu, sd) = standardise(feats, Xtr); w, b = logreg(Xs, [float(labels[i]) for i, _ in tr])
            for i, r in te:
                x = [(r[f] - m) / s for f, m, s in zip(feats, mu, sd)]; p = predict(w, b, x); correct += (p > 0.5) == labels[i]; ll += math.log(max(p if labels[i] else 1 - p, 1e-9))
        res["incremental"][fname] = dict(lodo_accuracy=correct / len(IV), mean_loglik=ll / len(IV), n=len(IV), domains=len(set(r["domain"] for r in IV)))
# ---- the silencing test at +2
SILSETS = ("wdd_forget", "diff_selected", "ratio_selected", "random"); res["silencing"] = {}
for gname, sel in (("heldout_domains_160m", lambda r: r["model"] == "pythia160" and r["domain"] in ("stackexchange", "wiki", "uspto")), ("all_160m", lambda r: r["model"] == "pythia160"), ("qwen05", lambda r: r["model"] == "qwen05"), ("pythia410", lambda r: r["model"] == "pythia410")):
    S = [s for s in sil_rows if sel(s) and all(k in s["sil"] for k in SILSETS)]
    if not S: continue
    delay = {k: [min(s["sil"][k]["back"], 101) - min(s["back_alone"], 101) for s in S] for k in SILSETS}
    res["silencing"][gname] = dict(n=len(S), median_delay={k: statistics.median(v) for k, v in delay.items()}, wdd_beats_diff=statistics.mean([d_w > d_d for d_w, d_d in zip(delay["wdd_forget"], delay["diff_selected"])]), wdd_beats_random=statistics.mean([d_w > d_r for d_w, d_r in zip(delay["wdd_forget"], delay["random"])]), by_method={m: {k: statistics.median([min(s["sil"][k]["back"], 101) - min(s["back_alone"], 101) for s in S if s["method"] == m]) for k in SILSETS} for m in sorted(set(s["method"] for s in S))})
# ---- summary and verdicts
g = res["groups"]; acc = lambda gn, p, lab="recovery_ge_0.8": (g[gn]["by_criterion"][lab]["auditors"][p]["fixed_rule_accuracy"] if gn in g else None)
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"
parts = [f"{len(rows)} new conditions"]
for gn in GROUPS:
    if gn in g: parts.append(f"{gn} (n {g[gn]['n']}, shallow {g[gn]['by_criterion']['recovery_ge_0.8']['n_shallow']}, majority {f2(g[gn]['by_criterion']['recovery_ge_0.8']['majority_accuracy'])}): fixed-rule accuracy wdd {f2(acc(gn, 'wdd'))}, diff {f2(acc(gn, 'diff'))}, ratio {f2(acc(gn, 'ratio'))}, probe {f2(acc(gn, 'probe'))}, rise {f2(acc(gn, 'rise'))}; AUC wdd {f2(g[gn]['by_criterion']['recovery_ge_0.8']['auditors']['wdd']['auc'])}, diff {f2(g[gn]['by_criterion']['recovery_ge_0.8']['auditors']['diff']['auc'])}, probe {f2(g[gn]['by_criterion']['recovery_ge_0.8']['auditors']['probe']['auc'])}")
if res["incremental"]: parts.append("incremental (leave-one-domain-out accuracy / mean log-lik): " + ", ".join(f"{k} {f2(v['lodo_accuracy'])} / {v['mean_loglik']:.3f}" for k, v in res["incremental"].items()))
for gn, v in res["silencing"].items(): parts.append(f"silencing at +2 ({gn}, n {v['n']}): median extra steps back " + ", ".join(f"{k} {d:+.0f}" for k, d in v["median_delay"].items()) + f"; wdd beats diff in {f2(v['wdd_beats_diff'])}, random in {f2(v['wdd_beats_random'])}")
summ = "held-out test: " + "; ".join(parts); res["summary"] = summ; res["_exp"] = "e617b_heldout_analysis" + ("_smoke" if SMOKE else ""); res["_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
json.dump(res, open(os.path.join(RES, res["_exp"] + ".json"), "w"), indent=1)
if not SMOKE: open(os.path.join(RES, "FINDINGS_box8.log"), "a").write(f"{res['_time']} e617b_heldout_analysis: {summ}\n")
print(summ)
