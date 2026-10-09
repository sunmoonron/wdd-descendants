"""e618a (session 117): the statistics of the held-out test, on the laptop. Over the 63 training conditions (e602, e610) and
the 255 held-out conditions (e617): cluster-bootstrap intervals (resampling jobs, each job being the three targets of one
model, domain, method and seed) for the AUC of each auditor per held-out group and for the gap between the difference-selected
set and the writers; a cluster-bootstrap interval for the incremental gain of the writers, and of the difference-selected
set, over a logistic model of the forget rise, the output KL and the drift (leave-one-domain-out); a regression of the
recovery magnitude (leave-one-domain-out R squared) instead of the shallow-or-not label; and, exploratory, a within-model
normalised score (the forget writers' ratio over the retain writers' ratio) with its cut tuned on the training conditions
and applied to the held-out groups. Pre-registered S1, S2 in e618_prereg.json."""
import json, glob, os, datetime, numpy as np, collections
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(root, "results"); rng = np.random.default_rng(0)
def load(pattern, new):
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        if "smoke" in f: continue
        r = json.load(open(f))
        for k, c in r["conditions"].items():
            if c["method"] == "drift" or c["recovery"] is None: continue
            a = c["audit"]["forget"]; sr = c["set_ratios"]
            if a["act_ratio"] is None: continue
            rows.append(dict(job=f"{r['model']}|{r['domain']}|{c['method']}|{r['seed']}", model=r["model"], domain=r["domain"], method=c["method"], seed=r["seed"], target=c["target"], wdd=a["act_ratio"], wdd_retain=c["audit"]["retain"]["act_ratio"], diff=sr["diff_selected"], ratio=sr["ratio_selected"], probe=c["probe"]["forget_acc"], rise=c["rise_forget"], kl=c.get("kl"), drift=c.get("drift"), rec=c["recovery"], new=new))
    return rows
TR = load("e602_baseline_*.json", False); NEW = load("e617_heldout_*.json", True); ALL = TR + NEW
KNOWN = ("ga", "gd", "npo", "rmu", "ssd"); NEWM = ("tv", "scrub")
GROUPS = {"heldout_domains_160m": lambda r: r["model"] == "pythia160" and r["domain"] in ("stackexchange", "wiki", "uspto") and r["method"] in KNOWN, "heldout_methods": lambda r: r["method"] in NEWM and r["model"] != "qwen05", "heldout_architecture_qwen05": lambda r: r["model"] == "qwen05", "heldout_410m_stackexchange": lambda r: r["model"] == "pythia410", "extra_domains_160m": lambda r: r["model"] == "pythia160" and r["domain"] in ("freelaw", "dm_math"), "all_new": lambda r: True}
def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, bool); npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0: return np.nan
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); sv = s[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return (ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)
def boot_jobs(rows, B, fn):
    jobs = sorted(set(r["job"] for r in rows)); byjob = collections.defaultdict(list)
    for r in rows: byjob[r["job"]].append(r)
    out = []
    for _ in range(B):
        pick = rng.choice(len(jobs), len(jobs), replace=True); samp = [r for i in pick for r in byjob[jobs[i]]]; out.append(fn(samp))
    return np.array(out, float)
ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
res = dict(n_train=len(TR), n_new=len(NEW), groups={}, _exp="e618a_stats")
PRED = ("wdd", "diff", "ratio", "probe", "rise")
for g, sel in GROUPS.items():
    G = [r for r in NEW if sel(r)]
    if len(G) < 6: continue
    y = [r["rec"] >= 0.8 for r in G]; point = {p: float(auc([r[p] for r in G], y)) for p in PRED}
    bs = boot_jobs(G, 1000, lambda S: [auc([r[p] for r in S], [r["rec"] >= 0.8 for r in S]) for p in PRED] + [auc([r["diff"] for r in S], [r["rec"] >= 0.8 for r in S]) - auc([r["wdd"] for r in S], [r["rec"] >= 0.8 for r in S])])
    res["groups"][g] = dict(n=len(G), n_jobs=len(set(r["job"] for r in G)), auc={p: dict(point=point[p], ci=ci(bs[:, i])) for i, p in enumerate(PRED)}, auc_diff_minus_wdd=dict(point=point["diff"] - point["wdd"], ci=ci(bs[:, len(PRED)]), p_gap_le_0=float(np.mean(bs[:, len(PRED)] <= 0))))
# ---- incremental validity with a cluster bootstrap
IV = [r for r in NEW if r["kl"] is not None and r["drift"] is not None]
def logreg_fit(X, y, l2=0.1, iters=400, lr=0.5):
    w = np.zeros(X.shape[1]); b = 0.0; n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30))); e = p - y; w -= lr * (X.T @ e / n + l2 * w / n); b -= lr * e.mean()
    return w, b
def lodo_acc(rows, feats):
    doms = sorted(set(r["domain"] for r in rows)); correct = 0
    for d in doms:
        tr = [r for r in rows if r["domain"] != d]; te = [r for r in rows if r["domain"] == d]
        if not tr or not te: continue
        Xtr = np.array([[r[f] for f in feats] for r in tr]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; w, b = logreg_fit((Xtr - mu) / sd, np.array([r["rec"] >= 0.8 for r in tr], float))
        Xte = (np.array([[r[f] for f in feats] for r in te]) - mu) / sd; correct += int((((Xte @ w + b) > 0) == np.array([r["rec"] >= 0.8 for r in te])).sum())
    return correct / len(rows)
FE = {"base": ("rise", "kl", "drift"), "plus_wdd": ("rise", "kl", "drift", "wdd"), "plus_diff": ("rise", "kl", "drift", "diff"), "plus_probe": ("rise", "kl", "drift", "probe"), "wdd_alone": ("wdd",), "diff_alone": ("diff",)}
point = {k: lodo_acc(IV, f) for k, f in FE.items()}
bs = boot_jobs(IV, 300, lambda S: [lodo_acc(S, FE["plus_wdd"]) - lodo_acc(S, FE["base"]), lodo_acc(S, FE["plus_diff"]) - lodo_acc(S, FE["base"]), lodo_acc(S, FE["plus_wdd"]) - lodo_acc(S, FE["plus_diff"])])
res["incremental"] = dict(n=len(IV), lodo_accuracy=point, gain_wdd=dict(point=point["plus_wdd"] - point["base"], ci=ci(bs[:, 0])), gain_diff=dict(point=point["plus_diff"] - point["base"], ci=ci(bs[:, 1])), wdd_minus_diff=dict(point=point["plus_wdd"] - point["plus_diff"], ci=ci(bs[:, 2])))
# ---- regression of the recovery magnitude, leave-one-domain-out R squared
def lodo_r2(rows, feats):
    doms = sorted(set(r["domain"] for r in rows)); pred = np.zeros(len(rows)); idx = {id(r): i for i, r in enumerate(rows)}
    for d in doms:
        tr = [r for r in rows if r["domain"] != d]; te = [r for r in rows if r["domain"] == d]
        Xtr = np.array([[r[f] for f in feats] + [1.0] for r in tr]); ytr = np.array([r["rec"] for r in tr]); w = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
        for r in te: pred[idx[id(r)]] = np.array([r[f] for f in feats] + [1.0]) @ w
    y = np.array([r["rec"] for r in rows]); return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
res["regression"] = {k: lodo_r2(IV, f) for k, f in FE.items()}
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
res["spearman_wdd_recovery"] = {g: spearman([r["wdd"] for r in NEW if sel(r)], [r["rec"] for r in NEW if sel(r)]) for g, sel in GROUPS.items() if sum(sel(r) for r in NEW) >= 6}
# ---- exploratory: a within-model normalised score, cut tuned on the training conditions
def norm_score(r): return r["wdd"] / r["wdd_retain"] if (r["wdd_retain"] is not None and r["wdd_retain"] > 0.05) else None
def tune(rows, key):
    pts = [(key(r), r["rec"] >= 0.8) for r in rows if key(r) is not None]; sc = sorted(set(s for s, _ in pts)); best = (0, None, 1)
    for i, t in enumerate(sc):
        for d in (1, -1):
            a = sum(((s > t) if d == 1 else (s <= t)) == l for s, l in pts) / len(pts)
            if a > best[0]: best = (a, (t + sc[i + 1]) / 2 if i + 1 < len(sc) else t, d)
    return best
acc_tr, cut, dirn = tune(TR, norm_score); res["normalised_score"] = dict(train_accuracy=acc_tr, cut=cut, direction=dirn, groups={})
for g, sel in GROUPS.items():
    G = [r for r in NEW if sel(r) and norm_score(r) is not None]
    if len(G) < 6: continue
    y = [r["rec"] >= 0.8 for r in G]; s = [norm_score(r) for r in G]; res["normalised_score"]["groups"][g] = dict(n=len(G), fixed_rule_accuracy=float(np.mean([((v > cut) if dirn == 1 else (v <= cut)) == l for v, l in zip(s, y)])), majority=float(max(np.mean(y), 1 - np.mean(y))), auc=float(auc(s, y)))
g_ = res["groups"]; inc = res["incremental"]; ns = res["normalised_score"]["groups"]; f2 = lambda x: f"{x:.2f}"
summ = ("held-out statistics (cluster bootstrap over jobs): AUC with 95% intervals: " + "; ".join(f"{g} (n {v['n']}, jobs {v['n_jobs']}) wdd {f2(v['auc']['wdd']['point'])} [{f2(v['auc']['wdd']['ci'][0])}, {f2(v['auc']['wdd']['ci'][1])}], diff {f2(v['auc']['diff']['point'])} [{f2(v['auc']['diff']['ci'][0])}, {f2(v['auc']['diff']['ci'][1])}], probe {f2(v['auc']['probe']['point'])}, gap diff-wdd {v['auc_diff_minus_wdd']['point']:+.2f} [{v['auc_diff_minus_wdd']['ci'][0]:+.2f}, {v['auc_diff_minus_wdd']['ci'][1]:+.2f}]" for g, v in g_.items())
        + f"; incremental (n {inc['n']}): base {f2(inc['lodo_accuracy']['base'])}, +wdd {f2(inc['lodo_accuracy']['plus_wdd'])} (gain {inc['gain_wdd']['point']:+.2f} [{inc['gain_wdd']['ci'][0]:+.2f}, {inc['gain_wdd']['ci'][1]:+.2f}]), +diff {f2(inc['lodo_accuracy']['plus_diff'])} (gain {inc['gain_diff']['point']:+.2f} [{inc['gain_diff']['ci'][0]:+.2f}, {inc['gain_diff']['ci'][1]:+.2f}]), wdd minus diff {inc['wdd_minus_diff']['point']:+.2f} [{inc['wdd_minus_diff']['ci'][0]:+.2f}, {inc['wdd_minus_diff']['ci'][1]:+.2f}], +probe {f2(inc['lodo_accuracy']['plus_probe'])}, wdd alone {f2(inc['lodo_accuracy']['wdd_alone'])}"
        + "; recovery-magnitude R2 (leave-one-domain-out): " + ", ".join(f"{k} {v:.2f}" for k, v in res["regression"].items())
        + f"; normalised score (forget over retain writers, cut {cut:.3f} {'above' if dirn == 1 else 'at or below'}, train acc {acc_tr:.2f}): " + ", ".join(f"{g} acc {v['fixed_rule_accuracy']:.2f} (majority {v['majority']:.2f}, AUC {v['auc']:.2f})" for g, v in ns.items()))
res["summary"] = summ; res["_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
json.dump(res, open(os.path.join(RES, "e618a_stats.json"), "w"), indent=1); open(os.path.join(RES, "FINDINGS_box8.log"), "a").write(f"{res['_time']} e618a_stats: {summ}\n"); print(summ)
