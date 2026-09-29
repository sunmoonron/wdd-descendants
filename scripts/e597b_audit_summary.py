"""e597b (session 111): the runs of e597 summarised. Per method (over targets, domains and models): the forget words'
still-writing share, their neurons' activation ratio, their rows' cosine and norm-matched change, the retain words'
still-writing share, the probe's forget accuracy, the KL split, and the three attacks (20-step relearning recovery,
steps back, benign recovery, 4-bit and 8-bit quantisation recovery). Across the training-method conditions (drift
excluded), the Spearman correlation of each candidate predictor with each attack outcome, so that the audit's number
can be compared with the probe and with the loss rise; the same within each model. A3 is decided on the 20-step
recovery over all training-method conditions."""
from s101_common import *
import glob
runs = [json.load(open(f)) for f in sorted(glob.glob("/workspace/wdd/results/e597_audit_*.json"))]
rows = []
for r in runs:
    for k, c in r["conditions"].items():
        a, rt = c["audit"]["forget"], c["audit"]["retain"]; rr = c["param_change"]["row_rel_change"]
        rows.append(dict(model=r["model"], domain=r["domain"], method=c["method"], target=c["target"], steps=c["steps"], rise_forget=c["rise_forget"], rise_retain=c["rise_retain"], still_writing=a["share_still_writing"], act_ratio=a["act_ratio"], act_halved=a["share_act_halved"], row_cos=a["row_cos"], norm_matched=(rr["forget_over_norm_matched"] if rr["forget_over_norm_matched"] < 100 else None), retain_norm_matched=(rr["retain_over_norm_matched"] if rr["retain_over_norm_matched"] < 100 else None), retain_writing=rt["share_still_writing"], retain_act_ratio=rt["act_ratio"], retain_row_cos=rt["row_cos"],
                         probe_acc=c["probe"]["forget_acc"], probe0=r["probe0"]["forget_acc"], early_share=c["early_share"], param_total=c["param_change"]["total"], recovery=c["recovery"], steps_back=c["steps_to_relearn"], recovery_benign=c["recovery_benign"], recovery_q4=c["recovery_q4"], recovery_q8=c["recovery_q8"], dampened=c.get("dampened_share")))
log(f"{len(runs)} runs, {len(rows)} conditions")
def spear(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 4: return None
    x, y = torch.tensor([p[0] for p in pairs], dtype=torch.float64), torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    if x.std() == 0 or y.std() == 0: return None
    return float(torch.corrcoef(torch.stack([x.argsort().argsort().double(), y.argsort().argsort().double()]))[0, 1])
PRED = ("still_writing", "act_ratio", "row_cos", "norm_matched", "probe_acc", "rise_forget", "rise_retain", "param_total", "early_share"); OUT = ("recovery", "steps_back", "recovery_benign", "recovery_q4", "recovery_q8")
train = [r for r in rows if r["method"] != "drift"]
def table(sub): return {o: {p: spear([r[p] for r in sub], [r[o] for r in sub]) for p in PRED} for o in OUT}
res = dict(n_runs=len(runs), n_conditions=len(rows), rows=rows, spearman_all=table(train), spearman_by_model={mname: table([r for r in train if r["model"] == mname]) for mname in sorted(set(r["model"] for r in train))}, spearman_no_ssd=table([r for r in train if r["method"] != "ssd"]), by_method={})
for mth in ("ga", "gd", "npo", "rmu", "ssd", "drift"):
    sub = [r for r in rows if r["method"] == mth]
    if not sub: continue
    res["by_method"][mth] = {k: (med([r[k] for r in sub if r[k] is not None]) if any(r[k] is not None for r in sub) else None) for k in ("still_writing", "act_ratio", "act_halved", "row_cos", "norm_matched", "retain_norm_matched", "retain_writing", "retain_act_ratio", "probe_acc", "early_share", "param_total", "recovery", "steps_back", "recovery_benign", "recovery_q4", "recovery_q8", "rise_retain")}; res["by_method"][mth]["n"] = len(sub)
    b = res["by_method"][mth]; log(f"{mth} (n={len(sub)}): still writing {b['still_writing']}, activation ratio {b['act_ratio']}, row cos {b['row_cos']}, norm-matched change {b['norm_matched']} (retain words {b['retain_norm_matched']}), retain writing {b['retain_writing']}, probe {b['probe_acc']}, KL early {b['early_share']}, recovery {b['recovery']}, steps back {b['steps_back']}, benign {b['recovery_benign']}, q4 {b['recovery_q4']}, q8 {b['recovery_q8']}, retain rise {b['rise_retain']}")
S = res["spearman_all"]
for o in OUT: log(f"Spearman with {o} over {len(train)} training-method conditions: " + ", ".join(f"{p} {S[o][p] if S[o][p] is None else round(S[o][p], 2)}" for p in PRED))
a3 = (abs(S["recovery"]["still_writing"] or 0) > abs(S["recovery"]["probe_acc"] or 0))
summ = (f"audit summary over {len(runs)} runs / {len(rows)} conditions: Spearman with the 20-step recovery: still-writing {S['recovery']['still_writing']}, activation ratio {S['recovery']['act_ratio']}, probe accuracy {S['recovery']['probe_acc']}, forget rise {S['recovery']['rise_forget']}, parameter change {S['recovery']['param_total']}; with benign recovery: still-writing {S['recovery_benign']['still_writing']}, probe {S['recovery_benign']['probe_acc']}; with 4-bit recovery: still-writing {S['recovery_q4']['still_writing']}, probe {S['recovery_q4']['probe_acc']}, parameter change {S['recovery_q4']['param_total']}; A3 {'holds' if a3 else 'fails'}; "
        + "; ".join(f"{m}: writing {b['still_writing']}, act {b['act_ratio']}, cos {b['row_cos']}, nm {b['norm_matched']}, probe {b['probe_acc']}, rec {b['recovery']}, benign {b['recovery_benign']}, q4 {b['recovery_q4']}" for m, b in res["by_method"].items()))
log(summ); record("e597b_audit_summary", res, summ)
