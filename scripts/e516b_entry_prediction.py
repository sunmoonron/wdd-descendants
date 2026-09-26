"""e516b: what governs entry into the vocabulary, the prediction half. From e516's records at thirteen Pythia
checkpoints: at an origin checkpoint t, every MLP row that is not a word (not among the 256 most used) is ranked by
each candidate quantity computed at t, and the ranking is scored by how well it picks the rows that are words at a
later checkpoint t' (AUC of entrants over non-entrants); no threshold is fitted and nothing from t' is used. The
same for exit: among the words at t, the rows still words at t' against those that left (AUC of stayers over
leavers). Candidates at t: usage (the rows already used a little; the baseline to beat), mean absolute write,
selectivity (one minus activity), write kurtosis, prominence when active, the share of positions at which the row's
own write clears the floor, the share at which its projection clears the floor, the largest projection over the
floor, and the product of magnitude and selectivity. Origins 512 to 32000, horizons to every later checkpoint.
Pre-registered (honest guesses), at block 12:
- the share of positions at which the row's own write clears the floor predicts entry at the next checkpoint better
  than mean absolute write at every origin from 1000 to 16000 (0.5);
- usage is the best single predictor at every origin and horizon (0.6);
- magnitude times selectivity beats either alone at the next checkpoint for origins 1000 to 8000 (0.5).
Arguments: name."""
import sys, os, json as _json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; CDIR = f"/workspace/wdd/cache/e516_{name}"; STEPS = ["step0", "step64", "step256", "step512", "step1000", "step2000", "step3000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
LAB = [0, 64, 256, 512, 1000, 2000, 3000, 4000, 8000, 16000, 32000, 64000, 143000]; NWORD = 256; ORIGINS = [512, 1000, 2000, 3000, 4000, 8000, 16000, 32000]
Q = ["usage", "magnitude", "selectivity", "kurtosis", "prominence_active", "write_over_floor_rate", "projection_over_floor_rate", "max_projection_over_floor", "magnitude_x_selectivity"]
t0 = time.time()
while not all(os.path.exists(f"{CDIR}/{s}.pt") for s in STEPS):
    if time.time() - t0 > 3600: raise SystemExit("records missing: " + ", ".join(s for s in STEPS if not os.path.exists(f"{CDIR}/{s}.pt")))
    time.sleep(30)
time.sleep(10)
data = {s: torch.load(f"{CDIR}/{s}.pt") for s in STEPS}
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def quantities(d):
    q = {k: d[k].float() for k in ("usage", "magnitude", "kurtosis", "prominence_active", "write_over_floor_rate", "projection_over_floor_rate", "max_projection_over_floor")}
    q["selectivity"] = 1 - d["activity"].float(); q["magnitude_x_selectivity"] = q["magnitude"] * q["selectivity"]; return q
res = dict(model=name, steps=LAB, origins=ORIGINS, quantities=Q, by_block={})
for b in data["main"]:
    W = {}
    for s in STEPS:
        u = data[s][b]["usage"]; used = torch.nonzero(u > 0)[:, 0]; W[s] = set(used[u[used].argsort(descending=True)[:NWORD]].tolist())
    R = data["main"][b]["usage"].numel(); entry, exit_ = {}, {}
    for o_ in ORIGINS:
        so = STEPS[LAB.index(o_)]; q = quantities(data[so][b]); inW = torch.zeros(R, dtype=torch.bool); inW[list(W[so])] = True; cand = torch.nonzero(~inW)[:, 0]; words = torch.nonzero(inW)[:, 0]
        entry[o_], exit_[o_] = {}, {}
        for j in range(LAB.index(o_) + 1, len(STEPS)):
            s1 = STEPS[j]; in1 = torch.zeros(R, dtype=torch.bool); in1[list(W[s1])] = True
            ent = cand[in1[cand]]; non = cand[~in1[cand]]; stay = words[in1[words]]; leave = words[~in1[words]]
            entry[o_][LAB[j]] = dict(n_entrants=int(ent.numel()), auc={k: auc(q[k][ent], q[k][non]) for k in Q})
            exit_[o_][LAB[j]] = dict(n_leavers=int(leave.numel()), n_stayers=int(stay.numel()), auc={k: auc(q[k][stay], q[k][leave]) for k in Q})
    res["by_block"][b] = dict(entry=entry, exit=exit_)
    nxt = lambda o_: LAB[LAB.index(o_) + 1]
    log(f"{name} block {b}: entry at the next checkpoint, AUC by quantity: " + " | ".join(f"from {o_}: " + ", ".join(f"{k} {entry[o_][nxt(o_)]['auc'][k]:.2f}" for k in Q if entry[o_][nxt(o_)]['auc'][k] is not None) for o_ in ORIGINS))
    log(f"{name} block {b}: entry by the end, AUC by quantity: " + " | ".join(f"from {o_}: " + ", ".join(f"{k} {entry[o_][143000]['auc'][k]:.2f}" for k in Q if entry[o_][143000]['auc'][k] is not None) for o_ in ORIGINS))
    log(f"{name} block {b}: staying at the next checkpoint (stayers over leavers), AUC by quantity: " + " | ".join(f"from {o_}: " + ", ".join(f"{k} {exit_[o_][nxt(o_)]['auc'][k]:.2f}" for k in Q if exit_[o_][nxt(o_)]['auc'][k] is not None) for o_ in ORIGINS))
B = res["by_block"][12 if 12 in res["by_block"] else list(res["by_block"])[-1]]; E = B["entry"]; nx = lambda o_: LAB[LAB.index(o_) + 1]
g = lambda o_, k, h=None: E[o_][h if h else nx(o_)]["auc"][k] or 0
res["checks"] = dict(write_over_floor_beats_magnitude_next=all(g(o_, "write_over_floor_rate") > g(o_, "magnitude") for o_ in (1000, 2000, 3000, 4000, 8000, 16000)),
                     usage_best_everywhere=all(g(o_, "usage", h) >= max(g(o_, k, h) for k in Q) for o_ in ORIGINS for h in E[o_]),
                     product_beats_both_next=all(g(o_, "magnitude_x_selectivity") > max(g(o_, "magnitude"), g(o_, "selectivity")) for o_ in (1000, 2000, 3000, 4000, 8000)))
summ = (f"{name}: " + " || ".join(f"block {b}: entry at the next checkpoint / by the end, AUC " + " | ".join(f"from {o_}: " + ", ".join(f"{k} {o['entry'][o_][nx(o_)]['auc'][k]:.2f}/{o['entry'][o_][143000]['auc'][k]:.2f}" for k in Q if o['entry'][o_][nx(o_)]['auc'][k] is not None and o['entry'][o_][143000]['auc'][k] is not None) for o_ in (1000, 4000, 16000)) for b, o in res["by_block"].items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e516b_entry_{name}", res, summ)
