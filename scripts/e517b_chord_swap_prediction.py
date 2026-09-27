"""e517b: the counterfactual chord, the prediction half. From e517's records at thirteen Pythia checkpoints: at an
origin checkpoint, the rows that are not words are ranked by their largest projection over the floor under each
counterfactual state (real; fake chord; permuted chord; no chord; own write removed), and the ranking is scored by
the AUC with which it picks the rows that are words at a later checkpoint, as e516b did for the real state. If the
chord carries rows across the floor, destroying it (fake, permuted, none) should destroy the prediction while
removing only the row's own write should leave most of it. Also the decomposition of the entrants' rise: for the
rows that enter between the origin and the horizon, the median of their largest real projection and of the floor at
that position, at the origin and at the horizon, so that the change of the ratio splits into the projection's rise
and the floor's move.
Pre-registered (honest guesses), block 12, entry at the next checkpoint, origins 2000-16000:
- the fake chord's AUC is at least 0.15 below the real state's at every origin (0.6);
- with the own write removed the AUC keeps at least 0.8 of the real state's excess over 0.5 (0.6);
- the permuted chord falls less than the fake chord (0.5);
- for the entrants, the projection's rise accounts for more than two thirds of the ratio's rise in log terms (0.5).
Arguments: name."""
import sys, os, json as _json, time, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; CDIR = f"/workspace/wdd/cache/e517_{name}"; STEPS = ["step0", "step64", "step256", "step512", "step1000", "step2000", "step3000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
LAB = [0, 64, 256, 512, 1000, 2000, 3000, 4000, 8000, 16000, 32000, 64000, 143000]; NWORD = 256; ORIGINS = [512, 1000, 2000, 3000, 4000, 8000, 16000, 32000]; V = ["real", "cross", "permuted", "fake", "nochord"]
t0 = time.time()
while not all(os.path.exists(f"{CDIR}/{s}.pt") for s in STEPS):
    if time.time() - t0 > 5400: raise SystemExit("records missing: " + ", ".join(s for s in STEPS if not os.path.exists(f"{CDIR}/{s}.pt")))
    time.sleep(30)
time.sleep(10)
data = {s: torch.load(f"{CDIR}/{s}.pt") for s in STEPS}
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
res = dict(model=name, steps=LAB, origins=ORIGINS, variants=V, by_block={})
for b in data["main"]:
    W = {}
    for s in STEPS:
        u = data[s][b]["usage"]; used = torch.nonzero(u > 0)[:, 0]; W[s] = set(used[u[used].argsort(descending=True)[:NWORD]].tolist())
    R = data["main"][b]["usage"].numel(); entry, decomp = {}, {}
    for o_ in ORIGINS:
        so = STEPS[LAB.index(o_)]; d = data[so][b]; inW = torch.zeros(R, dtype=torch.bool); inW[list(W[so])] = True; cand = torch.nonzero(~inW)[:, 0]
        qs = {v: d["variants"][v]["max_over_floor"].float() for v in V}; qs["magnitude"] = d["magnitude"].float(); qs["usage"] = d["usage"].float()
        entry[o_], decomp[o_] = {}, {}
        for j in range(LAB.index(o_) + 1, len(STEPS)):
            s1 = STEPS[j]; in1 = torch.zeros(R, dtype=torch.bool); in1[list(W[s1])] = True; ent = cand[in1[cand]]; non = cand[~in1[cand]]
            entry[o_][LAB[j]] = dict(n_entrants=int(ent.numel()), auc={k: auc(qs[k][ent], qs[k][non]) for k in qs})
            d1 = data[s1][b]; r0, r1 = d["variants"]["real"], d1["variants"]["real"]
            if ent.numel() >= 5:
                p0, f0, p1, f1 = r0["max_projection"][ent].float(), r0["floor_at_max"][ent].float(), r1["max_projection"][ent].float(), r1["floor_at_max"][ent].float()
                dl_ratio = float((torch.log(p1 / f1) - torch.log(p0 / f0)).median()); dl_p = float((torch.log(p1) - torch.log(p0)).median()); dl_f = float((torch.log(f1) - torch.log(f0)).median())
                decomp[o_][LAB[j]] = dict(entrants_projection=[float(p0.median()), float(p1.median())], entrants_floor=[float(f0.median()), float(f1.median())], entrants_ratio=[float((p0 / f0).median()), float((p1 / f1).median())], non_entrants_ratio=[float((r0["max_projection"][non] / r0["floor_at_max"][non]).median()), float((r1["max_projection"][non] / r1["floor_at_max"][non]).median())],
                                           dlog_ratio=dl_ratio, dlog_projection=dl_p, dlog_floor=dl_f, projection_share_of_rise=(dl_p / dl_ratio) if abs(dl_ratio) > 1e-6 else None)
    res["by_block"][b] = dict(entry=entry, decomposition=decomp)
    nxt = lambda o_: LAB[LAB.index(o_) + 1]; fmt = lambda x: "n/a" if x is None else f"{x:.2f}"
    log(f"{name} block {b}: entry at the next checkpoint, AUC by variant: " + " | ".join(f"from {o_}: " + ", ".join(f"{k} {fmt(entry[o_][nxt(o_)]['auc'][k])}" for k in list(V) + ["magnitude", "usage"]) for o_ in ORIGINS))
    log(f"{name} block {b}: entry by the end, AUC by variant: " + " | ".join(f"from {o_}: " + ", ".join(f"{k} {fmt(entry[o_][143000]['auc'][k])}" for k in list(V) + ["magnitude", "usage"]) for o_ in ORIGINS))
    log(f"{name} block {b}: the entrants' rise to the next checkpoint, median ratio origin to horizon, projection and floor: " + " | ".join(f"from {o_}: ratio {dc['entrants_ratio'][0]:.2f} to {dc['entrants_ratio'][1]:.2f} (non-entrants {dc['non_entrants_ratio'][0]:.2f} to {dc['non_entrants_ratio'][1]:.2f}), projection {dc['entrants_projection'][0]:.3f} to {dc['entrants_projection'][1]:.3f}, floor {dc['entrants_floor'][0]:.3f} to {dc['entrants_floor'][1]:.3f}, projection's share of the rise {fmt(dc['projection_share_of_rise'])}" for o_ in ORIGINS for dc in [decomp[o_].get(nxt(o_))] if dc))
B = res["by_block"][12 if 12 in res["by_block"] else list(res["by_block"])[-1]]; E = B["entry"]; nx = lambda o_: LAB[LAB.index(o_) + 1]; gg = lambda o_, k: (E[o_][nx(o_)]["auc"][k] or 0.5)
res["checks"] = dict(fake_falls_0_15=all(gg(o_, "real") - gg(o_, "fake") >= 0.15 for o_ in (2000, 3000, 4000, 8000, 16000)), cross_keeps_0_8=all((gg(o_, "cross") - 0.5) >= 0.8 * (gg(o_, "real") - 0.5) for o_ in (2000, 3000, 4000, 8000, 16000)),
                     permuted_falls_less_than_fake=all(gg(o_, "permuted") > gg(o_, "fake") for o_ in (2000, 3000, 4000, 8000, 16000)),
                     projection_carries_two_thirds=all((B["decomposition"][o_].get(nx(o_)) or {}).get("projection_share_of_rise") is not None and B["decomposition"][o_][nx(o_)]["projection_share_of_rise"] > 2 / 3 for o_ in (2000, 3000, 4000, 8000, 16000)))
summ = (f"{name}: " + " || ".join(f"block {b}: entry at the next checkpoint, AUC real/own removed/permuted/fake/no chord/magnitude/usage " + " | ".join(f"from {o_}: " + "/".join(("n/a" if o['entry'][o_][nx(o_)]['auc'][k] is None else f"{o['entry'][o_][nx(o_)]['auc'][k]:.2f}") for k in list(V) + ["magnitude", "usage"]) for o_ in ORIGINS) for b, o in res["by_block"].items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e517b_chordswap_{name}", res, summ)
