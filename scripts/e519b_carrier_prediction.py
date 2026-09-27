"""e519b: who supplies the projection, the prediction half. From e519's records at thirteen Pythia checkpoints: at an
origin, the rows that are not words are ranked by the largest projection over the floor that each part of the state
supplies alone (the token embedding; attention's output; the row's own write; the other largest writes; the crowd of
small writes; the centring) and by three combinations (attention with the embedding; all the MLP writes; the chord),
and each ranking is scored by the AUC with which it picks the rows that are words at a later checkpoint, as e516b
and e517b did for the real state. Also, at the position of each row's real maximum, the parts' shares of the real
projection, as medians for the entrants, the non-entrants and the rows already words, and the share of entrants
that are among the 64 largest writers at that position.
Pre-registered (honest guesses), block 12, entry at the next checkpoint, origins 2000-16000:
- attention's projection alone predicts entry at 0.8 or above (0.5);
- attention with the embedding comes within 0.05 of the real state (0.5);
- the crowd alone stays below 0.7 (0.5);
- at the entrants' maxima attention's share of the projection exceeds the MLP writes' share (own, other largest and
  crowd together) (0.5).
Arguments: name."""
import sys, os, json as _json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; CDIR = f"/workspace/wdd/cache/e519_{name}"; STEPS = ["step0", "step64", "step256", "step512", "step1000", "step2000", "step3000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
LAB = [0, 64, 256, 512, 1000, 2000, 3000, 4000, 8000, 16000, 32000, 64000, 143000]; NWORD = 256; ORIGINS = [512, 1000, 2000, 3000, 4000, 8000, 16000, 32000]
PARTS = ["embedding", "attention", "own", "chord_others", "crowd"]; V = ["real", "attention", "embedding", "attention_embedding", "crowd", "chord", "mlp_all", "own", "chord_others"]
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
    R = data["main"][b]["usage"].numel(); entry, shares = {}, {}
    for o_ in ORIGINS:
        so = STEPS[LAB.index(o_)]; d = data[so][b]; inW = torch.zeros(R, dtype=torch.bool); inW[list(W[so])] = True; cand = torch.nonzero(~inW)[:, 0]; words = torch.nonzero(inW)[:, 0]
        qs = {k: d["max_over_floor"][k].float() for k in V}; qs["magnitude"] = d["magnitude"].float(); qs["usage"] = d["usage"].float()
        entry[o_], shares[o_] = {}, {}
        for j in range(LAB.index(o_) + 1, len(STEPS)):
            s1 = STEPS[j]; in1 = torch.zeros(R, dtype=torch.bool); in1[list(W[s1])] = True; ent = cand[in1[cand]]; non = cand[~in1[cand]]
            entry[o_][LAB[j]] = dict(n_entrants=int(ent.numel()), auc={k: auc(qs[k][ent], qs[k][non]) for k in qs})
            if j == LAB.index(o_) + 1:
                sh = d["share_at_max"]; shares[o_] = {grp: {k: float(sh[k][ix].float().median()) for k in PARTS} | dict(n=int(ix.numel()), writer_at_max=float(d["is_writer_at_max"][ix].float().mean())) for grp, ix in (("entrants", ent), ("non_entrants", non), ("words", words)) if ix.numel() >= 5}
    res["by_block"][b] = dict(entry=entry, shares_at_max=shares)
    nxt = lambda o_: LAB[LAB.index(o_) + 1]; fm = lambda x: "n/a" if x is None else f"{x:.2f}"
    log(f"{name} block {b}: entry at the next checkpoint, AUC by part: " + " | ".join(f"from {o_}: " + ", ".join(f"{k} {fm(entry[o_][nxt(o_)]['auc'][k])}" for k in V + ["magnitude", "usage"]) for o_ in ORIGINS))
    log(f"{name} block {b}: entry by the end, AUC by part: " + " | ".join(f"from {o_}: " + ", ".join(f"{k} {fm(entry[o_][143000]['auc'][k])}" for k in V + ["magnitude", "usage"]) for o_ in ORIGINS))
    log(f"{name} block {b}: parts' shares of the real projection at the row's maximum (medians), entrants / non-entrants / words: " + " | ".join(f"from {o_}: " + ", ".join(f"{k} {fm(shares[o_].get('entrants', {}).get(k))}/{fm(shares[o_].get('non_entrants', {}).get(k))}/{fm(shares[o_].get('words', {}).get(k))}" for k in PARTS) + f"; entrants writers at their maximum {fm(shares[o_].get('entrants', {}).get('writer_at_max'))}" for o_ in ORIGINS))
B = res["by_block"][12 if 12 in res["by_block"] else list(res["by_block"])[-1]]; E = B["entry"]; nx = lambda o_: LAB[LAB.index(o_) + 1]; gg = lambda o_, k: (E[o_][nx(o_)]["auc"][k] or 0.5)
mlp_share = lambda o_: sum(B["shares_at_max"][o_]["entrants"][k] for k in ("own", "chord_others", "crowd")) if "entrants" in B["shares_at_max"][o_] else None
res["checks"] = dict(attention_alone_over_0_8=all(gg(o_, "attention") >= 0.8 for o_ in (2000, 3000, 4000, 8000, 16000)), attention_embedding_within_0_05=all(gg(o_, "real") - gg(o_, "attention_embedding") <= 0.05 for o_ in (2000, 3000, 4000, 8000, 16000)),
                     crowd_alone_under_0_7=all(gg(o_, "crowd") < 0.7 for o_ in (2000, 3000, 4000, 8000, 16000)), entrants_attention_share_over_mlp=all(mlp_share(o_) is not None and B["shares_at_max"][o_]["entrants"]["attention"] > mlp_share(o_) for o_ in (2000, 3000, 4000, 8000, 16000)))
summ = (f"{name}: " + " || ".join(f"block {b}: entry at the next checkpoint, AUC real/attention/embedding/attention+embedding/crowd/chord/all MLP/own " + " | ".join(f"from {o_}: " + "/".join(("n/a" if o['entry'][o_][nx(o_)]['auc'][k] is None else f"{o['entry'][o_][nx(o_)]['auc'][k]:.2f}") for k in ("real", "attention", "embedding", "attention_embedding", "crowd", "chord", "mlp_all", "own")) for o_ in ORIGINS) for b, o in res["by_block"].items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e519b_carrier_{name}", res, summ)
