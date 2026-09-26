"""e515b: word tracks, the analysis half. Reads e515's per-checkpoint records (every MLP row's usage as a native
word, mean absolute write, activity and cosine with its end direction, at 13 Pythia checkpoints from initialisation
to the end, blocks 6 and 12) and asks whether the vocabulary is replaced or whether rows cross a threshold.
Per block: the word set at each checkpoint is the 256 rows most used as native words (rows with any usage, if fewer);
the null set is the 256 rows of largest mean absolute write. Reported: the retention of the word set into the next
checkpoint, the same for the null set, the Jaccard of consecutive word sets, the Spearman of usage across consecutive
checkpoints over the rows in either set, the overlap of each checkpoint's word set with the end's and with the null
set; the transition of the early word sets (steps 512, 1000, 2000, 4000) into every later checkpoint; and the
hysteresis test: for each consecutive pair, the rows entering the word set against the rows leaving it, compared in
the state in which each is a word (entrants at entry, leavers at exit) in magnitude, selectivity (one minus
activity) and cosine with the end direction, as medians and as AUCs (0.5 means the two cross the boundary in the
same state; a threshold predicts that, a reorganisation does not).
Pre-registered (honest guesses):
- the word set's retention is below 0.5 between steps 512 and 4000 and above 0.8 after step 16000, at block 12 (0.6);
- the word set overlaps the largest-magnitude set by less than 0.5 at every checkpoint (0.5);
- the rows entering the word set between steps 1000 and 8000 are more selective at entry than the rows leaving it
  are at exit (AUC above 0.6) (0.5).
Arguments: name."""
import sys, os, json as _json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; CDIR = f"/workspace/wdd/cache/e515_{name}"; STEPS = ["step0", "step64", "step256", "step512", "step1000", "step2000", "step3000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
LAB = [0, 64, 256, 512, 1000, 2000, 3000, 4000, 8000, 16000, 32000, 64000, 143000]; NWORD = 256
t0 = time.time()
while not all(os.path.exists(f"{CDIR}/{s}.pt") for s in STEPS):
    if time.time() - t0 > 3600: raise SystemExit("tracks missing: " + ", ".join(s for s in STEPS if not os.path.exists(f"{CDIR}/{s}.pt")))
    time.sleep(30)
time.sleep(10)
data = {s: torch.load(f"{CDIR}/{s}.pt") for s in STEPS}
def auc(pos, neg):
    if pos.numel() == 0 or neg.numel() == 0: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def spear(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-9))
def setof(idx): return set(idx.tolist())
res = dict(model=name, steps=LAB, by_block={})
for b in data["main"]:
    U = {s: data[s][b]["usage"].reshape(-1) for s in STEPS}; M = {s: data[s][b]["magnitude"].reshape(-1) for s in STEPS}; A = {s: data[s][b]["activity"].reshape(-1) for s in STEPS}; C = {s: data[s][b]["cos_end"].reshape(-1) for s in STEPS}
    W, N = {}, {}
    for s in STEPS:
        used = torch.nonzero(U[s] > 0)[:, 0]; order = U[s][used].argsort(descending=True); W[s] = used[order[:NWORD]]; N[s] = M[s].topk(NWORD).indices
    Wend = setof(W["main"])
    per_step = []
    for i, s in enumerate(STEPS):
        ws, ns = setof(W[s]), setof(N[s]); e = dict(step=LAB[i], n_words=len(ws), overlap_with_end=len(ws & Wend) / max(len(ws), 1), overlap_with_magnitude_top=len(ws & ns) / max(len(ws), 1), median_usage=float(U[s][W[s]].median()) if len(ws) else None)
        if i + 1 < len(STEPS):
            s1 = STEPS[i + 1]; w1, n1 = setof(W[s1]), setof(N[s1]); both = torch.tensor(sorted(ws | w1))
            e.update(retention=len(ws & w1) / max(len(ws), 1), retention_magnitude_null=len(ns & n1) / max(len(ns), 1), jaccard=len(ws & w1) / max(len(ws | w1), 1), spearman_usage=spear(U[s][both], U[s1][both]) if both.numel() > 2 else None)
            ent = torch.tensor(sorted(w1 - ws), dtype=torch.long); lea = torch.tensor(sorted(ws - w1), dtype=torch.long)
            h = dict(n_entrants=int(ent.numel()), n_leavers=int(lea.numel()))
            if ent.numel() and lea.numel():
                h.update(entrants_at_entry=dict(magnitude=float(M[s1][ent].median()), selectivity=float((1 - A[s1][ent]).median()), cos_end=float(C[s1][ent].median())), leavers_at_exit=dict(magnitude=float(M[s][lea].median()), selectivity=float((1 - A[s][lea]).median()), cos_end=float(C[s][lea].median())),
                         entrants_before=dict(magnitude=float(M[s][ent].median()), selectivity=float((1 - A[s][ent]).median())), leavers_after=dict(magnitude=float(M[s1][lea].median()), selectivity=float((1 - A[s1][lea]).median())),
                         auc_entrants_over_leavers=dict(magnitude=auc(M[s1][ent], M[s][lea]), selectivity=auc(1 - A[s1][ent], 1 - A[s][lea]), cos_end=auc(C[s1][ent], C[s][lea])))
            e["hysteresis"] = h
        per_step.append(e)
    trans = {}
    for s0 in ("step512", "step1000", "step2000", "step4000"):
        E = setof(W[s0]); trans[LAB[STEPS.index(s0)]] = {LAB[j]: len(E & setof(W[STEPS[j]])) / max(len(E), 1) for j in range(STEPS.index(s0), len(STEPS))}
        e0 = torch.tensor(sorted(E), dtype=torch.long); trans[LAB[STEPS.index(s0)]]["at_end"] = dict(share_words=len(E & Wend) / max(len(E), 1), median_usage=float(U["main"][e0].median()), median_magnitude=float(M["main"][e0].median()), median_selectivity=float((1 - A["main"][e0]).median()), median_cos_end=float(C["main"][e0].median()))
    ever = set().union(*[setof(W[s]) for s in STEPS[:-1]]); res["by_block"][b] = dict(per_step=per_step, transition=trans, rows_ever_words_before_end=len(ever), share_of_end_words_never_words_before_step4000=len(Wend - set().union(*[setof(W[s]) for s in STEPS[:STEPS.index("step4000") + 1]])) / len(Wend), share_of_early_words_lost=len(ever - Wend) / max(len(ever), 1))
    ps = per_step
    log(f"{name} block {b}: retention of the word set by pair " + " ".join(f"{e['step']}:{e['retention']:.2f}({e['retention_magnitude_null']:.2f})" for e in ps if 'retention' in e) + " | overlap with the end's words " + " ".join(f"{e['step']}:{e['overlap_with_end']:.2f}" for e in ps) + " | overlap with the 256 largest " + " ".join(f"{e['step']}:{e['overlap_with_magnitude_top']:.2f}" for e in ps)
        + " | entrants over leavers, AUC magnitude/selectivity " + " ".join(f"{e['step']}:{e['hysteresis']['auc_entrants_over_leavers']['magnitude']:.2f}/{e['hysteresis']['auc_entrants_over_leavers']['selectivity']:.2f}" for e in ps if 'hysteresis' in e and 'auc_entrants_over_leavers' in e['hysteresis'])
        + f" | early words (step 1000) still words at the end {trans[1000]['at_end']['share_words']:.2f}; end words never words before step 4000 {res['by_block'][b]['share_of_end_words_never_words_before_step4000']:.2f}")
B12 = res["by_block"][12 if 12 in res["by_block"] else list(res["by_block"])[-1]]["per_step"]
rp = {e["step"]: e for e in B12}
res["checks"] = dict(retention_low_early_high_late=all(rp[s]["retention"] < 0.5 for s in (512, 1000, 2000, 3000)) and all(rp[s]["retention"] > 0.8 for s in (16000, 32000, 64000)),
                     words_not_the_largest=all(e["overlap_with_magnitude_top"] < 0.5 for e in B12),
                     entrants_more_selective=all((rp[s]["hysteresis"].get("auc_entrants_over_leavers") or {}).get("selectivity", 0) > 0.6 for s in (1000, 2000, 3000, 4000)))
summ = (f"{name}: " + " || ".join(f"block {b}: retention by pair " + " ".join(f"{e['step']}:{e['retention']:.2f}" for e in o["per_step"] if 'retention' in e) + " (magnitude null " + " ".join(f"{e['retention_magnitude_null']:.2f}" for e in o["per_step"] if 'retention' in e) + "); overlap with the end " + " ".join(f"{e['overlap_with_end']:.2f}" for e in o["per_step"]) + "; overlap with the largest " + " ".join(f"{e['overlap_with_magnitude_top']:.2f}" for e in o["per_step"]) + "; entrants over leavers AUC selectivity " + " ".join(f"{e['step']}:{e['hysteresis']['auc_entrants_over_leavers']['selectivity']:.2f}" for e in o["per_step"] if 'hysteresis' in e and 'auc_entrants_over_leavers' in e['hysteresis']) + ", magnitude " + " ".join(f"{e['hysteresis']['auc_entrants_over_leavers']['magnitude']:.2f}" for e in o["per_step"] if 'hysteresis' in e and 'auc_entrants_over_leavers' in e['hysteresis']) + f"; step-1000 words still words at the end {o['transition'][1000]['at_end']['share_words']:.2f}, end words new after step 4000 {o['share_of_end_words_never_words_before_step4000']:.2f}" for b, o in res["by_block"].items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e515b_turnover_{name}", res, summ)
