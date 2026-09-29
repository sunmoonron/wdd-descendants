"""e580b (session 105): which label-free statistic at time t predicts generalisation at t + delta? Over the twenty
trajectories of e580, every (run, t) with test accuracy under one half at t is a sample; the target is whether test
accuracy exceeds one half at t + delta (delta = 250, 500, 1000, 2000 steps), and, continuously, the gain in test
accuracy over the delta. Each statistic alone, leave-one-run-out: a logistic regression on the standardised statistic
(and its change over the previous 250 steps), the AUC pooled over held-out runs; and the Spearman correlation of the
statistic with the gain. The WDD statistics (native excess, advantage, top-32 share, rows over the floor) against the
baselines (training loss, weight norms, the states' participation ratio and top-8 share, ReLU density, the embedding's
Fourier share). Pre-registered (probabilities are honest guesses):
 P1 (0.5) at delta = 1000 the native excess predicts generalisation at AUC 0.75 or more, above training loss;
 P2 (0.6) the weight norm or the Fourier share does at least as well: the classic signals are not beaten;
 P3 (0.6) the excess's change over 250 steps is a better predictor than its level."""
from s101_common import *
import glob
t0 = time.time(); runs = {os.path.basename(f)[9:-5]: json.load(open(f)) for f in sorted(glob.glob("/workspace/wdd/results/e580_det_*.json"))}
STATS = ["wdd_excess", "wdd_advantage", "wdd_top32_share", "wdd_rows_over_floor", "train_loss", "weight_norm", "mlp_weight_norm", "state_pr", "top8_pc_share", "relu_density", "emb_fourier_top5"]
def samples(delta):
    S = []
    for name, d in runs.items():
        rows = d["log"]; step_of = {r["step"]: i for i, r in enumerate(rows)}
        for i, r in enumerate(rows):
            j = step_of.get(r["step"] + delta)
            if j is None or r["test_acc"] >= 0.5 or i == 0: continue
            prev = rows[i - 1]; feats = {k: r[k] for k in STATS}; feats.update({k + "_d": r[k] - prev[k] for k in STATS}); S.append(dict(run=name, y=float(rows[j]["test_acc"] >= 0.5), gain=rows[j]["test_acc"] - r["test_acc"], **feats))
    return S
def spearman(x, y):
    x = torch.tensor(x, dtype=torch.float64); y = torch.tensor(y, dtype=torch.float64); rx = x.argsort().argsort().double(); ry = y.argsort().argsort().double(); return float(torch.corrcoef(torch.stack([rx, ry]))[0, 1]) if x.std() > 0 and y.std() > 0 else None
res = {}
for delta in (250, 500, 1000, 2000):
    S = samples(delta); names = sorted(set(s["run"] for s in S)); y = torch.tensor([s["y"] for s in S]); base = float(y.mean()); out = dict(n=len(S), base_rate=base, stats={})
    for k in STATS + [k + "_d" for k in STATS]:
        x = torch.tensor([s[k] for s in S], dtype=torch.float32); sc = torch.zeros(len(S))
        for held in names:
            te = torch.tensor([s["run"] == held for s in S]); tr = ~te
            if y[tr].min() == y[tr].max(): sc[te] = 0; continue
            sc[te] = logreg(x[tr, None].to(DEV), y[tr].to(DEV), x[te, None].to(DEV), l2=1e-3, iters=100).cpu()
        out["stats"][k] = dict(auc=auc(sc, y.bool()), spearman_gain=spearman([s[k] for s in S], [s["gain"] for s in S]))
    res[delta] = out; top = sorted(out["stats"].items(), key=lambda kv: -(kv[1]["auc"] or 0))[:6]
    log(f"delta {delta} ({len(S)} samples, base rate {base:.2f}): " + ", ".join(f"{k} AUC {v['auc']:.3f} (rho {v['spearman_gain']:+.2f})" for k, v in top) + f" | wdd_excess {out['stats']['wdd_excess']['auc']:.3f}, train_loss {out['stats']['train_loss']['auc']:.3f}, weight_norm {out['stats']['weight_norm']['auc']:.3f}, emb_fourier {out['stats']['emb_fourier_top5']['auc']:.3f}")
r1 = res[1000]["stats"]; best = max(r1.items(), key=lambda kv: kv[1]["auc"] or 0)
summ = (f"detector benchmark over {len(runs)} trajectories: at delta 1000 (n {res[1000]['n']}, base rate {res[1000]['base_rate']:.2f}) AUC native excess {r1['wdd_excess']['auc']:.3f} (its change {r1['wdd_excess_d']['auc']:.3f}), advantage {r1['wdd_advantage']['auc']:.3f}, top-32 share {r1['wdd_top32_share']['auc']:.3f}; training loss {r1['train_loss']['auc']:.3f}, weight norm {r1['weight_norm']['auc']:.3f} (change {r1['weight_norm_d']['auc']:.3f}), MLP norm {r1['mlp_weight_norm']['auc']:.3f}, state PR {r1['state_pr']['auc']:.3f}, top-8 PC share {r1['top8_pc_share']['auc']:.3f}, ReLU density {r1['relu_density']['auc']:.3f}, Fourier {r1['emb_fourier_top5']['auc']:.3f} (change {r1['emb_fourier_top5_d']['auc']:.3f}); best {best[0]} {best[1]['auc']:.3f}; "
        f"at delta 250 excess {res[250]['stats']['wdd_excess']['auc']:.3f} vs weight norm {res[250]['stats']['weight_norm']['auc']:.3f} vs Fourier {res[250]['stats']['emb_fourier_top5']['auc']:.3f}; at 2000 {res[2000]['stats']['wdd_excess']['auc']:.3f} vs {res[2000]['stats']['weight_norm']['auc']:.3f} vs {res[2000]['stats']['emb_fourier_top5']['auc']:.3f} | {time.time() - t0:.0f}s")
log(summ); record("e580b_detector_analysis", res, summ)
