"""e629b (session 125, laptop): the transfer of the forecast rule across seeds and depths, from e629's saved score tables.
For horizon 2 (and 1, 4, 8 where present): the candidates at every checkpoint t are the rows that are not words at t, the
label is wordhood at t + k. A logistic rule on standardised scores is fitted on one run (all its checkpoints) and scored
on another: seed 0 block 6 to seeds 1 and 2 at block 6, seed 0 block 6 to seed 0 block 9, and the reverse directions;
within-run leave-one-checkpoint-out AUCs are the reference. Combos: wdd (S, class-side), feature (decoder cosine, mass,
feature partition), wdd+feature, all. Writes results/e629b_transfer.json."""
import os, sys, json, glob, torch
R = os.path.dirname(os.path.abspath(__file__)) + "/../results/"; T = R + "e629_tables/"
SCORES = ["S", "anymax", "trend", "usage", "norm", "align", "fmax", "fmass", "fany"]
COMBOS = {"wdd": ("S", "anymax"), "feature": ("fmax", "fmass", "fany"), "wdd_plus_feature": ("S", "anymax", "fmax", "fmass", "fany"), "all": tuple(SCORES)}
def auc(score, y):
    score = torch.as_tensor(score, dtype=torch.float64); y = torch.as_tensor(y).bool(); n1 = int(y.sum()); n0 = int((~y).sum())
    if n1 == 0 or n0 == 0: return None
    o = score.argsort(); s = score[o]; ranks = torch.arange(1, len(s) + 1, dtype=torch.float64); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        ranks[i:j + 1] = (i + j + 2) / 2; i = j + 1
    r = torch.empty_like(score); r[o] = ranks; return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def fit(Xtr, ytr, l2=1e-3, iters=100):
    Xtr, ytr = Xtr.to(DEV), ytr.to(DEV); mu = Xtr.mean(0); sd = Xtr.std(0).clamp_min(1e-6); Xs = (Xtr - mu) / sd; w = torch.zeros(Xs.shape[1], device=DEV, requires_grad=True); b = torch.zeros(1, device=DEV, requires_grad=True); yt = ytr.float()
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, yt) + l2 * w.pow(2).sum(); loss.backward(); return loss
    opt.step(closure); return dict(mu=mu.cpu(), sd=sd.cpu(), w=w.detach().cpu(), b=b.detach().cpu())
def apply(rule, X): return ((X - rule["mu"]) / rule["sd"]) @ rule["w"] + rule["b"]
def dataset(tab, k):
    steps = tab["steps"]; S = tab["scores"]; X, Y, Rn = [], [], []
    for i in range(len(steps) - k):
        n, nk = str(steps[i]), str(steps[i + k]); cand = torch.nonzero(~S[n]["words"])[:, 0]; Y.append(S[nk]["words"][cand].float()); Rn.append(torch.full((cand.numel(),), i)); X.append(torch.stack([S[n][q][cand].float() for q in SCORES], 1))
    return torch.cat(X), torch.cat(Y), torch.cat(Rn)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
runs = {os.path.basename(p).replace("e629_scores_", "").replace(".pt", ""): torch.load(p) for p in sorted(glob.glob(T + "e629_scores_*.pt")) if "smoke" not in p}
WITHIN = {}
for name in runs:
    tag, blk = name.rsplit("_B", 1); fn = R + f"e629_forecast_{tag}_B{blk}.json"
    if os.path.exists(fn): WITHIN[name] = {k: v["combos"] for k, v in json.load(open(fn))["horizons"].items()}
print("runs:", list(runs))
res = dict(runs=list(runs), horizons={})
for k in (1, 2):
    data = {}
    for name, tab in runs.items():
        if len(tab["steps"]) - k < 2: continue
        data[name] = dataset(tab, k)
    out = dict(within={name: WITHIN.get(name, {}).get(str(k), {}) for name in data}, transfer={})
    for src, (Xs, Ys, _) in data.items():
        for dst, (Xd, Yd, _) in data.items():
            if src == dst: continue
            out["transfer"][f"{src}->{dst}"] = {}
            for cn, keys in COMBOS.items():
                cols = [SCORES.index(q) for q in keys]; rule = fit(Xs[:, cols], Ys); out["transfer"][f"{src}->{dst}"][cn] = auc(apply(rule, Xd[:, cols]), Yd.bool())
    res["horizons"][k] = out
    for name in out["within"]: print(f"horizon {k} within {name}: " + ", ".join(f"{cn} {v:.3f}" for cn, v in out["within"][name].items()))
    for pair in out["transfer"]: print(f"horizon {k} transfer {pair}: " + ", ".join(f"{cn} {v:.3f}" for cn, v in out["transfer"][pair].items()))
json.dump(res, open(R + "e629b_transfer.json", "w"), indent=1); print("written e629b_transfer.json")
