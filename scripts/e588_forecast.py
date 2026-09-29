"""e588 (session 108): forecasting parameter recruitment from the class side, with no future in the class. At every
checkpoint t the states are clustered without any row (spherical k-means on the unit states, K clusters); for every
cluster and every row, the row's mean projection ratio over the cluster's positions; a row's class-side score is its
largest mean ratio over any cluster (its pull toward the class it is nearest to); a stricter variant, the largest over
the clusters in which the row is the best row, is degenerate (most rows own no cluster) and reported as such. The forecast: among the rows that are not words at t, which will be
words at t + k (k = 2, 4, 8 checkpoints)? Scored by AUC and precision at 100 against the row-side score the record
already has (the row's S at t, e548), the row's norm and its usage at t, and S with the class-side score combined;
pooled over t. The
recruitment clock: for the rows that are recruited, the checkpoint at which their pull, extrapolated linearly from
t - 1 and t, would cross the floor, against the checkpoint at which they are recruited; the lead (forecast minus
origin) against the actual lead, so that the origin's time does not inflate the correlation. Pythia-410m from e582's cache
(block 12, 12k positions, K = 512) and OLMo-1B from e550's cache (block 8, 2027 positions, K = 128) as the second
family. Argument: pythia410 or olmo1b. Pre-registered (probabilities are honest guesses):
 F1 (0.6) the class-side score forecasts recruitment two checkpoints ahead at AUC 0.8 or more on both models;
 F2 (0.5) it adds to the row's S (combined AUC above S alone by 0.03 or more): the class the row would own carries
    information the row's own maximum does not;
 F3 (0.5) the clock's forecast of the recruitment checkpoint correlates with the actual one at 0.4 or more."""
from s101_common import *
name = sys.argv[1] if len(sys.argv) > 1 else "pythia410"; t0 = time.time()
if name == "pythia410":
    CD = "/workspace/wdd/cache/e582_pythia410"; steps = list(range(1000, 16001, 1000)); ck = {n: torch.load(f"{CD}/step{n}.pt", map_location="cpu") for n in steps}; keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0)
    U = {n: unitr((ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV)) for n in steps}; A = {n: ck[n]["rows"].float().to(DEV) for n in steps}; NORM = {n: ck[n]["norms"].to(DEV) for n in steps}; KC = 512
else:
    CD = "/workspace/wdd/cache/e550_olmo1b"; steps = list(range(1000, 16001, 1000)); ck = {n: torch.load(f"{CD}/step{n}.pt", map_location="cpu") for n in steps}; keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0)
    U = {n: unitr((ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV)) for n in steps}; A = {n: ck[n]["rows"].float().to(DEV) for n in steps}; NORM = {n: ck[n]["norms"].to(DEV) for n in steps}; KC = 128
T = len(steps); N = int(keepc.sum()); m = A[steps[0]].shape[0]; log(f"{name}: {N} positions, {m} rows, {T} checkpoints, K = {KC}")
def kmeans(X, K, iters=15, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed); C = X[torch.randperm(X.shape[0], device=DEV, generator=g)[:K]].clone()
    for _ in range(iters):
        lab = (X @ C.T).argmax(1); C = torch.zeros_like(C).index_add_(0, lab, X); cnt = torch.bincount(lab, minlength=K).float(); empty = cnt == 0; C = unitr(C / cnt.clamp_min(1)[:, None]); C[empty] = X[torch.randperm(X.shape[0], device=DEV, generator=g)[:int(empty.sum())]]
    return (X @ C.T).argmax(1)
SC = {}
for i, n in enumerate(steps):
    st = stats(U[n], A[n], K); words = wordset(st["usage"]); S = st["S"]; ratio = st["ratio"].to(DEV).float(); lab = kmeans(U[n], KC); cnt = torch.bincount(lab, minlength=KC).float().clamp_min(1)
    M = torch.zeros(KC, m, device=DEV).index_add_(0, lab, ratio) / cnt[:, None]   # mean ratio of every row over every cluster
    best_row = M.argmax(1); own = torch.zeros(m, device=DEV); own.index_reduce_(0, best_row, M.max(1).values, "amax", include_self=True)   # the largest mean ratio over the clusters a row is the best row of
    SC[n] = dict(words=words, S=S, own=own.cpu(), anymax=M.max(0).values.cpu(), usage=st["usage"], norm=NORM[n].cpu(), n_clusters_owned=torch.bincount(best_row, minlength=m).cpu()); del ratio, M; torch.cuda.empty_cache(); log(f"{n}: clusters done ({time.time() - t0:.0f}s)")
res = dict(model=name, N=N, K=KC, horizons={})
for k in (2, 4, 8):
    P = {q: [] for q in ("S", "own", "anymax", "norm", "usage", "cluster_count")}; Y = []; runs = []
    for i in range(1, T - k):
        n, nk = steps[i], steps[i + k]; cand = torch.nonzero(~SC[n]["words"])[:, 0]; y = SC[nk]["words"][cand].float(); Y.append(y); runs.append(torch.full((cand.numel(),), i))
        for q in ("S", "own", "anymax", "norm", "usage"): P[q].append(SC[n][q][cand].float())
        P["cluster_count"].append(SC[n]["n_clusters_owned"][cand].float())
    Y = torch.cat(Y); R = torch.cat(runs); out = dict(n=int(Y.numel()), base_rate=float(Y.mean()), auc={}, precision100={})
    for q in ("S", "own", "anymax", "norm", "usage", "cluster_count"):
        x = torch.cat(P[q]); out["auc"][q] = auc(x, Y.bool()); top = x.topk(100 * len(set(R.tolist()))).indices; out["precision100"][q] = float(Y[top].mean())
    # combined, leave-one-checkpoint-out logistic on standardised S and own
    for combo, keys in (("S_plus_class", ("S", "anymax")), ("S_plus_usage", ("S", "usage")), ("all_three", ("S", "anymax", "usage"))):
        X2 = torch.stack([torch.cat(P[q]) for q in keys], 1); sc = torch.zeros(Y.numel())
        for i in set(R.tolist()):
            te = R == i; sc[te] = logreg(X2[~te].to(DEV), Y[~te].to(DEV), X2[te].to(DEV), l2=1e-3, iters=100).cpu()
        out["auc"][combo] = auc(sc, Y.bool())
    res["horizons"][k] = out
    log(f"horizon {k} ({out['n']} candidates, base rate {out['base_rate']:.3f}): AUC S {out['auc']['S']:.3f}, class-side {out['auc']['anymax']:.3f} (owned-cluster variant {out['auc']['own']:.3f}), norm {out['auc']['norm']:.3f}, usage {out['auc']['usage']:.3f}, S+class {out['auc']['S_plus_class']:.3f}, S+usage {out['auc']['S_plus_usage']:.3f}, all three {out['auc']['all_three']:.3f}; precision at 100 per checkpoint S {out['precision100']['S']:.2f}, class-side {out['precision100']['anymax']:.2f}, norm {out['precision100']['norm']:.2f}")
# the recruitment clock: entrants' arrival predicted from the pull's slope
pred, actual, origin = [], [], []
for i in range(2, T):
    n, n1 = steps[i], steps[i - 1]
    for j in range(i + 1, T):
        pass
ent_events = []
for j in range(2, T):
    new = SC[steps[j]]["words"] & ~SC[steps[j - 1]]["words"] & ~SC[steps[j - 2]]["words"]
    for r in torch.nonzero(new)[:, 0].tolist(): ent_events.append((int(r), j))
for r, j in ent_events:
    for i in (j - 2, j - 3, j - 4):
        if i < 1: continue
        p1, p0 = float(SC[steps[i]]["anymax"][r]), float(SC[steps[i - 1]]["anymax"][r]); slope = p1 - p0
        if slope > 1e-3 and p1 < 1: pred.append(i + (1 - p1) / slope); actual.append(j); origin.append(i)
        elif p1 >= 1: pred.append(i); actual.append(j); origin.append(i)
sp = None
if len(pred) > 10:
    px, ay, og = torch.tensor(pred), torch.tensor(actual, dtype=torch.float), torch.tensor(origin, dtype=torch.float); rk = lambda v: v.argsort().argsort().float(); sp = float(torch.corrcoef(torch.stack([rk(px), rk(ay)]))[0, 1]); err = (px - ay).abs()
    lead_p, lead_a = (px - og).clamp(max=12), ay - og; sp_lead = float(torch.corrcoef(torch.stack([rk(lead_p), rk(lead_a)]))[0, 1]) if lead_p.std() > 0 else None
    res["clock"] = dict(n=len(pred), spearman=sp, spearman_lead=sp_lead, median_abs_error=float(err.median()), share_within_1=float((err <= 1).float().mean()), share_within_2=float((err <= 2).float().mean()), median_lead_forecast=float(lead_p.median()), median_lead_actual=float(lead_a.median()))
    log(f"clock ({len(pred)} entrant forecasts from 2-4 checkpoints before): Spearman {sp:.2f} (of the leads, origin removed: {sp_lead}), median absolute error {float(err.median()):.1f} checkpoints, within 1 for {float((err <= 1).float().mean()):.2f}, within 2 for {float((err <= 2).float().mean()):.2f}; forecast lead {float(lead_p.median()):.1f} vs actual {float(lead_a.median()):.1f}")
H = res["horizons"]
summ = (f"{name} forecast of recruitment from row-free classes (K = {KC}): AUC at 2 / 4 / 8 checkpoints ahead, class-side {H[2]['auc']['anymax']:.3f} / {H[4]['auc']['anymax']:.3f} / {H[8]['auc']['anymax']:.3f} (owned-cluster variant {H[2]['auc']['own']:.3f}), row-side S {H[2]['auc']['S']:.3f} / {H[4]['auc']['S']:.3f} / {H[8]['auc']['S']:.3f}, usage {H[2]['auc']['usage']:.3f} / {H[4]['auc']['usage']:.3f} / {H[8]['auc']['usage']:.3f}, S+class {H[2]['auc']['S_plus_class']:.3f} / {H[4]['auc']['S_plus_class']:.3f} / {H[8]['auc']['S_plus_class']:.3f}, all three {H[2]['auc']['all_three']:.3f}, norm {H[2]['auc']['norm']:.3f}; precision at 100 per checkpoint at 2 ahead: class-side {H[2]['precision100']['anymax']:.2f}, S {H[2]['precision100']['S']:.2f}, usage {H[2]['precision100']['usage']:.2f}, norm {H[2]['precision100']['norm']:.2f} (base rate {H[2]['base_rate']:.3f}); "
        + (f"clock: Spearman {res['clock']['spearman']:.2f} (leads, origin removed {res['clock']['spearman_lead']}), within 2 checkpoints for {res['clock']['share_within_2']:.2f} of {res['clock']['n']}" if "clock" in res else "clock: too few entrants") + f" | {time.time() - t0:.0f}s")
log(summ); record(f"e588_forecast_{name}", res, summ)
