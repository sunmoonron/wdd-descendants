"""e629 (session 125): the row forecast against a feature-level forecast on the same checkpoints, with seeds, a second
depth and the controls that could dissolve it. Pythia-160m (seed 0, and PolyPythias seeds 1 and 2) at steps 1000 to
16000, the states after block B on 24 Pile sequences of 512 tokens, the MLP rows of blocks 0..B as the dictionary. At
every checkpoint t, per row: the row side (S, usage, norm); the class side of e588 (the row's largest mean projection
ratio over any row-free k-means cluster of the unit states, K = 512) and its trend from the previous checkpoint; the
feature side, from a TopK autoencoder trained on the same states (4096 latents, 32 active): the largest cosine of any
live feature's decoder with the row (can a learned dictionary name the parameter?), the feature mass pointing at the
row, and the class-side score with the features' active sets as the partition instead of k-means (does the partition's
source matter?); the generic side: the row's alignment with the cloud's second moment; the same row-side and class-side
scores for a fixed rotation of the rows forecasting the rotated dictionary's own words (session 83's generic part); a
static control, the step-1000 class-side score used at every later checkpoint; a label shuffle; and retrodiction, the
same scores read backwards (which non-words were words k checkpoints ago). Forecast: among the rows that are not words
at t, which are words at t + k (k = 1, 2, 4, 8): AUC, precision at 100 per checkpoint, leave-one-checkpoint-out logistic
combinations. Geometry (e587): for every eventual word recruited at index 2 or later, its rank among all rows at 0, 1, 2,
3, 5 and 8 checkpoints before recruitment by mean class ratio, by cosine with the class direction, by the decoder of the
feature best matching the class, and by the feature partition's mean ratio. The per-row score tables are saved for the
cross-seed and cross-depth transfer analysis. Arguments: tag (pythia160 | pythia160s1 | pythia160s2) block [--smoke].
Pre-registered in e629_prereg.json."""
import sys, os, time, json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
import wdd_common
wdd_common.MODELS.update({"pythia160s1": ("EleutherAI/pythia-160m-seed1", "neox"), "pythia160s2": ("EleutherAI/pythia-160m-seed2", "neox")}); MID.update({"pythia160s1": 6, "pythia160s2": 6})
tag = sys.argv[1]; B = int(sys.argv[2]); SMOKE = "--smoke" in sys.argv; t0 = time.time(); torch.set_grad_enabled(False)
steps = [1000, 2000, 3000, 4000] if SMOKE else list(range(1000, 16001, 1000)); NSEQ = 4 if SMOKE else 24; T = 512; KC = 64 if SMOKE else 512
F_, TOPK, SAE_STEPS, BS = 4096, 32, (60 if SMOKE else 2000), 2048; BLOCKS = [6, 9]; HORIZONS = [1, 2] if SMOKE else [1, 2, 4, 8]; LEADS = [0, 1, 2, 3, 5, 8]
sfx = "_smoke" if SMOKE else ""; CD = {b: f"/workspace/wdd/cache/e629_{tag}_B{b}{sfx}" for b in BLOCKS}
for b in BLOCKS: os.makedirs(CD[b], exist_ok=True)
ids = pile_ids(tag, NSEQ, T)
# ---------------- 1. the longitudinal cache (both depths in one pass per checkpoint)
for n in steps:
    if all(os.path.exists(f"{CD[b]}/step{n}.pt") for b in BLOCKS): continue
    model, _, fam = load_model(tag, revision=f"step{n}"); arch = Arch(model, fam); Xs = block_states(model, arch, ids, BLOCKS, chunk=4)
    for b in BLOCKS:
        X = Xs[b].reshape(-1, arch.D); keep = ~sinkmask(X); A, norms = rows_of(arch, b); out = f"{CD[b]}/step{n}.pt"; tmp = out + f".{os.getpid()}.tmp"
        torch.save(dict(step=n, X=X.half().cpu(), keep=keep.cpu(), rows=A.half().cpu(), norms=norms.cpu(), D=arch.D, DFF=arch.DFF, B=b), tmp); os.replace(tmp, out)
    del model, Xs; torch.cuda.empty_cache(); log(f"cached step {n} ({time.time() - t0:.0f}s)")
ck = {n: torch.load(f"{CD[B]}/step{n}.pt", map_location="cpu") for n in steps}; keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0); N = int(keepc.sum())
U = {n: unitr((ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV)) for n in steps}; A = {n: ck[n]["rows"].float().to(DEV) for n in steps}
AR = {n: unitr(rotate(A[n], seed=7)) for n in steps}; NORM = {n: ck[n]["norms"].to(DEV) for n in steps}; m = A[steps[0]].shape[0]; D = U[steps[0]].shape[1]; TT = len(steps); DFF = int(ck[steps[0]]["DFF"])
log(f"{tag} block {B}: {N} common positions, {m} rows, {TT} checkpoints, K = {KC}")
def kmeans(X, Kk, iters=15, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed); C = X[torch.randperm(X.shape[0], device=DEV, generator=g)[:Kk]].clone()
    for _ in range(iters):
        lab = (X @ C.T).argmax(1); C = torch.zeros_like(C).index_add_(0, lab, X); cnt = torch.bincount(lab, minlength=Kk).float(); empty = cnt == 0; C = unitr(C / cnt.clamp_min(1)[:, None]); C[empty] = X[torch.randperm(X.shape[0], device=DEV, generator=g)[:int(empty.sum())]]
    return (X @ C.T).argmax(1)
class TopKSAE(torch.nn.Module):
    def __init__(s, Dd):
        super().__init__(); s.We = torch.nn.Parameter(torch.randn(F_, Dd) / Dd ** 0.5); s.be = torch.nn.Parameter(torch.zeros(F_)); s.Wd = torch.nn.Parameter(s.We.detach().clone()); s.bd = torch.nn.Parameter(torch.zeros(Dd))
    def encode(s, x):
        z = (x - s.bd) @ s.We.T + s.be; top = z.topk(TOPK, dim=1); out = torch.zeros_like(z); out.scatter_(1, top.indices, torch.relu(top.values)); return out
    def forward(s, x): z = s.encode(x); return z @ s.Wd + s.bd, z
def train_sae(Xs, seed):
    torch.manual_seed(seed); sae = TopKSAE(Xs.shape[1]).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=1e-3); g = torch.Generator(device=DEV).manual_seed(seed); Nn = Xs.shape[0]
    with torch.enable_grad():
        for step in range(SAE_STEPS):
            idx = torch.randint(0, Nn, (BS,), device=DEV, generator=g); xb = Xs[idx]; xh, z = sae(xb); loss = (xh - xb).pow(2).sum(1).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad(): sae.Wd.data = sae.Wd.data / sae.Wd.data.norm(dim=1, keepdim=True).clamp_min(1e-6)
    xh, z = sae(Xs); fvu = float((xh - Xs).pow(2).sum(1).mean() / Xs.pow(2).sum(1).mean()); return sae, z, fvu
def class_scores(ratio, lab, Kk):
    cnt = torch.bincount(lab, minlength=Kk).float().clamp_min(1); M = torch.zeros(Kk, ratio.shape[1], device=DEV).index_add_(0, lab, ratio) / cnt[:, None]; return M.max(0).values
SC, GE = {}, {}
for i, n in enumerate(steps):
    st = stats(U[n], A[n], K); words = wordset(st["usage"]); ratio = st["ratio"].to(DEV).float(); lab = kmeans(U[n], KC); anymax = class_scores(ratio, lab, KC)
    C = U[n].T @ U[n] / N; align = ((A[n] @ C) * A[n]).sum(1)
    str_ = stats(U[n], AR[n], K); words_r = wordset(str_["usage"]); ratio_r = str_["ratio"].to(DEV).float(); anymax_r = class_scores(ratio_r, lab, KC); del ratio_r
    Xr = ck[n]["X"][keepc].float().to(DEV); mu = Xr.mean(0); Xc = Xr - mu; scale = float(Xc.norm(dim=1).mean()); Xs = Xc / scale; sae, Z, fvu = train_sae(Xs, seed=n); act = Z > 0; fs = act.sum(0); live = torch.nonzero(fs >= 10)[:, 0]
    Wd = unitr(sae.Wd.detach()[live]); n_f = fs[live].float(); FC = Wd @ A[n].T; fmax = FC.max(0).values; fmass = ((n_f[:, None] / N) * FC.clamp_min(0).pow(2)).sum(0)
    Fl = act[:, live].float(); Mf = (Fl.T @ ratio) / n_f[:, None]; fany = Mf.max(0).values
    SC[n] = dict(words=words, words_rot=words_r, S=st["S"], usage=st["usage"], norm=NORM[n].cpu(), anymax=anymax.cpu(), align=align.cpu(), S_rot=str_["S"], anymax_rot=anymax_r.cpu(), fmax=fmax.cpu(), fmass=fmass.cpu(), fany=fany.cpu(), n_live=int(live.numel()), sae_fvu=fvu, n_words=int(words.sum()))
    GE[n] = dict(L=st["L"], act=act[:, live].clone(), Wd=Wd.half())
    del st, str_, ratio, Mf, Fl, FC, Z, act, sae, Xr, Xc, Xs; torch.cuda.empty_cache()
    log(f"step {n}: {int(words.sum())} words (rotated {int(words_r.sum())}), SAE fvu {fvu:.3f} with {live.numel()} live features ({time.time() - t0:.0f}s)")
for i, n in enumerate(steps): SC[n]["trend"] = SC[n]["anymax"] - SC[steps[i - 1]]["anymax"] if i > 0 else torch.zeros(m)
SCORES = ["S", "anymax", "trend", "usage", "norm", "align", "fmax", "fmass", "fany"]
COMBOS = {"wdd": ("S", "anymax"), "wdd_trend": ("S", "anymax", "trend"), "feature": ("fmax", "fmass", "fany"), "generic": ("norm", "usage", "align"), "wdd_plus_feature": ("S", "anymax", "fmax", "fmass", "fany"), "all": tuple(SCORES)}
def loco(P, Y, R, keys):
    X2 = torch.stack([P[q] for q in keys], 1); sc = torch.zeros(Y.numel())
    for i in set(R.tolist()):
        te = R == i
        if int(Y[~te].sum()) == 0: continue
        sc[te] = logreg(X2[~te].to(DEV), Y[~te].to(DEV), X2[te].to(DEV), l2=1e-3, iters=100).cpu()
    return auc(sc, Y.bool())
res = dict(model=tag, block=B, N=N, m=m, K=KC, steps=steps, per_step={str(n): dict(n_words=SC[n]["n_words"], n_live=SC[n]["n_live"], sae_fvu=SC[n]["sae_fvu"]) for n in steps}, horizons={}, retro={})
g0 = torch.Generator().manual_seed(0)
for k in HORIZONS:
    P = {q: [] for q in SCORES + ["static", "S_rot", "anymax_rot"]}; Y, Yr, Rn, Cr = [], [], [], []
    for i in range(0, TT - k):
        n, nk = steps[i], steps[i + k]; cand = torch.nonzero(~SC[n]["words"])[:, 0]; Y.append(SC[nk]["words"][cand].float()); Rn.append(torch.full((cand.numel(),), i))
        for q in SCORES: P[q].append(SC[n][q][cand].float())
        P["static"].append(SC[steps[0]]["anymax"][cand].float())
        cr = torch.nonzero(~SC[n]["words_rot"])[:, 0]; Yr.append(SC[nk]["words_rot"][cr].float()); Cr.append(torch.full((cr.numel(),), i))
        for q in ("S_rot", "anymax_rot"): P[q].append(SC[n][q][cr].float())
    Y = torch.cat(Y); R = torch.cat(Rn); Yr = torch.cat(Yr); P = {q: torch.cat(v) for q, v in P.items()}; nck = len(set(R.tolist()))
    out = dict(n=int(Y.numel()), base_rate=float(Y.mean()), n_rot=int(Yr.numel()), base_rate_rot=float(Yr.mean()), auc={}, precision100={}, combos={})
    for q in SCORES + ["static"]:
        out["auc"][q] = auc(P[q], Y.bool()); top = P[q].topk(min(100 * nck, P[q].numel())).indices; out["precision100"][q] = float(Y[top].mean())
    for q in ("S_rot", "anymax_rot"): out["auc"][q] = auc(P[q], Yr.bool())
    out["auc"]["anymax_label_shuffle"] = auc(P["anymax"], Y[torch.randperm(Y.numel(), generator=g0)].bool())
    for name_, keys in COMBOS.items(): out["combos"][name_] = loco(P, Y, R, keys)
    res["horizons"][k] = out; a = out["auc"]; c = out["combos"]
    log(f"horizon {k} ({out['n']} candidates, base rate {out['base_rate']:.3f}): AUC class-side {a['anymax']:.3f} (trend {a['trend']:.3f}, static step-1000 {a['static']:.3f}, label shuffle {a['anymax_label_shuffle']:.3f}), S {a['S']:.3f}, usage {a['usage']:.3f}, norm {a['norm']:.3f}, align {a['align']:.3f}; feature side: decoder cos {a['fmax']:.3f}, mass {a['fmass']:.3f}, feature partition {a['fany']:.3f}; rotated system: S {a['S_rot']:.3f}, class-side {a['anymax_rot']:.3f} (base rate {out['base_rate_rot']:.3f}); combos wdd {c['wdd']:.3f}, wdd+trend {c['wdd_trend']:.3f}, feature {c['feature']:.3f}, generic {c['generic']:.3f}, wdd+feature {c['wdd_plus_feature']:.3f}, all {c['all']:.3f}; precision at 100: class-side {out['precision100']['anymax']:.2f}, decoder cos {out['precision100']['fmax']:.2f}, feature partition {out['precision100']['fany']:.2f}")
    # retrodiction: among non-words at t, which were words at t - k
    Pb = {q: [] for q in ("S", "anymax", "fany")}; Yb = []
    for i in range(k, TT):
        n, nb = steps[i], steps[i - k]; cand = torch.nonzero(~SC[n]["words"])[:, 0]; Yb.append(SC[nb]["words"][cand].float())
        for q in Pb: Pb[q].append(SC[n][q][cand].float())
    Yb = torch.cat(Yb); res["retro"][k] = dict(n=int(Yb.numel()), base_rate=float(Yb.mean()), auc={q: auc(torch.cat(v), Yb.bool()) for q, v in Pb.items()})
    log(f"retrodiction {k} back: class-side {res['retro'][k]['auc']['anymax']:.3f}, S {res['retro'][k]['auc']['S']:.3f}, feature partition {res['retro'][k]['auc']['fany']:.3f}")
# ---------------- geometry: the recruit's rank before recruitment, by four routes
last = steps[-1]; Ul, Al, Ll = U[last], A[last], GE[last]["L"].to(DEV); R16 = torch.cat([(Ul[s:s + 1024] @ Al.T).abs() / Ll[s:s + 1024, None] for s in range(0, N, 1024)]).half().cpu()
wl = torch.nonzero(SC[last]["words"])[:, 0]; csize = (R16[:, wl].float() > 1).sum(0)
def first_hold(flags):
    for i in range(len(flags)):
        if flags[i] and all(flags[i:]): return i
    return None
tracked = []
for w in wl[(csize >= 10) & ~SC[steps[0]]["words"][wl]].tolist():
    j = first_hold([bool(SC[n]["words"][w]) for n in steps])
    if j is not None and j >= 2: tracked.append((w, j))
CLS = {w: torch.nonzero(R16[:, w].float() > 1)[:, 0] for w, _ in tracked}; log(f"{len(tracked)} eventual words with a class of ten or more, recruited at index 2 or later")
def jaccard_best(n, pos):
    act = GE[n]["act"]; cm = torch.zeros(N, device=DEV); cm[pos.to(DEV)] = 1; Fl = act.float(); inter = Fl.T @ cm; jac = inter / (Fl.sum(0) + cm.sum() - inter).clamp_min(1); j, f = jac.max(0); return int(f), float(j)
def ranks(n, w, pos):
    p = pos.to(DEV); Up = U[n][p]; L = GE[n]["L"].to(DEV)[p]; r = (Up @ A[n].T).abs() / L[:, None]; mr = r.mean(0); d = unitr(Up.mean(0, keepdim=True))[0]; cos = A[n] @ d
    f, j = jaccard_best(n, pos); dec = GE[n]["Wd"][f].float(); cdec = A[n] @ dec; fpos = torch.nonzero(GE[n]["act"][:, f])[:, 0]; Uf = U[n][fpos]; Lf = GE[n]["L"].to(DEV)[fpos]; mf = ((Uf @ A[n].T).abs() / Lf[:, None]).mean(0)
    rk = lambda v: int((v > v[w]).sum()) + 1; return dict(rank_ratio=rk(mr), rank_cos=rk(cos), rank_dec=rk(cdec), rank_fpart=rk(mf), decoder_row_cos=float(cdec[w]), jaccard=j, ratio=float(mr[w]), cos=float(cos[w]))
res["geometry"] = dict(n_tracked=len(tracked), by_lead={})
for lead in LEADS:
    out = [ranks(steps[tr - lead], w, CLS[w]) for w, tr in tracked if tr - lead >= 0]
    if not out: continue
    r = dict(n=len(out)); 
    for key in ("rank_ratio", "rank_cos", "rank_dec", "rank_fpart"): r[key + "_median"] = med([o[key] for o in out]); r[key + "_top1"] = mean([float(o[key] == 1) for o in out]); r[key + "_top10"] = mean([float(o[key] <= 10) for o in out]); r[key + "_top100"] = mean([float(o[key] <= 100) for o in out])
    r["decoder_row_cos_median"] = med([o["decoder_row_cos"] for o in out]); r["jaccard_median"] = med([o["jaccard"] for o in out]); res["geometry"]["by_lead"][lead] = r
    log(f"{lead} before recruitment (n {r['n']}): rank by class ratio {r['rank_ratio_median']:.0f} (first {r['rank_ratio_top1']:.2f}, top 10 {r['rank_ratio_top10']:.2f}), by class cosine {r['rank_cos_median']:.0f} (top 10 {r['rank_cos_top10']:.2f}), by the best feature's decoder {r['rank_dec_median']:.0f} (top 10 {r['rank_dec_top10']:.2f}, decoder-row cosine {r['decoder_row_cos_median']:.2f}, Jaccard {r['jaccard_median']:.2f}), by the feature partition {r['rank_fpart_median']:.0f} (top 10 {r['rank_fpart_top10']:.2f})")
# ---------------- save the score tables for the transfer analysis
tab = dict(tag=tag, block=B, steps=steps, m=m, N=N, scores={str(n): {q: SC[n][q].half() if q not in ("words", "words_rot") else SC[n][q] for q in SCORES + ["words", "words_rot", "S_rot", "anymax_rot"]} for n in steps})
os.makedirs("/workspace/wdd/results/e629_tables", exist_ok=True); torch.save(tab, f"/workspace/wdd/results/e629_tables/e629_scores_{tag}_B{B}{sfx}.pt")
H = res["horizons"]; h2 = H[2] if 2 in H else H[HORIZONS[-1]]; gb = res["geometry"]["by_lead"]
summ = (f"{tag} block {B} ({N} positions, {m} rows, {TT} checkpoints): forecast two ahead AUC class-side {h2['auc']['anymax']:.3f} (S {h2['auc']['S']:.3f}, trend {h2['auc']['trend']:.3f}, static {h2['auc']['static']:.3f}, label shuffle {h2['auc']['anymax_label_shuffle']:.3f}), feature side decoder cos {h2['auc']['fmax']:.3f} / mass {h2['auc']['fmass']:.3f} / feature partition {h2['auc']['fany']:.3f}, generic norm {h2['auc']['norm']:.3f} / usage {h2['auc']['usage']:.3f} / align {h2['auc']['align']:.3f}, rotated system class-side {h2['auc']['anymax_rot']:.3f}; combos wdd {h2['combos']['wdd']:.3f}, feature {h2['combos']['feature']:.3f}, wdd+feature {h2['combos']['wdd_plus_feature']:.3f}, all {h2['combos']['all']:.3f}; "
        + (f"retrodiction two back class-side {res['retro'][2]['auc']['anymax']:.3f}; " if 2 in res["retro"] else "")
        + (f"geometry two before recruitment (n {gb[2]['n']}): class ratio first {gb[2]['rank_ratio_top1']:.2f} / top 10 {gb[2]['rank_ratio_top10']:.2f}, feature decoder top 10 {gb[2]['rank_dec_top10']:.2f} (decoder-row cosine {gb[2]['decoder_row_cos_median']:.2f}), feature partition top 10 {gb[2]['rank_fpart_top10']:.2f}" if 2 in gb else "geometry: too few recruits") + f" | {time.time() - t0:.0f}s")
log(summ); record(f"e629_forecast_{tag}_B{B}{sfx}", res, summ)
