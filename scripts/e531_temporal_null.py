"""e531: is the prospective criterion specific to the native rows, or would any fixed dictionary show it? The
program's positive claim is that a row's largest projection over the extreme-value floor at one checkpoint predicts
its entry into the vocabulary at a later one (e516b, e518), and that the entry is the row moving (e522). The floor's
calibration used a rotated dictionary, but only statically; no fixed dictionary was ever followed through training.
Here three fixed dictionaries are followed across the sixteen checkpoints a thousand steps apart, with the states
moving and the atoms not: the native dictionary rotated by one orthogonal map (fixed at step 8000), a dictionary of
random unit atoms with the same size and block labels, and the native dictionary frozen at step 4000 (the actual
rows, no longer moving). For each, at each checkpoint and block, OMP usage defines the word set (the 256 most-used
row atoms) and the same statistic S is computed; the prospective test is the AUC of S at t for entry at t+1000 and
t+4000, beside the number of entries per interval, the static AUC of S for wordhood at t, and the persistence of S
across the horizon. The moving native dictionary from e524's records is the reference.
Pre-registered (honest guesses), block 12, horizon 4000, means over origins:
- the native rows' prospective AUC exceeds the rotated dictionary's by 0.15 or more (0.5);
- the frozen native dictionary's is at least 0.10 below the moving native's (0.5);
- the fixed dictionaries have at most half the native's entries per interval (0.5);
- S is more persistent across the horizon for the native rows than for random atoms (0.5).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; CDIR = f"/workspace/wdd/cache/e524_{name}"; BL = [12, 6]; LB = 12; NWORD = 256; K = 16
steps = list(range(1000, 16001, 1000)); ck = {n: torch.load(f"{CDIR}/step{n}.pt") for n in steps}
idsB = eval_ids(name)[:8, :256].to(DEV)
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def spearman(a, b):
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double(); ra = ra - ra.mean(); rb = rb - rb.mean(); return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))
# states at every checkpoint, and the native dictionaries at 4000 (to freeze) and 8000 (to rotate)
Xs = {}; A4 = A8 = None
for n in steps:
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF
    for p in model.parameters(): p.requires_grad_(False)
    X0 = block_states(model, arch, idsB, BL, chunk=4); Xs[n] = {b: X0[b].reshape(-1, D) for b in BL}
    if n == 4000: A4, lab4 = build_dictionary(arch, blocks=list(range(LB + 1)))
    if n == 8000: A8, lab8 = build_dictionary(arch, blocks=list(range(LB + 1)))
    del model, X0; torch.cuda.empty_cache()
log(f"{name}: states at {len(steps)} checkpoints; dictionary of {A8.shape[0]} atoms")
gen = torch.Generator(device=DEV).manual_seed(0)
DICTS = {"frozen_native_4000": (unitr(A4), lab4), "rotated": (unitr(rotate(A8, seed=7)), lab8), "random": (unitr(torch.randn(A8.shape[0], D, device=DEV, generator=gen)), lab8)}
del A4, A8
def restrict(Au, lab, b):
    """the dictionary's atoms of blocks <= b (embeddings included), and the index of the block's MLP rows among them"""
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); m = (blk <= b); Ab = Au[m]; typb, blkb, idxb = typ[m], blk[m], idx[m]
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typb == T_MLP) & (blkb == bb); gidx[bb, idxb[mm]] = torch.nonzero(mm)[:, 0]
    return Ab, gidx.reshape(-1)
PRE = {}
for k, (Au, lab) in DICTS.items():
    PRE[k] = {}
    for b in BL:
        Ab, rows = restrict(Au, lab, b); m = Ab.shape[0]; Ar = unitr(rotate(Ab, seed=11)); PRE[k][b] = dict(A=Ab, rows=rows, CA=Ab.T @ Ab / m, CR=Ar.T @ Ar / m, Ar=Ar, gm=gabs(m), Am=Ab[rows])
USG = {k: {b: {} for b in BL} for k in DICTS}; SS = {k: {b: {} for b in BL} for k in DICTS}
for n in steps:
    for b in BL:
        X = Xs[n][b]; keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]; mu = Xk.mean(0); Xc = Xk - mu; U = unitr(Xc)
        for k in DICTS:
            d = PRE[k][b]; sel, _, _ = omp(Xc, d["A"], K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=d["A"].shape[0])[d["rows"]].float()
            s2 = ((U @ d["CA"]) * U).sum(1); s2r = ((U @ d["CR"]) * U).sum(1); Mr = torch.cat([(U[s:s + 128] @ d["Ar"].T).abs().max(1).values for s in range(0, N, 128)])
            r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * d["gm"]).mean()); L = s2.clamp_min(1e-12).sqrt() * d["gm"] * r_cal
            P = U @ d["Am"].T; S = (P.abs() / L[:, None]).max(0).values; USG[k][b][n] = usage.cpu(); SS[k][b][n] = S.cpu(); del P
    log(f"{name} step{n}: usage and S computed for {len(DICTS)} fixed dictionaries")
USG["native"] = {b: {n: ck[n]["blocks"][b]["usage"].float() for n in steps} for b in BL}; SS["native"] = {b: {n: ck[n]["blocks"][b]["S"].float() for n in steps} for b in BL}
def wordset(u):
    w = torch.zeros(u.numel(), dtype=torch.bool); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; return w
res = dict(model=name, steps=steps, dictionaries=list(USG), by_block={})
for b in BL:
    out = {}
    for k in USG:
        W = {n: wordset(USG[k][b][n]) for n in steps}; per = {}
        for H in (1000, 4000):
            aucs, nents, stat, pers, ovl = [], [], [], [], []
            for n in steps:
                if n + H not in W: continue
                S0 = SS[k][b][n]; nonw = ~W[n]; ent = nonw & W[n + H]; a = auc(S0[ent], S0[nonw & ~ent])
                if a is not None: aucs.append(a)
                nents.append(int(ent.sum())); stat.append(auc(S0[W[n]], S0[~W[n]])); pers.append(spearman(S0, SS[k][b][n + H])); ovl.append(float((W[n] & W[n + H]).sum() / NWORD))
            per[H] = dict(prospective_auc_by_origin=aucs, prospective_auc_mean=float(sum(aucs) / len(aucs)) if aucs else None, entries_per_interval=nents, entries_mean=float(sum(nents) / len(nents)), static_auc_mean=float(sum(stat) / len(stat)), persistence_of_S_mean=float(sum(pers) / len(pers)), word_overlap_mean=float(sum(ovl) / len(ovl)))
        out[k] = per
    res["by_block"][b] = out
    fm = lambda x: "n/a" if x is None else f"{x:.2f}"
    for H in (1000, 4000):
        log(f"{name} block {b}, horizon {H}: " + " | ".join(f"{k}: prospective AUC {fm(out[k][H]['prospective_auc_mean'])} (by origin " + "/".join(f"{a:.2f}" for a in out[k][H]["prospective_auc_by_origin"]) + f"), entries per interval {out[k][H]['entries_mean']:.0f}, static AUC {out[k][H]['static_auc_mean']:.2f}, persistence of S {out[k][H]['persistence_of_S_mean']:.2f}, word overlap {out[k][H]['word_overlap_mean']:.2f}" for k in USG))
B = res["by_block"][12]; g_ = lambda k, H, q: B[k][H][q]
res["checks"] = dict(native_over_rotated_0_15=g_("native", 4000, "prospective_auc_mean") - g_("rotated", 4000, "prospective_auc_mean") >= 0.15, frozen_below_native_0_10=g_("native", 4000, "prospective_auc_mean") - g_("frozen_native_4000", 4000, "prospective_auc_mean") >= 0.10,
                     fixed_half_the_entries=all(g_(k, 4000, "entries_mean") <= 0.5 * g_("native", 4000, "entries_mean") for k in ("rotated", "random", "frozen_native_4000")), native_S_more_persistent=g_("native", 4000, "persistence_of_S_mean") > g_("random", 4000, "persistence_of_S_mean"))
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name}: " + " | ".join(f"block {b}: prospective AUC at 1000/4000 steps, entries per 4000 steps, persistence of S at 4000: " + "; ".join(f"{k} {fm(o[k][1000]['prospective_auc_mean'])}/{fm(o[k][4000]['prospective_auc_mean'])}, {o[k][4000]['entries_mean']:.0f}, {o[k][4000]['persistence_of_S_mean']:.2f}" for k in ("native", "frozen_native_4000", "rotated", "random")) for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e531_temporal_null_{name}", res, summ)
