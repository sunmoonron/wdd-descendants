"""e535: does the vocabulary phenomenology survive the removal of the vocabulary machinery? Wordhood in this program is
OMP usage: a row is a word when it is among the 256 rows most often selected by orthogonal matching pursuit with sixteen
atoms. If the prospective criterion, the native rows' retention advantage and the entry statistics are properties of
the representation's geometry, they should appear under other definitions of wordhood; if they are artefacts of OMP,
they should not. Four definitions, each at every one of the sixteen Pythia checkpoints, for the native dictionary and
for a fixed random dictionary: OMP usage; nearest-neighbour usage (each state assigned to the single atom it is most
correlated with); top-16 correlation usage (each state's sixteen most correlated atoms, no orthogonalisation); and the
count of positions at which the atom clears the extreme-value floor (no reconstruction at all). For each, the words
are the 256 most-used row atoms; measured: the overlap of the words with the OMP words at the same checkpoint, the
prospective AUC of S for entry at one and four thousand steps, the entries per interval and the retention of the word
set, for the native dictionary and the random one.
Pre-registered (honest guesses), block 12:
- the OMP words and the nearest-neighbour words overlap at 0.6 or more (0.5);
- under the nearest-neighbour definition the prospective AUC of S at four thousand steps is within 0.05 of the OMP's (0.5);
- the native retention advantage holds under every definition, native at least twice random (0.6);
- the count-over-floor definition gives the highest prospective AUC (0.6).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; BL = [12, 6]; LB = 12; NWORD = 256; K = 16; DEFS = ["omp", "nearest_neighbour", "top16_correlation", "count_over_floor"]
steps = list(range(1000, 16001, 1000)); idsB = eval_ids(name)[:8, :256].to(DEV); gen = torch.Generator(device=DEV).manual_seed(0)
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def restrict(Au, lab, b, DFF):
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); mk = (blk <= b); Ab = Au[mk]; typb, blkb, idxb = typ[mk], blk[mk], idx[mk]
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typb == T_MLP) & (blkb == bb); gidx[bb, idxb[mm]] = torch.nonzero(mm)[:, 0]
    return Ab, gidx.reshape(-1)
def measures(Xc, A, rows):
    """usage under the four definitions (over the row atoms) and S per row"""
    U = unitr(Xc); m = A.shape[0]; N = U.shape[0]; Ar = unitr(rotate(A, seed=11)); CA = A.T @ A / m; CR = Ar.T @ Ar / m; gm = gabs(m)
    s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 128] @ Ar.T).abs().max(1).values for s in range(0, N, 128)]); r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); L = s2.clamp_min(1e-12).sqrt() * gm * r_cal
    sel, _, _ = omp(Xc, A, K, batch=1024, record_err=False); u_omp = torch.bincount(sel.reshape(-1), minlength=m)[rows].float()
    u_nn = torch.zeros(m, device=DEV); u_top = torch.zeros(m, device=DEV)
    for s in range(0, N, 128):
        P = (U[s:s + 128] @ A.T).abs(); u_nn += torch.bincount(P.argmax(1), minlength=m).float(); u_top += torch.bincount(P.topk(K, dim=1).indices.reshape(-1), minlength=m).float()
    Pr = U @ A[rows].T; ratio = Pr.abs() / L[:, None]; S = ratio.max(0).values; cnt = (ratio > 1).sum(0).float()
    return {"omp": u_omp.cpu(), "nearest_neighbour": u_nn[rows].cpu(), "top16_correlation": u_top[rows].cpu(), "count_over_floor": (cnt + 1e-3 * S).cpu()}, S.cpu()
def wordset(u):
    w = torch.zeros(u.numel(), dtype=torch.bool); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; return w
USG = {d: {b: {k: {} for k in DEFS} for b in BL} for d in ("native", "random")}; SS = {d: {b: {} for b in BL} for d in ("native", "random")}; Rd = None
for n in steps:
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF
    for p in model.parameters(): p.requires_grad_(False)
    X0 = block_states(model, arch, idsB, BL, chunk=4); A, lab = build_dictionary(arch, blocks=list(range(LB + 1))); Au = unitr(A); del A, model; torch.cuda.empty_cache()
    if Rd is None: Rd = unitr(torch.randn(Au.shape[0], D, device=DEV, generator=gen)); labR = lab
    for b in BL:
        X = X0[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0)
        for dname, (Ad, labd) in (("native", (Au, lab)), ("random", (Rd, labR))):
            Ab, rows = restrict(Ad, labd, b, DFF); us, S = measures(Xc, Ab, rows); SS[dname][b][n] = S
            for k in DEFS: USG[dname][b][k][n] = us[k]
    del Au, X0; torch.cuda.empty_cache(); log(f"{name} step{n}: usage under {len(DEFS)} definitions for native and random dictionaries")
res = dict(model=name, steps=steps, definitions=DEFS, by_block={})
for b in BL:
    out = {}
    for dname in ("native", "random"):
        W = {k: {n: wordset(USG[dname][b][k][n]) for n in steps} for k in DEFS}; out[dname] = {}
        for k in DEFS:
            ov = [float((W[k][n] & W["omp"][n]).sum() / NWORD) for n in steps]; per = dict(overlap_with_omp_words=float(sum(ov) / len(ov)), horizons={})
            for H in (1000, 4000):
                aucs, ents, ret = [], [], []
                for n in steps:
                    if n + H not in W[k]: continue
                    S0 = SS[dname][b][n]; nonw = ~W[k][n]; ent = nonw & W[k][n + H]; a = auc(S0[ent], S0[nonw & ~ent])
                    if a is not None: aucs.append(a)
                    ents.append(int(ent.sum())); ret.append(float((W[k][n] & W[k][n + H]).sum() / NWORD))
                per["horizons"][H] = dict(prospective_auc_mean=float(sum(aucs) / len(aucs)) if aucs else None, entries_mean=float(sum(ents) / len(ents)), retention_mean=float(sum(ret) / len(ret)))
            out[dname][k] = per
    res["by_block"][b] = out; fm = lambda x: "n/a" if x is None else f"{x:.2f}"
    for dname in ("native", "random"):
        log(f"{name} block {b} {dname}: " + " | ".join(f"{k}: overlap with OMP words {out[dname][k]['overlap_with_omp_words']:.2f}, prospective AUC at 1000/4000 {fm(out[dname][k]['horizons'][1000]['prospective_auc_mean'])}/{fm(out[dname][k]['horizons'][4000]['prospective_auc_mean'])}, entries per 4000 {out[dname][k]['horizons'][4000]['entries_mean']:.0f}, retention at 1000/4000 {out[dname][k]['horizons'][1000]['retention_mean']:.2f}/{out[dname][k]['horizons'][4000]['retention_mean']:.2f}" for k in DEFS))
B12 = res["by_block"][12]; g_ = lambda d, k, H, q: B12[d][k]["horizons"][H][q]
res["checks"] = dict(nn_overlap_over_0_6=B12["native"]["nearest_neighbour"]["overlap_with_omp_words"] >= 0.6, nn_auc_within_0_05=abs((g_("native", "nearest_neighbour", 4000, "prospective_auc_mean") or 0) - (g_("native", "omp", 4000, "prospective_auc_mean") or 0)) <= 0.05,
                     retention_gap_every_definition=all(g_("native", k, 4000, "retention_mean") >= 2 * g_("random", k, 4000, "retention_mean") for k in DEFS), count_over_floor_highest_auc=all(g_("native", "count_over_floor", 4000, "prospective_auc_mean") >= g_("native", k, 4000, "prospective_auc_mean") for k in DEFS))
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name}: " + " | ".join(f"block {b}: " + "; ".join(f"{k}: overlap with OMP {o['native'][k]['overlap_with_omp_words']:.2f}, AUC at 4000 native/random {fm(o['native'][k]['horizons'][4000]['prospective_auc_mean'])}/{fm(o['random'][k]['horizons'][4000]['prospective_auc_mean'])}, retention at 4000 native/random {o['native'][k]['horizons'][4000]['retention_mean']:.2f}/{o['random'][k]['horizons'][4000]['retention_mean']:.2f}, entries {o['native'][k]['horizons'][4000]['entries_mean']:.0f}/{o['random'][k]['horizons'][4000]['entries_mean']:.0f}" for k in DEFS) for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e535_wordhood_without_omp_{name}", res, summ)
