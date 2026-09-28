"""e537: which ingredient makes which statistic? e536 found the generic four fifths of the vocabulary phenomenology in
rows diffusing against a fixed cloud and the native levels supplied by a weak pull into the cloud. Two questions
follow. First, a factorial over the synthetic null: with the same marginal step size, isotropic diffusion; diffusion
whose steps are shaped by the cloud's covariance (geometry in the noise); temporally correlated diffusion, each row's
increment direction an AR(1) process with lag-1 correlation 0.3 and no relation to the cloud (memory only); memory
with covariance-shaped noise; the deterministic pull along the covariance (e536's B); and memory with the pull. Which
of the plateau above the floor, the breadth-first crossing, the increments' memory, the prospective criterion and the
retention does each produce? Second, the replay: Pythia's own thousand-step row increments (e524's records, blocks
0-12) permuted across rows, one permutation fixed over time (each row receives another row's whole increment sequence,
so the sizes and the autocorrelation of the increments are kept and only their coupling to the receiving row's own
start and context is destroyed) and permuted independently at every interval (the autocorrelation destroyed too),
against the real states at every checkpoint, beside the real rows and the rows frozen at 1000. If the native excess
survives the replay, it is the trajectory statistics; if it collapses, it is the coupling of each row's motion to its
own position.
Setup as e536 for the synthetic worlds (8192 rows, 2000 Gaussian states with the real block-12 covariance, sixteen
checkpoints, rows' cosine 0.96 per interval) and rows-only dictionaries on the 8 x 256 measurement text for the replay.
Pre-registered (honest guesses):
- memory alone raises retention over isotropic diffusion by under 0.10 (0.5);
- the deterministic pull gives the highest retention of the six worlds (0.6);
- only worlds with the pull show S rising past the floor after entry (1.1 or more two intervals after) (0.5);
- no world reproduces Pythia's plateau at 1.25-1.30 (0.6);
- the row-permuted replay drops the prospective AUC to the generic level (0.85 or below) and the retention to 0.35 or
  below (0.5), and the time-permuted replay does no worse (0.5).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; B = 12; NWORD = 256; K = 16; T = 16; R = 8192; N = 2000; COS_STEP = 0.96; RHO = 0.3; KMIN, KMAX = -4, 2; CDIR = f"/workspace/wdd/cache/e524_{name}"; LB = 12
idsB = eval_ids(name)[:8, :256].to(DEV); gen = torch.Generator(device=DEV).manual_seed(0)
def gabs(m, Tt=14.0, n=28001):
    t = torch.linspace(0, Tt, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def S_and_usage(Xc_, A):
    U = unitr(Xc_); m = A.shape[0]; Ar = unitr(rotate(A, seed=11)); CA = A.T @ A / m; CR = Ar.T @ Ar / m; gm = gabs(m); Nn = U.shape[0]
    s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 256] @ Ar.T).abs().max(1).values for s in range(0, Nn, 256)]); r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); L = s2.clamp_min(1e-12).sqrt() * gm * r_cal
    P = U @ A.T; ratio = P.abs() / L[:, None]; S = ratio.max(0).values; cnt = (ratio > 1).sum(0).float(); cnt75 = (ratio > 0.75).sum(0).float(); del P, ratio
    sel, _, _ = omp(Xc_, A, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m).float(); return S.cpu(), cnt.cpu(), cnt75.cpu(), usage.cpu()
def wordset(u):
    w = torch.zeros(u.numel(), dtype=torch.bool); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; return w
def analyse(hist, Ws, Rn, moving):
    words = [wordset(h["usage"]) for h in hist]; S_ = [h["S"] for h in hist]
    inc = [unitr(Ws[i + 1] - Ws[i]) for i in range(len(Ws) - 1)] if moving else None
    ac1 = float(torch.stack([(inc[i] * inc[i + 1]).sum(1) for i in range(len(inc) - 1)]).mean()) if moving else None
    ac4 = float(torch.stack([(inc[i] * inc[i + 4]).sum(1) for i in range(len(inc) - 4)]).mean()) if moving and len(inc) > 4 else None
    out = dict(rows_cos_1=float((Ws[1] * Ws[0]).sum(1).mean() / (Ws[1].norm(dim=1) * Ws[0].norm(dim=1)).mean()) if moving else 1.0, rows_cos_4=float(((Ws[4] * Ws[0]).sum(1) / (Ws[4].norm(dim=1) * Ws[0].norm(dim=1))).mean()) if moving else 1.0, lag1=ac1, lag4=ac4, horizons={})
    for H in (1, 4):
        aucs, ents, ret = [], [], []
        for t in range(len(words) - H):
            nonw = ~words[t]; ent = nonw & words[t + H]; a = auc(S_[t][ent], S_[t][nonw & ~ent])
            if a is not None: aucs.append(a)
            ents.append(int(ent.sum())); ret.append(float((words[t] & words[t + H]).sum() / NWORD))
        out["horizons"][H] = dict(prospective_auc_mean=float(sum(aucs) / len(aucs)) if aucs else None, entries_mean=float(sum(ents) / len(ents)), retention_mean=float(sum(ret) / len(ret)))
    ev_ = []
    for t in range(2, len(words) - 1):
        e = ~words[t - 2] & ~words[t - 1] & words[t] & words[t + 1]
        for r_ in torch.nonzero(e)[:, 0].tolist(): ev_.append((r_, t))
    matched = []; taken = torch.zeros(Rn, dtype=torch.bool)
    for r_, t in ev_:
        okm = ~words[t - 1] & ~words[t] & ~words[t + 1] & ~taken; okm[r_] = False; cand = torch.nonzero(okm)[:, 0]
        if cand.numel() == 0: continue
        j = cand[(S_[t - 1][cand] - S_[t - 1][r_]).abs().argmin()]; taken[j] = True; matched.append((int(j), t))
    def aligned(events):
        o = {}
        for kk in range(KMIN, KMAX + 1):
            vals = {q: [] for q in ("S", "cnt", "cnt75")}
            for r_, t in events:
                i = t + kk
                if 0 <= i < len(hist):
                    for q in vals: vals[q].append(float(hist[i][q][r_]))
            o[kk] = {q: (float(torch.tensor(v).median()) if v else None) for q, v in vals.items()}; o[kk]["n"] = len(vals["S"])
        return o
    out["n_clean_entries"] = len(ev_); out["entrants_aligned"] = aligned(ev_); out["matched_aligned"] = aligned(matched); return out
fm = lambda x: "n/a" if x is None else f"{x:.2f}"; f0 = lambda x: "n/a" if x is None else f"{x:.0f}"
def describe(tag, o):
    E, M = o["entrants_aligned"], o["matched_aligned"]
    log(f"{name} {tag}: rows' cosine across 1/4 intervals {o['rows_cos_1']:.3f}/{o['rows_cos_4']:.3f}, increments' lag-1/lag-4 cosine {fm(o['lag1'])}/{fm(o['lag4'])}; prospective AUC at 1/4 {fm(o['horizons'][1]['prospective_auc_mean'])}/{fm(o['horizons'][4]['prospective_auc_mean'])}, entries {o['horizons'][1]['entries_mean']:.0f}/{o['horizons'][4]['entries_mean']:.0f}, retention {o['horizons'][1]['retention_mean']:.2f}/{o['horizons'][4]['retention_mean']:.2f}; {o['n_clean_entries']} clean entries; entrants S / over floor / over 3/4 (matched) at k = -4..+2: " + " | ".join(f"{k:+d}: {fm(E[k]['S'])} / {f0(E[k]['cnt'])} / {f0(E[k]['cnt75'])} ({fm(M[k]['S'])} / {f0(M[k]['cnt'])} / {f0(M[k]['cnt75'])})" for k in range(KMIN, KMAX + 1)))
res = dict(model=name, synthetic={}, replay={})
# ---------- part A: the synthetic factorial ----------
model, tok, fam = load_model(name, revision="step8000"); arch = Arch(model, fam); D = arch.D
for p in model.parameters(): p.requires_grad_(False)
X = block_states(model, arch, idsB, [B], chunk=4)[B].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc8 = Xk - Xk.mean(0); C = (Xc8.T @ Xc8 / Xc8.shape[0]).double(); del model, X; torch.cuda.empty_cache()
ev, V = torch.linalg.eigh(C); ev = ev.clamp_min(0); Cn = (C / ev.max()).float()
cloud = ((torch.randn(N, D, device=DEV, generator=gen, dtype=torch.float64) * ev.sqrt()[None]) @ V.T).float()
sigma = math.sqrt(2 * (1 - COS_STEP))
def noise(g, shaped):
    z = torch.randn(R, D, device=DEV, generator=g, dtype=torch.float64)
    return unitr(((z * ev.sqrt()[None]) @ V.T).float()) if shaped else unitr(z.float())
WORLDS = {"isotropic": dict(rho=0.0, shaped=False, pull=0.0), "aligned_noise": dict(rho=0.0, shaped=True, pull=0.0), "memory": dict(rho=RHO, shaped=False, pull=0.0), "memory_aligned_noise": dict(rho=RHO, shaped=True, pull=0.0), "pull": dict(rho=0.0, shaped=False, pull=0.6), "memory_pull": dict(rho=RHO, shaped=False, pull=0.6)}
for wname, cfg in WORLDS.items():
    g = torch.Generator(device=DEV).manual_seed(1); W = unitr(torch.randn(R, D, device=DEV, generator=g)); d = noise(g, cfg["shaped"]); hist = []; Ws = []
    for t in range(T):
        S, cnt, cnt75, usage = S_and_usage(cloud, W); hist.append(dict(S=S, cnt=cnt, cnt75=cnt75, usage=usage)); Ws.append(W.clone())
        eta = noise(g, cfg["shaped"]); d = unitr(cfg["rho"] * d + math.sqrt(1 - cfg["rho"] ** 2) * eta) if cfg["rho"] > 0 else eta
        step = d - (d * W).sum(1, keepdim=True) * W; step = unitr(step) * sigma
        if cfg["pull"] > 0: pull = W @ Cn; pull = pull - (pull * W).sum(1, keepdim=True) * W; step = step + cfg["pull"] * sigma * unitr(pull)
        W = unitr(W + step)
    res["synthetic"][wname] = analyse(hist, Ws, R, True); describe(f"synthetic {wname}", res["synthetic"][wname])
del cloud; torch.cuda.empty_cache()
# ---------- part B: the replay of Pythia's increments against the real states ----------
steps = list(range(1000, 16001, 1000)); ck = {n: torch.load(f"{CDIR}/step{n}.pt") for n in steps}; Rn = (LB + 1) * 4096
Wreal = [(ck[n]["rows"].float() * ck[n]["norms"][:, None]).to(DEV) for n in steps]; dWs = [Wreal[i + 1] - Wreal[i] for i in range(T - 1)]
g = torch.Generator(device=DEV).manual_seed(2); perm = torch.randperm(Rn, device=DEV, generator=g)
Wrow = [Wreal[0].clone()]
for i in range(T - 1): Wrow.append(Wrow[-1] + dWs[i][perm])
Wtime = [Wreal[0].clone()]
for i in range(T - 1): Wtime.append(Wtime[-1] + dWs[i][torch.randperm(Rn, device=DEV, generator=g)])
Xs = {}
for n in steps:
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam)
    for p in model.parameters(): p.requires_grad_(False)
    X = block_states(model, arch, idsB, [B], chunk=4)[B].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xs[n] = Xk - Xk.mean(0); del model, X; torch.cuda.empty_cache()
log(f"{name}: replay dictionaries built; row norms at 16000, real {float(Wreal[-1].norm(dim=1).median()):.3f}, row-permuted {float(Wrow[-1].norm(dim=1).median()):.3f}, time-permuted {float(Wtime[-1].norm(dim=1).median()):.3f}")
for tag, Wseq, moving in (("real_rows", Wreal, True), ("frozen_at_1000", [Wreal[0]] * T, False), ("row_permuted_increments", Wrow, True), ("time_permuted_increments", Wtime, True)):
    hist = []
    for i, n in enumerate(steps):
        S, cnt, cnt75, usage = S_and_usage(Xs[n], unitr(Wseq[i])); hist.append(dict(S=S, cnt=cnt, cnt75=cnt75, usage=usage))
    res["replay"][tag] = analyse(hist, [unitr(w) for w in Wseq], Rn, moving); describe(f"replay {tag}", res["replay"][tag])
sy = res["synthetic"]; rp = res["replay"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(memory_alone_under_0_1=sy["memory"]["horizons"][4]["retention_mean"] - sy["isotropic"]["horizons"][4]["retention_mean"] < 0.10, pull_highest_retention=max(sy, key=lambda k: sy[k]["horizons"][4]["retention_mean"]) in ("pull", "memory_pull"),
                     only_pull_rises_past_floor=all((g_(sy[k]["entrants_aligned"][2]["S"]) >= 1.1) == (sy[k]["horizons"][4]["retention_mean"] > 0 and "pull" in k) for k in sy), no_world_reaches_1_25=all(g_(sy[k]["entrants_aligned"][2]["S"]) < 1.25 for k in sy),
                     replay_drops_to_generic=g_(rp["row_permuted_increments"]["horizons"][4]["prospective_auc_mean"]) <= 0.85 and rp["row_permuted_increments"]["horizons"][4]["retention_mean"] <= 0.35, time_permuted_no_worse=rp["time_permuted_increments"]["horizons"][4]["retention_mean"] >= rp["row_permuted_increments"]["horizons"][4]["retention_mean"] - 0.03)
summ = f"{name}: synthetic worlds, AUC at 4 / retention at 4 / lag-1 / entrants' S at k=+2: " + "; ".join(f"{k} {fm(o['horizons'][4]['prospective_auc_mean'])}/{o['horizons'][4]['retention_mean']:.2f}/{fm(o['lag1'])}/{fm(o['entrants_aligned'][2]['S'])}" for k, o in sy.items()) + " | replay against the real states: " + "; ".join(f"{k} {fm(o['horizons'][4]['prospective_auc_mean'])}/{o['horizons'][4]['retention_mean']:.2f}/{fm(o['lag1'])}/{fm(o['entrants_aligned'][2]['S'])}" for k, o in rp.items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e537_factorial_replay_{name}", res, summ)
