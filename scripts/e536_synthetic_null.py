"""e536: the synthetic null. Does the phenomenology of vocabulary formation (a prospective criterion, breadth rising
before the maximum, a fast crossing, turnover, retention) appear in a system with no training and no language: unit
rows diffusing on the sphere against a Gaussian state cloud with the real block-12 covariance? Four regimes, each run
for sixteen checkpoints with the rows' per-interval rotation matched to Pythia's (cosine 0.96 per thousand steps):
(A) rows diffuse isotropically, the cloud fixed; (B) rows diffuse with a weak pull along the cloud's covariance, the
cloud fixed (a caricature of co-adaptation); (C) rows fixed, the cloud rotating slowly; (D) both moving. At each
checkpoint OMP usage on the cloud defines the words (the 256 most-used rows), and S is the largest projection over the
extreme-value floor with the dictionary's own calibration. Measured as for Pythia: the prospective AUC of S for entry
at one and four intervals, entries per interval, retention of the word set, the increments' lag-1 cosine, and the
trajectories of clean entrants aligned at entry (S, positions over the floor and over three quarters of it) against
S-matched non-entrants.
Pre-registered (honest guesses), regime A:
- the prospective AUC at four intervals is 0.8 or above (0.6);
- breadth rises before the maximum: two intervals before entry the entrants' count over three quarters of the floor
  is at least 1.5 times the matched rows' (0.5);
- the crossing is as fast as Pythia's: the entrants' S goes from 1.07 or less at two intervals before to 1.2 or
  more at entry (0.4);
- retention over four intervals is 0.25 or less (0.6);
- regime B's retention exceeds regime A's by 0.1 or more (0.4).
Arguments: name (the model whose block-12 covariance at step 8000 shapes the cloud)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; B = 12; NWORD = 256; K = 16; T = 16; R = 8192; N = 2000; COS_STEP = 0.96; KMIN, KMAX = -4, 2
idsB = eval_ids(name)[:8, :256].to(DEV); gen = torch.Generator(device=DEV).manual_seed(0)
model, tok, fam = load_model(name, revision="step8000"); arch = Arch(model, fam); D = arch.D
for p in model.parameters(): p.requires_grad_(False)
X = block_states(model, arch, idsB, [B], chunk=4)[B].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); C = (Xc.T @ Xc / Xc.shape[0]).double(); del model, X; torch.cuda.empty_cache()
ev, V = torch.linalg.eigh(C); ev = ev.clamp_min(0); Cn = (C / ev.max()).float()
def gabs(m, Tt=14.0, n=28001):
    t = torch.linspace(0, Tt, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def sample_cloud(g):
    z = torch.randn(N, D, device=DEV, generator=g, dtype=torch.float64); return ((z * ev.sqrt()[None]) @ V.T).float()
def small_rotation(angle, g):
    A_ = torch.randn(D, D, device=DEV, generator=g); A_ = A_ - A_.T; A_ = A_ / A_.norm() * angle * math.sqrt(D / 2); return torch.linalg.matrix_exp(A_)
def S_and_usage(Xc_, A):
    """OMP usage of each row and its largest projection over the floor calibrated for this dictionary"""
    U = unitr(Xc_); m = A.shape[0]; Ar = unitr(rotate(A, seed=11)); CA = A.T @ A / m; CR = Ar.T @ Ar / m; gm = gabs(m); Nn = U.shape[0]
    s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 256] @ Ar.T).abs().max(1).values for s in range(0, Nn, 256)]); r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); L = s2.clamp_min(1e-12).sqrt() * gm * r_cal
    P = U @ A.T; ratio = P.abs() / L[:, None]; S = ratio.max(0).values; cnt = (ratio > 1).sum(0).float(); cnt75 = (ratio > 0.75).sum(0).float()
    sel, _, _ = omp(Xc_, A, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m).float(); return S.cpu(), cnt.cpu(), cnt75.cpu(), usage.cpu()
def wordset(u):
    w = torch.zeros(u.numel(), dtype=torch.bool); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; return w
sigma = math.sqrt(2 * (1 - COS_STEP))                                                                                    # step on the sphere giving the target cosine between consecutive checkpoints
REG = {"A_rows_diffuse_cloud_fixed": dict(rows=True, pull=0.0, cloud=False), "B_rows_diffuse_with_pull_cloud_fixed": dict(rows=True, pull=0.6, cloud=False), "C_rows_fixed_cloud_rotates": dict(rows=False, pull=0.0, cloud=True), "D_both_move": dict(rows=True, pull=0.0, cloud=True)}
res = dict(model=name, D=D, rows=R, states=N, checkpoints=T, cos_per_interval=COS_STEP, regimes={})
for rname, cfg in REG.items():
    g = torch.Generator(device=DEV).manual_seed(1); W = unitr(torch.randn(R, D, device=DEV, generator=g)); Xc_ = sample_cloud(g); Ws = []; hist = []
    for t in range(T):
        S, cnt, cnt75, usage = S_and_usage(Xc_, W); hist.append(dict(S=S, cnt=cnt, cnt75=cnt75, usage=usage)); Ws.append(W.clone())
        if cfg["rows"]:
            xi = torch.randn(R, D, device=DEV, generator=g) * sigma / math.sqrt(D) * math.sqrt(D); xi = xi / xi.norm(dim=1, keepdim=True) * sigma
            if cfg["pull"] > 0: pull = W @ Cn; pull = pull - (pull * W).sum(1, keepdim=True) * W; xi = xi + cfg["pull"] * sigma * pull / pull.norm(dim=1, keepdim=True).clamp_min(1e-9)
            W = unitr(W + xi)
        if cfg["cloud"]: Xc_ = Xc_ @ small_rotation(0.30, g)
    words = [wordset(h["usage"]) for h in hist]; S_ = [h["S"] for h in hist]
    # increments' lag-1 cosine, and the rows' cosine across four intervals
    inc = [unitr(Ws[i + 1] - Ws[i]) for i in range(T - 1)]; ac1 = float(torch.stack([(inc[i] * inc[i + 1]).sum(1) for i in range(T - 2)]).mean()) if cfg["rows"] else None
    cos4 = float((Ws[4] * Ws[0]).sum(1).mean()); cos1 = float((Ws[1] * Ws[0]).sum(1).mean())
    out = dict(rows_cos_1_interval=cos1, rows_cos_4_intervals=cos4, increments_lag1_cos=ac1, horizons={})
    for H in (1, 4):
        aucs, ents, ret = [], [], []
        for t in range(T - H):
            nonw = ~words[t]; ent = nonw & words[t + H]; a = auc(S_[t][ent], S_[t][nonw & ~ent])
            if a is not None: aucs.append(a)
            ents.append(int(ent.sum())); ret.append(float((words[t] & words[t + H]).sum() / NWORD))
        out["horizons"][H] = dict(prospective_auc_mean=float(sum(aucs) / len(aucs)) if aucs else None, entries_mean=float(sum(ents) / len(ents)), retention_mean=float(sum(ret) / len(ret)))
    # clean entries aligned at entry, against S-matched non-entrants
    ev_ = []
    for t in range(2, T - 1):
        e = ~words[t - 2] & ~words[t - 1] & words[t] & words[t + 1]
        for r_ in torch.nonzero(e)[:, 0].tolist(): ev_.append((r_, t))
    matched = []; taken = torch.zeros(R, dtype=torch.bool)
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
                if 0 <= i < T:
                    for q in vals: vals[q].append(float(hist[i][q][r_]))
            o[kk] = {q: (float(torch.tensor(v).median()) if v else None) for q, v in vals.items()}; o[kk]["n"] = len(vals["S"])
        return o
    out["n_clean_entries"] = len(ev_); out["entrants_aligned"] = aligned(ev_); out["matched_aligned"] = aligned(matched)
    res["regimes"][rname] = out; fm = lambda x: "n/a" if x is None else f"{x:.2f}"; E, M = out["entrants_aligned"], out["matched_aligned"]
    log(f"{name} synthetic {rname}: rows' cosine across 1/4 intervals {cos1:.3f}/{cos4:.3f}, increments' lag-1 cosine {fm(ac1)}; prospective AUC at 1/4 intervals {fm(out['horizons'][1]['prospective_auc_mean'])}/{fm(out['horizons'][4]['prospective_auc_mean'])}, entries per interval {out['horizons'][1]['entries_mean']:.0f}/{out['horizons'][4]['entries_mean']:.0f}, retention {out['horizons'][1]['retention_mean']:.2f}/{out['horizons'][4]['retention_mean']:.2f}; {len(ev_)} clean entries; aligned at entry (k = intervals from entry), entrants S / over floor / over 3/4 (matched): " + " | ".join(f"k={k:+d} n={E[k]['n']}: {fm(E[k]['S'])} / {fm(E[k]['cnt'])} / {fm(E[k]['cnt75'])} ({fm(M[k]['S'])} / {fm(M[k]['cnt'])} / {fm(M[k]['cnt75'])})" for k in range(KMIN, KMAX + 1)))
A_ = res["regimes"]["A_rows_diffuse_cloud_fixed"]; EA, MA = A_["entrants_aligned"], A_["matched_aligned"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(diffusion_auc_over_0_8=g_(A_["horizons"][4]["prospective_auc_mean"]) >= 0.8, breadth_first=g_(EA[-2]["cnt75"]) >= 1.5 * max(g_(MA[-2]["cnt75"]), 1e-6), fast_crossing=g_(EA[-2]["S"]) <= 1.07 and g_(EA[0]["S"]) >= 1.2, retention_under_0_25=A_["horizons"][4]["retention_mean"] <= 0.25, pull_raises_retention_0_1=res["regimes"]["B_rows_diffuse_with_pull_cloud_fixed"]["horizons"][4]["retention_mean"] - A_["horizons"][4]["retention_mean"] >= 0.1)
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name} synthetic null ({R} rows, {N} Gaussian states with the real block-12 covariance, {T} checkpoints, rows' cosine {COS_STEP} per interval): " + " | ".join(f"{k}: AUC at 1/4 {fm(o['horizons'][1]['prospective_auc_mean'])}/{fm(o['horizons'][4]['prospective_auc_mean'])}, entries {o['horizons'][4]['entries_mean']:.0f}, retention at 4 {o['horizons'][4]['retention_mean']:.2f}, lag-1 {fm(o['increments_lag1_cos'])}, entrants S at k=-2/-1/0/+1 {fm(o['entrants_aligned'][-2]['S'])}/{fm(o['entrants_aligned'][-1]['S'])}/{fm(o['entrants_aligned'][0]['S'])}/{fm(o['entrants_aligned'][1]['S'])}, over 3/4 {fm(o['entrants_aligned'][-2]['cnt75'])}/{fm(o['entrants_aligned'][-1]['cnt75'])}/{fm(o['entrants_aligned'][0]['cnt75'])} (matched {fm(o['matched_aligned'][-2]['cnt75'])}/{fm(o['matched_aligned'][-1]['cnt75'])}/{fm(o['matched_aligned'][0]['cnt75'])})" for k, o in res["regimes"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e536_synthetic_null_{name}", res, summ)
