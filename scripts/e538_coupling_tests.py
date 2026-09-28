"""e538: is the native fifth a property of the rows or of the trajectory ensemble, and can a conservation law give the
plateau? e537 named the ingredient, a temporally coherent motion directed into the cloud, and found it in Pythia's own
increment sequences. Five tests. In the synthetic world with memory and the pull: the pull's sign reversed (a
necessary-condition check), and the pull made a fixed-point law, proportional to the row's distance below a target
margin over the floor and vanishing there, at three targets, to see whether Pythia's plateau follows from a
conservation law rather than a fitted saturation. Against the real states with Pythia's real rows: the cloud turned by
one fixed random rotation at every checkpoint (each row's increment sequence intact, its alignment with the
contemporaneous cloud destroyed, the cloud's own slow evolution kept); the cloud resampled as Gaussian with each
checkpoint's covariance (the spectrum kept, the positions and tails not); the rows run backward in time against the
forward states; and the row-permuted replay of e537 with each received increment rescaled to the recipient's norm and
its component along the recipient row removed.
Pre-registered (honest guesses):
- the reversed pull drops retention to the isotropic level or below (0.31) and the criterion to 0.75 or below (0.7);
- the fixed-point law gives a plateau (the entrants' S two intervals after entry within 0.05 of one interval after,
  at the target 1.25) whose level tracks the target (0.5);
- the fixed rotation of the cloud drops the criterion to 0.86 or below and retention to 0.35 or below (0.6);
- the Gaussian-resampled cloud keeps retention within 0.10 of the real cloud's (0.4);
- the reversed-time rows give a criterion of 0.80 or below and retention of 0.35 or below (0.5);
- the radial-stripped replay is within 0.05 of the plain row-permuted replay in criterion and retention (0.6).
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
# ---------- part A: the synthetic worlds ----------
model, tok, fam = load_model(name, revision="step8000"); arch = Arch(model, fam); D = arch.D
for p in model.parameters(): p.requires_grad_(False)
X = block_states(model, arch, idsB, [B], chunk=4)[B].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc8 = Xk - Xk.mean(0); C = (Xc8.T @ Xc8 / Xc8.shape[0]).double(); del model, X; torch.cuda.empty_cache()
ev, V = torch.linalg.eigh(C); ev = ev.clamp_min(0); Cn = (C / ev.max()).float()
cloud = ((torch.randn(N, D, device=DEV, generator=gen, dtype=torch.float64) * ev.sqrt()[None]) @ V.T).float()
sigma = math.sqrt(2 * (1 - COS_STEP))
def noise(g):
    return unitr(torch.randn(R, D, device=DEV, generator=g))
WORLDS = {"memory_pull": dict(pull=0.6, target=None), "memory_pull_reversed": dict(pull=-0.6, target=None), "memory_fixed_point_1.10": dict(pull=0.6, target=1.10), "memory_fixed_point_1.25": dict(pull=0.6, target=1.25), "memory_fixed_point_1.40": dict(pull=0.6, target=1.40)}
for wname, cfg in WORLDS.items():
    g = torch.Generator(device=DEV).manual_seed(1); W = unitr(torch.randn(R, D, device=DEV, generator=g)); d = noise(g); hist = []; Ws = []
    for t in range(T):
        S, cnt, cnt75, usage = S_and_usage(cloud, W); hist.append(dict(S=S, cnt=cnt, cnt75=cnt75, usage=usage)); Ws.append(W.clone())
        eta = noise(g); d = unitr(RHO * d + math.sqrt(1 - RHO ** 2) * eta); step = d - (d * W).sum(1, keepdim=True) * W; step = unitr(step) * sigma
        pull = W @ Cn; pull = pull - (pull * W).sum(1, keepdim=True) * W; pull = unitr(pull)
        gain = torch.ones(R, 1, device=DEV) if cfg["target"] is None else ((cfg["target"] - S.to(DEV)) / (cfg["target"] - 0.9)).clamp(0, 1)[:, None]
        step = step + cfg["pull"] * sigma * gain * pull; W = unitr(W + step)
    res["synthetic"][wname] = analyse(hist, Ws, R, True); describe(f"synthetic {wname}", res["synthetic"][wname])
del cloud; torch.cuda.empty_cache()
# ---------- part B: the real rows against altered clouds, and altered rows against the real cloud ----------
steps = list(range(1000, 16001, 1000)); ck = {n: torch.load(f"{CDIR}/step{n}.pt") for n in steps}; Rn = (LB + 1) * 4096
Wreal = [(ck[n]["rows"].float() * ck[n]["norms"][:, None]).to(DEV) for n in steps]; dWs = [Wreal[i + 1] - Wreal[i] for i in range(T - 1)]
g = torch.Generator(device=DEV).manual_seed(2); perm = torch.randperm(Rn, device=DEV, generator=g)
Wstrip = [Wreal[0].clone()]
for i in range(T - 1):
    w = Wstrip[-1]; wu = unitr(w); dv = dWs[i][perm] * (w.norm(dim=1, keepdim=True) / Wreal[i][perm].norm(dim=1, keepdim=True).clamp_min(1e-9)); dv = dv - (dv * wu).sum(1, keepdim=True) * wu; Wstrip.append(w + dv)
Q = torch.linalg.qr(torch.randn(D, D, device=DEV, generator=g))[0]
Xs = {}; Xg = {}
for n in steps:
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam)
    for p in model.parameters(): p.requires_grad_(False)
    X = block_states(model, arch, idsB, [B], chunk=4)[B].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xs[n] = Xk - Xk.mean(0); del model, X; torch.cuda.empty_cache()
    Cn_ = (Xs[n].T @ Xs[n] / Xs[n].shape[0]).double(); e_, V_ = torch.linalg.eigh(Cn_); e_ = e_.clamp_min(0); Xg[n] = ((torch.randn(Xs[n].shape[0], D, device=DEV, generator=g, dtype=torch.float64) * e_.sqrt()[None]) @ V_.T).float()
log(f"{name}: clouds and rows built; radial-stripped rows' norm at 16000 {float(Wstrip[-1].norm(dim=1).median()):.3f} against real {float(Wreal[-1].norm(dim=1).median()):.3f}")
COND = {"real_rows": (Wreal, Xs, True), "cloud_rotated_fixed_Q": (Wreal, {n: Xs[n] @ Q for n in steps}, True), "cloud_gaussian_resampled": (Wreal, Xg, True), "reverse_time_rows": (Wreal[::-1], Xs, True), "row_permuted_radial_stripped": (Wstrip, Xs, True)}
for tag, (Wseq, Xd, moving) in COND.items():
    hist = []
    for i, n in enumerate(steps):
        S, cnt, cnt75, usage = S_and_usage(Xd[n], unitr(Wseq[i])); hist.append(dict(S=S, cnt=cnt, cnt75=cnt75, usage=usage))
    res["replay"][tag] = analyse(hist, [unitr(w) for w in Wseq], Rn, moving); describe(f"replay {tag}", res["replay"][tag])
sy = res["replay"]; sw = res["synthetic"]; g_ = lambda x: -9 if x is None else x
fp = sw["memory_fixed_point_1.25"]["entrants_aligned"]
res["checks"] = dict(reversed_pull_generic=sw["memory_pull_reversed"]["horizons"][4]["retention_mean"] <= 0.31 and g_(sw["memory_pull_reversed"]["horizons"][4]["prospective_auc_mean"]) <= 0.75, fixed_point_plateau=abs(g_(fp[2]["S"]) - g_(fp[1]["S"])) <= 0.05 and g_(sw["memory_fixed_point_1.40"]["entrants_aligned"][2]["S"]) > g_(sw["memory_fixed_point_1.10"]["entrants_aligned"][2]["S"]),
                     rotated_cloud_generic=g_(sy["cloud_rotated_fixed_Q"]["horizons"][4]["prospective_auc_mean"]) <= 0.86 and sy["cloud_rotated_fixed_Q"]["horizons"][4]["retention_mean"] <= 0.35, gaussian_cloud_within_0_1=abs(sy["cloud_gaussian_resampled"]["horizons"][4]["retention_mean"] - sy["real_rows"]["horizons"][4]["retention_mean"]) <= 0.10,
                     reverse_time_generic=g_(sy["reverse_time_rows"]["horizons"][4]["prospective_auc_mean"]) <= 0.80 and sy["reverse_time_rows"]["horizons"][4]["retention_mean"] <= 0.35, radial_stripped_same=abs(sy["row_permuted_radial_stripped"]["horizons"][4]["retention_mean"] - 0.47) <= 0.05 and abs(g_(sy["row_permuted_radial_stripped"]["horizons"][4]["prospective_auc_mean"]) - 0.88) <= 0.05)
summ = f"{name}: synthetic, AUC at 4 / retention at 4 / lag-1 / entrants' S at k=0/+1/+2: " + "; ".join(f"{k} {fm(o['horizons'][4]['prospective_auc_mean'])}/{o['horizons'][4]['retention_mean']:.2f}/{fm(o['lag1'])}/{fm(o['entrants_aligned'][0]['S'])}/{fm(o['entrants_aligned'][1]['S'])}/{fm(o['entrants_aligned'][2]['S'])}" for k, o in sw.items()) + " | against the real states: " + "; ".join(f"{k} {fm(o['horizons'][4]['prospective_auc_mean'])}/{o['horizons'][4]['retention_mean']:.2f}/{fm(o['lag1'])}/{fm(o['entrants_aligned'][0]['S'])}/{fm(o['entrants_aligned'][1]['S'])}/{fm(o['entrants_aligned'][2]['S'])}" for k, o in sy.items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e538_coupling_tests_{name}", res, summ)
