"""e544: is the native fifth in the tails? e543 found that a Gaussian cloud with the real covariance at every checkpoint,
at any temporal coupling, gives words at S 1.08 with retention 0.37-0.42 against 1.28 and 0.50 for the real states, and
e542 that a native row's extreme projections persist beyond its bulk where an isotropic atom's regress like its bulk.
The states here are unit vectors (the program's S is angular by construction), so the question is angular: is the
native fifth carried by the sparse set of (position, row) pairs at which a native row's projection is extreme? Two
surgeries on the clouds, the real rows against each:
- grafting: the coupled Gaussian cloud of e543 (the real covariance, a latent persistence matched to the real states')
  with the real projections transplanted at each position's top-k rows by the real projection over the floor (k = 4,
  16, 64, 256 of 53,248), the rest of the state Gaussian;
- surgery: the real states with the projections at each position's top-k rows replaced by the Gaussian cloud's values
  (the real bulk with Gaussian tails), at k = 16 and 64.
Each modified state is solved exactly per position (a k-by-k Gram system), re-solved after unit-normalisation. (First
run: a threshold-defined pair set with a damped additive correction did not converge, the residual at the pairs 0.1-0.2
and the words' S inflated to 1.6-2.3; replaced by the exact per-position solve on the top-k pairs.) Recorded per cloud: the share of pairs touched, the covariance kept (the overlap of the top-32
subspace with the real's) and the realised per-position persistence; the prospective criterion, entries per interval,
retention, the words' median S, the exit gate and S profiles, the maximum's provenance, the level-matched persistence.
Also, on the real states, a prospective panel: for the native non-words at t, the shape of the projection distribution
over positions (variance, excess kurtosis, the 95th and 99th percentiles over the floor, their ratio, the count over
three quarters of the floor) and the persistence of the row's top-decile positions from t-1, as predictors of entry
at t+4, alone and within the band S 0.9-1.1.
Pre-registered (honest guesses), block 12:
- grafting each position's top 64 rows restores the plateau (the words' median S within 0.05 of the real, partly by
  construction), the retention to within 0.05 of the real and the entries per interval to within 15 (0.5); the top 4
  restore less than half of the retention gap (0.5); the top 256 restore it fully (0.6);
- the surgery on the top 64 collapses the retention to within 0.05 of the Gaussian clouds' and the gate to under 0.65 (0.6);
- both keep the covariance (top-32 overlap with the real above 0.85, the Gaussian base's own level) and the
  per-position persistence within 0.05 of the real (0.7);
- the excess kurtosis of a non-word's projections predicts entry four intervals ahead at AUC 0.85 or more alone,
  and within the band S 0.9-1.1 at 0.6 or more (0.5).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
name = sys.argv[1]; B = 12; K = 16; CDIR = f"/workspace/wdd/cache/e524_{name}"; steps = list(range(1000, 16001, 1000)); T = len(steps); NIT = 3
ck = {n: torch.load(f"{CDIR}/step{n}.pt", map_location="cpu") for n in steps}; m, D = ck[steps[0]]["rows"].shape; keepc = torch.stack([ck[n]["blocks"][B]["keep"] for n in steps]).all(0); N = int(keepc.sum())
Us = [unitr(ck[n]["blocks"][B]["U"][keepc].float()).to(DEV) for n in steps]; A_cpu = [ck[n]["rows"] for n in steps]; Wreal = [(ck[n]["rows"].float() * ck[n]["norms"][:, None]).to(DEV) for n in steps]
g = torch.Generator(device=DEV).manual_seed(0); V, lam = [], []
for t in range(T):
    e, Vt = torch.linalg.eigh((Us[t].T @ Us[t] / N).double()); V.append(Vt.flip(1).float()); lam.append(e.flip(0).clamp_min(0).float())
SQ = [(V[t] * lam[t].sqrt()[None]) @ V[t].T for t in range(T)]; rho_uni = min(float(torch.stack([(Us[t] * Us[t + 1]).sum(1) for t in range(T - 1)]).median()), 0.999)
z = torch.randn(N, D, device=DEV, generator=g); Gs = []
for t in range(T):
    if t > 0: z = rho_uni * z + math.sqrt(1 - rho_uni ** 2) * torch.randn(N, D, device=DEV, generator=g)
    Gs.append(unitr(z @ SQ[t]))
log(f"{name}: real per-position persistence {rho_uni:.3f}; coupled Gaussian base built")
def surgered(t, k, kind):
    """kind 'graft': the Gaussian cloud with the real projections at each position's top-k rows (by the real projection over the floor); 'surgery': the real states with the Gaussian cloud's projections at those pairs. The modification is solved exactly per position (a k-by-k Gram system), re-solved after unit-normalisation."""
    A = A_cpu[t].float().to(DEV); st = stats(Us[t], A, K); L = st["L"].to(DEV); del st
    P = Us[t] @ A.T; top = (P.abs() / L[:, None]).topk(k, dim=1).indices; Q = Gs[t] @ A.T; tgt = (P if kind == "graft" else Q).gather(1, top); del P, Q
    X = (Gs[t] if kind == "graft" else Us[t]).clone(); Ak = A[top]; G = Ak @ Ak.transpose(1, 2) + 1e-4 * torch.eye(k, device=DEV)[None]
    for _ in range(NIT):
        cur = (Ak @ X[:, :, None])[:, :, 0]; c = torch.linalg.solve(G, (tgt - cur)[:, :, None])[:, :, 0]; X = unitr(X + (c[:, None, :] @ Ak)[:, 0])
    cur = (Ak @ X[:, :, None])[:, :, 0]; resid = float((tgt - cur).abs().mean()); del A, Ak, G; torch.cuda.empty_cache()
    return X, float(k) / m, float(k), resid
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def prospective(H, words, Hh):
    a = []
    for t in range(len(words) - Hh):
        nonw = ~words[t]; ent = nonw & words[t + Hh]; v = auc(H[t]["S"][ent], H[t]["S"][nonw & ~ent])
        if v is not None: a.append(v)
    return mean(a)
def provenance(H, words):
    same, beyond = [], []
    for t in range(T - 1):
        W = torch.nonzero(words[t])[:, 0]; R0 = H[t]["ratio"][:, W].float(); R1 = H[t + 1]["ratio"][:, W].float(); ar = torch.arange(len(W)); a1 = R1.argmax(0); rk = (R0 > R0[a1, ar][None, :]).sum(0); same += (rk == 0).float().tolist(); beyond += (rk >= 20).float().tolist()
    return dict(same=mean(same), beyond=mean(beyond))
def level_matched(H, words):
    r0s, r1s = [], []
    for t in range(T - 1):
        cols = torch.nonzero(words[t])[:, 0]; R0 = H[t]["ratio"][:, cols].float(); R1 = H[t + 1]["ratio"][:, cols].float(); mk = R0 >= 1.0; r0s.append(R0[mk]); r1s.append(R1[mk])
    r0 = torch.cat(r0s); r1 = torch.cat(r1s); out = {}
    for lo_, hi_ in ((1.0, 1.1), (1.1, 1.2), (1.2, 1.4)):
        mk = (r0 >= lo_) & (r0 < hi_)
        if mk.sum() >= 20: out[f"{lo_}-{hi_}"] = dict(n=int(mk.sum()), mean_after=float(r1[mk].mean()))
    return out
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"; res = dict(model=name, block=B, rho_position=rho_uni, clouds={})
COND = [("real", None, None), ("gaussian_coupled", None, None), ("graft_top4", 4, "graft"), ("graft_top16", 16, "graft"), ("graft_top64", 64, "graft"), ("graft_top256", 256, "graft"), ("surgery_top16", 16, "surgery"), ("surgery_top64", 64, "surgery")]
for cname, tau, kind in COND:
    H = []; shares, perpos, resids, ov = [], [], [], []; cloud = []
    for t in range(T):
        if kind is None: X = Us[t] if cname == "real" else Gs[t]
        else: X, sh, pp, rs = surgered(t, tau, kind); shares.append(sh); perpos.append(pp); resids.append(rs)
        cloud.append(X); H.append(stats(X, unitr(Wreal[t]), K))
        if cname != "real": e_, Vx = torch.linalg.eigh((X.T @ X / N).double()); Vx = Vx.flip(1).float(); ov.append(float(((V[t][:, :32].T @ Vx[:, :32]) ** 2).sum() / 32))
    words = [wordset(h["usage"]) for h in H]; prof, _, thr, ex, en = profiles(H); hy = hysteresis(H, words); pv = provenance(H, words); lm = level_matched(H, words); realised = float(torch.stack([(cloud[t] * cloud[t + 1]).sum(1) for t in range(T - 1)]).median())
    X_, E_, F_ = prof["exits_aligned"], prof["entries_aligned"], prof["fractions"]
    res["clouds"][cname] = dict(threshold=tau, kind=kind, share_of_pairs_touched=mean(shares), pairs_per_position=mean(perpos), residual_at_pairs=mean(resids), top32_overlap_with_real=mean(ov), realised_position_persistence=realised, prospective_auc_1=prospective(H, words, 1), prospective_auc_4=prospective(H, words, 4), profiles=prof, hysteresis=hy, provenance=pv, level_matched=lm, words_median_S=float(torch.stack([h["S"] for h in H])[torch.stack(words)].median()), words_median_cnt75=float(torch.stack([h["cnt75"] for h in H])[torch.stack(words)].median()))
    o = res["clouds"][cname]; log(f"{name} cloud {cname}: pairs touched per position {f2(o['pairs_per_position'])} (share {o['share_of_pairs_touched'] if o['share_of_pairs_touched'] is None else round(o['share_of_pairs_touched'], 5)}, residual {f3(o['residual_at_pairs'])}), top-32 overlap with the real {f3(o['top32_overlap_with_real'])}, per-position persistence {f3(realised)}; criterion at 1/4 {f2(o['prospective_auc_1'])}/{f2(o['prospective_auc_4'])}, entries per interval {prof['entries_per_interval']:.0f}, retention at 4 {f2(prof['retention_4'])}, words' median S {f2(o['words_median_S'])} with {o['words_median_cnt75']:.0f} positions over three quarters; leavers' S at k=-1/0/+2 {f2(X_[-1]['S'])}/{f2(X_[0]['S'])}/{f2(X_[2]['S'])}, over the floor at k=0/+2 {f2(F_['leavers_over_floor_k0'])}/{f2(F_['leavers_over_floor_k2'])}; entrants' S at -2/-1/0/+2 {f2(E_[-2]['S'])}/{f2(E_[-1]['S'])}/{f2(E_[0]['S'])}/{f2(E_[2]['S'])}, under the floor at -1/-2 {f2(F_['entrants_under_floor_km1'])}/{f2(F_['entrants_under_floor_km2'])}; S change across exits/entries {f2(prof['delta_S_exit'])}/{f2(prof['delta_S_entry'])}; hysteresis at 1.1-1.2 {f2(hy['1.1-1.2']['matched_difference'])}; maximum at the same position {f2(pv['same'])}, beyond twenty {f2(pv['beyond'])}; level-matched persistence at 1.0-1.1/1.1-1.2/1.2-1.4 " + "/".join(f2(lm[k]["mean_after"]) if k in lm else "n/a" for k in ("1.0-1.1", "1.1-1.2", "1.2-1.4")))
    if cname == "real": Hreal = H
    else: del H
    del cloud; torch.cuda.empty_cache()
# ---------- the prospective panel on the real states ----------
words = [wordset(h["usage"]) for h in Hreal]; feats = {k: [] for k in ("S", "cnt75", "variance", "kurtosis", "q95", "q99", "q99_over_q95", "top_decile_persistence")}; ys = []; Sb = []
for t in range(1, T - 4):
    nonw = ~words[t]; ent = nonw & words[t + 4]; cols = torch.nonzero(nonw)[:, 0]; R = Hreal[t]["ratio"][:, cols].float().to(DEV); R0 = Hreal[t - 1]["ratio"][:, cols].float().to(DEV)
    S_ = R.max(0).values; c75 = (R > 0.75).sum(0).float(); mu = R.mean(0); var = R.var(0); kurt = ((R - mu[None]) ** 4).mean(0) / var.clamp_min(1e-12) ** 2 - 3; q = R.quantile(torch.tensor([0.95, 0.99], device=DEV), dim=0); q95, q99 = q[0], q[1]
    kk = max(N // 10, 10); top = R0.topk(kk, dim=0).indices; r0 = R0.gather(0, top); r1 = R.gather(0, top); c0 = r0 - r0.mean(0); c1 = r1 - r1.mean(0); tp = (c0 * c1).sum(0) / (c0.norm(dim=0) * c1.norm(dim=0)).clamp_min(1e-9)
    for k_, v_ in (("S", S_), ("cnt75", c75), ("variance", var), ("kurtosis", kurt), ("q95", q95), ("q99", q99), ("q99_over_q95", q99 / q95.clamp_min(1e-6)), ("top_decile_persistence", tp)): feats[k_].append(v_.cpu())
    ys.append(ent[cols]); Sb.append(S_.cpu()); del R, R0; torch.cuda.empty_cache()
y = torch.cat(ys); Sb = torch.cat(Sb); band = (Sb >= 0.9) & (Sb < 1.1); panel = {}
for k_, v_ in feats.items():
    x = torch.cat(v_); panel[k_] = dict(auc_alone=auc(x[y], x[~y]), auc_within_S_band=auc(x[y & band], x[~y & band]), n_entrants=int(y.sum()), n_entrants_in_band=int((y & band).sum()))
res["prospective_panel"] = panel; log(f"{name} prospective panel (native non-words at t, entry at t+4; {int(y.sum())} entrants of {y.numel()}, {int((y & band).sum())} in the band S 0.9-1.1 of {int(band.sum())}): " + "; ".join(f"{k_}: alone {f2(v_['auc_alone'])}, within the band {f2(v_['auc_within_S_band'])}" for k_, v_ in panel.items()))
C = res["clouds"]; g_ = lambda x: -9 if x is None else x; ret = lambda k: g_(C[k]["profiles"]["retention_4"]); gap = ret("real") - ret("gaussian_coupled")
res["checks"] = dict(graft_top64_restores=abs(g_(C["graft_top64"]["words_median_S"]) - g_(C["real"]["words_median_S"])) <= 0.05 and abs(ret("graft_top64") - ret("real")) <= 0.05 and abs(g_(C["graft_top64"]["profiles"]["entries_per_interval"]) - g_(C["real"]["profiles"]["entries_per_interval"])) <= 15,
                     graft_top4_restores_under_half=(ret("graft_top4") - ret("gaussian_coupled")) < 0.5 * gap, graft_top256_restores_fully=abs(ret("graft_top256") - ret("real")) <= 0.05,
                     surgery_top64_collapses=abs(ret("surgery_top64") - ret("gaussian_coupled")) <= 0.05 and g_(C["surgery_top64"]["profiles"]["fractions"]["leavers_over_floor_k0"]) < 0.65,
                     covariance_kept=all(g_(C[k]["top32_overlap_with_real"]) > 0.85 and abs(g_(C[k]["realised_position_persistence"]) - rho_uni) <= 0.05 for k in ("graft_top64", "surgery_top64")),
                     kurtosis_predicts=g_(panel["kurtosis"]["auc_alone"]) >= 0.85 and g_(panel["kurtosis"]["auc_within_S_band"]) >= 0.6)
summ = f"{name} block {B}: " + " | ".join(f"{k}: criterion {f2(o['prospective_auc_4'])}, entries {o['profiles']['entries_per_interval']:.0f}, retention {f2(o['profiles']['retention_4'])}, words' S {f2(o['words_median_S'])}, gate {f2(o['profiles']['fractions']['leavers_over_floor_k0'])}, same peak {f2(o['provenance']['same'])}, persistence at 1.1-1.2 {f2(o['level_matched'].get('1.1-1.2', {}).get('mean_after'))}, top-32 overlap {f3(o['top32_overlap_with_real'])}, pairs per position {f2(o['pairs_per_position'])}" for k, o in C.items()) + " | panel: " + ", ".join(f"{k_} {f2(v_['auc_alone'])}/{f2(v_['auc_within_S_band'])}" for k_, v_ in panel.items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e544_tail_graft_{name}", res, summ)
