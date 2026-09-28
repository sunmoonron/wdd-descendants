"""e543 (v2: the clouds are built with the symmetric square root of the covariance and an operator coupling, after a first run whose coordinate-wise coupling in a reordering eigenbasis lost the persistence): does the covariance hold the maximum through the marginals or through the cloud's temporal dynamics? e541b
found atoms drawn with the state cloud's covariance persist like the native rows; e538 found Pythia's rows against a
Gaussian cloud resampled independently at every checkpoint keep most of the criterion and retention. Here the real
rows are run against five clouds with exactly the real covariance at every checkpoint but different dynamics:
the real states; independent Gaussian samples at every checkpoint (no per-position persistence); a coupled Gaussian
process with a uniform latent persistence matched to the real states' per-position cosine across a thousand steps; a
coupled process whose latent persistence follows the measured per-direction persistence of the real states (the
spectral dynamics); and a frozen latent (the cloud changes only through its covariance). For each: the prospective
criterion at one and four intervals, entries per interval and retention, the exit gate and S profiles, hysteresis, the
maximum's provenance, the level-matched persistence of the rows' positions, and the realised per-position persistence.
Pre-registered (honest guesses), block 12:
- the independent cloud keeps the criterion and most of the retention (as in e538) but has no per-position
  persistence: the rows' positions at 1.1-1.2 regress to under 1.0 a thousand steps later (0.7);
- the uniformly coupled cloud restores the per-position persistence to within 0.03 of the real (1.12 at 1.1-1.2) and
  the retention to within 0.05 of the real (0.5);
- the spectral cloud is within 0.02 of the uniform one in retention (0.5);
- the one-way gate is present in the independent cloud already, the leavers over the floor at k=0 in 0.6 or more,
  because the maximum persists through the marginal (0.5);
- the frozen latent has the highest retention, above the real (0.6).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
name = sys.argv[1]; B = 12; K = 16; CDIR = f"/workspace/wdd/cache/e524_{name}"; steps = list(range(1000, 16001, 1000)); T = len(steps)
ck = {n: torch.load(f"{CDIR}/step{n}.pt", map_location="cpu") for n in steps}; m, D = ck[steps[0]]["rows"].shape; keepc = torch.stack([ck[n]["blocks"][B]["keep"] for n in steps]).all(0); N = int(keepc.sum())
Us = [unitr(ck[n]["blocks"][B]["U"][keepc].float()).to(DEV) for n in steps]; Wreal = [(ck[n]["rows"].float() * ck[n]["norms"][:, None]).to(DEV) for n in steps]
g = torch.Generator(device=DEV).manual_seed(0); V, lam = [], []
for t in range(T):
    e, Vt = torch.linalg.eigh((Us[t].T @ Us[t] / N).double()); e = e.flip(0).clamp_min(0).float(); Vt = Vt.flip(1).float()
    if t > 0: sg = torch.sign((Vt * V[-1]).sum(0)); sg[sg == 0] = 1; Vt = Vt * sg[None]
    V.append(Vt); lam.append(e)
rho_t = []
for t in range(T - 1):
    Z0 = Us[t] @ V[t]; Z1 = Us[t + 1] @ V[t]; z0 = Z0 - Z0.mean(0); z1 = Z1 - Z1.mean(0); rho_t.append((z0 * z1).sum(0) / (z0.norm(dim=0) * z1.norm(dim=0)).clamp_min(1e-9))
rho_spec = torch.stack(rho_t).median(0).values.clamp(0, 0.999); rho_pos = float(torch.stack([(Us[t] * Us[t + 1]).sum(1) for t in range(T - 1)]).median()); rho_uni = min(rho_pos, 0.999)
log(f"{name}: real per-position cosine across a thousand steps {rho_pos:.3f}; per-direction persistence by rank 1-8 {float(rho_spec[:8].median()):.2f}, 9-32 {float(rho_spec[8:32].median()):.2f}, 33-128 {float(rho_spec[32:128].median()):.2f}, beyond 128 {float(rho_spec[128:].median()):.2f}")
SQ = [(V[t] * lam[t].sqrt()[None]) @ V[t].T for t in range(T)]  # symmetric square roots, continuous in the covariance (no eigenvector ordering)
R_ = [(V[t] * rho_spec[None]) @ V[t].T for t in range(T)]; Q_ = [(V[t] * (1 - rho_spec ** 2).sqrt()[None]) @ V[t].T for t in range(T)]
def make(kind):
    z = torch.randn(N, D, device=DEV, generator=g); out = []
    for t in range(T):
        if t > 0:
            eps = torch.randn(N, D, device=DEV, generator=g)
            if kind == "independent": z = eps
            elif kind == "coupled_uniform": z = rho_uni * z + math.sqrt(1 - rho_uni ** 2) * eps
            elif kind == "coupled_spectral": z = z @ R_[t - 1] + eps @ Q_[t - 1]
        out.append(unitr(z @ SQ[t]))
    return out
CLOUDS = {"real": Us, "independent": make("independent"), "coupled_uniform": make("coupled_uniform"), "coupled_spectral": make("coupled_spectral"), "frozen_latent": make("frozen_latent")}
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
    same, top5, beyond = [], [], []
    for t in range(T - 1):
        W = torch.nonzero(words[t])[:, 0]; R0 = H[t]["ratio"][:, W].float(); R1 = H[t + 1]["ratio"][:, W].float(); ar = torch.arange(len(W)); a1 = R1.argmax(0); rk = (R0 > R0[a1, ar][None, :]).sum(0)
        same += (rk == 0).float().tolist(); top5 += ((rk > 0) & (rk < 5)).float().tolist(); beyond += (rk >= 20).float().tolist()
    return dict(same=mean(same), top5=mean(top5), beyond=mean(beyond))
LB = ((1.0, 1.1), (1.1, 1.2), (1.2, 1.4))
def level_matched(H, words):
    r0s, r1s = [], []
    for t in range(T - 1):
        cols = torch.nonzero(words[t])[:, 0]; R0 = H[t]["ratio"][:, cols].float(); R1 = H[t + 1]["ratio"][:, cols].float(); mk = R0 >= 1.0; r0s.append(R0[mk]); r1s.append(R1[mk])
    r0 = torch.cat(r0s); r1 = torch.cat(r1s); out = {}
    for lo_, hi_ in LB:
        mk = (r0 >= lo_) & (r0 < hi_)
        if mk.sum() >= 20: out[f"{lo_}-{hi_}"] = dict(n=int(mk.sum()), mean_after=float(r1[mk].mean()))
    return out
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"; res = dict(model=name, block=B, rho_position=rho_pos, rho_spectral_by_rank={"1-8": float(rho_spec[:8].median()), "9-32": float(rho_spec[8:32].median()), "33-128": float(rho_spec[32:128].median()), "129-1024": float(rho_spec[128:].median())}, worlds={})
for wname, cloud in CLOUDS.items():
    H = []
    for t in range(T): H.append(stats(cloud[t], unitr(Wreal[t]), K))
    words = [wordset(h["usage"]) for h in H]; prof, _, thr, ex, en = profiles(H); hy = hysteresis(H, words); pv = provenance(H, words); lm = level_matched(H, words)
    realised = float(torch.stack([(cloud[t] * cloud[t + 1]).sum(1) for t in range(T - 1)]).median()); X, E, F = prof["exits_aligned"], prof["entries_aligned"], prof["fractions"]
    res["worlds"][wname] = dict(realised_position_persistence=realised, prospective_auc_1=prospective(H, words, 1), prospective_auc_4=prospective(H, words, 4), profiles=prof, hysteresis=hy, provenance=pv, level_matched=lm, words_median_S=float(torch.stack([h["S"] for h in H])[torch.stack(words)].median()))
    o = res["worlds"][wname]; log(f"{name} cloud {wname}: per-position persistence {f3(realised)}; criterion at 1/4 {f2(o['prospective_auc_1'])}/{f2(o['prospective_auc_4'])}, entries per interval {prof['entries_per_interval']:.0f}, retention at 4 {f2(prof['retention_4'])}, words' median S {f2(o['words_median_S'])}; leavers' S at k=-1/0/+2 {f2(X[-1]['S'])}/{f2(X[0]['S'])}/{f2(X[2]['S'])}, over the floor at k=0/+2 {f2(F['leavers_over_floor_k0'])}/{f2(F['leavers_over_floor_k2'])}, usage over the threshold at -1/0 {f2(X[-1]['usage_over_threshold'])}/{f2(X[0]['usage_over_threshold'])}; entrants' S at -2/-1/0/+2 {f2(E[-2]['S'])}/{f2(E[-1]['S'])}/{f2(E[0]['S'])}/{f2(E[2]['S'])}, under the floor at -1/-2 {f2(F['entrants_under_floor_km1'])}/{f2(F['entrants_under_floor_km2'])}; S change across exits/entries {f2(prof['delta_S_exit'])}/{f2(prof['delta_S_entry'])}; hysteresis at 1.1-1.2 {f2(hy['1.1-1.2']['matched_difference'])}; maximum at the same position {f2(pv['same'])}, beyond twenty {f2(pv['beyond'])}; level-matched persistence at 1.0-1.1/1.1-1.2/1.2-1.4 " + "/".join(f2(lm[k]["mean_after"]) if k in lm else "n/a" for k in ("1.0-1.1", "1.1-1.2", "1.2-1.4")))
    del H; torch.cuda.empty_cache()
w = res["worlds"]; g_ = lambda x: -9 if x is None else x; lmv = lambda k, b: g_(w[k]["level_matched"].get(b, {}).get("mean_after"))
res["checks"] = dict(independent_no_position_persistence=lmv("independent", "1.1-1.2") < 1.0 and g_(w["independent"]["prospective_auc_4"]) >= 0.8, uniform_restores=abs(lmv("coupled_uniform", "1.1-1.2") - lmv("real", "1.1-1.2")) <= 0.03 and abs(g_(w["coupled_uniform"]["profiles"]["retention_4"]) - g_(w["real"]["profiles"]["retention_4"])) <= 0.05,
                     spectral_equals_uniform=abs(g_(w["coupled_spectral"]["profiles"]["retention_4"]) - g_(w["coupled_uniform"]["profiles"]["retention_4"])) <= 0.02, gate_in_independent=g_(w["independent"]["profiles"]["fractions"]["leavers_over_floor_k0"]) >= 0.6, frozen_highest_retention=g_(w["frozen_latent"]["profiles"]["retention_4"]) > g_(w["real"]["profiles"]["retention_4"]))
summ = f"{name} block {B}: real per-position persistence {rho_pos:.3f}; " + " | ".join(f"{k}: persistence {f3(o['realised_position_persistence'])}, criterion {f2(o['prospective_auc_1'])}/{f2(o['prospective_auc_4'])}, entries {o['profiles']['entries_per_interval']:.0f}, retention {f2(o['profiles']['retention_4'])}, words' S {f2(o['words_median_S'])}, leavers over the floor {f2(o['profiles']['fractions']['leavers_over_floor_k0'])}, S across exits/entries {f2(o['profiles']['delta_S_exit'])}/{f2(o['profiles']['delta_S_entry'])}, same peak {f2(o['provenance']['same'])}, persistence at 1.1-1.2 {f2(o['level_matched'].get('1.1-1.2', {}).get('mean_after'))}" for k, o in w.items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e543_cloud_dynamics_null_{name}", res, summ)
