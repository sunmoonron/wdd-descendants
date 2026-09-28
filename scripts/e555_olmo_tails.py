"""e555: the tails on the second model. Sessions 92-93 found on Pythia that a Gaussian cloud with the real covariance
loses the native fifth and that the top four real projections per position grafted onto it restore it, while the real
states with those projections Gaussianised lose it. From the e550 cache of OLMo-1B (steps 1000-16000, block 8): the
real rows against the real states; a coupled Gaussian cloud with the real covariance at every checkpoint (the symmetric
square root, a latent persistence matched to the real per-position cosine); the same with the real projections
transplanted at each position's top-4 rows (solved exactly per position); and the real states with each position's
top-16 projections replaced by the Gaussian cloud's. Recorded as in e544: the criterion, entries, retention, the words'
S, the gate, the maximum's provenance, the level-matched persistence, the covariance kept.
Pre-registered (honest guesses), OLMo-1B block 8: the Gaussian cloud keeps under 0.4 of the retention (real 0.64) and
under 0.6 of the gate (0.6); the top-4 graft restores the retention to within 0.1 of the real and the gate to 0.8 or
more (0.6); the surgery collapses the retention below 0.3 (0.6).
Arguments: none."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmo_cache_common import *
name = "olmo1b"; C = load_olmo_cache(); T, N, m, D, B = C["T"], C["N"], C["m"], C["D"], C["B"]; Us, RU, NORM, Hn = C["Us"], C["RU"], C["NORM"], C["H"]; steps = C["steps"]; K = 16; NIT = 3
g = torch.Generator(device=DEV).manual_seed(0); V, lam = [], []
for t in range(T):
    e, Vt = torch.linalg.eigh((Us[t].T @ Us[t] / N).double()); V.append(Vt.flip(1).float()); lam.append(e.flip(0).clamp_min(0).float())
SQ = [(V[t] * lam[t].sqrt()[None]) @ V[t].T for t in range(T)]; rho = min(float(torch.stack([(Us[t] * Us[t + 1]).sum(1) for t in range(T - 1)]).median()), 0.999); z = torch.randn(N, D, device=DEV, generator=g); Gs = []
for t in range(T):
    if t > 0: z = rho * z + math.sqrt(1 - rho ** 2) * torch.randn(N, D, device=DEV, generator=g)
    Gs.append(unitr(z @ SQ[t]))
Wreal = lambda t: (RU[t].float() * NORM[t][:, None]).to(DEV)
def surgered(t, k, kind):
    A = RU[t].float().to(DEV); L = Hn[t]["L"].to(DEV); P = Us[t] @ A.T; top = (P.abs() / L[:, None]).topk(k, dim=1).indices; Q = Gs[t] @ A.T; tgt = (P if kind == "graft" else Q).gather(1, top); del P, Q
    X = (Gs[t] if kind == "graft" else Us[t]).clone(); Ak = A[top]; G = Ak @ Ak.transpose(1, 2) + 1e-4 * torch.eye(k, device=DEV)[None]
    for _ in range(NIT):
        cur = (Ak @ X[:, :, None])[:, :, 0]; c = torch.linalg.solve(G, (tgt - cur)[:, :, None])[:, :, 0]; X = unitr(X + (c[:, None, :] @ Ak)[:, 0])
    cur = (Ak @ X[:, :, None])[:, :, 0]; resid = float((tgt - cur).abs().mean()); del A, Ak, G; torch.cuda.empty_cache(); return X, resid
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
    same = []
    for t in range(T - 1):
        W = torch.nonzero(words[t])[:, 0]; R0 = H[t]["ratio"][:, W].float(); R1 = H[t + 1]["ratio"][:, W].float(); ar = torch.arange(len(W)); a1 = R1.argmax(0); same += ((R0 > R0[a1, ar][None, :]).sum(0) == 0).float().tolist()
    return mean(same)
def level_matched(H, words):
    r0s, r1s = [], []
    for t in range(T - 1):
        cols = torch.nonzero(words[t])[:, 0]; R0 = H[t]["ratio"][:, cols].float(); R1 = H[t + 1]["ratio"][:, cols].float(); mk = (R0 >= 1.1) & (R0 < 1.2); r0s.append(R0[mk]); r1s.append(R1[mk])
    r1 = torch.cat(r1s); return float(r1.mean()) if r1.numel() >= 20 else None
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"; res = dict(model=name, block=B, rho_position=rho, clouds={})
for cname, k, kind in (("real", None, None), ("gaussian_coupled", None, None), ("graft_top4", 4, "graft"), ("surgery_top16", 16, "surgery")):
    H = []; resids = []; ov = []; cloud = []
    for t in range(T):
        if kind is None: X = Us[t] if cname == "real" else Gs[t]
        else: X, rs = surgered(t, k, kind); resids.append(rs)
        cloud.append(X); H.append(stats(X, unitr(Wreal(t)), K))
        if cname != "real": e_, Vx = torch.linalg.eigh((X.T @ X / N).double()); Vx = Vx.flip(1).float(); ov.append(float(((V[t][:, :32].T @ Vx[:, :32]) ** 2).sum() / 32))
    words = [wordset(h["usage"]) for h in H]; prof, _, thr, ex, en = profiles(H); F_, E_, X_ = prof["fractions"], prof["entries_aligned"], prof["exits_aligned"]; realised = float(torch.stack([(cloud[t] * cloud[t + 1]).sum(1) for t in range(T - 1)]).median())
    res["clouds"][cname] = dict(residual_at_pairs=mean(resids), top32_overlap_with_real=mean(ov), realised_position_persistence=realised, prospective_auc_1=prospective(H, words, 1), prospective_auc_4=prospective(H, words, 4), entries_per_interval=prof["entries_per_interval"], retention_4=prof["retention_4"], words_median_S=float(torch.stack([h["S"] for h in H])[torch.stack(words)].median()), gate=F_["leavers_over_floor_k0"], entrants_under_floor_km2=F_["entrants_under_floor_km2"], delta_S_exit=prof["delta_S_exit"], delta_S_entry=prof["delta_S_entry"], leavers_S=[X_[k_]["S"] for k_ in (-1, 0, 2)], entrants_S=[E_[k_]["S"] for k_ in (-2, -1, 0, 2)], same_peak=provenance(H, words), persistence_1_1_1_2=level_matched(H, words))
    o = res["clouds"][cname]; log(f"{name} cloud {cname}: residual {f3(o['residual_at_pairs'])}, top-32 overlap with the real {f3(o['top32_overlap_with_real'])}, per-position persistence {f3(realised)}; criterion at 1/4 {f2(o['prospective_auc_1'])}/{f2(o['prospective_auc_4'])}, entries {o['entries_per_interval']:.0f}, retention {f2(o['retention_4'])}, words' S {f2(o['words_median_S'])}, leavers' S at -1/0/+2 " + "/".join(f2(x) for x in o["leavers_S"]) + f", gate {f2(o['gate'])}, entrants' S at -2/-1/0/+2 " + "/".join(f2(x) for x in o["entrants_S"]) + f", under the floor at -2 {f2(o['entrants_under_floor_km2'])}, S across exits/entries {f2(o['delta_S_exit'])}/{f2(o['delta_S_entry'])}, same peak {f2(o['same_peak'])}, persistence at 1.1-1.2 {f2(o['persistence_1_1_1_2'])}")
    del H, cloud; torch.cuda.empty_cache()
Cc = res["clouds"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(gaussian_loses=g_(Cc["gaussian_coupled"]["retention_4"]) < 0.4 and g_(Cc["gaussian_coupled"]["gate"]) < 0.6, graft_restores=abs(g_(Cc["graft_top4"]["retention_4"]) - g_(Cc["real"]["retention_4"])) <= 0.1 and g_(Cc["graft_top4"]["gate"]) >= 0.8, surgery_collapses=g_(Cc["surgery_top16"]["retention_4"]) < 0.3)
summ = f"{name} block {B}: " + " | ".join(f"{k}: criterion {f2(o['prospective_auc_4'])}, entries {o['entries_per_interval']:.0f}, retention {f2(o['retention_4'])}, words' S {f2(o['words_median_S'])}, gate {f2(o['gate'])}, same peak {f2(o['same_peak'])}, persistence at 1.1-1.2 {f2(o['persistence_1_1_1_2'])}" for k, o in Cc.items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e555_olmo_tails_{name}", res, summ)
