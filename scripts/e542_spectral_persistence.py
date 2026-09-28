"""e542: which directions of the cloud persist, and do the rows sit in them? e541 and e541b found that what holds a
native word's maximum above the floor is the persistence of its alignments, shared by any direction drawn with the
state cloud's covariance. This asks what in the cloud persists. From the e524 cache (unit centred states at every
position kept throughout, the unit rows of blocks 0-12, sixteen checkpoints, block 12):
- the states' per-direction persistence: for the eigenvectors of the covariance at t, the correlation across positions
  between the states' components at t and at t+1, as a function of the eigenvalue rank; and the dynamic modes, the
  eigenvectors of the symmetrised lag-one cross-covariance, against the variance modes (the overlap of the top-k
  subspaces);
- where the rows sit: the weight of the native words, native non-words, isotropic random atoms and covariance-matched
  atoms in the top-k variance subspace and in the top-k dynamic subspace (k = 8, 32, 128, 512), and their
  persistence-weighted alignment (the sum over directions of the persistence times the squared component), which is
  the expected persistence of a row's projections;
- transport: the top-32 subspace's rotation per thousand steps, and for the native words the three norms, the weight
  of the row at t in the subspace at t, of the row at t+1 in the subspace at t+1, and of the row at t in the subspace
  at t+1 (does the row move into a moving subspace, or does the subspace carry the row);
- lead and lag: the native words' weight in the subspace frozen at step 4000 and in the final subspace (step 16000)
  at every checkpoint against the contemporaneous, the entrants' weight around entry (k = -4..+2) in the contemporaneous
  and final subspaces, and, underpowered with fifteen differences, the cross-correlation at lags -4..4 between the
  change of the rows' subspace weight and the change of the subspace's variance share.
Pre-registered (honest guesses), block 12:
- persistence falls with the eigenvalue rank: the top 32 directions have a median persistence above 0.9 and the
  directions beyond rank 256 below 0.5 (0.6);
- the dynamic modes are the variance modes: the top-32 subspaces overlap at 0.9 or more (0.6);
- the native words' persistence-weighted alignment is at least twice the isotropic atoms', with the covariance-matched
  atoms between (0.6);
- the row moves into a slowly moving subspace: the row at t has the same weight in the subspace at t+1 as at t within
  0.02 (0.6);
- the rows do not lead the cloud: at steps 1000-4000 the native words' weight in the final top-32 subspace is no
  higher than in the contemporaneous one (0.6).
Added for the second run (v2), after the first showed the states persisting in most directions: the empirical persistence
of a row's whole projection profile, and the conditional persistence by level (the mean at t+1 over the mean at t within
bins of the level), for native words, native non-words, isotropic and covariance-matched atoms. Pre-registered:
- the whole-profile persistence is the same for native words and isotropic atoms within 0.05 (0.5);
- the extremes persist only for the native rows: the conditional persistence at 1.0-1.2 is above 0.95 for native words
  and below 0.9 for isotropic atoms, while in the bulk (0.4-0.6) the two are within 0.05 (0.6).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
name = sys.argv[1]; B = 12; K = 16; CDIR = f"/workspace/wdd/cache/e524_{name}"; steps = list(range(1000, 16001, 1000)); T = len(steps); KS = [8, 32, 128, 512]
ck = {n: torch.load(f"{CDIR}/step{n}.pt", map_location="cpu") for n in steps}; m, D = ck[steps[0]]["rows"].shape; keepc = torch.stack([ck[n]["blocks"][B]["keep"] for n in steps]).all(0); N = int(keepc.sum())
Us = [unitr(ck[n]["blocks"][B]["U"][keepc].float()).to(DEV) for n in steps]; Wn = [ck[n]["rows"].float().to(DEV) for n in steps]
g = torch.Generator(device=DEV).manual_seed(0); Arand = unitr(torch.randn(m, D, device=DEV, generator=g))
C8 = (Us[7].T @ Us[7] / N).double(); e8, V8 = torch.linalg.eigh(C8); Acov = unitr(((torch.randn(m, D, device=DEV, generator=g, dtype=torch.float64) * e8.clamp_min(0).sqrt()[None]) @ V8.T).float())
# words from OMP over the rows
words = []
for t in range(T):
    st = stats(Us[t], Wn[t], K); del st["ratio"]; words.append(wordset(st["usage"]))
log(f"{name}: words at {T} checkpoints, {N} positions")
# eigenbases, aligned in sign from one checkpoint to the next
V, lam = [], []
for t in range(T):
    e, Vt = torch.linalg.eigh((Us[t].T @ Us[t] / N).double()); e = e.flip(0).clamp_min(0).float(); Vt = Vt.flip(1).float()
    if t > 0: sg = torch.sign((Vt * V[-1]).sum(0)); sg[sg == 0] = 1; Vt = Vt * sg[None]
    V.append(Vt); lam.append(e)
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
res = dict(model=name, block=B, steps=steps, persistence_by_rank={}, dynamic_vs_variance={}, row_weights={}, persistence_weighted={}, transport={}, lead_lag={}, entrants={})
RB = [(0, 8), (8, 32), (32, 128), (128, 256), (256, 512), (512, 1024)]; rho_all = []; overlap = {k: [] for k in KS}; dyn_weights = {k: {} for k in KS}
def wsq(A, Vk): return ((A @ Vk) ** 2).sum(1)
def subspace_overlap(Va, Vb, k): return float(((Va[:, :k].T @ Vb[:, :k]) ** 2).sum() / k)
for t in range(T - 1):
    Z0 = Us[t] @ V[t]; Z1 = Us[t + 1] @ V[t]; z0 = Z0 - Z0.mean(0); z1 = Z1 - Z1.mean(0); rho = (z0 * z1).sum(0) / (z0.norm(dim=0) * z1.norm(dim=0)).clamp_min(1e-9); rho_all.append(rho.cpu())
    M = (Us[t + 1].T @ Us[t] / N); Ms = ((M + M.T) / 2).double(); em, Q = torch.linalg.eigh(Ms); Q = Q.flip(1).float()
    for k in KS:
        overlap[k].append(subspace_overlap(V[t], Q, k))
        for lab, A in (("native_words", Wn[t][words[t]]), ("native_nonwords", Wn[t][~words[t]]), ("random", Arand), ("covariance", Acov)): dyn_weights[k].setdefault(lab, []).append(float(wsq(A, Q[:, :k]).mean()))
rho_all = torch.stack(rho_all); res["persistence_by_rank"] = {f"{a + 1}-{b}": float(rho_all[:, a:b].median()) for a, b in RB}; res["persistence_by_rank"]["mean_over_ranks"] = float(rho_all.mean())
res["persistence_by_rank"]["n_directions_over_0_9"] = float((rho_all.median(0).values > 0.9).sum()); res["persistence_by_rank"]["n_directions_over_0_5"] = float((rho_all.median(0).values > 0.5).sum())
res["dynamic_vs_variance"] = {"overlap_top_k": {str(k): mean(v) for k, v in overlap.items()}, "weights_in_dynamic_top_k": {str(k): {lab: mean(v) for lab, v in d.items()} for k, d in dyn_weights.items()}}
log(f"{name} persistence of the states' components by eigenvalue rank (median over directions and transitions): " + ", ".join(f"{k}: {f2(v)}" for k, v in res["persistence_by_rank"].items() if "-" in k) + f"; directions with median persistence over 0.9: {res['persistence_by_rank']['n_directions_over_0_9']:.0f}, over 0.5: {res['persistence_by_rank']['n_directions_over_0_5']:.0f}; overlap of the dynamic and variance top-k subspaces: " + ", ".join(f"k={k}: {f2(v)}" for k, v in res["dynamic_vs_variance"]["overlap_top_k"].items()))
# where the rows sit, and the persistence-weighted alignment
rho_med = rho_all.median(0).values.to(DEV).clamp(0, 1); W_ = {k: {} for k in KS}; PW = {}
for t in range(T - 1):
    for lab, A in (("native_words", Wn[t][words[t]]), ("native_nonwords", Wn[t][~words[t]]), ("random", Arand), ("covariance", Acov)):
        for k in KS: W_[k].setdefault(lab, []).append(float(wsq(A, V[t][:, :k]).mean()))
        Z2 = ((A @ V[t]) ** 2) * lam[t][None]; PW.setdefault(lab, []).append(float(((Z2 * rho_all[t].to(DEV).clamp(0, 1)[None]).sum(1) / Z2.sum(1).clamp_min(1e-12)).mean()))
res["row_weights"] = {str(k): {lab: mean(v) for lab, v in d.items()} for k, d in W_.items()}; res["persistence_weighted"] = {lab: mean(v) for lab, v in PW.items()}
log(f"{name} weight in the top-k variance subspace (mean over rows and checkpoints): " + "; ".join(f"k={k}: " + ", ".join(f"{lab} {f3(v)}" for lab, v in d.items()) for k, d in res["row_weights"].items()) + " | in the top-k dynamic subspace: " + "; ".join(f"k={k}: " + ", ".join(f"{lab} {f3(v)}" for lab, v in d.items()) for k, d in res["dynamic_vs_variance"]["weights_in_dynamic_top_k"].items()))
# empirical persistence of a row's whole projection profile, and the conditional persistence by level (the mean at t+1 over the mean at t within bins of the level at t)
LV = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.5)]; prof = {}; cond = {}
gs = torch.Generator(device=DEV).manual_seed(5); sub_r = torch.randperm(m, device=DEV, generator=gs)[:8192]; sub_c = torch.randperm(m, device=DEV, generator=gs)[:8192]
for t in range(T - 1):
    stn = stats(Us[t], Wn[t], K); Ln0 = stn["L"].to(DEV); stn1 = stats(Us[t + 1], Wn[t + 1], K); Ln1 = stn1["L"].to(DEV); del stn, stn1
    str0 = stats(Us[t], Arand, K); Lr0 = str0["L"].to(DEV); str1 = stats(Us[t + 1], Arand, K); Lr1 = str1["L"].to(DEV); del str0, str1
    stc0 = stats(Us[t], Acov, K); Lc0 = stc0["L"].to(DEV); stc1 = stats(Us[t + 1], Acov, K); Lc1 = stc1["L"].to(DEV); del stc0, stc1
    nw_sub = torch.nonzero(~words[t])[:, 0].to(DEV); nw_sub = nw_sub[torch.randperm(nw_sub.numel(), device=DEV, generator=gs)[:8192]]
    for lab, A0, A1, L0, L1 in (("native_words", Wn[t][words[t]], Wn[t + 1][words[t]], Ln0, Ln1), ("native_nonwords", Wn[t][nw_sub], Wn[t + 1][nw_sub], Ln0, Ln1), ("random", Arand[sub_r], Arand[sub_r], Lr0, Lr1), ("covariance", Acov[sub_c], Acov[sub_c], Lc0, Lc1)):
        P0 = Us[t] @ A0.T; P1 = Us[t + 1] @ A1.T; c0 = P0 - P0.mean(0); c1 = P1 - P1.mean(0); prof.setdefault(lab, []).append(float(((c0 * c1).sum(0) / (c0.norm(dim=0) * c1.norm(dim=0)).clamp_min(1e-9)).mean()))
        R0 = P0.abs() / L0[:, None]; R1 = P1.abs() / L1[:, None]
        for lo_, hi_ in LV:
            mk = (R0 >= lo_) & (R0 < hi_)
            if mk.sum() >= 50: cond.setdefault(lab, {}).setdefault(f"{lo_}-{hi_}", []).append((float(R1[mk].mean()), float(R0[mk].mean()), int(mk.sum())))
        del P0, P1, c0, c1, R0, R1
res["profile_persistence"] = {lab: mean(v) for lab, v in prof.items()}; res["conditional_persistence"] = {lab: {b: dict(ratio=sum(a for a, _, _ in v) / sum(b_ for _, b_, _ in v), n=sum(n_ for _, _, n_ in v)) for b, v in d.items()} for lab, d in cond.items()}
log(f"{name} empirical persistence of a row's whole projection profile (correlation across positions of its projections at t and t+1): " + ", ".join(f"{lab} {f3(v)}" for lab, v in res["profile_persistence"].items()))
log(f"{name} conditional persistence by level (mean at t+1 over mean at t): " + " | ".join(f"{lab}: " + ", ".join(f"{b} {v['ratio']:.3f} ({v['n']})" for b, v in d.items()) for lab, d in res["conditional_persistence"].items()))
log(f"{name} persistence-weighted alignment (expected persistence of a row's projections): " + ", ".join(f"{lab} {f3(v)}" for lab, v in res["persistence_weighted"].items()))
# transport: the subspace's rotation and the three norms
k = 32; rot = [subspace_overlap(V[t], V[t + 1], k) for t in range(T - 1)]; rot4 = [subspace_overlap(V[t], V[t + 4], k) for t in range(T - 4)]; rot_first_last = subspace_overlap(V[0], V[-1], k); rot_mid_last = subspace_overlap(V[7], V[-1], k)
a_, b_, c_, d_ = [], [], [], []
for t in range(T - 1):
    Wt = Wn[t][words[t]]; Wt1 = Wn[t + 1][words[t]]; a_.append(float(wsq(Wt, V[t][:, :k]).median())); b_.append(float(wsq(Wt1, V[t + 1][:, :k]).median())); c_.append(float(wsq(Wt, V[t + 1][:, :k]).median())); d_.append(float(wsq(Wt1, V[t][:, :k]).median()))
res["transport"] = dict(top32_overlap_1=mean(rot), top32_overlap_4=mean(rot4), top32_overlap_first_last=rot_first_last, top32_overlap_mid_last=rot_mid_last, row_t_in_subspace_t=mean(a_), row_t1_in_subspace_t1=mean(b_), row_t_in_subspace_t1=mean(c_), row_t1_in_subspace_t=mean(d_))
o = res["transport"]; log(f"{name} transport: top-32 subspace overlap across 1000 / 4000 steps {f3(o['top32_overlap_1'])} / {f3(o['top32_overlap_4'])}, first to last {f3(o['top32_overlap_first_last'])}, step 8000 to last {f3(o['top32_overlap_mid_last'])}; native words' weight in the top 32: row t in subspace t {f3(o['row_t_in_subspace_t'])}, row t+1 in subspace t+1 {f3(o['row_t1_in_subspace_t1'])}, row t in subspace t+1 {f3(o['row_t_in_subspace_t1'])}, row t+1 in subspace t {f3(o['row_t1_in_subspace_t'])}")
# lead and lag: frozen and final bases, entrants, cross-correlation
Vf = V[3][:, :k]; Vl = V[-1][:, :k]; cont, froz, fin, share, allw = [], [], [], [], []
for t in range(T):
    Wt = Wn[t][words[t]]; cont.append(float(wsq(Wt, V[t][:, :k]).mean())); froz.append(float(wsq(Wt, Vf).mean())); fin.append(float(wsq(Wt, Vl).mean())); share.append(float(lam[t][:k].sum() / lam[t].sum())); allw.append(float(wsq(Wn[t], V[t][:, :k]).mean()))
res["lead_lag"] = dict(native_words_weight_contemporaneous=cont, native_words_weight_frozen_4000=froz, native_words_weight_final=fin, top32_variance_share=share, all_rows_weight_contemporaneous=allw)
dc = torch.tensor(cont).diff(); ds = torch.tensor(share).diff(); da = torch.tensor(allw).diff(); xc = {}
for lag in range(-4, 5):
    if lag >= 0: x, y = da[lag:], ds[:len(ds) - lag]
    else: x, y = da[:len(da) + lag], ds[-lag:]
    xc[str(lag)] = float(torch.corrcoef(torch.stack([x, y]))[0, 1]) if len(x) > 3 else None
res["lead_lag"]["xcorr_rows_weight_change_vs_share_change"] = xc
log(f"{name} lead and lag: native words' weight in the top 32, contemporaneous / frozen at 4000 / final, by checkpoint: " + " ".join(f"{s // 1000}k {c:.2f}/{f:.2f}/{l:.2f}" for s, c, f, l in zip(steps, cont, froz, fin)) + "; top-32 variance share " + " ".join(f"{x:.2f}" for x in share) + "; cross-correlation of the change of all rows' weight with the change of the share at lags -4..4 (rows lead at positive lags): " + ", ".join(f"{l}: {f2(v)}" for l, v in xc.items()))
ex, en = events(words); acc = {kk: {"cont": [], "final": []} for kk in range(-4, 3)}
for r, t in en:
    for kk in range(-4, 3):
        i = t + kk
        if 0 <= i < T: w = Wn[i][r]; acc[kk]["cont"].append(float(wsq(w[None], V[i][:, :k]))); acc[kk]["final"].append(float(wsq(w[None], Vl)))
res["entrants"] = {str(kk): {q: med(v) for q, v in d.items()} | {"n": len(d["cont"])} for kk, d in acc.items()}
log(f"{name} entrants' weight in the top 32 at k=-4..+2 (contemporaneous / final subspace): " + ", ".join(f"{kk}: {f3(d['cont'])}/{f3(d['final'])}" for kk, d in res["entrants"].items()))
pr = res["persistence_by_rank"]; pw = res["persistence_weighted"]; g_ = lambda x: -9 if x is None else x
cp = res["conditional_persistence"]; pp = res["profile_persistence"]
res["checks_v2"] = dict(profile_persistence_equal=abs(g_(pp.get("native_words")) - g_(pp.get("random"))) <= 0.05, extremes_persist_only_native=g_(cp.get("native_words", {}).get("1.0-1.2", {}).get("ratio")) > 0.95 and g_(cp.get("random", {}).get("1.0-1.2", {}).get("ratio")) < 0.9 and abs(g_(cp.get("native_words", {}).get("0.4-0.6", {}).get("ratio")) - g_(cp.get("random", {}).get("0.4-0.6", {}).get("ratio"))) <= 0.05)
res["checks"] = dict(persistence_falls_with_rank=g_(pr["1-8"]) > 0.9 and g_(pr["9-32"]) > 0.9 and g_(pr["257-512"]) < 0.5, dynamic_modes_are_variance_modes=g_(res["dynamic_vs_variance"]["overlap_top_k"]["32"]) >= 0.9,
                     native_persistence_weighted_twice_random=g_(pw["native_words"]) >= 2 * g_(pw["random"]) and g_(pw["random"]) <= g_(pw["covariance"]) <= g_(pw["native_words"]), row_moves_into_slow_subspace=abs(g_(o["row_t_in_subspace_t1"]) - g_(o["row_t_in_subspace_t"])) <= 0.02,
                     rows_do_not_lead=all(fin[t] <= cont[t] for t in range(4)))
summ = f"{name} block {B}: states' persistence by eigenvalue rank 1-8 {f2(pr['1-8'])}, 9-32 {f2(pr['9-32'])}, 33-128 {f2(pr['33-128'])}, 129-256 {f2(pr['129-256'])}, 257-512 {f2(pr['257-512'])}, 513-1024 {f2(pr['513-1024'])} ({pr['n_directions_over_0_9']:.0f} directions over 0.9, {pr['n_directions_over_0_5']:.0f} over 0.5); dynamic against variance top-32 overlap {f2(res['dynamic_vs_variance']['overlap_top_k']['32'])}; weight in the top 32: native words {f3(res['row_weights']['32']['native_words'])}, native non-words {f3(res['row_weights']['32']['native_nonwords'])}, covariance atoms {f3(res['row_weights']['32']['covariance'])}, random {f3(res['row_weights']['32']['random'])}; persistence-weighted alignment native words {f3(pw['native_words'])}, non-words {f3(pw['native_nonwords'])}, covariance {f3(pw['covariance'])}, random {f3(pw['random'])}; top-32 subspace overlap across 1000 steps {f3(o['top32_overlap_1'])}, first to last {f3(o['top32_overlap_first_last'])}; three norms {f3(o['row_t_in_subspace_t'])} / {f3(o['row_t1_in_subspace_t1'])} / {f3(o['row_t_in_subspace_t1'])}; native words' weight at 1000-4000 in the contemporaneous / final subspace " + " ".join(f"{c:.2f}/{l:.2f}" for c, l in zip(cont[:4], fin[:4])) + f"; entrants' weight at k=-4/0/+2 {f3(res['entrants']['-4']['cont'])}/{f3(res['entrants']['0']['cont'])}/{f3(res['entrants']['2']['cont'])} | profile persistence native words {f3(pp.get('native_words'))} random {f3(pp.get('random'))}; conditional persistence at 1.0-1.2 native words {cp.get('native_words', {}).get('1.0-1.2', {}).get('ratio', -1):.3f} random {cp.get('random', {}).get('1.0-1.2', {}).get('ratio', -1):.3f} covariance {cp.get('covariance', {}).get('1.0-1.2', {}).get('ratio', -1):.3f} | checks {_json.dumps(res['checks'])} | checks v2 {_json.dumps(res['checks_v2'])}"
log(summ); record(f"e542_spectral_persistence_{name}", res, summ)
