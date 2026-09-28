"""e541b: is the persistence of a native row's alignment the cloud's covariance? e541 found that at equal level a
native row's projection at a position does not regress to the mean over a thousand steps (1.01, 1.12, 1.26 at t+1 from
1.0-1.1, 1.1-1.2, 1.2-1.4) while a random atom's does (0.93, 1.03, 1.10), for native non-words nearly as much as for
words; that is what holds the native maximum above the floor while the margins erode. e538 found the rows' motion
aimed at the cloud's covariance directions, and e532 that atoms drawn with the state cloud's covariance have the most
persistent projections and the most churning words. So: the same analysis for a third dictionary of random atoms drawn
with the state cloud's covariance (at step 8000, unit-normalised), against the native rows and isotropic random atoms:
level-matched persistence of a position's projection (words and non-words); the persistence of S across one and four
thousand steps (correlation over all atoms and over words); the provenance of the maximum; the order-statistics null
for the covariance words with their own kernel (the calibration) and for the native words with the covariance kernel,
which reaches higher levels than the isotropic one; and each dictionary's retention and entries per interval.
Pre-registered (honest guesses), block 12:
- the covariance-matched atoms' positions persist at equal level like the native rows', the mean at t+1 within 0.03
  of the native words' in the bins 1.0-1.1 and 1.1-1.2 (0.6);
- the covariance kernel predicts the native words' S at t+1 within 0.05 in the bins it populates (0.5);
- the covariance words' retention at four thousand steps stays under 0.3 despite the persistence (0.7);
- the covariance words' maximum relocates at the native rate, the same position at t+1 in 0.35-0.50 of cases (0.7).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
name = sys.argv[1]; B = 12; K = 16; NDRAW = 20; CDIR = f"/workspace/wdd/cache/e524_{name}"; steps = list(range(1000, 16001, 1000)); T = len(steps)
ck = {n: torch.load(f"{CDIR}/step{n}.pt", map_location="cpu") for n in steps}; m, D = ck[steps[0]]["rows"].shape; keepc = torch.stack([ck[n]["blocks"][B]["keep"] for n in steps]).all(0); N = int(keepc.sum())
g = torch.Generator(device=DEV).manual_seed(0); g2 = torch.Generator(device=DEV).manual_seed(1); Arand = unitr(torch.randn(m, D, device=DEV, generator=g))
Us = [unitr(ck[n]["blocks"][B]["U"][keepc].float()) for n in steps]; U8 = Us[7].to(DEV); C8 = (U8.T @ U8 / N).double(); ev, V = torch.linalg.eigh(C8); ev = ev.clamp_min(0)
Acov = unitr(((torch.randn(m, D, device=DEV, generator=g, dtype=torch.float64) * ev.sqrt()[None]) @ V.T).float()); del U8
DICTS = ["native", "random", "covariance"]; H = {d: [] for d in DICTS}
for i, n in enumerate(steps):
    U = Us[i].to(DEV); An = ck[n]["rows"].float().to(DEV)
    for d, A in (("native", An), ("random", Arand), ("covariance", Acov)): H[d].append(stats(U, A, K))
    del U, An; torch.cuda.empty_cache()
words = {d: [wordset(h["usage"]) for h in H[d]] for d in DICTS}; log(f"{name}: statistics for {len(DICTS)} dictionaries at {T} checkpoints, {N} positions")
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"; res = dict(model=name, block=B, summary={}, level_matched={}, provenance={}, null={}, kernels={})
for d in DICTS:
    S_ = torch.stack([h["S"] for h in H[d]]); Wd = torch.stack(words[d]); c1 = float(torch.corrcoef(torch.stack([S_[:-1].reshape(-1), S_[1:].reshape(-1)]))[0, 1]); c4 = float(torch.corrcoef(torch.stack([S_[:-4].reshape(-1), S_[4:].reshape(-1)]))[0, 1])
    w1 = float(torch.corrcoef(torch.stack([S_[:-1][Wd[:-1]], S_[1:][Wd[:-1]]]))[0, 1]); w4 = float(torch.corrcoef(torch.stack([S_[:-4][Wd[:-4]], S_[4:][Wd[:-4]]]))[0, 1])
    res["summary"][d] = dict(S_persistence_1_all=c1, S_persistence_4_all=c4, S_persistence_1_words=w1, S_persistence_4_words=w4, retention_4=mean([float((Wd[t] & Wd[t + 4]).sum() / NWORD) for t in range(T - 4)]), entries_per_interval=mean([float((~Wd[t] & Wd[t + 1]).sum()) for t in range(T - 1)]), words_median_S=float(S_[Wd].median()), words_median_cnt75=float(torch.stack([h["cnt75"] for h in H[d]])[Wd].median()))
    o = res["summary"][d]; log(f"{name} {d}: persistence of S across 1000/4000 steps, all atoms {f2(o['S_persistence_1_all'])}/{f2(o['S_persistence_4_all'])}, words {f2(o['S_persistence_1_words'])}/{f2(o['S_persistence_4_words'])}; retention at 4 {f2(o['retention_4'])}, entries per interval {o['entries_per_interval']:.0f}, words' median S {f2(o['words_median_S'])}, positions over three quarters {o['words_median_cnt75']:.0f}")
LB = ((0.9, 1.0), (1.0, 1.1), (1.1, 1.2), (1.2, 1.4), (1.4, 9.0))
def level_matched(d, which):
    r0s, r1s = [], []
    for t in range(T - 1):
        if which == "words": cols = torch.nonzero(words[d][t])[:, 0]
        else: cols = torch.nonzero(~(words[d][t] | words[d][t + 1]))[:, 0]; cols = cols[torch.randperm(cols.numel(), generator=torch.Generator().manual_seed(t))[:4096]]
        R0 = H[d][t]["ratio"][:, cols].float(); R1 = H[d][t + 1]["ratio"][:, cols].float(); mk = R0 >= 0.9; r0s.append(R0[mk]); r1s.append(R1[mk])
    r0 = torch.cat(r0s); r1 = torch.cat(r1s); out = {}
    for lo_, hi_ in LB:
        mk = (r0 >= lo_) & (r0 < hi_)
        if mk.sum() >= 20: out[f"{lo_}-{hi_}"] = dict(n=int(mk.sum()), mean_after=float(r1[mk].mean()), p_over_floor_after=float((r1[mk] >= 1).float().mean()))
    return out
for d in DICTS:
    for which in ("words", "nonwords"):
        o = level_matched(d, which); res["level_matched"][f"{d}_{which}"] = o; log(f"{name} level-matched persistence, {d} {which}: " + "; ".join(f"{k}: {v['n']} pairs, mean at t+1 {f2(v['mean_after'])}, over the floor {f2(v['p_over_floor_after'])}" for k, v in o.items()))
def provenance(d):
    acc = {q: [] for q in ("same", "top5", "top20", "beyond", "old_peak_after", "new_max_after")}
    for t in range(T - 1):
        W = torch.nonzero(words[d][t])[:, 0]; R0 = H[d][t]["ratio"][:, W].float(); R1 = H[d][t + 1]["ratio"][:, W].float(); ar = torch.arange(len(W)); a0 = R0.argmax(0); a1 = R1.argmax(0); rk = (R0 > R0[a1, ar][None, :]).sum(0)
        acc["same"] += (rk == 0).float().tolist(); acc["top5"] += ((rk > 0) & (rk < 5)).float().tolist(); acc["top20"] += ((rk >= 5) & (rk < 20)).float().tolist(); acc["beyond"] += (rk >= 20).float().tolist(); acc["old_peak_after"] += R1[a0, ar].tolist(); acc["new_max_after"] += R1.max(0).values.tolist()
    return dict(n=len(acc["same"]), same=mean(acc["same"]), top5=mean(acc["top5"]), top20=mean(acc["top20"]), beyond=mean(acc["beyond"]), old_peak_after=med(acc["old_peak_after"]), new_max_after=med(acc["new_max_after"]))
for d in DICTS:
    o = provenance(d); res["provenance"][d] = o; log(f"{name} {d} provenance of the maximum ({o['n']}): same position {f2(o['same'])}, top five {f2(o['top5'])}, top twenty {f2(o['top20'])}, beyond {f2(o['beyond'])}; old peak at t+1 {f2(o['old_peak_after'])} against the new maximum {f2(o['new_max_after'])}")
EDGES = torch.arange(0.0, 2.0001, 0.05); nb = len(EDGES) - 1; E_dev = EDGES[1:-1].to(DEV)
def kernel(d):
    samples = [[] for _ in range(nb)]
    for t in range(T - 1):
        R0 = H[d][t]["ratio"].to(DEV).float(); R1 = H[d][t + 1]["ratio"].to(DEV).float(); r0 = R0.reshape(-1); r1 = R1.reshape(-1)
        idx = torch.cat([torch.nonzero(r0 >= 0.7)[:, 0], torch.randint(r0.numel(), (200000,), device=DEV, generator=g2)]); x0 = r0[idx]; dl = r1[idx] - x0; b = torch.bucketize(x0, E_dev)
        for bb in range(nb):
            sb = dl[b == bb]
            if sb.numel(): samples[bb].append(sb[torch.randperm(sb.numel(), device=DEV, generator=g2)[:20000]].cpu())
        del R0, R1, r0, r1; torch.cuda.empty_cache()
    S_ = [torch.cat(s) if s else torch.zeros(0) for s in samples]; counts = [int(s.numel()) for s in S_]; pop = [i for i in range(nb) if counts[i] >= 100]
    return dict(delta=S_, counts=counts, fallback=[min(pop, key=lambda j: abs(j - i)) for i in range(nb)], populated_up_to=float(EDGES[max(pop) + 1]))
def null_S(d, ker):
    fb = torch.tensor(ker["fallback"], device=DEV); act, nul, act1, nul1, s0 = [], [], [], [], []
    for t in range(T - 1):
        W = torch.nonzero(words[d][t])[:, 0]; R0 = H[d][t]["ratio"][:, W].float().to(DEV); S1 = H[d][t + 1]["ratio"][:, W].float().max(0).values; nW = len(W)
        b = fb[torch.bucketize(R0.reshape(-1), E_dev)]; draws = torch.empty(NDRAW, R0.numel(), device=DEV)
        for bb in b.unique().tolist():
            msk = b == bb; smp = ker["delta"][bb].to(DEV); draws[:, msk] = smp[torch.randint(smp.numel(), (NDRAW, int(msk.sum())), device=DEV, generator=g2)]
        Sn = (R0.reshape(-1)[None, :] + draws).reshape(NDRAW, N, nW).max(1).values
        act.append(S1); nul.append(Sn.median(0).values.cpu()); act1.append((S1 >= 1).float()); nul1.append((Sn >= 1).float().mean(0).cpu()); s0.append(R0.max(0).values.cpu()); del R0, draws, Sn
    act, nul, act1, nul1, s0 = (torch.cat(x) for x in (act, nul, act1, nul1, s0))
    out = dict(n=int(act.numel()), median_S_before=float(s0.median()), median_S_after_actual=float(act.median()), median_S_after_null=float(nul.median()), p_over_floor_actual=float(act1.mean()), p_over_floor_null=float(nul1.mean()), by_S_bin={})
    for lo_, hi_ in ((0.9, 1.0), (1.0, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.5), (1.5, 9.0)):
        mk = (s0 >= lo_) & (s0 < hi_)
        if mk.sum() >= 10: out["by_S_bin"][f"{lo_}-{hi_}"] = dict(n=int(mk.sum()), actual=float(act[mk].median()), null=float(nul[mk].median()), p_actual=float(act1[mk].mean()), p_null=float(nul1[mk].mean()))
    return out
KER = {"covariance_atoms": kernel("covariance")}; log(f"{name} kernel from covariance atoms: populated up to level {KER['covariance_atoms']['populated_up_to']:.2f}; samples per bin from 0.9: " + ", ".join(str(KER['covariance_atoms']['counts'][i]) for i in range(18, nb)))
res["kernels"]["covariance_atoms"] = dict(counts=KER["covariance_atoms"]["counts"], populated_up_to=KER["covariance_atoms"]["populated_up_to"], median_delta_by_bin={f"{float(EDGES[i]):.2f}": float(KER["covariance_atoms"]["delta"][i].median()) for i in range(nb) if KER["covariance_atoms"]["counts"][i] >= 100})
for d in ("covariance", "native"):
    o = null_S(d, KER["covariance_atoms"]); res["null"][f"{d}_words_with_covariance_atoms"] = o
    log(f"{name} null for {d} words with the covariance-atom kernel ({o['n']}): S {f2(o['median_S_before'])} -> actual {f2(o['median_S_after_actual'])}, null {f2(o['median_S_after_null'])}; over the floor actual {f2(o['p_over_floor_actual'])}, null {f2(o['p_over_floor_null'])}; by S at t: " + "; ".join(f"{k}: {v['n']}, actual {f2(v['actual'])} null {f2(v['null'])}, over the floor {f2(v['p_actual'])} against {f2(v['p_null'])}" for k, v in o["by_S_bin"].items()))
lm = res["level_matched"]; g_ = lambda x: -9 if x is None else x; nn = res["null"]["native_words_with_covariance_atoms"]; pop_bins = [k for k, v in nn["by_S_bin"].items() if float(k.split("-")[1]) <= KER["covariance_atoms"]["populated_up_to"]]
res["checks"] = dict(covariance_persists_like_native=all(abs(g_(lm["covariance_words"][k]["mean_after"]) - g_(lm["native_words"][k]["mean_after"])) <= 0.03 for k in ("1.0-1.1", "1.1-1.2") if k in lm["covariance_words"] and k in lm["native_words"]),
                     covariance_kernel_predicts_native=bool(pop_bins) and all(abs(nn["by_S_bin"][k]["actual"] - nn["by_S_bin"][k]["null"]) <= 0.05 for k in pop_bins), covariance_retention_under_0_3=res["summary"]["covariance"]["retention_4"] < 0.3, covariance_maximum_relocates=0.35 <= g_(res["provenance"]["covariance"]["same"]) <= 0.50)
summ = f"{name} block {B}: level-matched persistence at 1.0-1.1 / 1.1-1.2 / 1.2-1.4, words: native " + "/".join(f2(lm["native_words"][k]["mean_after"]) for k in ("1.0-1.1", "1.1-1.2", "1.2-1.4") if k in lm["native_words"]) + ", covariance " + "/".join(f2(lm["covariance_words"][k]["mean_after"]) for k in ("1.0-1.1", "1.1-1.2", "1.2-1.4") if k in lm["covariance_words"]) + ", random " + "/".join(f2(lm["random_words"][k]["mean_after"]) for k in ("1.0-1.1", "1.1-1.2", "1.2-1.4") if k in lm["random_words"]) + "; " + "; ".join(f"{d}: S persistence at 1000/4000 all {f2(o['S_persistence_1_all'])}/{f2(o['S_persistence_4_all'])} words {f2(o['S_persistence_1_words'])}/{f2(o['S_persistence_4_words'])}, retention {f2(o['retention_4'])}, entries {o['entries_per_interval']:.0f}, words' S {f2(o['words_median_S'])}, same peak {f2(res['provenance'][d]['same'])}" for d, o in res["summary"].items()) + f"; covariance kernel populated to {KER['covariance_atoms']['populated_up_to']:.2f}, native words' S actual {f2(nn['median_S_after_actual'])} null {f2(nn['median_S_after_null'])} (covariance words {f2(res['null']['covariance_words_with_covariance_atoms']['median_S_after_actual'])} / {f2(res['null']['covariance_words_with_covariance_atoms']['median_S_after_null'])}) | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e541b_covariance_persistence_{name}", res, summ)
