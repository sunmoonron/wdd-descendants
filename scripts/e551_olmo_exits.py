"""e551: the gate, the criterion and the persistence on a second model. From the e550 cache of OLMo-1B (steps 1000-16000,
block 8, the MLP rows of blocks 0-8 as the dictionary, OMP over the rows alone) and a fixed isotropic random dictionary
of the same size against the same states: e538b's profiles (the one-way gate: leavers over the floor at the first
checkpoint out, entrants under it before entry, the S change across exits and entries), the prospective criterion at
one and four intervals, entries per interval and retention, the maximum's provenance (e541), the level-matched
persistence of a position's projection (e541) and the conditional persistence by level (e542, the native rows'
extremes persisting beyond their bulk where a random atom's regress like its bulk).
Pre-registered (honest guesses), OLMo-1B block 8:
- the one-way gate replicates: the rows' leavers are over the floor at the first checkpoint out in 0.7 or more of
  cases, the random dictionary's in under 0.5 (0.6);
- the native fifth replicates: the rows' prospective criterion at four intervals is 0.88 or more and at least 0.05
  above the random dictionary's, their retention 0.4 or more against 0.25 or less (0.6);
- the level-matched persistence replicates: the rows' positions at 1.1-1.2 are at 1.05 or more a thousand steps later,
  the random dictionary's at 1.0 or less (0.6);
- the extremes persist beyond the bulk for the rows (conditional persistence at 1.2-1.5 at least 0.03 above the bulk's
  at 0.4-0.6) and not for random atoms (0.5).
Arguments: none."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
CDIR = "/workspace/wdd/cache/e550_olmo1b"; steps = list(range(1000, 16001, 1000)); T = len(steps); K = 16; name = "olmo1b"
ck = {n: torch.load(f"{CDIR}/step{n}.pt", map_location="cpu") for n in steps}; B = ck[steps[0]]["block"]; D = ck[steps[0]]["D"]; m = ck[steps[0]]["rows"].shape[0]; keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0); N = int(keepc.sum())
g = torch.Generator(device=DEV).manual_seed(0); Arand = unitr(torch.randn(m, D, device=DEV, generator=g)); H = {"native": [], "random": []}
for n in steps:
    U = unitr(ck[n]["U"][keepc].float().to(DEV)); An = ck[n]["rows"].float().to(DEV)
    for d, A in (("native", An), ("random", Arand)): H[d].append(stats(U, A, K))
    del U, An; torch.cuda.empty_cache(); log(f"{name} step{n}: {N} positions, {m} rows; native words' median S {float(H['native'][-1]['S'][wordset(H['native'][-1]['usage'])].median()):.2f}, random {float(H['random'][-1]['S'][wordset(H['random'][-1]['usage'])].median()):.2f}")
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def prospective(Hd, words, Hh):
    a = []
    for t in range(len(words) - Hh):
        nonw = ~words[t]; ent = nonw & words[t + Hh]; v = auc(Hd[t]["S"][ent], Hd[t]["S"][nonw & ~ent])
        if v is not None: a.append(v)
    return mean(a)
def provenance(Hd, words):
    same, beyond = [], []
    for t in range(T - 1):
        W = torch.nonzero(words[t])[:, 0]; R0 = Hd[t]["ratio"][:, W].float(); R1 = Hd[t + 1]["ratio"][:, W].float(); ar = torch.arange(len(W)); a1 = R1.argmax(0); rk = (R0 > R0[a1, ar][None, :]).sum(0); same += (rk == 0).float().tolist(); beyond += (rk >= 20).float().tolist()
    return dict(same=mean(same), beyond=mean(beyond))
LB = ((1.0, 1.1), (1.1, 1.2), (1.2, 1.4)); LV = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.5)]
def level_matched(Hd, words):
    r0s, r1s = [], []
    for t in range(T - 1):
        cols = torch.nonzero(words[t])[:, 0]; R0 = Hd[t]["ratio"][:, cols].float(); R1 = Hd[t + 1]["ratio"][:, cols].float(); mk = R0 >= 1.0; r0s.append(R0[mk]); r1s.append(R1[mk])
    r0 = torch.cat(r0s); r1 = torch.cat(r1s); out = {}
    for lo_, hi_ in LB:
        mk = (r0 >= lo_) & (r0 < hi_)
        if mk.sum() >= 20: out[f"{lo_}-{hi_}"] = dict(n=int(mk.sum()), mean_after=float(r1[mk].mean()))
    return out
def conditional(Hd, words, sub=8192):
    acc = {}
    for t in range(T - 1):
        cols = torch.nonzero(words[t])[:, 0]; cols = cols[torch.randperm(cols.numel(), generator=torch.Generator().manual_seed(t))[:sub]]; R0 = Hd[t]["ratio"][:, cols].float().to(DEV); R1 = Hd[t + 1]["ratio"][:, cols].float().to(DEV)
        for lo_, hi_ in LV:
            mk = (R0 >= lo_) & (R0 < hi_)
            if mk.sum() >= 50: acc.setdefault(f"{lo_}-{hi_}", []).append((float(R1[mk].sum()), float(R0[mk].sum()), int(mk.sum())))
        del R0, R1
    return {b: dict(ratio=sum(a for a, _, _ in v) / sum(b_ for _, b_, _ in v), n=sum(n_ for _, _, n_ in v)) for b, v in acc.items()}
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; res = dict(model=name, block=B, n_positions=N, n_rows=m, by_dictionary={})
for d in ("native", "random"):
    Hd = H[d]; prof, words, thr, ex, en = profiles(Hd); pv = provenance(Hd, words); lm = level_matched(Hd, words); cp = conditional(Hd, words); X_, E_, F_ = prof["exits_aligned"], prof["entries_aligned"], prof["fractions"]
    res["by_dictionary"][d] = dict(profiles=prof, prospective_auc_1=prospective(Hd, words, 1), prospective_auc_4=prospective(Hd, words, 4), provenance=pv, level_matched=lm, conditional_persistence=cp, words_median_S=float(torch.stack([h["S"] for h in Hd])[torch.stack(words)].median()), words_median_cnt75=float(torch.stack([h["cnt75"] for h in Hd])[torch.stack(words)].median()))
    o = res["by_dictionary"][d]; log(f"{name} {d}: criterion at 1/4 {f2(o['prospective_auc_1'])}/{f2(o['prospective_auc_4'])}, entries per interval {prof['entries_per_interval']:.0f}, retention at 4 {f2(prof['retention_4'])}, words' median S {f2(o['words_median_S'])} with {o['words_median_cnt75']:.0f} positions over three quarters; {prof['n_clean_exits']} clean exits, {prof['n_clean_entries']} entries; leavers' S at k=-2/-1/0/+1/+2 {f2(X_[-2]['S'])}/{f2(X_[-1]['S'])}/{f2(X_[0]['S'])}/{f2(X_[1]['S'])}/{f2(X_[2]['S'])}, usage over the threshold at -1/0/+2 {f2(X_[-1]['usage_over_threshold'])}/{f2(X_[0]['usage_over_threshold'])}/{f2(X_[2]['usage_over_threshold'])}, over the floor at k=0/+2 {f2(F_['leavers_over_floor_k0'])}/{f2(F_['leavers_over_floor_k2'])}; entrants' S at -2/-1/0/+1/+2 {f2(E_[-2]['S'])}/{f2(E_[-1]['S'])}/{f2(E_[0]['S'])}/{f2(E_[1]['S'])}/{f2(E_[2]['S'])}, under the floor at -1/-2 {f2(F_['entrants_under_floor_km1'])}/{f2(F_['entrants_under_floor_km2'])}; S change across exits/entries {f2(prof['delta_S_exit'])}/{f2(prof['delta_S_entry'])}; maximum at the same position {f2(pv['same'])}, beyond twenty {f2(pv['beyond'])}; level-matched persistence at 1.0-1.1/1.1-1.2/1.2-1.4 " + "/".join(f2(lm[k]["mean_after"]) if k in lm else "n/a" for k in ("1.0-1.1", "1.1-1.2", "1.2-1.4")) + "; conditional persistence by level " + ", ".join(f"{b} {v['ratio']:.3f} ({v['n']})" for b, v in cp.items()))
nat, rnd = res["by_dictionary"]["native"], res["by_dictionary"]["random"]; g_ = lambda x: -9 if x is None else x; cpn, cpr = nat["conditional_persistence"], rnd["conditional_persistence"]
res["checks"] = dict(gate_replicates=g_(nat["profiles"]["fractions"]["leavers_over_floor_k0"]) >= 0.7 and g_(rnd["profiles"]["fractions"]["leavers_over_floor_k0"]) < 0.5, native_fifth_replicates=g_(nat["prospective_auc_4"]) >= 0.88 and g_(nat["prospective_auc_4"]) - g_(rnd["prospective_auc_4"]) >= 0.05 and g_(nat["profiles"]["retention_4"]) >= 0.4 and g_(rnd["profiles"]["retention_4"]) <= 0.25,
                     level_matched_replicates=g_(nat["level_matched"].get("1.1-1.2", {}).get("mean_after")) >= 1.05 and g_(rnd["level_matched"].get("1.1-1.2", {}).get("mean_after", 0)) <= 1.0, extremes_beyond_bulk=("1.2-1.5" in cpn and "0.4-0.6" in cpn and cpn["1.2-1.5"]["ratio"] >= cpn["0.4-0.6"]["ratio"] + 0.03) and (("1.2-1.5" not in cpr) or cpr["1.2-1.5"]["ratio"] < cpr["0.4-0.6"]["ratio"] + 0.03))
summ = f"{name} block {B}: " + " | ".join(f"{d}: criterion {f2(o['prospective_auc_4'])}, entries {o['profiles']['entries_per_interval']:.0f}, retention {f2(o['profiles']['retention_4'])}, words' S {f2(o['words_median_S'])}, leavers over the floor {f2(o['profiles']['fractions']['leavers_over_floor_k0'])}, entrants under it at -2 {f2(o['profiles']['fractions']['entrants_under_floor_km2'])}, S across exits/entries {f2(o['profiles']['delta_S_exit'])}/{f2(o['profiles']['delta_S_entry'])}, same peak {f2(o['provenance']['same'])}, persistence at 1.1-1.2 {f2(o['level_matched'].get('1.1-1.2', {}).get('mean_after'))}, conditional persistence bulk/extremes {o['conditional_persistence'].get('0.4-0.6', {}).get('ratio', -1):.3f}/{o['conditional_persistence'].get('1.2-1.5', o['conditional_persistence'].get('1.0-1.2', {})).get('ratio', -1):.3f}" for d, o in res["by_dictionary"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e551_olmo_exits_{name}", res, summ)
