"""e538b: the shape of an exit. e538's reversed-time replay hinted that a word leaves the vocabulary with its projection
still above the floor. This reads it directly, from the e524 cache (every row atom of blocks 0-12, sixteen checkpoints,
blocks 12 and 6, no GPU): clean exits (a word at k=-2 and -1, not a word at k=0 and +1) aligned on the first non-word
checkpoint, with S, the count of positions over the floor, the count over three quarters of the floor and the usage as
a fraction of the 256th-largest usage (the word threshold) at k=-4..+2, against S-matched words that stay; the clean
entries on the same footing; the fractions of leavers above the floor at k=0 and +2 and of entrants below it at k=-1
and -2; and the per-row change of S across the event (k=-1 to +1) for leavers and entrants.
Pre-registered (honest guesses), block 12:
- the leavers' median S at k=0 (the first checkpoint out) is still at or above the floor (0.6);
- the leavers' usage falls under half the threshold by k=+1 while their median S falls by less than 0.1 from k=-1 to +1 (0.5);
- at k=-1, at equal S, the leavers have fewer positions over three quarters of the floor than the words that stay (0.5);
- the fall of S across an exit is smaller in size than the rise across an entry (0.7).
Arguments: name."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; CDIR = f"/workspace/wdd/cache/e524_{name}"; NWORD = 256; KMIN, KMAX = -4, 2; BL = [12, 6]
steps = list(range(1000, 16001, 1000)); ck = {n: torch.load(f"{CDIR}/step{n}.pt", map_location="cpu") for n in steps}
def wordset(u):
    w = torch.zeros(u.numel(), dtype=torch.bool); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; return w
def med(v): return float(torch.tensor(v).median()) if v else None
res = dict(model=name, steps=steps, by_block={})
for b in BL:
    H = [ck[n]["blocks"][b] for n in steps]; U_ = [h["usage"].float() for h in H]; words = [wordset(u) for u in U_]; thr = [max(float(u[w].min()), 1.0) for u, w in zip(U_, words)]; S_ = [h["S"].float() for h in H]; Rn = S_[0].numel()
    ex, en = [], []
    for t in range(2, len(words) - 1):
        e = words[t - 2] & words[t - 1] & ~words[t] & ~words[t + 1]; ex += [(r, t) for r in torch.nonzero(e)[:, 0].tolist()]
        e2 = ~words[t - 2] & ~words[t - 1] & words[t] & words[t + 1]; en += [(r, t) for r in torch.nonzero(e2)[:, 0].tolist()]
    def match(events, stay):
        taken = torch.zeros(Rn, dtype=torch.bool); out = []
        for r, t in events:
            okm = stay(t) & ~taken; okm[r] = False; cand = torch.nonzero(okm)[:, 0]
            if cand.numel() == 0: continue
            j = cand[(S_[t - 1][cand] - S_[t - 1][r]).abs().argmin()]; taken[j] = True; out.append((int(j), t))
        return out
    mex = match(ex, lambda t: words[t - 2] & words[t - 1] & words[t] & words[t + 1]); men = match(en, lambda t: ~words[t - 1] & ~words[t] & ~words[t + 1])
    def aligned(events):
        o = {}
        for kk in range(KMIN, KMAX + 1):
            vals = {q: [] for q in ("S", "cnt", "cnt75", "usage_over_threshold")}
            for r, t in events:
                i = t + kk
                if 0 <= i < len(H): vals["S"].append(float(S_[i][r])); vals["cnt"].append(float(H[i]["cnt"][r])); vals["cnt75"].append(float(H[i]["cnt75"][r])); vals["usage_over_threshold"].append(float(U_[i][r]) / thr[i])
            o[kk] = {q: med(v) for q, v in vals.items()}; o[kk]["n"] = len(vals["S"])
        return o
    Aex, Amex, Aen, Amen = aligned(ex), aligned(mex), aligned(en), aligned(men)
    d_ex = med([float(S_[t + 1][r] - S_[t - 1][r]) for r, t in ex]); d_en = med([float(S_[t + 1][r] - S_[t - 1][r]) for r, t in en])
    fr = dict(leavers_over_floor_k0=float(torch.tensor([float(S_[t][r] >= 1) for r, t in ex]).mean()), leavers_over_floor_k2=float(torch.tensor([float(S_[t + 2][r] >= 1) for r, t in ex if t + 2 < len(H)]).mean()),
              entrants_under_floor_km1=float(torch.tensor([float(S_[t - 1][r] < 1) for r, t in en]).mean()), entrants_under_floor_km2=float(torch.tensor([float(S_[t - 2][r] < 1) for r, t in en]).mean()),
              leavers_S_below_km1_at_k0=float(torch.tensor([float(S_[t][r] < S_[t - 1][r]) for r, t in ex]).mean()))
    res["by_block"][b] = dict(n_clean_exits=len(ex), n_clean_entries=len(en), exits_aligned=Aex, exits_matched_stayers=Amex, entries_aligned=Aen, entries_matched=Amen, delta_S_exit=d_ex, delta_S_entry=d_en, fractions=fr, word_threshold_usage=thr)
    f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; ks = list(range(KMIN, KMAX + 1)); pr = lambda A, q: "/".join(f2(A[k][q]) for k in ks)
    log(f"{name} block {b}: {len(ex)} clean exits, {len(en)} clean entries; leavers' S at k={KMIN}..{KMAX} {pr(Aex, 'S')} (matched stayers {pr(Amex, 'S')}), over the floor {pr(Aex, 'cnt')}, over three quarters {pr(Aex, 'cnt75')} (stayers {pr(Amex, 'cnt75')}), usage over the threshold {pr(Aex, 'usage_over_threshold')} (stayers {pr(Amex, 'usage_over_threshold')}); entrants' S {pr(Aen, 'S')} (matched {pr(Amen, 'S')}), over three quarters {pr(Aen, 'cnt75')}, usage over the threshold {pr(Aen, 'usage_over_threshold')}; S change across the event, leavers {f2(d_ex)}, entrants {f2(d_en)}; fractions {_json.dumps({k: round(v, 2) for k, v in fr.items()})}")
B12 = res["by_block"][12]; Aex, Amex = B12["exits_aligned"], B12["exits_matched_stayers"]
res["checks"] = dict(leavers_S_k0_over_floor=Aex[0]["S"] >= 1.0, usage_halved_S_kept=(Aex[1]["usage_over_threshold"] < 0.5) and (Aex[-1]["S"] - Aex[1]["S"] < 0.1), leavers_narrower_at_equal_S=Aex[-1]["cnt75"] < Amex[-1]["cnt75"], exit_fall_smaller_than_entry_rise=abs(B12["delta_S_exit"]) < B12["delta_S_entry"])
summ = f"{name}: " + " | ".join(f"block {b}: leavers' S at k=-1/0/+2 {o['exits_aligned'][-1]['S']:.2f}/{o['exits_aligned'][0]['S']:.2f}/{o['exits_aligned'][2]['S']:.2f}, usage over the threshold {o['exits_aligned'][-1]['usage_over_threshold']:.2f}/{o['exits_aligned'][0]['usage_over_threshold']:.2f}/{o['exits_aligned'][2]['usage_over_threshold']:.2f}, entrants' S {o['entries_aligned'][-1]['S']:.2f}/{o['entries_aligned'][0]['S']:.2f}/{o['entries_aligned'][2]['S']:.2f}; S change across exits {o['delta_S_exit']:+.2f}, entries {o['delta_S_entry']:+.2f}; leavers over the floor at k=0/+2 {o['fractions']['leavers_over_floor_k0']:.2f}/{o['fractions']['leavers_over_floor_k2']:.2f}, entrants under it at k=-1/-2 {o['fractions']['entrants_under_floor_km1']:.2f}/{o['fractions']['entrants_under_floor_km2']:.2f}" for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e538b_exit_profile_{name}", res, summ)
