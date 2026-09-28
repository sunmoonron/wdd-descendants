"""e541: what holds the native maximum? e539b found that across an exit the cosine at a leaver's old peak falls as
much as at its lost positions while its S holds: the maximum relocates. Five mechanisms could hold a word's maximum
above the floor while its margins erode: the same position staying high because its state persists (peak memory), the
maximum moving among a dense upper tail of positions (extreme-value persistence, no mechanism beyond breadth), the
states repeatedly revisiting the row's direction (a row-local attractor), and, for the erosion, positions disappearing
in chunks because their contexts do (usage clusters). From the e524 cache (block 12, OMP over the rows alone, Pythia's
rows and a fixed random dictionary of the same size, sixteen checkpoints):
- provenance: for every word at t, whether its position of largest projection at t+1 is the same position, among its
  top five or top twenty at t, or beyond, for stayers and for leavers, with the old peak's value at t+1 against the new;
- the order-statistics null: a transition kernel of per-position changes of the projection over the floor, conditional
  on the level, estimated from the random dictionary's atoms (and, as a second kernel, from the native non-words), then
  applied independently to every position of a word's profile at t and the maximum taken; the predicted S at t+1 and the
  predicted share over the floor against the actual, for native words (both kernels) and random words (the calibration);
- level-matched persistence of the projection at a word's positions: native words against random words, native
  non-words against random atoms, in bins of the level at t;
- the states' own persistence at the peak positions of native words, of random words, and at all positions;
- usage clusters: for the clean leavers, the pairwise cosine among the states of the lost positions against the kept
  positions and random positions, and the share of pairs sharing a token; the same for entrants' gained positions.
Pre-registered (honest guesses), block 12:
- the maximum relocates: for native words the position of the maximum is the same at t+1 in fewer than half the
  cases and among the top five at t in 0.7 or more (0.6);
- breadth alone holds it: the random-atom kernel applied to the native words' profiles predicts their median S at
  t+1 within 0.05 of the actual, and the share over the floor within 0.10 (0.5);
- at equal level (1.0-1.2) the native words' positions persist more than random words' by at least 0.05 in the mean
  projection at t+1 (0.5);
- lost positions are not clustered: their pairwise state cosine within 0.02 of the kept positions', their token
  sharing within 0.02 of random pairs' (0.6);
- the states at native words' peaks are no more persistent than at other positions, within 0.02 (0.6).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
from lr_common import eval_ids
name = sys.argv[1]; B = 12; K = 16; NDRAW = 20; CDIR = f"/workspace/wdd/cache/e524_{name}"; steps = list(range(1000, 16001, 1000)); T = len(steps)
ck = {n: torch.load(f"{CDIR}/step{n}.pt", map_location="cpu") for n in steps}; m, D = ck[steps[0]]["rows"].shape; keepc = torch.stack([ck[n]["blocks"][B]["keep"] for n in steps]).all(0); N = int(keepc.sum())
ids = eval_ids(name)[:8, 1:256].reshape(-1).cpu()[keepc]  # block_states keeps positions 1: of each sequence
g = torch.Generator(device=DEV).manual_seed(0); g2 = torch.Generator(device=DEV).manual_seed(1); Arand = unitr(torch.randn(m, D, device=DEV, generator=g))
Us = [unitr(ck[n]["blocks"][B]["U"][keepc].float()) for n in steps]; H = {"native": [], "random": []}
for i, n in enumerate(steps):
    U = Us[i].to(DEV); An = ck[n]["rows"].float().to(DEV)
    for d, A in (("native", An), ("random", Arand)): H[d].append(stats(U, A, K))
    del U, An; torch.cuda.empty_cache()
words = {d: [wordset(h["usage"]) for h in H[d]] for d in H}; log(f"{name}: statistics at {T} checkpoints, {N} positions kept throughout")
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
# ---------- provenance of the maximum, and the states' persistence at the peaks ----------
res = dict(model=name, block=B, provenance={}, peak_state_persistence={}, null={}, level_matched={}, clusters={})
pos_pers = [(Us[t] * Us[t + 1]).sum(1) for t in range(T - 1)]; allpos = torch.cat(pos_pers)
for d in ("native", "random"):
    acc = {grp: {q: [] for q in ("same", "top5", "top20", "beyond", "old_peak_after", "new_max_after", "S_before", "S_after", "n")} for grp in ("all", "stayer", "leaver")}; pk = []
    for t in range(T - 1):
        W = torch.nonzero(words[d][t])[:, 0]; R0 = H[d][t]["ratio"][:, W].float(); R1 = H[d][t + 1]["ratio"][:, W].float(); nW = len(W); ar = torch.arange(nW)
        a0 = R0.argmax(0); a1 = R1.argmax(0); rank = (R0 > R0[a1, ar][None, :]).sum(0); stay = words[d][t + 1][W]; pk.append(pos_pers[t][a0])
        for grp, msk in (("all", torch.ones(nW, dtype=torch.bool)), ("stayer", stay), ("leaver", ~stay)):
            if msk.sum() == 0: continue
            rk = rank[msk]; acc[grp]["same"] += (rk == 0).float().tolist(); acc[grp]["top5"] += ((rk > 0) & (rk < 5)).float().tolist(); acc[grp]["top20"] += ((rk >= 5) & (rk < 20)).float().tolist(); acc[grp]["beyond"] += (rk >= 20).float().tolist()
            acc[grp]["old_peak_after"] += R1[a0, ar][msk].tolist(); acc[grp]["new_max_after"] += R1.max(0).values[msk].tolist(); acc[grp]["S_before"] += R0.max(0).values[msk].tolist(); acc[grp]["S_after"] += R1.max(0).values[msk].tolist(); acc[grp]["n"].append(int(msk.sum()))
    res["provenance"][d] = {grp: dict(n=sum(v["n"]), same=mean(v["same"]), top5=mean(v["top5"]), top20=mean(v["top20"]), beyond=mean(v["beyond"]), old_peak_after=med(v["old_peak_after"]), new_max_after=med(v["new_max_after"]), S_before=med(v["S_before"]), S_after=med(v["S_after"])) for grp, v in acc.items() if v["n"]}
    res["peak_state_persistence"][d] = float(torch.cat(pk).median())
    log(f"{name} {d} provenance of the maximum: " + " | ".join(f"{grp} ({o['n']}): same position {f2(o['same'])}, top five {f2(o['top5'])}, top twenty {f2(o['top20'])}, beyond {f2(o['beyond'])}; old peak at t+1 {f2(o['old_peak_after'])} against the new maximum {f2(o['new_max_after'])}; S {f2(o['S_before'])} -> {f2(o['S_after'])}" for grp, o in res["provenance"][d].items()) + f"; states' persistence at the peaks {f3(res['peak_state_persistence'][d])}")
res["peak_state_persistence"]["all_positions"] = float(allpos.median()); log(f"{name} states' persistence across a thousand steps, all positions {f3(res['peak_state_persistence']['all_positions'])}")
# ---------- the order-statistics null ----------
EDGES = torch.arange(0.0, 2.0001, 0.05); nb = len(EDGES) - 1; E_dev = EDGES[1:-1].to(DEV)
def kernel(d, exclude_words):
    samples = [[] for _ in range(nb)]
    for t in range(T - 1):
        R0 = H[d][t]["ratio"].to(DEV).float(); R1 = H[d][t + 1]["ratio"].to(DEV).float()
        if exclude_words: cols = torch.nonzero(~(words[d][t] | words[d][t + 1]))[:, 0].to(DEV); R0 = R0[:, cols]; R1 = R1[:, cols]
        r0 = R0.reshape(-1); r1 = R1.reshape(-1); idx = torch.cat([torch.nonzero(r0 >= 0.7)[:, 0], torch.randint(r0.numel(), (200000,), device=DEV, generator=g2)]); x0 = r0[idx]; dl = r1[idx] - x0; b = torch.bucketize(x0, E_dev)
        for bb in range(nb):
            sb = dl[b == bb]
            if sb.numel(): samples[bb].append(sb[torch.randperm(sb.numel(), device=DEV, generator=g2)[:20000]].cpu())
        del R0, R1, r0, r1; torch.cuda.empty_cache()
    S_ = [torch.cat(s) if s else torch.zeros(0) for s in samples]; counts = [int(s.numel()) for s in S_]; pop = [i for i in range(nb) if counts[i] >= 100]
    fb = [min(pop, key=lambda j: abs(j - i)) for i in range(nb)]; return dict(delta=S_, counts=counts, fallback=fb, populated_up_to=float(EDGES[max(pop) + 1]))
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
    out = dict(n=int(act.numel()), median_S_before=float(s0.median()), median_S_after_actual=float(act.median()), median_S_after_null=float(nul.median()), p_over_floor_actual=float(act1.mean()), p_over_floor_null=float(nul1.mean()), corr_actual_null=float(torch.corrcoef(torch.stack([act, nul]))[0, 1]), by_S_bin={})
    for lo_, hi_ in ((0.9, 1.0), (1.0, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.5), (1.5, 9.0)):
        mk = (s0 >= lo_) & (s0 < hi_)
        if mk.sum() >= 10: out["by_S_bin"][f"{lo_}-{hi_}"] = dict(n=int(mk.sum()), actual=float(act[mk].median()), null=float(nul[mk].median()), p_actual=float(act1[mk].mean()), p_null=float(nul1[mk].mean()))
    return out
KER = {"random_atoms": kernel("random", False), "native_nonwords": kernel("native", True)}
for kn, ker in KER.items(): log(f"{name} kernel from {kn}: populated up to level {ker['populated_up_to']:.2f}; samples per bin from 0.9: " + ", ".join(str(ker["counts"][i]) for i in range(18, nb)))
res["kernels"] = {kn: dict(counts=ker["counts"], populated_up_to=ker["populated_up_to"], median_delta_by_bin={f"{float(EDGES[i]):.2f}": float(ker["delta"][i].median()) for i in range(nb) if ker["counts"][i] >= 100}) for kn, ker in KER.items()}
for d, kn in (("native", "random_atoms"), ("native", "native_nonwords"), ("random", "random_atoms")):
    o = null_S(d, KER[kn]); res["null"][f"{d}_words_with_{kn}"] = o
    log(f"{name} null for {d} words with the {kn} kernel ({o['n']} word-checkpoints): S {f2(o['median_S_before'])} -> actual {f2(o['median_S_after_actual'])}, null {f2(o['median_S_after_null'])}; over the floor at t+1 actual {f2(o['p_over_floor_actual'])}, null {f2(o['p_over_floor_null'])}; correlation {f2(o['corr_actual_null'])}; by S at t: " + "; ".join(f"{k}: {v['n']}, actual {f2(v['actual'])} null {f2(v['null'])}, over the floor {f2(v['p_actual'])} against {f2(v['p_null'])}" for k, v in o["by_S_bin"].items()))
# ---------- level-matched persistence of a position's projection ----------
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
for d, which in (("native", "words"), ("random", "words"), ("native", "nonwords"), ("random", "nonwords")):
    o = level_matched(d, which); res["level_matched"][f"{d}_{which}"] = o
    log(f"{name} level-matched persistence, {d} {which}: " + "; ".join(f"{k}: {v['n']} pairs, mean at t+1 {f2(v['mean_after'])}, over the floor {f2(v['p_over_floor_after'])}" for k, v in o.items()))
# ---------- usage clusters ----------
def clusters(d):
    ex, en = events(words[d]); out = {}
    def pair_stats(P, U):
        if P.numel() < 2: return None, None
        C = U[P] @ U[P].T; iu = torch.triu_indices(P.numel(), P.numel(), 1); tok = ids[P]; return float(C[iu[0], iu[1]].mean()), float((tok[iu[0]] == tok[iu[1]]).float().mean())
    for kind, evs in (("exits", ex), ("entries", en)):
        acc = {k: [] for k in ("event_cos", "event_tok", "other_cos", "other_tok", "random_cos", "random_tok", "n_event", "n_other")}; gr = torch.Generator().manual_seed(3)
        for r, t in evs:
            prev = (H[d][t - 1]["sel"] == r).any(1); cur = (H[d][t]["sel"] == r).any(1); P = torch.nonzero(prev & ~cur if kind == "exits" else ~prev & cur)[:, 0]; Q = torch.nonzero(prev & cur)[:, 0]; U = Us[t - 1]
            c1, k1 = pair_stats(P, U); c2, k2 = pair_stats(Q, U); c3, k3 = pair_stats(torch.randperm(N, generator=gr)[:max(P.numel(), 2)], U)
            if c1 is not None: acc["event_cos"].append(c1); acc["event_tok"].append(k1); acc["random_cos"].append(c3); acc["random_tok"].append(k3); acc["n_event"].append(float(P.numel()))
            if c2 is not None: acc["other_cos"].append(c2); acc["other_tok"].append(k2); acc["n_other"].append(float(Q.numel()))
        out[kind] = {k: med(v) for k, v in acc.items()} | {"n_events": len(acc["event_cos"])}
    return out
for d in ("native", "random"):
    o = clusters(d); res["clusters"][d] = o
    log(f"{name} {d} usage clusters: " + " | ".join(f"{kind} ({v['n_events']}): pairwise state cosine among the {'lost' if kind == 'exits' else 'gained'} positions {f3(v['event_cos'])} (token sharing {f3(v['event_tok'])}), among the kept {f3(v['other_cos'])} ({f3(v['other_tok'])}), random positions {f3(v['random_cos'])} ({f3(v['random_tok'])})" for kind, v in o.items()))
# ---------- checks ----------
pv = res["provenance"]["native"]["all"]; nn = res["null"]["native_words_with_random_atoms"]; lm = res["level_matched"]; cl = res["clusters"]["native"]["exits"]; g_ = lambda x: -9 if x is None else x
lm_diff = [g_(lm["native_words"][k]["mean_after"]) - g_(lm["random_words"][k]["mean_after"]) for k in ("1.0-1.1", "1.1-1.2") if k in lm["native_words"] and k in lm["random_words"]]
res["checks"] = dict(maximum_relocates=g_(pv["same"]) < 0.5 and g_(pv["same"]) + g_(pv["top5"]) >= 0.7, breadth_holds_it=abs(nn["median_S_after_actual"] - nn["median_S_after_null"]) <= 0.05 and abs(nn["p_over_floor_actual"] - nn["p_over_floor_null"]) <= 0.10,
                     native_positions_persist_more=bool(lm_diff) and all(x >= 0.05 for x in lm_diff), lost_not_clustered=abs(g_(cl["event_cos"]) - g_(cl["other_cos"])) <= 0.02 and g_(cl["event_tok"]) <= g_(cl["random_tok"]) + 0.02,
                     peaks_not_more_persistent=abs(res["peak_state_persistence"]["native"] - res["peak_state_persistence"]["all_positions"]) <= 0.02)
summ = f"{name} block {B}: native words' maximum at t+1 at the same position {f2(pv['same'])}, top five {f2(pv['top5'])}, beyond twenty {f2(pv['beyond'])} (random words {f2(res['provenance']['random']['all']['same'])}/{f2(res['provenance']['random']['all']['top5'])}/{f2(res['provenance']['random']['all']['beyond'])}); old peak at t+1 {f2(pv['old_peak_after'])} against the new maximum {f2(pv['new_max_after'])}; order-statistics null with the random-atom kernel: native S {f2(nn['median_S_before'])} -> actual {f2(nn['median_S_after_actual'])}, null {f2(nn['median_S_after_null'])}, over the floor {f2(nn['p_over_floor_actual'])} against {f2(nn['p_over_floor_null'])} (native-nonword kernel {f2(res['null']['native_words_with_native_nonwords']['median_S_after_null'])}, {f2(res['null']['native_words_with_native_nonwords']['p_over_floor_null'])}; random words actual {f2(res['null']['random_words_with_random_atoms']['median_S_after_actual'])} null {f2(res['null']['random_words_with_random_atoms']['median_S_after_null'])}); level-matched persistence at 1.0-1.1 / 1.1-1.2, native words {'/'.join(f2(lm['native_words'][k]['mean_after']) for k in ('1.0-1.1', '1.1-1.2') if k in lm['native_words'])} against random words {'/'.join(f2(lm['random_words'][k]['mean_after']) for k in ('1.0-1.1', '1.1-1.2') if k in lm['random_words'])}; states' persistence at native peaks {f3(res['peak_state_persistence']['native'])} against all positions {f3(res['peak_state_persistence']['all_positions'])}; lost positions' pairwise cosine {f3(cl['event_cos'])} against kept {f3(cl['other_cos'])} and random {f3(cl['random_cos'])}, token sharing {f3(cl['event_tok'])} against {f3(cl['random_tok'])} | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e541_maximum_persistence_{name}", res, summ)
