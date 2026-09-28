"""Shared analysis for the exit and takeover experiments (e539, e540): OMP statistics with the per-position
selections and the sixteenth-largest projection over the floor at every position (the cutoff), clean entry and exit
events, e538b's aligned profiles, the hysteresis table, the own-versus-cutoff decomposition at the positions a row
loses or gains, and the takeover audit (who takes a leaver's positions, who an entrant displaces)."""
import sys, os, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
NWORD = 256; SB = [0.9, 1.0, 1.1, 1.2, 1.3, 1.5]; CB = [0, 10, 20, 35, 10 ** 9]
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def med(v): return float(torch.tensor(v, dtype=torch.float64).median()) if len(v) else None
def mean(v): return float(torch.tensor(v, dtype=torch.float64).mean()) if len(v) else None
def wmed(v, w):
    if not len(v): return None
    v = torch.tensor(v, dtype=torch.float64); w = torch.tensor(w, dtype=torch.float64); o = v.argsort(); c = w[o].cumsum(0); return float(v[o][int((c >= c[-1] / 2).nonzero()[0])])
def wordset(u):
    w = torch.zeros(u.numel(), dtype=torch.bool); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; return w
def stats(U, A, K=16, seed=11):
    """U unit states [N, D] and A unit atoms [m, D] on the GPU. Returns, on the CPU: S, the counts over the floor and over three quarters of it, OMP usage, the selections [N, K], the cutoff (the K-th largest projection over the floor at each position) and the whole ratio matrix [N, m] in half precision."""
    m = A.shape[0]; Ar = unitr(rotate(A, seed=seed)); CA = A.T @ A / m; CR = Ar.T @ Ar / m; gm = gabs(m); N = U.shape[0]
    s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 256] @ Ar.T).abs().max(1).values for s in range(0, N, 256)]); r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); L = s2.clamp_min(1e-12).sqrt() * gm * r_cal
    ratio = (U @ A.T).abs() / L[:, None]; S = ratio.max(0).values; cnt = (ratio > 1).sum(0).float(); cnt75 = (ratio > 0.75).sum(0).float(); cut = ratio.topk(K, dim=1).values[:, -1]
    sel, _, _ = omp(U, A, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m).float()
    out = dict(S=S.cpu(), cnt=cnt.cpu(), cnt75=cnt75.cpu(), usage=usage.cpu(), sel=sel.cpu(), cut=cut.cpu(), ratio=ratio.half().cpu(), L=L.cpu()); del ratio; return out
def events(words):
    ex, en = [], []
    for t in range(2, len(words) - 1):
        e = words[t - 2] & words[t - 1] & ~words[t] & ~words[t + 1]; ex += [(int(r), t) for r in torch.nonzero(e)[:, 0]]
        e2 = ~words[t - 2] & ~words[t - 1] & words[t] & words[t + 1]; en += [(int(r), t) for r in torch.nonzero(e2)[:, 0]]
    return ex, en
def match(events_, words, S_, stay):
    Rn = S_[0].numel(); taken = torch.zeros(Rn, dtype=torch.bool); out = []
    for r, t in events_:
        okm = stay(words, t) & ~taken; okm[r] = False; cand = torch.nonzero(okm)[:, 0]
        if cand.numel() == 0: continue
        j = cand[(S_[t - 1][cand] - S_[t - 1][r]).abs().argmin()]; taken[j] = True; out.append((int(j), t))
    return out
def aligned(events_, H, thr, KMIN=-4, KMAX=2):
    o = {}
    for kk in range(KMIN, KMAX + 1):
        vals = {q: [] for q in ("S", "cnt", "cnt75", "usage_over_threshold")}
        for r, t in events_:
            i = t + kk
            if 0 <= i < len(H): vals["S"].append(float(H[i]["S"][r])); vals["cnt"].append(float(H[i]["cnt"][r])); vals["cnt75"].append(float(H[i]["cnt75"][r])); vals["usage_over_threshold"].append(float(H[i]["usage"][r]) / thr[i])
        o[kk] = {q: med(v) for q, v in vals.items()}; o[kk]["n"] = len(vals["S"])
    return o
def profiles(H):
    """e538b's exit and entry profiles from a history of dicts with S, cnt, cnt75 and usage; also returns the word sets, the thresholds and the events."""
    words = [wordset(h["usage"]) for h in H]; thr = [max(float(h["usage"][w].min()), 1.0) for h, w in zip(H, words)]; S_ = [h["S"] for h in H]; ex, en = events(words)
    mex = match(ex, words, S_, lambda W, t: W[t - 2] & W[t - 1] & W[t] & W[t + 1]); men = match(en, words, S_, lambda W, t: ~W[t - 1] & ~W[t] & ~W[t + 1])
    fr = dict(leavers_over_floor_k0=mean([float(S_[t][r] >= 1) for r, t in ex]), leavers_over_floor_k2=mean([float(S_[t + 2][r] >= 1) for r, t in ex if t + 2 < len(H)]), entrants_under_floor_km1=mean([float(S_[t - 1][r] < 1) for r, t in en]), entrants_under_floor_km2=mean([float(S_[t - 2][r] < 1) for r, t in en]))
    prof = dict(n_clean_exits=len(ex), n_clean_entries=len(en), exits_aligned=aligned(ex, H, thr), exits_matched_stayers=aligned(mex, H, thr), entries_aligned=aligned(en, H, thr), entries_matched=aligned(men, H, thr), delta_S_exit=med([float(S_[t + 1][r] - S_[t - 1][r]) for r, t in ex]), delta_S_entry=med([float(S_[t + 1][r] - S_[t - 1][r]) for r, t in en]), fractions=fr, retention_4=mean([float((words[t] & words[t + 4]).sum() / NWORD) for t in range(len(words) - 4)]), entries_per_interval=mean([float((~words[t] & words[t + 1]).sum()) for t in range(len(words) - 1)]))
    return prof, words, thr, ex, en
def hysteresis(H, words):
    """The probability of being a word at the next checkpoint for words and for non-words at this one, within bins of S and of the count over three quarters of the floor; the matched difference weights each cell by its smaller count."""
    out = {}; T = len(H)
    for i in range(len(SB) - 1):
        cw = [0, 0]; cn = [0, 0]; diffs = []; wts = []
        for j in range(len(CB) - 1):
            nw = [0, 0]; nn = [0, 0]
            for t in range(T - 1):
                S = H[t]["S"]; c = H[t]["cnt75"]; inb = (S >= SB[i]) & (S < SB[i + 1]) & (c >= CB[j]) & (c < CB[j + 1]); nxt = words[t + 1]; w = inb & words[t]; n_ = inb & ~words[t]
                nw[0] += int(w.sum()); nw[1] += int((w & nxt).sum()); nn[0] += int(n_.sum()); nn[1] += int((n_ & nxt).sum())
            cw[0] += nw[0]; cw[1] += nw[1]; cn[0] += nn[0]; cn[1] += nn[1]
            if nw[0] >= 5 and nn[0] >= 5: diffs.append(nw[1] / nw[0] - nn[1] / nn[0]); wts.append(min(nw[0], nn[0]))
        out[f"{SB[i]:.1f}-{SB[i + 1]:.1f}"] = dict(n_words=cw[0], p_word_stays=(cw[1] / cw[0]) if cw[0] else None, n_nonwords=cn[0], p_nonword_enters=(cn[1] / cn[0]) if cn[0] else None, matched_difference=(sum(d * w for d, w in zip(diffs, wts)) / sum(wts)) if wts else None)
    return out
def decompose(events_, H, kind):
    """At the positions a leaver loses (kind 'exit': selected at t-1, not at t) or an entrant gains ('entry'): the row's own projection over the floor and the cutoff before and after, their changes, the row's rank among all atoms, and the share of positions where the cutoff moved more than the row's own projection."""
    acc = {q: [] for q in ("n_positions", "own_before", "own_after", "cut_before", "cut_after", "d_own", "d_cut", "rank_before", "rank_after", "cutoff_moved_more")}
    for r, t in events_:
        prev = (H[t - 1]["sel"] == r).any(1); cur = (H[t]["sel"] == r).any(1); P = torch.nonzero(prev & ~cur if kind == "exit" else ~prev & cur)[:, 0]
        if P.numel() == 0: continue
        ob = H[t - 1]["ratio"][P, r].float(); oa = H[t]["ratio"][P, r].float(); cb = H[t - 1]["cut"][P]; ca = H[t]["cut"][P]; rb = (H[t - 1]["ratio"][P].float() > ob[:, None]).sum(1).float(); ra = (H[t]["ratio"][P].float() > oa[:, None]).sum(1).float()
        acc["n_positions"].append(float(P.numel())); acc["own_before"].append(float(ob.median())); acc["own_after"].append(float(oa.median())); acc["cut_before"].append(float(cb.median())); acc["cut_after"].append(float(ca.median())); acc["d_own"].append(float((oa - ob).median())); acc["d_cut"].append(float((ca - cb).median())); acc["rank_before"].append(float(rb.median())); acc["rank_after"].append(float(ra.median())); acc["cutoff_moved_more"].append(float(((ca - cb).abs() > (oa - ob).abs()).float().mean()))
    return {q: med(v) for q, v in acc.items()} | {"n_events": len(acc["n_positions"])}
def takeover(ex, en, H, words, A_at, seed=0):
    """Who takes a leaver's lost positions (the atoms new at those positions at t), and who an entrant displaces at its gained positions. Concentration (the effective number of takers over the number of lost positions, the top taker's share), the takers' class at t (established word, entrant, non-word), their cosine to the leaver and their rank among the leaver's neighbours, their projection at the lost positions before and at the exit, and a null that draws as many random positions and takes their newcomers."""
    g = torch.Generator().manual_seed(seed); T = len(H); N = H[0]["sel"].shape[0]; exset = {t: set(r for r, tt in ex if tt == t) for t in range(T)}; enset = {t: set(r for r, tt in en if tt == t) for t in range(T)}
    acc = {k: [] for k in ("eff_over_lost", "top_share", "share_established", "share_entrant", "share_nonword", "cos_to_leaver", "neighbour_rank", "taker_ratio_before2", "taker_ratio_before", "taker_ratio_at", "taker_usage_over_threshold_before", "lost_with_entrant", "n_lost")}
    nul = {k: [] for k in ("eff_over_lost", "top_share", "share_established", "share_entrant", "share_nonword", "cos_to_leaver", "neighbour_rank")}
    dis = {k: [] for k in ("share_leaver", "share_word_stays", "share_nonword_before", "gained_with_leaver", "n_gained", "displaced_ratio_before", "displaced_ratio_at")}
    thr = [max(float(h["usage"][w].min()), 1.0) for h, w in zip(H, words)]
    for t in range(2, T - 1):
        if not exset[t] and not enset[t]: continue
        A = A_at(t); est = words[t - 1] & words[t]; ent = ~words[t - 1] & words[t]; sp, sc = H[t - 1]["sel"], H[t]["sel"]
        newc = [sc[p][~torch.isin(sc[p], sp[p])] for p in range(N)]; oldc = [sp[p][~torch.isin(sp[p], sc[p])] for p in range(N)]
        def taker_stats(r, P, store, full):
            w = {}
            for p in P.tolist():
                nw = newc[p]
                for a in nw.tolist(): w[a] = w.get(a, 0.0) + 1.0 / nw.numel()
            if not w: return
            at = torch.tensor(list(w.keys())); wt = torch.tensor(list(w.values())); tot = float(wt.sum()); store["eff_over_lost"].append(float(tot ** 2 / (wt ** 2).sum() / P.numel())); store["top_share"].append(float(wt.max() / tot))
            store["share_established"].append(float(wt[est[at]].sum() / tot)); store["share_entrant"].append(float(wt[ent[at]].sum() / tot)); store["share_nonword"].append(float(wt[~words[t][at]].sum() / tot))
            cosall = A @ A[r]; ca = cosall[at]; store["cos_to_leaver"].append(float((ca * wt).sum() / tot)); store["neighbour_rank"].append(wmed([float((cosall > c).sum() - 1) for c in ca.tolist()], wt.tolist()))
            if full:
                pr = []; pr2 = []; pa = []; pu = []
                for p in P.tolist():
                    for a in newc[p].tolist(): pr2.append(float(H[t - 2]["ratio"][p, a])); pr.append(float(H[t - 1]["ratio"][p, a])); pa.append(float(H[t]["ratio"][p, a])); pu.append(float(H[t - 1]["usage"][a]) / thr[t - 1])
                store["taker_ratio_before2"].append(med(pr2)); store["taker_ratio_before"].append(med(pr)); store["taker_ratio_at"].append(med(pa)); store["taker_usage_over_threshold_before"].append(med(pu)); store["lost_with_entrant"].append(mean([float(bool(ent[newc[p]].any())) for p in P.tolist()])); store["n_lost"].append(float(P.numel()))
        for r in exset[t]:
            P = torch.nonzero((sp == r).any(1) & ~(sc == r).any(1))[:, 0]
            if P.numel() == 0: continue
            taker_stats(r, P, acc, True); taker_stats(r, torch.randperm(N, generator=g)[:P.numel()], nul, False)
        for r in enset[t]:
            G = torch.nonzero(~(sp == r).any(1) & (sc == r).any(1))[:, 0]
            if G.numel() == 0: continue
            w = {}
            for p in G.tolist():
                oc = oldc[p]
                for a in oc.tolist(): w[a] = w.get(a, 0.0) + 1.0 / oc.numel()
            if not w: continue
            at = torch.tensor(list(w.keys())); wt = torch.tensor(list(w.values())); tot = float(wt.sum()); lv = torch.tensor([a in exset[t] for a in at.tolist()])
            dis["share_leaver"].append(float(wt[lv].sum() / tot)); dis["share_word_stays"].append(float(wt[est[at]].sum() / tot)); dis["share_nonword_before"].append(float(wt[~words[t - 1][at]].sum() / tot)); dis["gained_with_leaver"].append(mean([float(any(a in exset[t] for a in oldc[p].tolist())) for p in G.tolist()])); dis["n_gained"].append(float(G.numel()))
            dis["displaced_ratio_before"].append(med([float(H[t - 1]["ratio"][p, a]) for p in G.tolist() for a in oldc[p].tolist()])); dis["displaced_ratio_at"].append(med([float(H[t]["ratio"][p, a]) for p in G.tolist() for a in oldc[p].tolist()]))
    return dict(takers={k: med(v) for k, v in acc.items()} | {"n_events": len(acc["eff_over_lost"])}, takers_null={k: med(v) for k, v in nul.items()}, displaced={k: med(v) for k, v in dis.items()} | {"n_events": len(dis["share_leaver"])})
def full_analysis(H, A_at, tag, name):
    prof, words, thr, ex, en = profiles(H); hy = hysteresis(H, words); dx = decompose(ex, H, "exit"); dn = decompose(en, H, "entry"); tk = takeover(ex, en, H, words, A_at)
    f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f0 = lambda x: "n/a" if x is None else f"{x:.0f}"; E, X, F = prof["entries_aligned"], prof["exits_aligned"], prof["fractions"]
    log(f"{name} {tag}: {prof['n_clean_exits']} clean exits, {prof['n_clean_entries']} clean entries, {prof['entries_per_interval']:.0f} entries per interval, retention at 4 {f2(prof['retention_4'])}; leavers' S at k=-2/-1/0/+1/+2 {f2(X[-2]['S'])}/{f2(X[-1]['S'])}/{f2(X[0]['S'])}/{f2(X[1]['S'])}/{f2(X[2]['S'])}, usage over the threshold {f2(X[-1]['usage_over_threshold'])}/{f2(X[0]['usage_over_threshold'])}/{f2(X[2]['usage_over_threshold'])}, over the floor at k=0/+2 {f2(F['leavers_over_floor_k0'])}/{f2(F['leavers_over_floor_k2'])}; entrants' S {f2(E[-2]['S'])}/{f2(E[-1]['S'])}/{f2(E[0]['S'])}/{f2(E[1]['S'])}/{f2(E[2]['S'])}, under the floor at -1/-2 {f2(F['entrants_under_floor_km1'])}/{f2(F['entrants_under_floor_km2'])}; S change across exits {f2(prof['delta_S_exit'])}, entries {f2(prof['delta_S_entry'])}")
    log(f"{name} {tag} hysteresis (S bin: words' stay rate / non-words' entry rate / matched difference, counts): " + "; ".join(f"{k}: {f2(v['p_word_stays'])}/{f2(v['p_nonword_enters'])}/{f2(v['matched_difference'])} ({v['n_words']}/{v['n_nonwords']})" for k, v in hy.items()))
    log(f"{name} {tag} decomposition at lost positions (exits, {dx['n_events']} events): positions {f0(dx['n_positions'])}, own {f2(dx['own_before'])}->{f2(dx['own_after'])}, cutoff {f2(dx['cut_before'])}->{f2(dx['cut_after'])}, d_own {f2(dx['d_own'])}, d_cut {f2(dx['d_cut'])}, rank {f0(dx['rank_before'])}->{f0(dx['rank_after'])}, cutoff moved more at {f2(dx['cutoff_moved_more'])} of positions | at gained positions (entries, {dn['n_events']}): positions {f0(dn['n_positions'])}, own {f2(dn['own_before'])}->{f2(dn['own_after'])}, cutoff {f2(dn['cut_before'])}->{f2(dn['cut_after'])}, d_own {f2(dn['d_own'])}, d_cut {f2(dn['d_cut'])}, rank {f0(dn['rank_before'])}->{f0(dn['rank_after'])}, cutoff moved more at {f2(dn['cutoff_moved_more'])}")
    tk_, nu, di = tk["takers"], tk["takers_null"], tk["displaced"]
    log(f"{name} {tag} takers ({tk_['n_events']} exits): lost positions {f0(tk_['n_lost'])}, effective takers over lost {f2(tk_['eff_over_lost'])} (null {f2(nu['eff_over_lost'])}), top share {f2(tk_['top_share'])} ({f2(nu['top_share'])}), established/entrant/non-word {f2(tk_['share_established'])}/{f2(tk_['share_entrant'])}/{f2(tk_['share_nonword'])} (null {f2(nu['share_established'])}/{f2(nu['share_entrant'])}/{f2(nu['share_nonword'])}), cosine to the leaver {f2(tk_['cos_to_leaver'])} ({f2(nu['cos_to_leaver'])}), neighbour rank {f0(tk_['neighbour_rank'])} ({f0(nu['neighbour_rank'])}), takers' ratio at the position at t-2/t-1/t {f2(tk_['taker_ratio_before2'])}/{f2(tk_['taker_ratio_before'])}/{f2(tk_['taker_ratio_at'])}, takers' usage over the threshold before {f2(tk_['taker_usage_over_threshold_before'])}, lost positions with an entrant among the takers {f2(tk_['lost_with_entrant'])} | displaced by entrants ({di['n_events']}): gained positions {f0(di['n_gained'])}, leaver/word that stays/non-word before {f2(di['share_leaver'])}/{f2(di['share_word_stays'])}/{f2(di['share_nonword_before'])}, gained positions with a leaver {f2(di['gained_with_leaver'])}, displaced ratio before/at {f2(di['displaced_ratio_before'])}/{f2(di['displaced_ratio_at'])}")
    return dict(profiles=prof, hysteresis=hy, decomposition=dict(exits=dx, entries=dn), takeover=tk)
