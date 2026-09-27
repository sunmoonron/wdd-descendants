"""e524b: how a row moves into the vocabulary, a thousand steps at a time (an analysis over e524's records). e522 and
e523 found the rise of a future word's alignment to be its row rotating toward the states and into the cloud's
principal subspace, measured over intervals of four to sixteen thousand steps. Over sixteen checkpoints a thousand
steps apart this run follows every row's increment against the cloud it faces at the time, and aligns the entrants'
trajectories at their entry. Per row and interval: the increment's fraction inside the current top-32 principal
subspace, its cosine with the state at the row's current best position, its cosine with the row itself (growth
against rotation), its cosine with the next increment (a steady drift gives a positive value, a random walk zero),
the rank of the old best position under the new row (with the old states, and with both moved), the profile's
stability across positions under the row's motion alone and under the states' motion alone (Spearman over positions),
and the relative size of the increment. Per row and checkpoint: the largest projection over the floor, the counts of
positions over the floor and over three quarters of it, the row's fraction in the top-32 subspace and its mean
squared projection. Entry events are dated at a thousand steps (a clean entry: a non-word at the two checkpoints
before, a word at the checkpoint and the next), the trajectories aligned at the event from five thousand steps
before to three after, against S-matched non-entrants and against all non-words at each time. The relayed take's
four readings, steady, episodic, concentrated at entry, or continuous tracking, are read from the aligned medians.
Pre-registered (honest guesses), block 12, clean entrants:
- the increments are a steady drift: the cosine of consecutive increments is above 0.2 at every aligned interval
  from four thousand steps before entry to two after (0.5);
- the motion concentrates at entry: the increment's cosine with the current best state in the interval ending at
  entry is at least 1.5 times its value four intervals earlier (0.4);
- continuous tracking: the row's fraction in the top-32 subspace rises at every aligned checkpoint from four
  thousand steps before entry to one after (0.4);
- the profile is preserved: the profile's stability under the row's motion is higher for entrants than for matched
  rows at every aligned interval up to entry (0.5);
- breadth first: two thousand steps before entry the entrants' count over three quarters of the floor is at least
  1.5 times the matched rows' (0.5).
Arguments: name."""
import sys, os, json as _json, math, glob, re; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; CDIR = f"/workspace/wdd/cache/e524_{name}"; BL = [12, 6]; NWORD = 256; KMIN, KMAX = -5, 3
files = sorted(glob.glob(f"{CDIR}/step*.pt"), key=lambda f: int(re.search(r"step(\d+)", f).group(1)))
steps = [int(re.search(r"step(\d+)", f).group(1)) for f in files]; n = len(steps); assert n >= 6, steps
log(f"{name}: {n} checkpoints {steps[0]}-{steps[-1]}")
recs = [torch.load(f) for f in files]
def colrank(M): return M.argsort(0).argsort(0).float()
def col_spearman(A, B):
    ra = colrank(A); rb = colrank(B); ra = ra - ra.mean(0); rb = rb - rb.mean(0); return (ra * rb).sum(0) / (ra.norm(dim=0) * rb.norm(dim=0)).clamp_min(1e-9)
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def spearman(a, b):
    if a.numel() < 5: return None
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double(); ra = ra - ra.mean(); rb = rb - rb.mean(); return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))
res = dict(model=name, steps=steps, by_block={})
LEVELS = ("S", "cnt", "cnt75", "q", "m2"); INCS = ("g", "c", "growth", "ac", "rank_old", "rank_both", "persist", "stab_row", "stab_state", "rel", "dS", "dB75")
for b in BL:
    R = recs[0]["blocks"][b]["n_rows"]; D = recs[0]["rows"].shape[1]
    W = [ (r["rows"][:R].float() * r["norms"][:R, None]).to(DEV) for r in recs ]                                      # row vectors per checkpoint
    Bk = [r["blocks"][b] for r in recs]
    words = []
    for k in range(n):
        u = Bk[k]["usage"]; w = torch.zeros(R, dtype=torch.bool); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; words.append(w)
    lev = {q: torch.stack([Bk[k][q].float() if q != "q" else ((recs[k]["rows"][:R].float().to(DEV) @ Bk[k]["V"].to(DEV)).pow(2).sum(1)).cpu() for k in range(n)]) for q in LEVELS}   # [n, R]
    inc = {q: torch.full((n - 1, R), float("nan")) for q in INCS}
    dh_prev = None
    for k in range(n - 1):
        U0 = Bk[k]["U"].float().to(DEV); keep0 = Bk[k]["keep"].to(DEV); L0 = Bk[k]["L"].to(DEV); V0 = Bk[k]["V"].to(DEV); a0 = Bk[k]["a"].to(DEV)
        U1 = Bk[k + 1]["U"].float().to(DEV); keep1 = Bk[k + 1]["keep"].to(DEV); L1 = Bk[k + 1]["L"].to(DEV)
        w0, w1 = W[k], W[k + 1]; u0, u1 = unitr(w0), unitr(w1); dw = w1 - w0; dh = unitr(dw)
        inc["rel"][k] = (dw.norm(dim=1) / w0.norm(dim=1).clamp_min(1e-9)).cpu(); inc["g"][k] = (dh @ V0).pow(2).sum(1).cpu(); inc["c"][k] = (dh * U0[a0]).sum(1).cpu(); inc["growth"][k] = (dh * u0).sum(1).cpu()
        if dh_prev is not None: inc["ac"][k - 1] = (dh_prev * dh).sum(1).cpu()
        dh_prev = dh
        P00 = U0 @ u0.T; P10 = U0 @ u1.T; P01 = U1 @ u0.T; P11 = U1 @ u1.T                                              # states x rows: (state time, row time)
        r10 = P10.abs() / L0[:, None]; r10[~keep0] = 0; old = r10.gather(0, a0[None])[0]; inc["rank_old"][k] = ((r10 > old[None]).sum(0) + 1).float().cpu()
        r11 = P11.abs() / L1[:, None]; r11[~keep1] = 0; old = r11.gather(0, a0[None])[0]; rk = ((r11 > old[None]).sum(0) + 1).float(); rk[~keep1[a0]] = float("nan"); inc["rank_both"][k] = rk.cpu()
        inc["persist"][k] = (inc["rank_both"][k] == 1).float(); inc["persist"][k][inc["rank_both"][k].isnan()] = float("nan")
        inc["stab_row"][k] = col_spearman(P00[keep0], P10[keep0]).cpu(); both = keep0 & keep1; inc["stab_state"][k] = col_spearman(P00[both], P01[both]).cpu()
        inc["dS"][k] = lev["S"][k + 1] - lev["S"][k]; inc["dB75"][k] = lev["cnt75"][k + 1] - lev["cnt75"][k]
        del U0, U1, P00, P10, P01, P11, r10, r11; torch.cuda.empty_cache()
    # entry events
    ev = []
    for k in range(2, n - 1):
        e = ~words[k - 2] & ~words[k - 1] & words[k] & words[k + 1]
        for r in torch.nonzero(e)[:, 0].tolist(): ev.append((r, k))
    any_entries = sum(int((~words[k - 1] & words[k]).sum()) for k in range(1, n))
    matched = []; taken = torch.zeros(R, dtype=torch.bool)
    for r, k in ev:
        okm = ~words[k - 1] & ~words[k] & ~words[k + 1] & ~taken; okm[r] = False; cand = torch.nonzero(okm)[:, 0]; j = cand[(lev["S"][k - 1][cand] - lev["S"][k - 1][r]).abs().argmin()]; taken[j] = True; matched.append((int(j), k))
    def aligned(events):
        out = {}
        for kk in range(KMIN, KMAX + 1):
            vals = {q: [] for q in LEVELS + INCS}
            for r, k in events:
                i = k + kk
                if 0 <= i < n:
                    for q in LEVELS: vals[q].append(float(lev[q][i][r]))
                if 0 <= i < n - 1:
                    for q in INCS: vals[q].append(float(inc[q][i][r]))
            out[kk] = {q: (float(torch.tensor(v).nanmedian()) if len(v) else None) for q, v in vals.items()}; out[kk]["n"] = len(vals["S"])
        return out
    ent_al = aligned(ev); mat_al = aligned(matched)
    # all non-words by absolute interval
    allnw = {}
    for k in range(n - 1):
        m = ~words[k] & ~words[k + 1]; allnw[steps[k]] = {q: float(inc[q][k][m].nanmedian()) for q in INCS}; allnw[steps[k]].update({q: float(lev[q][k][m].median()) for q in LEVELS}); allnw[steps[k]]["n"] = int(m.sum())
        allnw[steps[k]]["spearman_g_dB75"] = spearman(inc["g"][k][m].nan_to_num(0), inc["dB75"][k][m]); allnw[steps[k]]["spearman_c_dS"] = spearman(inc["c"][k][m].nan_to_num(0), inc["dS"][k][m])
    # entrants against matched at each aligned interval: AUC of the increment quantities
    aucs = {}
    for kk in range(KMIN, 1):
        aucs[kk] = {}
        for q in ("g", "c", "stab_row", "ac", "cnt75", "q"):
            src = inc if q in INCS else lev; e_v = torch.tensor([float(src[q][k + kk][r]) for r, k in ev if 0 <= k + kk < (n - 1 if q in INCS else n)]); m_v = torch.tensor([float(src[q][k + kk][r]) for r, k in matched if 0 <= k + kk < (n - 1 if q in INCS else n)])
            aucs[kk][q] = auc(e_v[~e_v.isnan()], m_v[~m_v.isnan()])
    res["by_block"][b] = dict(n_rows=R, n_clean_entries=len(ev), n_any_entries=any_entries, entrants=ent_al, matched=mat_al, all_non_words=allnw, auc_entrants_vs_matched=aucs, cloud_top32_variance={steps[k]: float(Bk[k]["evfrac"][31]) for k in range(n)})
    fm = lambda x: "n/a" if x is None else f"{x:.2f}"; fi = lambda x: "n/a" if x is None else f"{x:.0f}"
    log(f"{name} block {b}: {len(ev)} clean entries ({any_entries} entries of any kind) over {n} checkpoints; aligned medians, entrants (matched): k = thousand steps from entry, levels at k, increments over (k, k+1)")
    for kk in range(KMIN, KMAX + 1):
        e_, m_ = ent_al[kk], mat_al[kk]
        log(f"  k={kk:+d} n={e_['n']}: S {fm(e_['S'])} ({fm(m_['S'])}), over floor {fm(e_['cnt'])} ({fm(m_['cnt'])}), over 3/4 {fm(e_['cnt75'])} ({fm(m_['cnt75'])}), row in top-32 {fm(e_['q'])} ({fm(m_['q'])}), mean sq proj {fm(e_['m2'])} ({fm(m_['m2'])}) | increment: in top-32 {fm(e_['g'])} ({fm(m_['g'])}), toward best state {fm(e_['c'])} ({fm(m_['c'])}), growth {fm(e_['growth'])} ({fm(m_['growth'])}), consecutive cosine {fm(e_['ac'])} ({fm(m_['ac'])}), rank of old best under new row {fi(e_['rank_old'])} ({fi(m_['rank_old'])}), both moved {fi(e_['rank_both'])} ({fi(m_['rank_both'])}), persists {fm(e_['persist'])} ({fm(m_['persist'])}), profile stability row/state {fm(e_['stab_row'])}/{fm(e_['stab_state'])} ({fm(m_['stab_row'])}/{fm(m_['stab_state'])}), relative size {fm(e_['rel'])} ({fm(m_['rel'])})")
    log(f"{name} block {b}: all non-words by interval: " + " | ".join(f"{s}: S {v['S']:.2f}, over 3/4 {v['cnt75']:.1f}, top-32 {v['q']:.3f}; inc in top-32 {v['g']:.3f}, toward best {v['c']:.3f}, growth {v['growth']:.2f}, consecutive {fm(v['ac'])}, stability row/state {v['stab_row']:.2f}/{v['stab_state']:.2f}, rel {v['rel']:.2f}; Spearman g vs dB {fm(v['spearman_g_dB75'])}, c vs dS {fm(v['spearman_c_dS'])}" for s, v in allnw.items()))
    log(f"{name} block {b}: AUC entrants vs matched by aligned interval: " + " | ".join(f"k={kk:+d}: " + ", ".join(f"{q} {fm(v)}" for q, v in aucs[kk].items()) for kk in aucs))
E = res["by_block"][12]["entrants"]; M_ = res["by_block"][12]["matched"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(steady_drift=all(g_(E[k]["ac"]) > 0.2 for k in range(-4, 3)), concentrated_at_entry=g_(E[-1]["c"]) >= 1.5 * g_(E[-5]["c"]) and g_(E[-5]["c"]) > 0, continuous_tracking=all(g_(E[k + 1]["q"]) > g_(E[k]["q"]) for k in range(-4, 1)),
                     profile_preserved=all(g_(E[k]["stab_row"]) > g_(M_[k]["stab_row"]) for k in range(KMIN, 0)), breadth_first=g_(E[-2]["cnt75"]) >= 1.5 * g_(M_[-2]["cnt75"]))
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name}: " + " | ".join(f"block {b}: {o['n_clean_entries']} clean entries; entrants at k=-4/-2/-1/0/+2: S {'/'.join(fm(o['entrants'][k]['S']) for k in (-4, -2, -1, 0, 2))}, over 3/4 {'/'.join(fm(o['entrants'][k]['cnt75']) for k in (-4, -2, -1, 0, 2))} (matched {'/'.join(fm(o['matched'][k]['cnt75']) for k in (-4, -2, -1, 0, 2))}), row in top-32 {'/'.join(fm(o['entrants'][k]['q']) for k in (-4, -2, -1, 0, 2))}; increment in top-32 {'/'.join(fm(o['entrants'][k]['g']) for k in (-4, -2, -1, 0, 2))} (matched {'/'.join(fm(o['matched'][k]['g']) for k in (-4, -2, -1, 0, 2))}), toward best state {'/'.join(fm(o['entrants'][k]['c']) for k in (-4, -2, -1, 0, 2))} (matched {'/'.join(fm(o['matched'][k]['c']) for k in (-4, -2, -1, 0, 2))}), consecutive cosine {'/'.join(fm(o['entrants'][k]['ac']) for k in (-4, -2, -1, 0, 2))} (matched {'/'.join(fm(o['matched'][k]['ac']) for k in (-4, -2, -1, 0, 2))}), stability row {'/'.join(fm(o['entrants'][k]['stab_row']) for k in (-4, -2, -1, 0, 2))} (matched {'/'.join(fm(o['matched'][k]['stab_row']) for k in (-4, -2, -1, 0, 2))}), persists {'/'.join(fm(o['entrants'][k]['persist']) for k in (-4, -2, -1, 0, 2))}" for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e524b_trajectories_{name}", res, summ)
