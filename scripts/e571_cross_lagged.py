"""e571 (session 102): the causal loop between the row and the cloud at the one-checkpoint resolution. e558b found that
over 8000 steps an entrant's rise is the row's turn into a direction the cloud already speaks. Here every interval
of the session-88 cache (Pythia-410m, block 12, steps 1000-16000, rows of blocks 0-12, 2027 common positions) gives
four S matrices: the old rows in the old cloud, the new rows in the new cloud, the new rows in the old cloud and the old
rows in the new cloud (each cloud with its own floor). Per interval and row: the row part (new row in the old cloud
minus old row in the old cloud) and the cloud part (new row in the new cloud minus new row in the old cloud), and the
other order. Aligned on clean entries and exits (k = -4..+2) for entrants, S-matched near misses, leavers and random
rows. Also the cloud's variance along the row's direction (does the cloud grow along the row: neuron -> data) and the
row's turn toward the cloud's variance (does the row turn to where the cloud has variance: data -> neuron), each against
a random-direction null of the same size. Pre-registered (probabilities are honest guesses):
 X1 (0.6) at the entry interval the row part exceeds the cloud part, as in e558b, and it peaks at k = 0 or -1;
 X2 (0.5) before entry (k <= -2) the cloud part leads: the cloud rises along the direction first, then the row turns;
 X3 (0.6) the leavers' fall is the cloud part at every k."""
from s101_common import *
t0 = time.time(); CD = "/workspace/wdd/cache/e524_pythia410"; steps = list(range(1000, 16001, 1000)); T = len(steps)
ck = {n: torch.load(f"{CD}/step{n}.pt", map_location="cpu") for n in steps}; keepc = torch.stack([ck[n]["blocks"][12]["keep"] for n in steps]).all(0); N = int(keepc.sum())
U = {i: unitr(ck[n]["blocks"][12]["U"][keepc].float().to(DEV)) for i, n in enumerate(steps)}; Rw = {i: ck[n]["rows"].float().to(DEV) for i, n in enumerate(steps)}; m = Rw[0].shape[0]
H = {}; cal = {}
for i in range(T):
    H[i] = stats(U[i], Rw[i], K); cal[i] = floor_calibration(U[i], Rw[i]); torch.cuda.empty_cache()
words = [wordset(H[i]["usage"]) for i in range(T)]; S_ = [H[i]["S"] for i in range(T)]; ex, en = events(words)
def Smax(Ui, rows, Li):
    return torch.cat([((Ui[s:s + 256] @ rows.T).abs() / Li[s:s + 256, None]).max(0).values[None] for s in range(0, Ui.shape[0], 256)]).max(0).values.cpu()
# cross terms per interval i-1 -> i: S(old cloud | new rows), S(new cloud | old rows)
Sx = {}
for i in range(1, T):
    Lo, Ln = floor_of(U[i - 1], cal[i - 1]), floor_of(U[i], cal[i]); Sx[i] = dict(new_in_old=Smax(U[i - 1], Rw[i], Lo), old_in_new=Smax(U[i], Rw[i - 1], Ln)); torch.cuda.empty_cache()
log(f"cross S computed for {T - 1} intervals ({N} positions) in {time.time() - t0:.0f}s")
def parts(i, rows):
    """interval i-1 -> i for the given rows: row part then cloud part, and cloud part then row part"""
    s_oo, s_nn = S_[i - 1][rows], S_[i][rows]; s_no, s_on = Sx[i]["new_in_old"][rows], Sx[i]["old_in_new"][rows]
    return dict(rise=s_nn - s_oo, row_first=s_no - s_oo, cloud_after_row=s_nn - s_no, cloud_first=s_on - s_oo, row_after_cloud=s_nn - s_on)
mex = match(ex, words, S_, lambda W, t: W[t - 2] & W[t - 1] & W[t] & W[t + 1]); men = match(en, words, S_, lambda W, t: ~W[t - 1] & ~W[t] & ~W[t + 1])
g = torch.Generator().manual_seed(0); rnd = [(int(r), t) for r, t in zip(torch.randint(0, m, (400,), generator=g).tolist(), torch.randint(3, T - 1, (400,), generator=g).tolist())]
def aligned_parts(evs, KMIN=-4, KMAX=2):
    out = {}
    for kk in range(KMIN, KMAX + 1):
        acc = {q: [] for q in ("rise", "row_first", "cloud_after_row", "cloud_first", "row_after_cloud")}
        for r, t in evs:
            i = t + kk
            if 1 <= i < T:
                p = parts(i, torch.tensor([r]))
                for q in acc: acc[q].append(float(p[q][0]))
        out[kk] = {q: med(v) for q, v in acc.items()}; out[kk]["n"] = len(acc["rise"])
    return out
res = dict(n_positions=N, n_entries=len(en), n_exits=len(ex), aligned={g_: aligned_parts(e_) for g_, e_ in (("entrants", en), ("near_misses", men), ("leavers", ex), ("matched_stayers", mex), ("random", rnd))})
for g_, a in res["aligned"].items():
    log(f"{g_}: " + " | ".join(f"k={kk}: rise {a[kk]['rise']:+.3f} = row {a[kk]['row_first']:+.3f} + cloud {a[kk]['cloud_after_row']:+.3f} (cloud-first {a[kk]['cloud_first']:+.3f} + row {a[kk]['row_after_cloud']:+.3f}; n {a[kk]['n']})" for kk in range(-4, 3)))
# variance-based loop: does the cloud grow along the row's old direction, does the row turn to where the cloud has variance
def var_along(Ui, rows): return torch.cat([((Ui @ rows[s:s + 4096].T) ** 2).mean(0) for s in range(0, rows.shape[0], 4096)]).cpu()
loop = {kk: {q: [] for q in ("cloud_growth_along_old_row", "cloud_growth_random", "row_turn_toward_variance", "row_turn_random")} for kk in range(-4, 3)}
gg = torch.Generator(device=DEV).manual_seed(1)
for i in range(1, T):
    idx = [(r, kk) for r, t in en for kk in range(-4, 3) if t + kk == i]
    if not idx: continue
    rows = torch.tensor([r for r, _ in idx], device=DEV); Ro, Rn = Rw[i - 1][rows], Rw[i][rows]
    vo_o, vn_o = var_along(U[i - 1], Ro), var_along(U[i], Ro)                  # cloud at old/new time along the old row
    dR = Rn - Ro; ang = dR.norm(dim=1, keepdim=True); Rr = unitr(Ro + ang * unitr(torch.randn(Ro.shape, device=DEV, generator=gg)))   # a random turn of the same size
    vo_n, vo_r = var_along(U[i - 1], Rn), var_along(U[i - 1], Rr)               # old cloud along the new row / along a random turn
    base = float(((U[i] @ unitr(torch.randn(256, U[i].shape[1], device=DEV, generator=gg)).T) ** 2).mean() / ((U[i - 1] @ unitr(torch.randn(256, U[i].shape[1], device=DEV, generator=gg)).T) ** 2).mean())
    for q, (r, kk) in enumerate(idx):
        loop[kk]["cloud_growth_along_old_row"].append(float(vn_o[q] / vo_o[q].clamp_min(1e-12))); loop[kk]["cloud_growth_random"].append(base)
        loop[kk]["row_turn_toward_variance"].append(float(vo_n[q] / vo_o[q].clamp_min(1e-12))); loop[kk]["row_turn_random"].append(float(vo_r[q] / vo_o[q].clamp_min(1e-12)))
res["variance_loop"] = {kk: {q: med(v) for q, v in d.items()} for kk, d in loop.items()}
for kk in range(-4, 3):
    v = res["variance_loop"][kk]; log(f"k={kk}: cloud variance along the old row grows x{v['cloud_growth_along_old_row']:.3f} (random directions x{v['cloud_growth_random']:.3f}); the row's turn moves it to x{v['row_turn_toward_variance']:.3f} of the old variance (a random turn of the same size x{v['row_turn_random']:.3f})")
E, Lv = res["aligned"]["entrants"], res["aligned"]["leavers"]; V = res["variance_loop"]
summ = (f"cross-lagged at one-checkpoint resolution ({len(en)} entries, {len(ex)} exits): entrants' rise at k=-2/-1/0 {E[-2]['rise']:+.3f}/{E[-1]['rise']:+.3f}/{E[0]['rise']:+.3f} = row {E[-2]['row_first']:+.3f}/{E[-1]['row_first']:+.3f}/{E[0]['row_first']:+.3f} + cloud {E[-2]['cloud_after_row']:+.3f}/{E[-1]['cloud_after_row']:+.3f}/{E[0]['cloud_after_row']:+.3f} (cloud-first order: cloud {E[-2]['cloud_first']:+.3f}/{E[-1]['cloud_first']:+.3f}/{E[0]['cloud_first']:+.3f}, row {E[-2]['row_after_cloud']:+.3f}/{E[-1]['row_after_cloud']:+.3f}/{E[0]['row_after_cloud']:+.3f}); "
        f"leavers at k=-1/0/+1 rise {Lv[-1]['rise']:+.3f}/{Lv[0]['rise']:+.3f}/{Lv[1]['rise']:+.3f} = row {Lv[-1]['row_first']:+.3f}/{Lv[0]['row_first']:+.3f}/{Lv[1]['row_first']:+.3f} + cloud {Lv[-1]['cloud_after_row']:+.3f}/{Lv[0]['cloud_after_row']:+.3f}/{Lv[1]['cloud_after_row']:+.3f}; random rows' row/cloud parts {res['aligned']['random'][0]['row_first']:+.3f}/{res['aligned']['random'][0]['cloud_after_row']:+.3f}; "
        f"variance loop at k=-1/0: cloud along the old row x{V[-1]['cloud_growth_along_old_row']:.3f}/x{V[0]['cloud_growth_along_old_row']:.3f} (random x{V[-1]['cloud_growth_random']:.3f}/x{V[0]['cloud_growth_random']:.3f}), the row's turn to x{V[-1]['row_turn_toward_variance']:.3f}/x{V[0]['row_turn_toward_variance']:.3f} of the old variance (random turn x{V[-1]['row_turn_random']:.3f}/x{V[0]['row_turn_random']:.3f}) | {time.time() - t0:.0f}s")
log(summ); record("e571_cross_lagged", res, summ)
