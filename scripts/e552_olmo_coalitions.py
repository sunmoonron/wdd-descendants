"""e552: the coalitions, their precedence and the neuron's selectivity on a second model. From the e550 cache of OLMo-1B
(steps 1000-16000, block 8, the MLP rows of blocks 0-8, OMP over the rows alone), the core of sessions 94-96:
- e545 at step 8000: for every top-4 extreme pair the contribution of each of the 73,728 MLP writers (activation times
  the write's norm times its cosine with the row); the concentration of an extreme (effective writers, top-10 share,
  the attention outputs' share); the similarity of a row's writer vectors at two of its extreme positions against the
  same row at an extreme and a typical position and against different rows; random directions in place of the rows;
  and the persistence of a word's writer vector from step 8000 to 16000 against the selection-matched null (the same
  positions weighted by a random native row's alignment);
- e546: for every clean entrant and an S-matched near miss, at the row's top-4 positions at the event and k = -4..+2,
  S, the coherence of the contributions, the within-row similarity of the writer vectors (contemporaneous alignment)
  and of the activations alone, the own neuron's share;
- e547b: the z-score of the neuron's own activation at its future extreme contexts against all positions, k = -8..+2,
  for entrants, near misses and random non-words.
Pre-registered (honest guesses), OLMo-1B block 8:
- the writer vectors of a row at two of its extremes cohere at 0.35 or more against 0.05 or less across rows, and
  random directions' at most half the rows' (0.6);
- a word's writer vector persists from step 8000 to 16000 at 0.7 or more against 0.6 or less for the null (0.5);
- the attention outputs supply 0.4 or more of an extreme (0.5);
- entrants' within-row similarity at k=-4 is at least 0.15 above the near misses' (0.6);
- the neuron's selectivity precedes entry: z 1.0 or more at k=-8 for entrants, rising to 2.0 or more at entry, under
  0.5 for random non-words (0.5).
Arguments: none."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmo_cache_common import *
name = "olmo1b"; C = load_olmo_cache(); T, N, m, D, DFF, B = C["T"], C["N"], C["m"], C["D"], C["DFF"], C["B"]; ACT, ATT, XS, RU, NORM, WORDS, SS, RATIO = (C[k] for k in ("ACT", "ATT", "XS", "RU", "NORM", "WORDS", "SS", "RATIO")); steps = C["steps"]; KTOP = 4; NRAND = 64
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"; f0 = lambda x: "n/a" if x is None else f"{x:.0f}"
def gvec(i, r_): Ru = RU[i].float().to(DEV); return NORM[i].to(DEV) * (Ru @ Ru[r_])
def cosr(a, b): return ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-9))
def within(Cm):
    Cn = Cm / Cm.norm(dim=1, keepdim=True).clamp_min(1e-9); S_ = Cn @ Cn.T; iu = torch.triu_indices(Cm.shape[0], Cm.shape[0], 1, device=Cm.device); return float(S_[iu[0], iu[1]].mean())
def coherence(Cm): return float(((Cm.sum(1) ** 2) / (Cm ** 2).sum(1).clamp_min(1e-12) - 1).median())
def top_positions(i, r_): return RATIO[i][:, r_].float().topk(KTOP).indices
res = dict(model=name, block=B, n_positions=N, n_rows=m)
# ---------- e545 at step 8000 ----------
i8 = steps.index(8000); i16 = steps.index(16000); ratio8 = RATIO[i8].float().to(DEV); words8 = WORDS[i8].to(DEV); top = ratio8.topk(KTOP, dim=1).indices; pairs = [(int(p_), int(r_)) for p_ in range(N) for r_ in top[p_].tolist()]; byrow = {}
for p_, r_ in pairs: byrow.setdefault(r_, []).append(p_)
multi = [r_ for r_, P in byrow.items() if len(P) >= 2]; ACT8 = ACT[i8]; X8 = XS[i8].to(DEV); A8 = ATT[i8].to(DEV); Ru8 = RU[i8].float().to(DEV); n8 = NORM[i8].to(DEV)
conc = {k: [] for k in ("eff", "top10", "share_attn", "share_mlp")}; samp = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(1))[:1500].tolist()
for j in samp:
    p_, r_ = pairs[j]; c = ACT8[p_].float().to(DEV) * (n8 * (Ru8 @ Ru8[r_])); tot_mlp = float(c.sum()); proj = float(X8[p_] @ Ru8[r_]); a_att = float(A8[p_] @ Ru8[r_]); ca = c.abs(); conc["eff"].append(float(ca.sum() ** 2 / (c ** 2).sum())); conc["top10"].append(float(ca.sort(descending=True).values[:10].sum() / ca.sum())); conc["share_attn"].append(a_att / proj if abs(proj) > 1e-6 else None); conc["share_mlp"].append(tot_mlp / proj if abs(proj) > 1e-6 else None)
conc = {k: med([v for v in vals if v is not None]) for k, vals in conc.items()}
within_, typical, act_within, act_typical = [], [], [], []; rs = torch.Generator(device=DEV).manual_seed(2)
for r_ in multi:
    P = byrow[r_]; gv = n8 * (Ru8 @ Ru8[r_]); Ap = ACT8[P].float().to(DEV); Cm = Ap * gv[None]; within_.append(within(Cm)); q_ = torch.randint(N, (len(P),), device=DEV, generator=rs); Aq = ACT8[q_.cpu()].float().to(DEV); Cq = Aq * gv[None]
    Cn = Cm / Cm.norm(dim=1, keepdim=True).clamp_min(1e-9); Cqn = Cq / Cq.norm(dim=1, keepdim=True).clamp_min(1e-9); typical.append(float((Cn * Cqn).sum(1).mean())); act_within.append(within(Ap)); An = Ap / Ap.norm(dim=1, keepdim=True).clamp_min(1e-9); Aqn = Aq / Aq.norm(dim=1, keepdim=True).clamp_min(1e-9); act_typical.append(float((An * Aqn).sum(1).mean()))
cross, act_cross = [], []; rl = list(byrow.keys()); cs = torch.Generator().manual_seed(3)
for _ in range(1500):
    i1, i2 = torch.randint(len(rl), (2,), generator=cs).tolist()
    if i1 == i2: continue
    r1, r2 = rl[i1], rl[i2]; p1, p2 = byrow[r1][0], byrow[r2][0]; c1 = ACT8[p1].float().to(DEV) * (n8 * (Ru8 @ Ru8[r1])); c2 = ACT8[p2].float().to(DEV) * (n8 * (Ru8 @ Ru8[r2])); cross.append(float(cosr(c1, c2))); a1, a2 = ACT8[p1].float().to(DEV), ACT8[p2].float().to(DEV); act_cross.append(float(cosr(a1, a2)))
Q = unitr(torch.randn(NRAND, D, device=DEV, generator=rs)); PQ = (C["Us"][i8] @ Q.T).abs(); topq = PQ.topk(8, dim=0).indices; rwithin, rcross = [], []
for j in range(NRAND):
    gv = n8 * (Ru8 @ Q[j]); P = topq[:, j]; Cm = ACT8[P.cpu()].float().to(DEV) * gv[None]; rwithin.append(within(Cm))
for _ in range(500):
    j1, j2 = torch.randint(NRAND, (2,), generator=cs).tolist()
    if j1 == j2: continue
    c1 = ACT8[int(topq[0, j1])].float().to(DEV) * (n8 * (Ru8 @ Q[j1])); c2 = ACT8[int(topq[0, j2])].float().to(DEV) * (n8 * (Ru8 @ Q[j2])); rcross.append(float(cosr(c1, c2)))
wl = [r_ for r_ in byrow if bool(words8[r_])]; real_f, null_f, act_f = [], [], []; rs2 = torch.Generator().manual_seed(4); Ru16 = RU[i16].float().to(DEV); n16 = NORM[i16].to(DEV)
for r_ in wl:
    if not WORDS[i16][r_]: continue
    P8 = torch.tensor(byrow[r_]); q_ = int(torch.randint(m, (1,), generator=rs2)); a8 = ACT8[P8].float().to(DEV); b16 = ACT[i16][P8].float().to(DEV); g8 = n8 * (Ru8 @ Ru8[r_]); g16 = n16 * (Ru16 @ Ru16[r_]); h8 = n8 * (Ru8 @ Ru8[q_]); h16 = n16 * (Ru16 @ Ru16[q_])
    real_f.append(float(cosr((a8 * g8[None]).mean(0), (b16 * g16[None]).mean(0)))); null_f.append(float(cosr((a8 * h8[None]).mean(0), (b16 * h16[None]).mean(0)))); act_f.append(float(cosr(a8.mean(0), b16.mean(0))))
res["step8000"] = dict(n_pairs=len(pairs), n_rows_with_extremes=len(byrow), n_rows_multi=len(multi), n_words_multi=int(sum(bool(words8[r_]) for r_ in multi)), concentration=conc, within_row=med(within_), extreme_vs_typical=med(typical), cross_row=med(cross), activation_within=med(act_within), activation_typical=med(act_typical), activation_cross=med(act_cross), random_within=med(rwithin), random_cross=med(rcross), persistence_real=med(real_f), persistence_null=med(null_f), persistence_activations=med(act_f), n_persist=len(real_f))
o = res["step8000"]; log(f"{name} step 8000: {o['n_pairs']} extreme pairs on {o['n_rows_with_extremes']} rows, {o['n_rows_multi']} with two or more positions ({o['n_words_multi']} words); effective writers {f0(o['concentration']['eff'])}, top-10 share {f2(o['concentration']['top10'])}, shares MLP/attention {f2(o['concentration']['share_mlp'])}/{f2(o['concentration']['share_attn'])}; writer-vector similarity within a row {f3(o['within_row'])}, extreme against typical {f3(o['extreme_vs_typical'])}, across rows {f3(o['cross_row'])} (activations alone {f3(o['activation_within'])}/{f3(o['activation_typical'])}/{f3(o['activation_cross'])}); random directions within {f3(o['random_within'])}, across {f3(o['random_cross'])}; persistence of a word's writer vector 8000-16000 real {f3(o['persistence_real'])}, selection-matched null {f3(o['persistence_null'])}, activations alone {f3(o['persistence_activations'])} ({o['n_persist']} words)")
del ratio8, X8, A8; torch.cuda.empty_cache()
# ---------- e546 and e547b: entrants, near misses, random non-words ----------
words = [WORDS[i] for i in range(T)]; S_ = [SS[i] for i in range(T)]; ex, en = events(words); men = match(en, words, S_, lambda W, t: ~W[t - 1] & ~W[t] & ~W[t + 1]); gr_ = torch.Generator().manual_seed(5); rnd = []
for r_, t in en:
    cand = torch.nonzero(~words[t - 1] & ~words[t] & ~words[t + 1])[:, 0]; rnd.append((int(cand[int(torch.randint(cand.numel(), (1,), generator=gr_))]), t))
log(f"{name}: {len(en)} clean entrants, {len(men)} near misses, {len(rnd)} random non-words")
def profile(evs, tag):
    KR = list(range(-8, 3)); acc = {k: {q: [] for q in ("S", "coherence", "within_now", "within_act", "own_share", "z_activation")} for k in KR}
    for r_, t in evs:
        P = top_positions(t, r_)
        for k in KR:
            i = t + k
            if not (0 <= i < T): continue
            act = ACT[i][P].float().to(DEV); gnow = gvec(i, r_); cnow = act * gnow[None]; col = ACT[i][:, r_].float().to(DEV); z = (col[P.to(DEV)].mean() - col.mean()) / col.std().clamp_min(1e-6); own = act[:, r_] * NORM[i][r_].to(DEV)
            a = acc[k]; a["S"].append(float(SS[i][r_]))
            if k >= -4: a["coherence"].append(coherence(cnow)); a["within_now"].append(within(cnow)); a["within_act"].append(within(act)); a["own_share"].append(float((own / cnow.sum(1).clamp_min(1e-6)).median()))
            a["z_activation"].append(float(z))
    out = {str(k): {q: med(v) for q, v in a.items()} | {"n": len(a["S"])} for k, a in acc.items()}
    log(f"{name} {tag} ({len(evs)}): k=-8..+2 S " + "/".join(f2(out[str(k)]["S"]) for k in KR) + "; z-score of the neuron's activation at its future contexts " + "/".join(f2(out[str(k)]["z_activation"]) for k in KR) + "; k=-4..+2 coherence " + "/".join(f2(out[str(k)]["coherence"]) for k in range(-4, 3)) + ", within-row similarity of the writer vectors " + "/".join(f2(out[str(k)]["within_now"]) for k in range(-4, 3)) + ", activations alone " + "/".join(f2(out[str(k)]["within_act"]) for k in range(-4, 3)) + ", own neuron's share " + "/".join(f2(out[str(k)]["own_share"]) for k in range(-4, 3)))
    return dict(by_k=out, n_events=len(evs))
res["entrants"] = profile(en, "entrants"); res["near_misses"] = profile(men, "near misses"); res["random_nonwords"] = profile(rnd, "random non-words")
E, M_, Rn = res["entrants"]["by_k"], res["near_misses"]["by_k"], res["random_nonwords"]["by_k"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(coalition_coheres=g_(o["within_row"]) >= 0.35 and g_(o["cross_row"]) <= 0.05 and g_(o["random_within"]) <= 0.5 * g_(o["within_row"]), coalition_persists=g_(o["persistence_real"]) >= 0.7 and g_(o["persistence_null"]) <= 0.6, attention_half=g_(o["concentration"]["share_attn"]) >= 0.4, precedence=g_(E["-4"]["within_now"]) - g_(M_["-4"]["within_now"]) >= 0.15, selectivity_precedes=g_(E["-8"]["z_activation"]) >= 1.0 and g_(E["0"]["z_activation"]) >= 2.0 and all(g_(Rn[str(k)]["z_activation"]) < 0.5 for k in range(-8, 3)))
summ = f"{name} block {B}: coalition at 8000: within-row {f3(o['within_row'])}, cross {f3(o['cross_row'])}, random directions {f3(o['random_within'])}, persistence {f3(o['persistence_real'])} (null {f3(o['persistence_null'])}), effective writers {f0(o['concentration']['eff'])}, attention share {f2(o['concentration']['share_attn'])} | entrants at k=-8/-4/-1/0/+2: S {f2(E['-8']['S'])}/{f2(E['-4']['S'])}/{f2(E['-1']['S'])}/{f2(E['0']['S'])}/{f2(E['2']['S'])}, within-row at -4/0 {f2(E['-4']['within_now'])}/{f2(E['0']['within_now'])}, coherence at -4/0 {f2(E['-4']['coherence'])}/{f2(E['0']['coherence'])}, z {f2(E['-8']['z_activation'])}/{f2(E['-4']['z_activation'])}/{f2(E['-1']['z_activation'])}/{f2(E['0']['z_activation'])}/{f2(E['2']['z_activation'])} | near misses: S {f2(M_['-8']['S'])}/{f2(M_['-4']['S'])}/{f2(M_['-1']['S'])}/{f2(M_['0']['S'])}/{f2(M_['2']['S'])}, within-row at -4/0 {f2(M_['-4']['within_now'])}/{f2(M_['0']['within_now'])}, z {f2(M_['-8']['z_activation'])}/{f2(M_['-4']['z_activation'])}/{f2(M_['0']['z_activation'])} | random non-words z {f2(Rn['-8']['z_activation'])}/{f2(Rn['0']['z_activation'])} | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e552_olmo_coalitions_{name}", res, summ)
