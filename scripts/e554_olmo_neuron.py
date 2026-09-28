"""e554: the OLMo word's neuron, the attention share and the birth threshold, re-measured against the critique session
99 invites. Three objections to e552 anticipated: the z-score of a signed SwiGLU activation is not the selectivity a
GELU z-score is, so the neuron's part must be measured by the size of its activation and by its own write's rank among
the contributors; the attention outputs' share at the extremes (0.14 against Pythia's 0.54) may be OLMo's share
everywhere, not a fact about extremes; and the extreme-value floor may not be OLMo's birth threshold (its entrants
enter at S 1.40 and its near misses stay at 1.22 without entering), so the gate and the near-miss control need the
words' own level. From the e550 cache (and the e524 Pythia cache for the threshold comparison):
- for entrants, near misses and random non-words at k = -8..+2: the z-score of the neuron's absolute activation at its
  future extreme contexts, the share of positions with a larger absolute activation, and its own write's rank among
  the 73,728 contributors at those contexts (median over positions; the share of events with the own write in the top
  hundred);
- at step 8000: the attention outputs', the MLP writes' and the embedding's share of the state (their projection on
  the state over its squared norm) at the words' extreme positions and at random positions;
- per checkpoint, on both models: the words' minimum, tenth-percentile and median S, the number of non-words over the
  floor and over the words' tenth percentile, and the entrants' S at entry over the words' tenth percentile.
Pre-registered (honest guesses):
- the OLMo entrant's neuron is not selective by absolute activation either (z under 0.5 at k=-8, under 1.0 at entry)
  and its own write is not among the top hundred contributors to its extreme (median rank above 100) (0.6);
- attention's share of the state is low on OLMo everywhere (under 0.3 at random positions as at the extremes) (0.6);
- the floor is not OLMo's birth threshold: more than 2,000 non-words sit over the floor at step 8000 (Pythia under
  1,000), and the entrants enter above the words' tenth percentile (0.6).
Arguments: none."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmo_cache_common import *
name = "olmo1b"; C = load_olmo_cache(); T, N, m, D, DFF, B = C["T"], C["N"], C["m"], C["D"], C["DFF"], C["B"]; ACT, ATT, EMB, XS, RU, NORM, WORDS, SS, RATIO = (C[k] for k in ("ACT", "ATT", "EMB", "XS", "RU", "NORM", "WORDS", "SS", "RATIO")); steps = C["steps"]; KTOP = 4
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"; f0 = lambda x: "n/a" if x is None else f"{x:.0f}"
RUd = {}
def rud(i):
    if i not in RUd: RUd[i] = RU[i].float().to(DEV)
    return RUd[i]
def top_positions(i, r_): return RATIO[i][:, r_].float().topk(KTOP).indices
words = [WORDS[i] for i in range(T)]; S_ = [SS[i] for i in range(T)]; ex, en = events(words); men = match(en, words, S_, lambda W, t: ~W[t - 1] & ~W[t] & ~W[t + 1]); gr_ = torch.Generator().manual_seed(5); rnd = []
for r_, t in en:
    cand = torch.nonzero(~words[t - 1] & ~words[t] & ~words[t + 1])[:, 0]; rnd.append((int(cand[int(torch.randint(cand.numel(), (1,), generator=gr_))]), t))
KR = list(range(-8, 3)); res = dict(model=name, block=B, groups={})
for tag, evs in (("entrants", en), ("near_misses", men), ("random_nonwords", rnd)):
    acc = {k: {q: [] for q in ("S", "z_abs", "rank_abs", "signed_over_mean", "own_rank", "own_top100")} for k in KR}
    for r_, t in evs:
        P = top_positions(t, r_); Pd = P.to(DEV)
        for k in KR:
            i = t + k
            if not (0 <= i < T): continue
            col = ACT[i][:, r_].float().to(DEV); a = col.abs(); z = (a[Pd].mean() - a.mean()) / a.std().clamp_min(1e-6); rk = (a[None, :] > a[Pd][:, None]).float().mean(); sm = col[Pd].mean() / col.abs().mean().clamp_min(1e-6)
            Ru = rud(i); g = NORM[i].to(DEV) * (Ru @ Ru[r_]); cm = ACT[i][P].float().to(DEV) * g[None]; own = cm[:, r_].abs(); ranks = (cm.abs() > own[:, None]).sum(1).float()
            a_ = acc[k]; a_["S"].append(float(SS[i][r_])); a_["z_abs"].append(float(z)); a_["rank_abs"].append(float(rk)); a_["signed_over_mean"].append(float(sm)); a_["own_rank"].append(float(ranks.median())); a_["own_top100"].append(float((ranks < 100).float().mean()))
    out = {str(k): {q: med(v) for q, v in a_.items()} | {"n": len(a_["S"])} for k, a_ in acc.items()}; res["groups"][tag] = dict(by_k=out, n_events=len(evs))
    log(f"{name} {tag} ({len(evs)}): k=-8..+2 S " + "/".join(f2(out[str(k)]["S"]) for k in KR) + "; z-score of the absolute activation at the future contexts " + "/".join(f2(out[str(k)]["z_abs"]) for k in KR) + "; share of positions with a larger absolute activation " + "/".join(f2(out[str(k)]["rank_abs"]) for k in KR) + "; signed activation there over the mean absolute " + "/".join(f2(out[str(k)]["signed_over_mean"]) for k in KR) + "; own write's rank among contributors " + "/".join(f0(out[str(k)]["own_rank"]) for k in KR) + ", in the top hundred for " + "/".join(f2(out[str(k)]["own_top100"]) for k in KR))
# attention's, the MLP's and the embedding's share of the state at extremes and at random positions, step 8000
i8 = steps.index(8000); X8 = XS[i8].to(DEV); A8 = ATT[i8].to(DEV); E8 = EMB[i8].to(DEV); M8 = X8 - A8 - E8; n2 = (X8 ** 2).sum(1).clamp_min(1e-9); sh_att = (A8 * X8).sum(1) / n2; sh_mlp = (M8 * X8).sum(1) / n2; sh_emb = (E8 * X8).sum(1) / n2
wl = torch.nonzero(WORDS[i8])[:, 0]; ratio8 = RATIO[i8].float().to(DEV); Pw = ratio8[:, wl.to(DEV)].topk(KTOP, dim=0).indices.reshape(-1); Ru8 = rud(i8); proj_att, proj_x = [], []
for j, r_ in enumerate(wl.tolist()):
    P = ratio8[:, r_].topk(KTOP).indices; u = Ru8[r_]; proj_att.append(float(((A8[P] @ u) / (X8[P] @ u).clamp_min(1e-6)).median())); proj_x.append(float((X8[P] @ u).median()))
res["shares_step8000"] = dict(state_share_attention_at_extremes=float(sh_att[Pw].median()), state_share_attention_random=float(sh_att.median()), state_share_mlp_at_extremes=float(sh_mlp[Pw].median()), state_share_mlp_random=float(sh_mlp.median()), state_share_embedding_at_extremes=float(sh_emb[Pw].median()), state_share_embedding_random=float(sh_emb.median()), projection_share_attention_at_extremes=med(proj_att), attention_norm_over_state_norm=float((A8.norm(dim=1) / X8.norm(dim=1)).median()))
o = res["shares_step8000"]; log(f"{name} step 8000: attention's share of the state (projection on the state over its squared norm) at the words' extreme positions {f2(o['state_share_attention_at_extremes'])} against {f2(o['state_share_attention_random'])} at all positions; the MLP writes' {f2(o['state_share_mlp_at_extremes'])} against {f2(o['state_share_mlp_random'])}; the embedding's {f2(o['state_share_embedding_at_extremes'])} against {f2(o['state_share_embedding_random'])}; attention's share of the words' projections at their extremes {f2(o['projection_share_attention_at_extremes'])}; attention's norm over the state's {f2(o['attention_norm_over_state_norm'])}")
del X8, A8, E8, M8, ratio8; torch.cuda.empty_cache()
# the birth threshold on both models
def threshold_stats(SSd, WORDSd, entr, Tn):
    out = {}
    for i in range(Tn):
        S = SSd[i]; W = WORDSd[i]; ws = S[W]; q10 = float(ws.quantile(0.1)); out[i] = dict(words_min_S=float(ws.min()), words_q10_S=q10, words_median_S=float(ws.median()), nonwords_over_floor=int(((~W) & (S >= 1.0)).sum()), nonwords_over_words_q10=int(((~W) & (S >= q10)).sum()), n_rows=int(S.numel()))
    ent_rel = [float(SSd[t][r_] / out[t]["words_q10_S"]) for r_, t in entr]; return out, med(ent_rel)
thr_o, ent_o = threshold_stats(SS, WORDS, en, T); res["threshold_olmo"] = dict(by_checkpoint={str(steps[i]): v for i, v in thr_o.items()}, entrants_S_over_words_q10=ent_o)
log(f"{name} birth threshold: at steps 1000/4000/8000/16000 the words' minimum / tenth-percentile / median S " + "; ".join(f"{steps[i]//1000}k {thr_o[i]['words_min_S']:.2f} / {thr_o[i]['words_q10_S']:.2f} / {thr_o[i]['words_median_S']:.2f}" for i in (0, 3, 7, 15)) + "; non-words over the floor " + "/".join(str(thr_o[i]["nonwords_over_floor"]) for i in (0, 3, 7, 15)) + ", over the words' tenth percentile " + "/".join(str(thr_o[i]["nonwords_over_words_q10"]) for i in (0, 3, 7, 15)) + f"; entrants' S at entry over the words' tenth percentile {f2(ent_o)}")
PC = "/workspace/wdd/cache/e524_pythia410"; pst = list(range(1000, 16001, 1000)); pck = {n: torch.load(f"{PC}/step{n}.pt", map_location="cpu") for n in pst}; pkeep = torch.stack([pck[n]["blocks"][12]["keep"] for n in pst]).all(0); PS, PW = {}, {}
for i, n in enumerate(pst):
    U = unitr(pck[n]["blocks"][12]["U"][pkeep].float().to(DEV)); st = stats(U, pck[n]["rows"].float().to(DEV), 16); PS[i] = st["S"]; PW[i] = wordset(st["usage"]); del st, U; torch.cuda.empty_cache()
pwords = [PW[i] for i in range(len(pst))]; pex, pen = events(pwords); thr_p, ent_p = threshold_stats(PS, PW, pen, len(pst)); res["threshold_pythia"] = dict(by_checkpoint={str(pst[i]): v for i, v in thr_p.items()}, entrants_S_over_words_q10=ent_p)
log(f"pythia410 birth threshold: at steps 1000/4000/8000/16000 the words' minimum / tenth-percentile / median S " + "; ".join(f"{pst[i]//1000}k {thr_p[i]['words_min_S']:.2f} / {thr_p[i]['words_q10_S']:.2f} / {thr_p[i]['words_median_S']:.2f}" for i in (0, 3, 7, 15)) + "; non-words over the floor " + "/".join(str(thr_p[i]["nonwords_over_floor"]) for i in (0, 3, 7, 15)) + ", over the words' tenth percentile " + "/".join(str(thr_p[i]["nonwords_over_words_q10"]) for i in (0, 3, 7, 15)) + f"; entrants' S at entry over the words' tenth percentile {f2(ent_p)}")
E = res["groups"]["entrants"]["by_k"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(neuron_not_selective_by_absolute=g_(E["-8"]["z_abs"]) < 0.5 and g_(E["0"]["z_abs"]) < 1.0 and g_(E["0"]["own_rank"]) > 100, attention_low_everywhere=g_(o["state_share_attention_random"]) < 0.3 and g_(o["state_share_attention_at_extremes"]) < 0.3, floor_not_birth_threshold=thr_o[7]["nonwords_over_floor"] > 2000 and thr_p[7]["nonwords_over_floor"] < 1000 and g_(ent_o) >= 1.0)
summ = f"{name} block {B}: entrants' absolute-activation z at k=-8/-4/0/+2 {f2(E['-8']['z_abs'])}/{f2(E['-4']['z_abs'])}/{f2(E['0']['z_abs'])}/{f2(E['2']['z_abs'])}, own write's rank {f0(E['-8']['own_rank'])}/{f0(E['-4']['own_rank'])}/{f0(E['0']['own_rank'])}/{f0(E['2']['own_rank'])} (near misses z {f2(res['groups']['near_misses']['by_k']['-8']['z_abs'])}/{f2(res['groups']['near_misses']['by_k']['0']['z_abs'])}, random {f2(res['groups']['random_nonwords']['by_k']['-8']['z_abs'])}/{f2(res['groups']['random_nonwords']['by_k']['0']['z_abs'])}); attention's share of the state at extremes / everywhere {f2(o['state_share_attention_at_extremes'])} / {f2(o['state_share_attention_random'])}, MLP {f2(o['state_share_mlp_at_extremes'])} / {f2(o['state_share_mlp_random'])}; threshold at 8000: OLMo words' q10 S {thr_o[7]['words_q10_S']:.2f}, non-words over the floor {thr_o[7]['nonwords_over_floor']}, entrants over q10 {f2(ent_o)}; Pythia words' q10 S {thr_p[7]['words_q10_S']:.2f}, non-words over the floor {thr_p[7]['nonwords_over_floor']}, entrants over q10 {f2(ent_p)} | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e554_olmo_neuron_{name}", res, summ)
