"""e546: does the ensemble come before the word? e545 found each word's extremes written by a persistent, word-specific
ensemble of thousands of writers. Three temporal readings: the ensemble forms first and the projection follows (the
ensemble causes entry); the projection becomes extreme first and the decomposition merely names the writers afterward;
or a common context raises both together. At all sixteen checkpoints, block 12, the state at every kept position is
split into its writes (53,248 MLP writers, thirteen attention outputs, the embedding) and the rows taken from the same
model. For every clean entrant (a non-word at t-2 and t-1, a word at t and t+1) and an S-matched non-entrant (the
near miss: the same S at t-1, a non-word through t+1), at the row's top-4 positions at t and at k = -4..+2: the row's S;
the MLP part of the projection; the coherence of the contributions, the squared sum over the sum of squares less one
(how far the sum exceeds an incoherent sum); the within-row similarity of the writer vectors with the contemporaneous
alignment and with the alignment fixed at entry; the persistence to entry of the activation pattern, of the
entry-aligned writer vector and of the contemporaneous one; the attention outputs' share; the own neuron's share. The
rise of the MLP projection from k=-4 to 0 is split into the activations' change, the alignment's change from the row's
rotation, the alignment's change from the writers' motion, and the remainder. Also, at step 8000: the coherence for
persistent words, their random positions and random rows; the ensemble-by-row specificity matrix (the aggregate write
of a word's top-500 writers at its extreme positions, projected on every word); and a selection-matched null for the
ensembles' persistence from step 8000 to 16000 (the same positions weighted by a random native row's alignment).
Pre-registered (honest guesses):
- the context is there first: at k=-4 the activation pattern at an entrant's entry-time positions has cosine 0.5 or
  more with its entry-time pattern, and so does the near miss's (stability is generic) (0.6);
- the rise is the alignment's, not the activations': from k=-4 to 0 the alignment's change accounts for more than half
  of the MLP projection's rise, the row's rotation for more of it than the writers' motion, the activations' change for
  less than a quarter (0.5);
- coherence rises across the entry, at least doubling from k=-4 to 0, and is at least twice the near miss's at k=0 (0.5);
- the ensembles' persistence is mostly the contexts': the selection-matched null persists at 0.7 or more against the
  real 0.88 (0.5);
- the ensembles are private: a word's aggregate ensemble write has a cosine with its own row at least three times its
  mean cosine with other words (0.7).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
from lr_common import eval_ids
name = sys.argv[1]; B = 12; K = 16; KTOP = 4; steps = list(range(1000, 16001, 1000)); T = len(steps); CDIR = f"/workspace/wdd/cache/e524_{name}"; NTOP_E = 500
idsB = eval_ids(name)[:8, :256].to(DEV); keepc = torch.stack([torch.load(f"{CDIR}/step{n}.pt", map_location="cpu")["blocks"][B]["keep"] for n in steps]).all(0); N = int(keepc.sum())
class Stop(Exception): pass
def capture(model, arch, fam):
    cap = {"act": {}, "attn": {}, "out": None, "emb": None}; layers = arch.layers; attn_mod = lambda bb: layers[bb].attention if fam == "neox" else (layers[bb].attn if fam == "gpt2" else layers[bb].self_attn); emb_mod = model.gpt_neox.embed_in if fam == "neox" else (model.transformer.wte if fam == "gpt2" else model.model.embed_tokens)
    def h_attn(bb):
        def f(m, i, o): cap["attn"][bb] = (o[0] if isinstance(o, tuple) else o).detach().float()
        return f
    def h_act(bb):
        def f(m, a): cap["act"][bb] = a[0].detach().float(); return None
        return f
    def h_out(m, i, o): cap["out"] = (o[0] if isinstance(o, tuple) else o).detach().float(); raise Stop
    def h_emb(m, i, o): cap["emb"] = o.detach().float()
    hs = [attn_mod(bb).register_forward_hook(h_attn(bb)) for bb in range(B + 1)] + [arch.mlp_lin(bb).register_forward_pre_hook(h_act(bb)) for bb in range(B + 1)] + [emb_mod.register_forward_hook(h_emb), layers[B].register_forward_hook(h_out)]
    try: model(idsB)
    except Stop: pass
    finally: [h.remove() for h in hs]
    return cap
def flat(t): return t.reshape(8, 256, -1)[:, 1:].reshape(2040, -1)[keepc]
ACT, ATT, XS, RU, NORM, WORDS, SS, RATIO = {}, {}, {}, {}, {}, {}, {}, {}
for i, n in enumerate(steps):
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF
    for p_ in model.parameters(): p_.requires_grad_(False)
    cap = capture(model, arch, fam); Wv = torch.cat([arch.wdir(bb).float() for bb in range(B + 1)]); NORM[i] = Wv.norm(dim=1).cpu(); Ru = unitr(Wv); RU[i] = Ru.half().cpu(); m = Ru.shape[0]
    X = flat(cap["out"]); XS[i] = X.cpu(); ACT[i] = torch.cat([flat(cap["act"][bb]) for bb in range(B + 1)], 1).half().cpu(); ATT[i] = torch.stack([flat(cap["attn"][bb]) for bb in range(B + 1)], 1).sum(1).cpu()
    U = unitr(X - X.mean(0)); st = stats(U, Ru, K); RATIO[i] = st["ratio"]; SS[i] = st["S"]; WORDS[i] = wordset(st["usage"]); del st, model, cap, Wv, Ru, X, U; torch.cuda.empty_cache(); log(f"{name} step{n}: captured ({int(WORDS[i].sum())} words)")
def top_positions(i, r_): return RATIO[i][:, r_].float().topk(KTOP).indices
def gvec(i, r_, j=None):
    """alignment of every writer with row r_: the writers' norms and directions from checkpoint i, the row's direction from checkpoint j (default i)."""
    j = i if j is None else j; Ru = RU[i].float().to(DEV); return (NORM[i].to(DEV) * (Ru @ RU[j][r_].float().to(DEV)))
def cosr(a, b): return ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-9))
def within(Cm):
    Cn = Cm / Cm.norm(dim=1, keepdim=True).clamp_min(1e-9); S_ = Cn @ Cn.T; iu = torch.triu_indices(Cm.shape[0], Cm.shape[0], 1, device=Cm.device); return float(S_[iu[0], iu[1]].mean())
def coherence(Cm): return float(((Cm.sum(1) ** 2) / (Cm ** 2).sum(1).clamp_min(1e-12) - 1).median())
# ---------- events: clean entrants and S-matched near misses ----------
words = [WORDS[i] for i in range(T)]; S_ = [SS[i] for i in range(T)]; ex, en = events(words); men = match(en, words, S_, lambda W, t: ~W[t - 1] & ~W[t] & ~W[t + 1]); log(f"{name}: {len(en)} clean entrants, {len(men)} matched near misses")
QS = ("S", "mlp_projection", "coherence", "within_now", "within_fixed", "within_act", "persist_act", "persist_fixed", "persist_now", "attention_share", "own_share")
def profile(evs, tag):
    acc = {k: {q: [] for q in QS} for k in range(-4, 3)}; dec = {q: [] for q in ("total", "activations", "row_rotation", "writers_motion", "remainder")}
    for r_, t in evs:
        P = top_positions(t, r_).to(DEV); Ru_t = RU[t][r_].float().to(DEV); g_fix = gvec(t, r_); act0 = ACT[t][P.cpu()].float().to(DEV); c_fix0 = act0 * g_fix[None]; c_now0 = c_fix0
        for k in range(-4, 3):
            i = t + k
            if not (0 <= i < T): continue
            act = ACT[i][P.cpu()].float().to(DEV); g_now = gvec(i, r_); c_now = act * g_now[None]; c_fix = act * g_fix[None]; Ru_i = RU[i][r_].float().to(DEV); X = XS[i][P.cpu()].to(DEV); A = ATT[i][P.cpu()].to(DEV)
            proj = X @ Ru_i; att = A @ Ru_i; own = act[:, r_] * NORM[i][r_].to(DEV)
            a = acc[k]; a["S"].append(float(SS[i][r_])); a["mlp_projection"].append(float(c_now.sum(1).median())); a["coherence"].append(coherence(c_now)); a["within_now"].append(within(c_now)); a["within_fixed"].append(within(c_fix)); a["within_act"].append(within(act))
            a["persist_act"].append(float(cosr(act, act0).median())); a["persist_fixed"].append(float(cosr(c_fix, c_fix0).median())); a["persist_now"].append(float(cosr(c_now, c_now0).median())); a["attention_share"].append(float((att / proj.clamp_min(1e-6)).median())); a["own_share"].append(float((own / c_now.sum(1).clamp_min(1e-6)).median()))
            if k == -4:
                act_m4, g_m4 = act, g_now; g_row = gvec(i, r_, j=t)
        if t - 4 >= 0:
            total = (act0 * g_fix[None]).sum(1) - (act_m4 * g_m4[None]).sum(1); d_act = ((act0 - act_m4) * g_m4[None]).sum(1); d_row = (act_m4 * (g_row - g_m4)[None]).sum(1); d_wr = (act_m4 * (g_fix - g_row)[None]).sum(1); rem = total - d_act - d_row - d_wr
            for q, v in (("total", total), ("activations", d_act), ("row_rotation", d_row), ("writers_motion", d_wr), ("remainder", rem)): dec[q].append(float(v.median()))
    out = {str(k): {q: med(v) for q, v in a.items()} | {"n": len(a["S"])} for k, a in acc.items()}; dsum = {q: med(v) for q, v in dec.items()} | {"n": len(dec["total"])}
    f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
    log(f"{name} {tag} ({len(evs)} events), k=-4..+2: S " + "/".join(f2(out[str(k)]["S"]) for k in range(-4, 3)) + "; MLP projection " + "/".join(f3(out[str(k)]["mlp_projection"]) for k in range(-4, 3)) + "; coherence " + "/".join(f2(out[str(k)]["coherence"]) for k in range(-4, 3)) + "; within-row similarity of the writer vectors, contemporaneous alignment " + "/".join(f2(out[str(k)]["within_now"]) for k in range(-4, 3)) + ", entry alignment " + "/".join(f2(out[str(k)]["within_fixed"]) for k in range(-4, 3)) + ", activations alone " + "/".join(f2(out[str(k)]["within_act"]) for k in range(-4, 3)) + "; persistence to entry of the activations " + "/".join(f2(out[str(k)]["persist_act"]) for k in range(-4, 3)) + ", of the entry-aligned writer vector " + "/".join(f2(out[str(k)]["persist_fixed"]) for k in range(-4, 3)) + ", of the contemporaneous one " + "/".join(f2(out[str(k)]["persist_now"]) for k in range(-4, 3)) + "; attention share " + "/".join(f2(out[str(k)]["attention_share"]) for k in range(-4, 3)) + "; own neuron's share " + "/".join(f2(out[str(k)]["own_share"]) for k in range(-4, 3)) + f"; rise of the MLP projection from k=-4 to 0 ({dsum['n']}): total {f3(dsum['total'])} = activations {f3(dsum['activations'])} + row's rotation {f3(dsum['row_rotation'])} + writers' motion {f3(dsum['writers_motion'])} + remainder {f3(dsum['remainder'])}")
    return dict(by_k=out, decomposition=dsum, n_events=len(evs))
res = dict(model=name, block=B, entrants=profile(en, "entrants"), near_misses=profile(men, "S-matched near misses"))
# ---------- coherence of the groups at step 8000, the specificity matrix, the matched-null persistence ----------
i8, i16 = steps.index(8000), steps.index(16000); m = RU[i8].shape[0]; gr = torch.Generator(device=DEV).manual_seed(0); f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
persist_words = [r_ for r_ in torch.nonzero(WORDS[i8] & WORDS[i8 - 1])[:, 0].tolist()]; rand_rows = torch.nonzero(~WORDS[i8])[:, 0][torch.randperm(int((~WORDS[i8]).sum()), generator=torch.Generator().manual_seed(1))[:256]].tolist()
def coh_group(rows_, pos_fn):
    v = []
    for r_ in rows_:
        P = pos_fn(r_); act = ACT[i8][P.cpu()].float().to(DEV); v.append(coherence(act * gvec(i8, r_)[None]))
    return med(v)
grp = dict(persistent_words_at_their_top_positions=coh_group(persist_words, lambda r_: top_positions(i8, r_)), persistent_words_at_random_positions=coh_group(persist_words, lambda r_: torch.randint(N, (KTOP,), generator=torch.Generator().manual_seed(2))), random_rows_at_their_top_positions=coh_group(rand_rows, lambda r_: top_positions(i8, r_)), entrants_at_k0=res["entrants"]["by_k"]["0"]["coherence"], near_misses_at_k0=res["near_misses"]["by_k"]["0"]["coherence"])
res["coherence_groups_step8000"] = grp; log(f"{name} coherence at step 8000: " + ", ".join(f"{k} {f2(v)}" for k, v in grp.items()))
wl = torch.nonzero(WORDS[i8])[:, 0].tolist(); Ru8 = RU[i8].float().to(DEV); Wv8 = Ru8 * NORM[i8].to(DEV)[:, None]; Uw = Ru8[wl]; spec = torch.zeros(len(wl), len(wl))
for a_, r_ in enumerate(wl):
    P = top_positions(i8, r_); act = ACT[i8][P].float().to(DEV); g = gvec(i8, r_); mc = (act * g[None]).mean(0); E = mc.topk(NTOP_E).indices; xE = act[:, E] @ Wv8[E]; spec[a_] = (unitr(xE) @ Uw.T).mean(0).cpu()
diag = spec.diag(); off = spec - torch.diag(diag); offm = off.sum(1) / (len(wl) - 1); res["specificity"] = dict(n_words=len(wl), diagonal_median=float(diag.median()), off_diagonal_median=float(offm.median()), ratio_median=float((diag / offm.abs().clamp_min(1e-6)).median()), share_diagonal_is_largest=float((spec.argmax(1) == torch.arange(len(wl))).float().mean()), off_diagonal_max_median=float(off.max(1).values.median()))
sp = res["specificity"]; log(f"{name} specificity at step 8000 ({len(wl)} words): a word's top-500 ensemble aggregate write has cosine {f3(sp['diagonal_median'])} with its own row, {f3(sp['off_diagonal_median'])} with other words on average, {f3(sp['off_diagonal_max_median'])} with the most similar other word; the own row is the largest entry for {f2(sp['share_diagonal_is_largest'])} of words")
del Ru8, Wv8, Uw; torch.cuda.empty_cache()
real_e, real_f, null_f, act_f = [], [], [], []; rs = torch.Generator().manual_seed(3)
for r_ in wl:
    if not WORDS[i16][r_]: continue
    P8 = top_positions(i8, r_); P16 = top_positions(i16, r_); q_ = int(torch.randint(m, (1,), generator=rs))
    a8, a16, b16 = ACT[i8][P8].float().to(DEV), ACT[i16][P16].float().to(DEV), ACT[i16][P8].float().to(DEV); g8, g16 = gvec(i8, r_), gvec(i16, r_); h8, h16 = gvec(i8, q_), gvec(i16, q_)
    real_e.append(float(cosr((a8 * g8[None]).mean(0), (a16 * g16[None]).mean(0)))); real_f.append(float(cosr((a8 * g8[None]).mean(0), (b16 * g16[None]).mean(0)))); null_f.append(float(cosr((a8 * h8[None]).mean(0), (b16 * h16[None]).mean(0)))); act_f.append(float(cosr(a8.mean(0), b16.mean(0))))
res["persistence_null_8000_16000"] = dict(n=len(real_e), real_each_checkpoints_positions=med(real_e), real_fixed_positions=med(real_f), null_random_row_alignment_fixed_positions=med(null_f), activations_alone_fixed_positions=med(act_f))
pn = res["persistence_null_8000_16000"]; log(f"{name} persistence of a word's writer vector from step 8000 to 16000 ({pn['n']} words): real with each checkpoint's positions {f3(pn['real_each_checkpoints_positions'])}, with the step-8000 positions fixed {f3(pn['real_fixed_positions'])}; the same positions weighted by a random native row's alignment {f3(pn['null_random_row_alignment_fixed_positions'])}; the activations alone {f3(pn['activations_alone_fixed_positions'])}")
E_, M_ = res["entrants"]["by_k"], res["near_misses"]["by_k"]; dE = res["entrants"]["decomposition"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(context_first=g_(E_["-4"]["persist_act"]) >= 0.5 and g_(M_["-4"]["persist_act"]) >= 0.5, rise_is_alignment=(g_(dE["row_rotation"]) + g_(dE["writers_motion"])) > 0.5 * g_(dE["total"]) and g_(dE["row_rotation"]) > g_(dE["writers_motion"]) and g_(dE["activations"]) < 0.25 * g_(dE["total"]),
                     coherence_rises=g_(E_["0"]["coherence"]) >= 2 * g_(E_["-4"]["coherence"]) and g_(E_["0"]["coherence"]) >= 2 * g_(M_["0"]["coherence"]), persistence_mostly_context=g_(pn["null_random_row_alignment_fixed_positions"]) >= 0.7, ensembles_private=g_(sp["diagonal_median"]) >= 3 * abs(g_(sp["off_diagonal_median"])))
summ = f"{name} block {B}: entrants ({res['entrants']['n_events']}) at k=-4/-2/-1/0/+2: S " + "/".join(f2(E_[str(k)]["S"]) for k in (-4, -2, -1, 0, 2)) + ", coherence " + "/".join(f2(E_[str(k)]["coherence"]) for k in (-4, -2, -1, 0, 2)) + ", within-row similarity (contemporaneous / entry alignment / activations) at -4 and 0 " + f"{f2(E_['-4']['within_now'])}/{f2(E_['-4']['within_fixed'])}/{f2(E_['-4']['within_act'])} and {f2(E_['0']['within_now'])}/{f2(E_['0']['within_fixed'])}/{f2(E_['0']['within_act'])}, persistence to entry at -4 (activations / entry-aligned / contemporaneous) {f2(E_['-4']['persist_act'])}/{f2(E_['-4']['persist_fixed'])}/{f2(E_['-4']['persist_now'])}, attention share " + "/".join(f2(E_[str(k)]["attention_share"]) for k in (-4, -2, -1, 0, 2)) + f"; rise of the MLP projection: total {f3(dE['total'])} = activations {f3(dE['activations'])} + row {f3(dE['row_rotation'])} + writers {f3(dE['writers_motion'])} + remainder {f3(dE['remainder'])} | near misses ({res['near_misses']['n_events']}): S " + "/".join(f2(M_[str(k)]["S"]) for k in (-4, -2, -1, 0, 2)) + ", coherence " + "/".join(f2(M_[str(k)]["coherence"]) for k in (-4, -2, -1, 0, 2)) + f", persistence at -4 {f2(M_['-4']['persist_act'])}/{f2(M_['-4']['persist_fixed'])}/{f2(M_['-4']['persist_now'])}, rise total {f3(res['near_misses']['decomposition']['total'])} = {f3(res['near_misses']['decomposition']['activations'])} + {f3(res['near_misses']['decomposition']['row_rotation'])} + {f3(res['near_misses']['decomposition']['writers_motion'])} + {f3(res['near_misses']['decomposition']['remainder'])} | coherence groups at 8000: " + ", ".join(f"{k} {f2(v)}" for k, v in grp.items()) + f" | specificity: own row {f3(sp['diagonal_median'])}, other words {f3(sp['off_diagonal_median'])} (most similar {f3(sp['off_diagonal_max_median'])}), own largest for {f2(sp['share_diagonal_is_largest'])} | persistence 8000-16000: real {f3(pn['real_each_checkpoints_positions'])} / fixed positions {f3(pn['real_fixed_positions'])}, random-row null {f3(pn['null_random_row_alignment_fixed_positions'])}, activations alone {f3(pn['activations_alone_fixed_positions'])} | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e546_ensemble_precedence_{name}", res, summ)
