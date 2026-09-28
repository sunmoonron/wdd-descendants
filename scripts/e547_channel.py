"""e547: the channel. Reading sessions 72-95 together: a word's extremes are written by a private coalition of thousands
of writers (e545) that is in place before the word (e546), the row's own write is a minor part of it, and the rise is a
quarter the row's rotation, a quarter the arrival of transported writes and an eighth the local writers' motion (e546b).
The reading tested here: the coalition's aggregate write is a direction of the residual stream at a class of contexts,
a channel written by many neurons; a native word is an MLP neuron whose output row has rotated onto that channel; the
neuron fires at the channel's contexts and gradient descent turns its output row toward the states there (e523), so
the destination of the drift, which nothing at the origin could predict (e530), is the channel. Measured at all
sixteen checkpoints, block 12, for every clean entrant, an S-matched near miss and a random non-word, with the
coalition defined at the event as the 500 writers with the largest mean contribution at the row's top-4 positions,
the row's own neuron excluded, and the channel as the coalition's aggregate write at those positions:
- the channel's stability, the cosine of the channel at k with the channel at the event, k = -8..+2;
- the row's alignment with the channel, the cosine of the row at k with the channel at the event and with the channel
  at k; the row's displacement from k=-4 (and -8) to the event projected on the channel at the event, on the mean
  state at the positions, and on random directions;
- the neuron's selectivity for its contexts, its activation at the positions over its activation at random positions;
- the channel's purity, the aggregate write's cosine with the row at the positions and at random positions, and its
  norm at random positions relative to the positions;
- the coalition's layer profile (the share of its contribution by block) and the attention outputs' share by block;
- at step 8000, coalition families: the distribution of pairwise cosines of the words' writer vectors, the components
  at cosine 0.3, whether family members share extreme positions and tokens, and the cosine of their rows.
Pre-registered (honest guesses):
- the channel is stable before entry: its cosine at k=-4 with the event's channel is 0.7 or more for entrants and at
  least 0.2 lower for near misses (0.5);
- the row aligns with a pre-existing channel: the row's cosine with the event's channel rises by more than 0.1 from
  k=-4 to 0 while the channel's own cosine stays above 0.7 (0.5); co-development, both moving, (0.3); row first (0.2);
- the destination is the channel: the row's displacement from k=-4 has cosine 0.2 or more with the event's channel, at
  least four times its cosine with random directions, and larger than with the mean state (0.5);
- the neuron's selectivity precedes entry: its activation at the positions is at least twice its activation at random
  positions at k=-4 for entrants and under 1.5 times for near misses (0.5);
- the channel is pure: the aggregate write has cosine 0.5 or more with the row at the positions and under 0.2 at random
  positions, where its norm is under half (0.6);
- families are few: under a tenth of word pairs have writer-vector cosine above 0.3 and the components at 0.3 are mostly
  singletons; members of a family share extreme positions and tokens more than random pairs (0.6).
Arguments: name."""
"""(capture and statistics as in e546: e545 found each word's extremes written by a persistent, word-specific
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
# ---------- events ----------
words = [WORDS[i] for i in range(T)]; S_ = [SS[i] for i in range(T)]; ex, en = events(words); men = match(en, words, S_, lambda W, t: ~W[t - 1] & ~W[t] & ~W[t + 1])
gr_ = torch.Generator().manual_seed(5); rnd = []
for r_, t in en:
    cand = torch.nonzero(~words[t - 1] & ~words[t] & ~words[t + 1])[:, 0]; rnd.append((int(cand[int(torch.randint(cand.numel(), (1,), generator=gr_))]), t))
log(f"{name}: {len(en)} clean entrants, {len(men)} matched near misses, {len(rnd)} random non-words")
ids = eval_ids(name)[:8, 1:256].reshape(-1).cpu()[keepc]; NC = 500; GV = {}
def gv(i, r_):
    if (i, r_) not in GV: GV[(i, r_)] = gvec(i, r_).cpu()
    return GV[(i, r_)].to(DEV)
def wv(i, C): return RU[i][C].float().to(DEV) * NORM[i][C].to(DEV)[:, None]
Qrand = unitr(torch.randn(16, D, device=DEV, generator=torch.Generator(device=DEV).manual_seed(6)))
def channel_profile(evs, tag):
    KR = list(range(-8, 3)); acc = {k: {q: [] for q in ("channel_stability", "row_cos_event_channel", "row_cos_current_channel", "selectivity", "S")} for k in KR}
    disp = {q: [] for q in ("cos_channel_4", "cos_mean_state_4", "cos_random_4", "share_along_channel_4", "cos_channel_8", "cos_random_8")}; pur = {q: [] for q in ("cos_at_positions", "cos_at_random", "norm_ratio_random")}; layers = torch.zeros(B + 1); attl = torch.zeros(B + 1); nl = 0
    for r_, t in evs:
        P = top_positions(t, r_).cpu(); Pd = P.to(DEV); act0 = ACT[t][P].float().to(DEV); mc = (act0 * gv(t, r_)[None]).mean(0); mc[r_] = -1e9; C = mc.topk(NC).indices; Cc = C.cpu()
        W0 = wv(t, Cc); d0 = (act0[:, C] @ W0).mean(0); u_d0 = d0 / d0.norm().clamp_min(1e-9); u_r0 = RU[t][r_].float().to(DEV)
        for k in KR:
            i = t + k
            if not (0 <= i < T): continue
            act = ACT[i][P].float().to(DEV); Wi = wv(i, Cc); d = (act[:, C] @ Wi).mean(0); u_r = RU[i][r_].float().to(DEV); q_ = torch.randint(N, (KTOP,), generator=torch.Generator().manual_seed(7 + k)); actq = ACT[i][q_].float().to(DEV)
            a = acc[k]; a["channel_stability"].append(float(cosr(d, d0))); a["row_cos_event_channel"].append(float(cosr(u_r, u_d0))); a["row_cos_current_channel"].append(float(cosr(u_r, d))); a["selectivity"].append(float(act[:, r_].mean() / actq[:, r_].mean().clamp_min(1e-6))); a["S"].append(float(SS[i][r_]))
        for lag, key in ((4, "4"), (8, "8")):
            i = t - lag
            if i < 0: continue
            dr = (RU[t][r_].float() * NORM[t][r_]).to(DEV) - (RU[i][r_].float() * NORM[i][r_]).to(DEV); ms = XS[t][P].to(DEV).mean(0)
            disp["cos_channel_" + key].append(float(cosr(dr, u_d0))); disp["cos_random_" + key].append(float(cosr(dr[None].expand(16, -1), Qrand).abs().mean()))
            if key == "4": disp["cos_mean_state_4"].append(float(cosr(dr, ms))); disp["share_along_channel_4"].append(float((dr @ u_d0) / dr.norm().clamp_min(1e-9)))
        q_ = torch.randint(N, (KTOP,), generator=torch.Generator().manual_seed(9)); actq = ACT[t][q_].float().to(DEV); dq = actq[:, C] @ W0; dp = act0[:, C] @ W0
        pur["cos_at_positions"].append(float(cosr(dp, u_r0[None].expand(KTOP, -1)).median())); pur["cos_at_random"].append(float(cosr(dq, u_r0[None].expand(KTOP, -1)).median())); pur["norm_ratio_random"].append(float(dq.norm(dim=1).median() / dp.norm(dim=1).median().clamp_min(1e-9)))
        contrib = (act0 * gv(t, r_)[None]).mean(0).cpu(); blk = torch.arange(m) // DFF; tot = float(contrib.abs().sum()); layers += torch.bincount(blk, weights=contrib.abs(), minlength=B + 1) / max(tot, 1e-9)
        A0 = ATT[t][P].to(DEV) @ u_r0; nl += 1
    out = {str(k): {q: med(v) for q, v in a.items()} | {"n": len(a["S"])} for k, a in acc.items()}; dd = {q: med(v) for q, v in disp.items()} | {"n4": len(disp["cos_channel_4"]), "n8": len(disp["cos_channel_8"])}; pp = {q: med(v) for q, v in pur.items()}; lp = (layers / max(nl, 1)).tolist()
    f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
    log(f"{name} {tag} ({len(evs)}): k=-8..+2 channel stability " + "/".join(f2(out[str(k)]["channel_stability"]) for k in KR) + "; row's cosine with the event's channel " + "/".join(f2(out[str(k)]["row_cos_event_channel"]) for k in KR) + ", with the current channel " + "/".join(f2(out[str(k)]["row_cos_current_channel"]) for k in KR) + "; neuron's selectivity " + "/".join(f2(out[str(k)]["selectivity"]) for k in KR) + "; S " + "/".join(f2(out[str(k)]["S"]) for k in KR) + f"; displacement from k=-4 ({dd['n4']}): cosine with the channel {f3(dd['cos_channel_4'])}, with the mean state {f3(dd['cos_mean_state_4'])}, with random directions {f3(dd['cos_random_4'])}, share along the channel {f3(dd['share_along_channel_4'])}; from k=-8 ({dd['n8']}): {f3(dd['cos_channel_8'])} against random {f3(dd['cos_random_8'])}; purity: the coalition's aggregate write's cosine with the row at the positions {f3(pp['cos_at_positions'])}, at random positions {f3(pp['cos_at_random'])}, norm at random over at the positions {f3(pp['norm_ratio_random'])}; coalition's contribution by block 0-12: " + " ".join(f"{x:.2f}" for x in lp))
    return dict(by_k=out, displacement=dd, purity=pp, layer_profile=lp, n_events=len(evs))
res = dict(model=name, block=B, entrants=channel_profile(en, "entrants"), near_misses=channel_profile(men, "near misses"), random_nonwords=channel_profile(rnd, "random non-words"))
# ---------- coalition families at step 8000 ----------
i8 = steps.index(8000); wl = torch.nonzero(WORDS[i8])[:, 0].tolist(); Mrows = []; posl = []
for r_ in wl:
    P = top_positions(i8, r_); act = ACT[i8][P].float().to(DEV); Mrows.append((act * gv(i8, r_)[None]).mean(0)); posl.append(set(P.tolist()))
Mm = torch.stack(Mrows); Mn = Mm / Mm.norm(dim=1, keepdim=True).clamp_min(1e-9); Cw = (Mn @ Mn.T).cpu(); iu = torch.triu_indices(len(wl), len(wl), 1); pc = Cw[iu[0], iu[1]]
Rw = RU[i8][wl].float().to(DEV); Cr = (Rw @ Rw.T).cpu(); rc = Cr[iu[0], iu[1]]
adj = (Cw > 0.3); adj.fill_diagonal_(False); comp = -torch.ones(len(wl), dtype=torch.long); nc = 0
for s0 in range(len(wl)):
    if comp[s0] >= 0: continue
    stack = [s0]; comp[s0] = nc
    while stack:
        u = stack.pop()
        for v in torch.nonzero(adj[u])[:, 0].tolist():
            if comp[v] < 0: comp[v] = nc; stack.append(v)
    nc += 1
sizes = torch.bincount(comp); fam_pairs = [(i_, j_) for i_, j_ in zip(iu[0].tolist(), iu[1].tolist()) if Cw[i_, j_] > 0.3]; rp = torch.randperm(len(iu[0]), generator=torch.Generator().manual_seed(8))[:max(len(fam_pairs), 1)].tolist()
shared_pos = lambda pairs: mean([len(posl[i_] & posl[j_]) / KTOP for i_, j_ in pairs]) if pairs else None; shared_tok = lambda pairs: mean([float(len(set(ids[list(posl[i_])].tolist()) & set(ids[list(posl[j_])].tolist())) > 0) for i_, j_ in pairs]) if pairs else None
res["families_step8000"] = dict(n_words=len(wl), pair_cosine_median=float(pc.median()), pair_cosine_p90=float(pc.quantile(0.9)), pair_cosine_p99=float(pc.quantile(0.99)), share_pairs_over_0_3=float((pc > 0.3).float().mean()), share_pairs_over_0_1=float((pc > 0.1).float().mean()), n_components_at_0_3=int(nc), n_singletons=int((sizes == 1).sum()), largest_component=int(sizes.max()), family_pairs=len(fam_pairs), family_pairs_shared_positions=shared_pos(fam_pairs), random_pairs_shared_positions=shared_pos([(int(iu[0][x]), int(iu[1][x])) for x in rp]), family_pairs_shared_token=shared_tok(fam_pairs), random_pairs_shared_token=shared_tok([(int(iu[0][x]), int(iu[1][x])) for x in rp]), family_pairs_row_cosine=mean([float(Cr[i_, j_]) for i_, j_ in fam_pairs]) if fam_pairs else None, all_pairs_row_cosine=float(rc.median()))
fm = res["families_step8000"]; f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
log(f"{name} families at step 8000 ({fm['n_words']} words): pairwise writer-vector cosine median {f3(fm['pair_cosine_median'])}, 90th {f3(fm['pair_cosine_p90'])}, 99th {f3(fm['pair_cosine_p99'])}, share over 0.1 {f3(fm['share_pairs_over_0_1'])}, over 0.3 {f3(fm['share_pairs_over_0_3'])}; components at 0.3: {fm['n_components_at_0_3']} ({fm['n_singletons']} singletons, largest {fm['largest_component']}); family pairs {fm['family_pairs']}: shared extreme positions {f2(fm['family_pairs_shared_positions'])} (random pairs {f2(fm['random_pairs_shared_positions'])}), a shared token {f2(fm['family_pairs_shared_token'])} ({f2(fm['random_pairs_shared_token'])}), rows' cosine {f3(fm['family_pairs_row_cosine'])} (all pairs {f3(fm['all_pairs_row_cosine'])})")
E, M_, Rn = res["entrants"], res["near_misses"], res["random_nonwords"]; g_ = lambda x: -9 if x is None else x; bk = lambda o, k, q: g_(o["by_k"][str(k)][q])
res["checks"] = dict(channel_stable_before=bk(E, -4, "channel_stability") >= 0.7 and bk(M_, -4, "channel_stability") <= bk(E, -4, "channel_stability") - 0.2, row_aligns_with_preexisting=(bk(E, 0, "row_cos_event_channel") - bk(E, -4, "row_cos_event_channel")) > 0.1 and bk(E, -4, "channel_stability") >= 0.7,
                     destination_is_channel=g_(E["displacement"]["cos_channel_4"]) >= 0.2 and g_(E["displacement"]["cos_channel_4"]) >= 4 * g_(E["displacement"]["cos_random_4"]) and g_(E["displacement"]["cos_channel_4"]) > g_(E["displacement"]["cos_mean_state_4"]),
                     selectivity_precedes=bk(E, -4, "selectivity") >= 2 and bk(M_, -4, "selectivity") < 1.5, channel_pure=g_(E["purity"]["cos_at_positions"]) >= 0.5 and g_(E["purity"]["cos_at_random"]) < 0.2 and g_(E["purity"]["norm_ratio_random"]) < 0.5,
                     families_few=g_(fm["share_pairs_over_0_3"]) < 0.1 and fm["n_singletons"] > fm["n_words"] / 2 and (fm["family_pairs_shared_positions"] is None or g_(fm["family_pairs_shared_positions"]) > g_(fm["random_pairs_shared_positions"])))
summ = f"{name} block {B}: " + " | ".join(f"{tag}: channel stability at k=-8/-4/-1 {f2(bk(o, -8, 'channel_stability'))}/{f2(bk(o, -4, 'channel_stability'))}/{f2(bk(o, -1, 'channel_stability'))}, row's cosine with the event's channel at -8/-4/-1/0/+2 {f2(bk(o, -8, 'row_cos_event_channel'))}/{f2(bk(o, -4, 'row_cos_event_channel'))}/{f2(bk(o, -1, 'row_cos_event_channel'))}/{f2(bk(o, 0, 'row_cos_event_channel'))}/{f2(bk(o, 2, 'row_cos_event_channel'))}, selectivity at -4/0 {f2(bk(o, -4, 'selectivity'))}/{f2(bk(o, 0, 'selectivity'))}, displacement cosine with the channel / mean state / random {f3(o['displacement']['cos_channel_4'])}/{f3(o['displacement']['cos_mean_state_4'])}/{f3(o['displacement']['cos_random_4'])}, purity at positions / random {f3(o['purity']['cos_at_positions'])}/{f3(o['purity']['cos_at_random'])}" for tag, o in (("entrants", E), ("near misses", M_), ("random non-words", Rn))) + f" | families: share of pairs over 0.3 {f3(fm['share_pairs_over_0_3'])}, components {fm['n_components_at_0_3']} ({fm['n_singletons']} singletons, largest {fm['largest_component']}) | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e547_channel_{name}", res, summ)
