"""e547b: what an entrant's neuron does at its future contexts, and the approach measured against a selection control.
e547's alignment-selected channel proved generic (its stability, the row's approach to it and the displacement's cosine
with it were the same for entrants, near misses and random non-words), and its selectivity statistic broke on
activations whose mean at random positions is near zero or negative. This reruns the two parts that can be made
clean, with the three groups as each other's controls: the neuron's own activation at the row's top-4 positions at the
event, as a z-score against its activation over all positions at the same checkpoint, at k = -8..+2; and the row's
approach to its future contexts measured against a class direction defined at k=-8 (the mean centred state at those
positions at k=-8, which the row at k=-8 has not been selected on), the row's cosine with it at k = -8..+2, and the
displacement's cosine with it from k=-8 to 0. Random non-words calibrate what selection alone gives.
Pre-registered (honest guesses):
- the neuron's selectivity precedes entry: the z-score of its activation at its future contexts is 1.0 or more at
  k=-8 for entrants, rising to 2.0 or more at entry, and under 0.5 for random non-words throughout (0.5);
- the approach is not selection: the entrants' displacement from k=-8 has a cosine with the k=-8 class direction at
  least 0.1 above the random non-words' (0.5).
Arguments: name."""
"""(e547: Reading sessions 72-95 together: a word's extremes are written by a private coalition of thousands
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
KR = list(range(-8, 3))
def prof(evs, tag):
    acc = {k: {q: [] for q in ("z_activation", "rank_activation", "row_cos_class_m8", "row_cos_class_now", "S")} for k in KR}; disp = {q: [] for q in ("cos_class_m8_from_m8", "cos_class_m8_from_m4", "cos_class_now_from_m4")}
    for r_, t in evs:
        P = top_positions(t, r_).cpu(); Pd = P.to(DEV)
        if t - 8 >= 0: Xm8 = XS[t - 8].to(DEV); cls8 = (Xm8[Pd] - Xm8.mean(0)).mean(0)
        else: cls8 = None
        Xt = XS[t].to(DEV); clsn = (Xt[Pd] - Xt.mean(0)).mean(0)
        for k in KR:
            i = t + k
            if not (0 <= i < T): continue
            col = ACT[i][:, r_].float().to(DEV); z = (col[Pd].mean() - col.mean()) / col.std().clamp_min(1e-6); rk = (col[None, :] > col[Pd][:, None]).float().mean(); u_r = RU[i][r_].float().to(DEV)
            a = acc[k]; a["z_activation"].append(float(z)); a["rank_activation"].append(float(rk)); a["row_cos_class_now"].append(float(cosr(u_r, clsn))); a["S"].append(float(SS[i][r_]))
            if cls8 is not None: a["row_cos_class_m8"].append(float(cosr(u_r, cls8)))
        if cls8 is not None:
            d8 = (RU[t][r_].float() * NORM[t][r_]).to(DEV) - (RU[t - 8][r_].float() * NORM[t - 8][r_]).to(DEV); disp["cos_class_m8_from_m8"].append(float(cosr(d8, cls8)))
        if t - 4 >= 0:
            d4 = (RU[t][r_].float() * NORM[t][r_]).to(DEV) - (RU[t - 4][r_].float() * NORM[t - 4][r_]).to(DEV); disp["cos_class_now_from_m4"].append(float(cosr(d4, clsn)))
            if cls8 is not None: disp["cos_class_m8_from_m4"].append(float(cosr(d4, cls8)))
    out = {str(k): {q: med(v) for q, v in a.items()} | {"n": len(a["S"])} for k, a in acc.items()}; dd = {q: med(v) for q, v in disp.items()} | {"n8": len(disp["cos_class_m8_from_m8"]), "n4": len(disp["cos_class_now_from_m4"])}
    f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
    log(f"{name} {tag} ({len(evs)}): k=-8..+2 z-score of the neuron's activation at its future contexts " + "/".join(f2(out[str(k)]["z_activation"]) for k in KR) + "; share of positions with a higher activation " + "/".join(f2(out[str(k)]["rank_activation"]) for k in KR) + "; row's cosine with the class direction defined at k=-8 " + "/".join(f2(out[str(k)]["row_cos_class_m8"]) for k in KR) + ", with the class direction at the event " + "/".join(f2(out[str(k)]["row_cos_class_now"]) for k in KR) + "; S " + "/".join(f2(out[str(k)]["S"]) for k in KR) + f"; displacement's cosine with the k=-8 class direction, from k=-8 ({dd['n8']}) {f3(dd['cos_class_m8_from_m8'])}, from k=-4 {f3(dd['cos_class_m8_from_m4'])}; with the event's class direction from k=-4 ({dd['n4']}) {f3(dd['cos_class_now_from_m4'])}")
    return dict(by_k=out, displacement=dd, n_events=len(evs))
res = dict(model=name, block=B, entrants=prof(en, "entrants"), near_misses=prof(men, "near misses"), random_nonwords=prof(rnd, "random non-words"))
E, M_, Rn = res["entrants"], res["near_misses"], res["random_nonwords"]; g_ = lambda x: -9 if x is None else x; bk = lambda o, k, q: g_(o["by_k"][str(k)][q])
res["checks"] = dict(selectivity_precedes=bk(E, -8, "z_activation") >= 1.0 and bk(E, 0, "z_activation") >= 2.0 and all(bk(Rn, k, "z_activation") < 0.5 for k in KR), approach_beyond_selection=g_(E["displacement"]["cos_class_m8_from_m8"]) >= g_(Rn["displacement"]["cos_class_m8_from_m8"]) + 0.1)
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
summ = f"{name} block {B}: " + " | ".join(f"{tag}: z-score at k=-8/-4/-1/0/+2 {f2(bk(o, -8, 'z_activation'))}/{f2(bk(o, -4, 'z_activation'))}/{f2(bk(o, -1, 'z_activation'))}/{f2(bk(o, 0, 'z_activation'))}/{f2(bk(o, 2, 'z_activation'))}, row's cosine with the k=-8 class direction at -8/-4/0 {f2(bk(o, -8, 'row_cos_class_m8'))}/{f2(bk(o, -4, 'row_cos_class_m8'))}/{f2(bk(o, 0, 'row_cos_class_m8'))}, displacement cosine with it from -8 {f3(o['displacement']['cos_class_m8_from_m8'])}" for tag, o in (("entrants", E), ("near misses", M_), ("random non-words", Rn))) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e547b_selectivity_{name}", res, summ)
