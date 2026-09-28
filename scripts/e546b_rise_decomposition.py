"""e546b: the rise of an entrant's projection, decomposed additively. e546 split the rise of the MLP part of the projection from four intervals before entry to entry into the activations' change, the row's rotation, the writers' motion and a remainder, but reported medians of the parts, which do not add. This reports, per event, each part's share of the rise (its median over the row's top-4 positions divided by the total there), then medians and means over events, for the entrants and the S-matched near misses, and the same split for the change of the whole projection (the attention part itself split into the attention outputs' own motion, the row's rotation seen through them, and a cross term). Also the shares at one interval before entry (k=-1 to 0, the crossing) separately from the approach (k=-4 to -1).
Pre-registered (honest guesses): the row's rotation is the largest positive share of the rise in the approach and in the crossing (0.5); the activations' change is negative in the median (the contexts drift away while the row turns toward them) (0.4).
Arguments: name."""
"""(built from e546: e545 found each word's extremes written by a persistent, word-specific
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
words = [WORDS[i] for i in range(T)]; S_ = [SS[i] for i in range(T)]; ex, en = events(words); men = match(en, words, S_, lambda W, t: ~W[t - 1] & ~W[t] & ~W[t + 1]); log(f"{name}: {len(en)} clean entrants, {len(men)} matched near misses")
GV = {}
def gv(i, r_, j=None):
    key = (i, r_, i if j is None else j)
    if key not in GV: GV[key] = gvec(i, r_, j).cpu()
    return GV[key].to(DEV)
def split(evs, k0, k1, tag):
    sh = {q: [] for q in ("total", "activations", "row_rotation", "writers_motion", "remainder", "attention", "attention_motion", "row_rotation_through_attention", "attention_cross", "row_rotation_total", "embedding_and_bias")}; ab = {q: [] for q in sh}
    for r_, t in evs:
        i0, i1 = t + k0, t + k1
        if not (0 <= i0 < T and 0 <= i1 < T): continue
        P = top_positions(t, r_); a0 = ACT[i0][P].float().to(DEV); a1 = ACT[i1][P].float().to(DEV); g0 = gv(i0, r_); g1 = gv(i1, r_); grow = gv(i0, r_, j=i1)
        m0 = (a0 * g0[None]).sum(1); m1 = (a1 * g1[None]).sum(1); d_act = ((a1 - a0) * g0[None]).sum(1); d_row = (a0 * (grow - g0)[None]).sum(1); d_wr = (a0 * (g1 - grow)[None]).sum(1); rem = (m1 - m0) - d_act - d_row - d_wr
        u0 = RU[i0][r_].float().to(DEV); u1 = RU[i1][r_].float().to(DEV); p0 = XS[i0][P].to(DEV) @ u0; p1 = XS[i1][P].to(DEV) @ u1; A0 = ATT[i0][P].to(DEV); A1 = ATT[i1][P].to(DEV); d_att = A1 @ u1 - A0 @ u0; d_att_motion = (A1 - A0) @ u0; d_att_row = A0 @ (u1 - u0); d_att_cross = d_att - d_att_motion - d_att_row; d_emb = (p1 - p0) - (m1 - m0) - d_att
        tot = (p1 - p0); tm = float(tot.median())
        if abs(tm) < 1e-6: continue
        for q, v in (("total", tot), ("activations", d_act), ("row_rotation", d_row), ("writers_motion", d_wr), ("remainder", rem), ("attention", d_att), ("attention_motion", d_att_motion), ("row_rotation_through_attention", d_att_row), ("attention_cross", d_att_cross), ("row_rotation_total", d_row + d_att_row), ("embedding_and_bias", d_emb)): ab[q].append(float(v.median())); sh[q].append(float(v.median()) / tm)
    out = {q: dict(median_share=med(v), mean_share=mean(v), median_value=med(ab[q]), mean_value=mean(ab[q])) for q, v in sh.items()} | {"n": len(sh["total"])}
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"; f2 = lambda x: "n/a" if x is None else f"{x:.2f}"
    log(f"{name} {tag}, k={k0} to {k1} ({out['n']} events): change of the projection, median {f3(out['total']['median_value'])} (mean {f3(out['total']['mean_value'])}); shares of it, median (mean): activations {f2(out['activations']['median_share'])} ({f2(out['activations']['mean_share'])}), row's rotation {f2(out['row_rotation']['median_share'])} ({f2(out['row_rotation']['mean_share'])}), writers' motion {f2(out['writers_motion']['median_share'])} ({f2(out['writers_motion']['mean_share'])}), remainder {f2(out['remainder']['median_share'])} ({f2(out['remainder']['mean_share'])}), attention in all {f2(out['attention']['median_share'])} ({f2(out['attention']['mean_share'])}), of which the attention outputs' own motion {f2(out['attention_motion']['median_share'])} ({f2(out['attention_motion']['mean_share'])}), the row's rotation seen through them {f2(out['row_rotation_through_attention']['median_share'])} ({f2(out['row_rotation_through_attention']['mean_share'])}) and the cross term {f2(out['attention_cross']['median_share'])}; the row's rotation in all {f2(out['row_rotation_total']['median_share'])} ({f2(out['row_rotation_total']['mean_share'])}); embedding and biases {f2(out['embedding_and_bias']['median_share'])} ({f2(out['embedding_and_bias']['mean_share'])})")
    return out
res = dict(model=name, block=B, n_entrants=len(en), n_near_misses=len(men), entrants={}, near_misses={})
for tag, evs, store in (("entrants", en, res["entrants"]), ("near misses", men, res["near_misses"])):
    for k0, k1, lab in ((-4, 0, "approach_and_crossing"), (-4, -1, "approach"), (-1, 0, "crossing")): store[lab] = split(evs, k0, k1, tag)
E_ = res["entrants"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(row_largest_in_approach=g_(E_["approach"]["row_rotation_total"]["median_share"]) >= max(g_(E_["approach"][q]["median_share"]) for q in ("activations", "writers_motion", "attention_motion", "embedding_and_bias")), row_largest_in_crossing=g_(E_["crossing"]["row_rotation_total"]["median_share"]) >= max(g_(E_["crossing"][q]["median_share"]) for q in ("activations", "writers_motion", "attention_motion", "embedding_and_bias")), activations_negative=g_(E_["approach_and_crossing"]["activations"]["median_share"]) < 0)
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name} block {B}: " + " | ".join(f"{grp} {lab}: median shares activations {f2(o['activations']['median_share'])}, row through the MLP writes {f2(o['row_rotation']['median_share'])}, writers' motion {f2(o['writers_motion']['median_share'])}, remainder {f2(o['remainder']['median_share'])}, attention's own motion {f2(o['attention_motion']['median_share'])}, row through attention {f2(o['row_rotation_through_attention']['median_share'])}, row in all {f2(o['row_rotation_total']['median_share'])}, embedding {f2(o['embedding_and_bias']['median_share'])} (n {o['n']})" for grp, d in (("entrants", res["entrants"]), ("near misses", res["near_misses"])) for lab, o in d.items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e546b_rise_decomposition_{name}", res, summ)
