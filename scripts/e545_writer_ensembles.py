"""e545: who writes a row's extremes? e544 found the native fifth to be the top four projections at each position, and
e544b that the vocabulary is the set of rows the states make extreme somewhere, not where. This asks what makes a row
extreme at a position and whether the same writers do it at its different positions. Three hypotheses: each extreme is
made by whatever writes happen to enter the tail there (the writers vary); the same population of writers makes a
row's extremes wherever they occur (a stable ensemble per row); or a few persistent write programs make the extremes
of many rows (a low-rank ensemble). At three checkpoints (steps 4000, 8000, 16000), block 12, the 8 x 256 evaluation
tokens: the state at every kept position is split into its writes (the MLP neurons of blocks 0-12 as 53,248 writers,
the thirteen attention outputs and the embedding), the top-4 extreme pairs of e544 are taken, and for every extreme
pair the contribution of each MLP writer to the projection (its activation times its write's norm times the cosine of
its write with the row). Recorded: the linearity check; the concentration of an extreme (the effective number of
writers, the top-10 and top-100 shares, the shares of MLP, attention and embedding); the similarity of the writer
vectors of the same row at two of its extreme positions against the same row at an extreme and a typical position, and
against different rows at their extremes; the same for random directions in place of the rows (big writers or
direction-selective writers); the words-by-writers matrix (a word's mean writer vector over its extreme positions),
its effective rank against random position sets of the same sizes, and its persistence across checkpoints.
Pre-registered (honest guesses):
- an extreme is made by many writers: the effective number of MLP writers at least 200 and the top ten's share of the
  absolute contribution at most 0.25 (0.6);
- the same population writes a row's extremes: the within-row similarity exceeds the cross-row similarity by at least
  0.2 in cosine (0.6);
- the ensembles are direction-selective: for random directions the within-direction excess over the cross baseline is
  at most half the native rows' (0.5);
- the ensembles persist: for words with extremes at both steps 8000 and 16000 the cosine of their writer vectors is
  at least 0.5 (0.5);
- the ensembles are shared: the words-by-writers matrix has an effective rank at most a quarter of random position
  sets' (0.4).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
from lr_common import eval_ids
name = sys.argv[1]; B = 12; K = 16; KTOP = 4; STEPS = [4000, 8000, 16000]; NRAND = 64; CDIR = f"/workspace/wdd/cache/e524_{name}"
idsB = eval_ids(name)[:8, :256].to(DEV); keepc = torch.stack([torch.load(f"{CDIR}/step{n}.pt", map_location="cpu")["blocks"][B]["keep"] for n in range(1000, 16001, 1000)]).all(0); N = int(keepc.sum())
class Stop(Exception): pass
def capture(model, arch, fam):
    cap = {"act": {}, "attn": {}, "mlp": {}, "out": None, "emb": None}; layers = arch.layers; attn_mod = lambda bb: layers[bb].attention if fam == "neox" else (layers[bb].attn if fam == "gpt2" else layers[bb].self_attn); emb_mod = model.gpt_neox.embed_in if fam == "neox" else (model.transformer.wte if fam == "gpt2" else model.model.embed_tokens)
    def h_attn(bb):
        def f(m, i, o): cap["attn"][bb] = (o[0] if isinstance(o, tuple) else o).detach().float()
        return f
    def h_mlp(bb):
        def f(m, i, o): cap["mlp"][bb] = (o[0] if isinstance(o, tuple) else o).detach().float()
        return f
    def h_act(bb):
        def f(m, a): cap["act"][bb] = a[0].detach().float(); return None
        return f
    def h_out(m, i, o): cap["out"] = (o[0] if isinstance(o, tuple) else o).detach().float(); raise Stop
    def h_emb(m, i, o): cap["emb"] = o.detach().float()
    hs = [attn_mod(bb).register_forward_hook(h_attn(bb)) for bb in range(B + 1)] + [layers[bb].mlp.register_forward_hook(h_mlp(bb)) for bb in range(B + 1)] + [arch.mlp_lin(bb).register_forward_pre_hook(h_act(bb)) for bb in range(B + 1)] + [emb_mod.register_forward_hook(h_emb), layers[B].register_forward_hook(h_out)]
    try: model(idsB)
    except Stop: pass
    finally: [h.remove() for h in hs]
    return cap
def flat(t): return t.reshape(8, 256, -1)[:, 1:].reshape(2040, -1)[keepc]
def wmedcos(Xa, Xb):
    return float(((Xa * Xb).sum(1) / (Xa.norm(dim=1) * Xb.norm(dim=1)).clamp_min(1e-9)).median())
def eff_rank(Mx):
    sv = torch.linalg.svdvals(Mx.float()); p = sv ** 2; return float(p.sum() ** 2 / (p ** 2).sum())
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"; res = dict(model=name, block=B, steps=STEPS, by_step={}); MW = {}; gr = torch.Generator(device=DEV).manual_seed(0)
for n in STEPS:
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF
    for p_ in model.parameters(): p_.requires_grad_(False)
    cap = capture(model, arch, fam); Wv = torch.cat([arch.wdir(bb).float() for bb in range(B + 1)]); norms = Wv.norm(dim=1); Ru = unitr(Wv); m = Ru.shape[0]
    X = flat(cap["out"]); ACT = torch.cat([flat(cap["act"][bb]) for bb in range(B + 1)], 1); ATT = torch.stack([flat(cap["attn"][bb]) for bb in range(B + 1)], 1); EMB = flat(cap["emb"]); MLPo = torch.stack([flat(cap["mlp"][bb]) for bb in range(B + 1)], 1)
    rec = EMB + ATT.sum(1) + MLPo.sum(1); lin_state = wmedcos(rec, X); mlp_rec = torch.stack([ACT[:, bb * DFF:(bb + 1) * DFF] @ Wv[bb * DFF:(bb + 1) * DFF] for bb in range(B + 1)], 1); lin_mlp = float(((mlp_rec * MLPo).sum(2) / (mlp_rec.norm(dim=2) * MLPo.norm(dim=2)).clamp_min(1e-9)).median())
    del model, cap, mlp_rec; torch.cuda.empty_cache()
    U = unitr(X - X.mean(0)); st = stats(U, Ru, K); ratio = st["ratio"].to(DEV).float(); words = wordset(st["usage"]).to(DEV); del st
    top = ratio.topk(KTOP, dim=1).indices; pairs = [(int(p_), int(r_)) for p_ in range(N) for r_ in top[p_].tolist()]; byrow = {}
    for p_, r_ in pairs: byrow.setdefault(r_, []).append(p_)
    multi = [r_ for r_, P in byrow.items() if len(P) >= 2]; nw_multi = int(sum(bool(words[r_]) for r_ in multi))
    def contrib(P, direction, gvec):
        """MLP writer contributions at positions P to the projection on a unit direction: activation times the write's norm times the cosine of the write with the direction."""
        return ACT[P] * gvec[None, :]
    G = Ru @ Ru.T if m <= 8192 else None
    def gvec_of(r_): return norms * (Ru @ Ru[r_])
    # concentration and linearity of the projection, over all extreme pairs (sampled)
    conc = {k: [] for k in ("eff", "top10", "top100", "top1000_signed", "share_mlp", "share_attn", "share_emb", "n_pos_over_0")}; lin = []
    samp = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(1))[:2000].tolist()
    for j in samp:
        p_, r_ = pairs[j]; c = ACT[p_] * gvec_of(r_); tot_mlp = float(c.sum()); a_att = float((ATT[p_] @ Ru[r_]).sum()); a_emb = float(EMB[p_] @ Ru[r_]); tot = tot_mlp + a_att + a_emb; proj = float(X[p_] @ Ru[r_]); lin.append(abs(tot - proj) / max(abs(proj), 1e-6))
        ca = c.abs(); conc["eff"].append(float(ca.sum() ** 2 / (c ** 2).sum())); srt = ca.sort(descending=True).values; conc["top10"].append(float(srt[:10].sum() / ca.sum())); conc["top100"].append(float(srt[:100].sum() / ca.sum())); conc["top1000_signed"].append(float(c.sort(descending=True).values[:1000].sum() / max(tot_mlp, 1e-6))); conc["share_mlp"].append(tot_mlp / tot if tot else None); conc["share_attn"].append(a_att / tot if tot else None); conc["share_emb"].append(a_emb / tot if tot else None); conc["n_pos_over_0"].append(float((c > 0).sum()))
    conc = {k: med([v for v in vals if v is not None]) for k, vals in conc.items()}
    # within-row, extreme-vs-typical and cross-row similarities of the writer vectors
    within, typical, act_within, act_typical = [], [], [], []; rs = torch.Generator(device=DEV).manual_seed(2)
    for r_ in multi:
        P = byrow[r_]; gv = gvec_of(r_); C = ACT[P] * gv[None]; Cn = C / C.norm(dim=1, keepdim=True).clamp_min(1e-9); S_ = Cn @ Cn.T; iu = torch.triu_indices(len(P), len(P), 1, device=DEV); within.append(float(S_[iu[0], iu[1]].mean()))
        q_ = torch.randint(N, (len(P),), device=DEV, generator=rs); Ct = ACT[q_] * gv[None]; Ctn = Ct / Ct.norm(dim=1, keepdim=True).clamp_min(1e-9); typical.append(float((Cn * Ctn).sum(1).mean()))
        An = ACT[P] / ACT[P].norm(dim=1, keepdim=True).clamp_min(1e-9); Sa = An @ An.T; act_within.append(float(Sa[iu[0], iu[1]].mean())); Atn = ACT[q_] / ACT[q_].norm(dim=1, keepdim=True).clamp_min(1e-9); act_typical.append(float((An * Atn).sum(1).mean()))
    cross = []; act_cross = []; rl = list(byrow.keys()); cs = torch.Generator().manual_seed(3)
    for _ in range(2000):
        i1, i2 = torch.randint(len(rl), (2,), generator=cs).tolist()
        if i1 == i2: continue
        r1, r2 = rl[i1], rl[i2]; p1 = byrow[r1][0]; p2 = byrow[r2][0]; c1 = ACT[p1] * gvec_of(r1); c2 = ACT[p2] * gvec_of(r2); cross.append(float(c1 @ c2 / (c1.norm() * c2.norm()).clamp_min(1e-9))); act_cross.append(float(ACT[p1] @ ACT[p2] / (ACT[p1].norm() * ACT[p2].norm()).clamp_min(1e-9)))
    sim = dict(n_rows_with_extremes=len(byrow), n_rows_multi=len(multi), n_words_multi=nw_multi, within_row=med(within), extreme_vs_typical=med(typical), cross_row=med(cross), activation_within=med(act_within), activation_typical=med(act_typical), activation_cross=med(act_cross))
    # random directions
    Q = unitr(torch.randn(NRAND, D, device=DEV, generator=gr)); PQ = (U @ Q.T).abs(); topq = PQ.topk(8, dim=0).indices; rwithin, rtyp = [], []
    for j in range(NRAND):
        gv = norms * (Ru @ Q[j]); P = topq[:, j]; C = ACT[P] * gv[None]; Cn = C / C.norm(dim=1, keepdim=True).clamp_min(1e-9); S_ = Cn @ Cn.T; iu = torch.triu_indices(8, 8, 1, device=DEV); rwithin.append(float(S_[iu[0], iu[1]].mean()))
        q_ = torch.randint(N, (8,), device=DEV, generator=rs); Ct = ACT[q_] * gv[None]; Ctn = Ct / Ct.norm(dim=1, keepdim=True).clamp_min(1e-9); rtyp.append(float((Cn * Ctn).sum(1).mean()))
    rcross = []
    for _ in range(1000):
        j1, j2 = torch.randint(NRAND, (2,), generator=cs).tolist()
        if j1 == j2: continue
        c1 = ACT[topq[0, j1]] * (norms * (Ru @ Q[j1])); c2 = ACT[topq[0, j2]] * (norms * (Ru @ Q[j2])); rcross.append(float(c1 @ c2 / (c1.norm() * c2.norm()).clamp_min(1e-9)))
    rand = dict(within_direction=med(rwithin), extreme_vs_typical=med(rtyp), cross_direction=med(rcross))
    # words-by-writers matrix and its effective rank, against random position sets
    wl = [r_ for r_ in byrow if bool(words[r_])]; Mrows = torch.stack([(ACT[byrow[r_]] * gvec_of(r_)[None]).mean(0) for r_ in wl]); Arows = torch.stack([ACT[byrow[r_]].mean(0) for r_ in wl]); Rrows = torch.stack([ACT[torch.randint(N, (len(byrow[r_]),), device=DEV, generator=rs)].mean(0) for r_ in wl])
    rank = dict(n_words=len(wl), words_by_writers=eff_rank(Mrows), activation_means=eff_rank(Arows), random_position_sets=eff_rank(Rrows)); MW[n] = {r_: Mrows[i].cpu() for i, r_ in enumerate(wl)}
    res["by_step"][n] = dict(linearity_state=lin_state, linearity_mlp=lin_mlp, linearity_projection_relative_error=med(lin), concentration=conc, similarity=sim, random_directions=rand, rank=rank)
    log(f"{name} step{n}: state reconstructed from the writes at cosine {f3(lin_state)}, MLP outputs from activations and write rows at {f3(lin_mlp)}, projection from the contributions with relative error {f3(res['by_step'][n]['linearity_projection_relative_error'])}; {len(pairs)} extreme pairs on {len(byrow)} rows, {len(multi)} rows with two or more extreme positions ({nw_multi} words); concentration: effective writers {conc['eff']:.0f}, top-10 share {f2(conc['top10'])}, top-100 {f2(conc['top100'])}, the top 1000 signed over the MLP total {f2(conc['top1000_signed'])}, shares MLP/attention/embedding {f2(conc['share_mlp'])}/{f2(conc['share_attn'])}/{f2(conc['share_emb'])}, writers with positive contribution {conc['n_pos_over_0']:.0f}; writer-vector similarity within a row's extremes {f3(sim['within_row'])}, extreme against typical {f3(sim['extreme_vs_typical'])}, across rows {f3(sim['cross_row'])} (activation vectors alone {f3(sim['activation_within'])} / {f3(sim['activation_typical'])} / {f3(sim['activation_cross'])}); random directions: within {f3(rand['within_direction'])}, extreme against typical {f3(rand['extreme_vs_typical'])}, across {f3(rand['cross_direction'])}; effective rank of the words-by-writers matrix {rank['words_by_writers']:.1f}, of the activation means {rank['activation_means']:.1f}, of random position sets {rank['random_position_sets']:.1f} ({len(wl)} words)")
    del ACT, ATT, EMB, MLPo, X, U, ratio, Wv, Ru; torch.cuda.empty_cache()
# persistence of the writer vectors across checkpoints
pers = {}
for n1, n2 in ((4000, 8000), (8000, 16000), (4000, 16000)):
    common = [r_ for r_ in MW[n1] if r_ in MW[n2]]; same = [float(MW[n1][r_] @ MW[n2][r_] / (MW[n1][r_].norm() * MW[n2][r_].norm()).clamp_min(1e-9)) for r_ in common]
    other = []; ks = list(MW[n2].keys()); gg = torch.Generator().manual_seed(4)
    for r_ in common:
        r2 = ks[int(torch.randint(len(ks), (1,), generator=gg))]
        if r2 != r_: other.append(float(MW[n1][r_] @ MW[n2][r2] / (MW[n1][r_].norm() * MW[n2][r2].norm()).clamp_min(1e-9)))
    pers[f"{n1}-{n2}"] = dict(n_common=len(common), same_word=med(same), different_words=med(other))
res["persistence"] = pers; log(f"{name} persistence of a word's writer vector across checkpoints (same word / different words, common words): " + "; ".join(f"{k}: {f3(v['same_word'])} / {f3(v['different_words'])} ({v['n_common']})" for k, v in pers.items()))
b8 = res["by_step"][8000]; g_ = lambda x: -9 if x is None else x; nat_ex = g_(b8["similarity"]["within_row"]) - g_(b8["similarity"]["cross_row"]); rnd_ex = g_(b8["random_directions"]["within_direction"]) - g_(b8["random_directions"]["cross_direction"])
res["checks"] = dict(many_writers=g_(b8["concentration"]["eff"]) >= 200 and g_(b8["concentration"]["top10"]) <= 0.25, same_population=nat_ex >= 0.2, direction_selective=rnd_ex <= 0.5 * nat_ex, ensembles_persist=g_(pers["8000-16000"]["same_word"]) >= 0.5, ensembles_shared=g_(b8["rank"]["activation_means"]) <= 0.25 * g_(b8["rank"]["random_position_sets"]))
summ = f"{name} block {B}: " + " | ".join(f"step {n}: effective writers {o['concentration']['eff']:.0f}, top-10 share {f2(o['concentration']['top10'])}, MLP/attention/embedding shares {f2(o['concentration']['share_mlp'])}/{f2(o['concentration']['share_attn'])}/{f2(o['concentration']['share_emb'])}; writer similarity within row / extreme-typical / across rows {f3(o['similarity']['within_row'])}/{f3(o['similarity']['extreme_vs_typical'])}/{f3(o['similarity']['cross_row'])} (activations {f3(o['similarity']['activation_within'])}/{f3(o['similarity']['activation_typical'])}/{f3(o['similarity']['activation_cross'])}); random directions {f3(o['random_directions']['within_direction'])}/{f3(o['random_directions']['extreme_vs_typical'])}/{f3(o['random_directions']['cross_direction'])}; effective rank words-by-writers {o['rank']['words_by_writers']:.1f}, activation means {o['rank']['activation_means']:.1f}, random sets {o['rank']['random_position_sets']:.1f}" for n, o in res["by_step"].items()) + " | persistence " + "; ".join(f"{k} {f3(v['same_word'])}/{f3(v['different_words'])}" for k, v in pers.items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e545_writer_ensembles_{name}", res, summ)
