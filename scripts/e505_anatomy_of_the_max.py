"""e505: the anatomy of the maximum. e503 tested THEORY 3m's two halves quantitatively and both missed: at the end of
GPT-2's training the writers' maximum projection is 1.6-1.9 times the largest write's own coefficient (prominence),
and the non-writers' maximum is 1.45-1.96 times the Gumbel level computed from their own second moment along the
state, while the second-order alignment of the state cloud with the atoms accounts for a factor of only 1.18-1.24.
On Pythia the non-writers' maximum sits at the Gaussian level at step 256 (1.07-1.11) and is 1.4-1.6 times above it
by step 512: the accent is a tail of the projection distribution, not its variance. So where do the excesses come
from? Every state is exactly the sum of its writes, so the projection of a state onto any atom is the Gram matrix
applied to its full ledger: own coefficient, plus the other writes times their cosines with the atom, plus the parts
of the state that the MLP ledger does not hold (attention, biases, the centring). This run decomposes the winning
projections into those parts, state by state.
Per typical state x: the 64 largest MLP writes (|activation x row norm|) plus the token embedding (and the position
embedding where the model has one) are the writers; W64 is their write-sum, Wall the sum of every MLP row's write
plus the embeddings, rest = x - mean - Wall (attention, biases, the centring). For the overall argmax atom, the top
writer atom and the top non-writer atom, the projection <x - mean, a> is split into: own coefficient (zero for a
non-writer), the cross term from the other top-64 writes (<W64, a> less own), the cross term from the remaining MLP
rows (<Wall - W64, a>), and the rest's part; each as a share of the projection. Also the type of the top non-writer
atom and its largest |cosine| with any of the state's writers (is it a near-duplicate of a writer?), and two
predictions that use the ledger alone: is the argmax of |<W64, a>| over atoms a writer, and does it coincide with the
state's actual argmax atom; the same for Wall.
Setup: a model at a checkpoint (arguments), blocks NB/4, NB/2, 3NB/4; 8 x 256 evaluation tokens, typical positions.
Pre-registered (honest guesses):
- for observed writer wins at the end of training, the own coefficient is under 0.6 of the winning projection and the
  other top-64 writes' cross term is the largest of the remaining parts at the middle block (0.5);
- the top non-writer atom is a near-duplicate of a writer: its largest |cosine| with a writer is above 0.5 for more
  than half of the states at the middle block, and its projection's largest part is the top-64 cross term (0.5);
- the argmax of the write-sum W64 is a writer for the same share of states as observed within 0.1 (0.5).
Arguments: name [revision]."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; NW = 64
blocks = sorted({NB // 4, NB // 2, (3 * NB) // 4}); LB = max(blocks); ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
acts = {b: [] for b in range(LB + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)]
try: S_ = block_states(model, arch, ids, blocks, chunk=4)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(LB + 1)}; del acts; rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}
tokens = ids[:, 1:].reshape(-1); positions = torch.arange(1, T_, device=DEV).repeat(B_); has_pos = len(arch.emb) > 1
TYPES = {T_TOK: "token", T_POS: "position", T_MLP: "mlp", T_ATT: "head", T_BIAS: "bias"}
def spear(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-9))
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); U = unitr(Xc); xn = Xc.norm(dim=-1); N = U.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); anorm = A.float().norm(dim=-1); Au = unitr(A); del A; m = Au.shape[0]
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV)
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    Cled = torch.cat([A_all[bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); top = Cled.abs().topk(NW, dim=1); writers = gidx.reshape(-1)[top.indices]; wcoef = Cled.gather(1, top.indices)
    mt = typ == T_TOK; tok_lut = torch.full((int(idx[mt].max()) + 1,), -1, device=DEV, dtype=torch.long); tok_lut[idx[mt]] = torch.nonzero(mt)[:, 0]; tatom = tok_lut[tokens[keep]]
    writers = torch.cat([writers, tatom[:, None]], 1); wcoef = torch.cat([wcoef, anorm[tatom][:, None]], 1)
    if has_pos:
        mp = typ == T_POS; pos_lut = torch.full((int(idx[mp].max()) + 1,), -1, device=DEV, dtype=torch.long); pos_lut[idx[mp]] = torch.nonzero(mp)[:, 0]; patom = pos_lut[positions[keep]]
        writers = torch.cat([writers, patom[:, None]], 1); wcoef = torch.cat([wcoef, anorm[patom][:, None]], 1)
    assert bool((writers >= 0).all()); nw = writers.shape[1]
    W64 = torch.einsum("nk,nkd->nd", wcoef, Au[writers])                                                       # the top writes' sum, in the atoms' coordinates
    Wall = sum(A_all[bb][keep] @ arch.wdir(bb).float() for bb in range(b + 1)) + arch.emb[0].detach().float()[tokens[keep]] + (arch.emb[1].detach().float()[positions[keep]] if has_pos else 0)
    rest = Xc - Wall
    # projections, the three atoms of interest, and the ledger-only predictions
    Cw_sum, Cn_sum, arg_all, arg_w, arg_n, pred64, predall = [], [], [], [], [], [], []
    for s in range(0, N, 128):
        C = U[s:s + 128] @ Au.T; Ca = C.abs(); wm = torch.zeros_like(C, dtype=torch.bool); wm.scatter_(1, writers[s:s + 128], True)
        arg_all.append(Ca.argmax(1)); arg_w.append(torch.where(wm, Ca, torch.zeros_like(Ca)).argmax(1)); arg_n.append(torch.where(wm, torch.zeros_like(Ca), Ca).argmax(1))
        P64 = (W64[s:s + 128] @ Au.T).abs(); Pall = (Wall[s:s + 128] @ Au.T).abs(); pred64.append(P64.argmax(1)); predall.append(Pall.argmax(1))
    arg_all, arg_w, arg_n, pred64, predall = map(torch.cat, (arg_all, arg_w, arg_n, pred64, predall))
    is_writer = lambda atom: (writers == atom[:, None]).any(1)
    win = is_writer(arg_all)
    def decompose(atom):
        """the projection of the centred state onto the atom, split into own coefficient, top-64 cross, other MLP cross, rest"""
        a = Au[atom]; proj = (Xc * a).sum(1); own = torch.where(writers == atom[:, None], wcoef, torch.zeros_like(wcoef)).sum(1)
        cross64 = (W64 * a).sum(1) - own; crossmlp = ((Wall - W64) * a).sum(1); restp = (rest * a).sum(1); den = proj.abs().clamp_min(1e-9) * torch.sign(proj)
        return dict(projection=proj / xn, own=own / den, cross_top64=cross64 / den, cross_other_mlp=crossmlp / den, rest=restp / den)
    def summarise(dc, mask):
        return {k: float(v[mask].median()) for k, v in dc.items()} | dict(n=int(mask.sum()))
    d_all, d_w, d_n = decompose(arg_all), decompose(arg_w), decompose(arg_n)
    # the top non-writer atom: its type, and its largest |cosine| with any of the state's writers
    nn_cos = (Au[arg_n][:, None, :] * Au[writers]).sum(-1).abs().max(1).values; ntype = typ[arg_n]
    out = dict(n_states=N, n_atoms=m, n_writers=nw, win_rate_observed=float(win.float().mean()),
               argmax_when_writer=summarise(d_all, win), argmax_when_nonwriter=summarise(d_all, ~win), top_writer_atom=summarise(d_w, torch.ones_like(win)), top_nonwriter_atom=summarise(d_n, torch.ones_like(win)),
               nonwriter_top_type={TYPES[t]: float((ntype == t).float().mean()) for t in TYPES}, nonwriter_near_duplicate=dict(median_max_cos_with_a_writer=float(nn_cos.median()), share_above_0_5=float((nn_cos > 0.5).float().mean()), share_above_0_8=float((nn_cos > 0.8).float().mean()), spearman_cos_with_projection=spear(nn_cos, d_n["projection"])),
               prediction_from_ledger=dict(w64_argmax_is_writer=float(is_writer(pred64).float().mean()), w64_argmax_matches=float((pred64 == arg_all).float().mean()), w64_accuracy=float((is_writer(pred64) == win).float().mean()),
                                           wall_argmax_is_writer=float(is_writer(predall).float().mean()), wall_argmax_matches=float((predall == arg_all).float().mean()), wall_accuracy=float((is_writer(predall) == win).float().mean())),
               top_nonwriter_over_top_writer=float((d_n["projection"].abs() / d_w["projection"].abs().clamp_min(1e-9)).median()))
    res["by_block"][b] = out; o = out; aw = o["argmax_when_writer"]; an = o["top_nonwriter_atom"]; nd = o["nonwriter_near_duplicate"]; pr = o["prediction_from_ledger"]
    log(f"{name}{' ' + rev if rev else ''} block {b} ({N} states, {m} atoms, {nw} writers): writer wins {o['win_rate_observed']:.2f} | writer wins' projection ({aw['n']}): own {aw['own']:.2f}, top-64 cross {aw['cross_top64']:.2f}, other MLP cross {aw['cross_other_mlp']:.2f}, rest {aw['rest']:.2f} | top non-writer atom: projection {an['projection']:.3f} (over the top writer's {o['top_nonwriter_over_top_writer']:.2f}), top-64 cross {an['cross_top64']:.2f}, other MLP cross {an['cross_other_mlp']:.2f}, rest {an['rest']:.2f}; type " + ", ".join(f"{k} {v:.2f}" for k, v in o["nonwriter_top_type"].items() if v > 0) + f"; max cos with a writer {nd['median_max_cos_with_a_writer']:.2f} (above 0.5: {nd['share_above_0_5']:.2f}, above 0.8: {nd['share_above_0_8']:.2f}) | from the ledger alone: W64's argmax is a writer {pr['w64_argmax_is_writer']:.2f} (accuracy {pr['w64_accuracy']:.2f}, same atom {pr['w64_argmax_matches']:.2f}), Wall's {pr['wall_argmax_is_writer']:.2f} (accuracy {pr['wall_accuracy']:.2f}, same atom {pr['wall_argmax_matches']:.2f})")
    del Au, Cled, W64, Wall, rest; torch.cuda.empty_cache()
mid = res["by_block"][NB // 2]; Bk = res["by_block"]
res["checks"] = dict(own_under_0_6_and_top64_cross_largest=mid["argmax_when_writer"]["own"] < 0.6 and mid["argmax_when_writer"]["cross_top64"] >= max(mid["argmax_when_writer"]["cross_other_mlp"], mid["argmax_when_writer"]["rest"]),
                     nonwriter_is_near_duplicate=mid["nonwriter_near_duplicate"]["share_above_0_5"] > 0.5 and mid["top_nonwriter_atom"]["cross_top64"] >= max(mid["top_nonwriter_atom"]["cross_other_mlp"], mid["top_nonwriter_atom"]["rest"]),
                     w64_argmax_predicts_win_rate=all(abs(o["prediction_from_ledger"]["w64_argmax_is_writer"] - o["win_rate_observed"]) < 0.1 for o in Bk.values()))
summ = (f"{name}{' ' + rev if rev else ''}: by block " + " | ".join(f"{b}: writer wins {o['win_rate_observed']:.2f}; a writer win's projection is own {o['argmax_when_writer']['own']:.2f} + top-64 cross {o['argmax_when_writer']['cross_top64']:.2f} + other MLP {o['argmax_when_writer']['cross_other_mlp']:.2f} + rest {o['argmax_when_writer']['rest']:.2f}; the top non-writer's is top-64 cross {o['top_nonwriter_atom']['cross_top64']:.2f} + other MLP {o['top_nonwriter_atom']['cross_other_mlp']:.2f} + rest {o['top_nonwriter_atom']['rest']:.2f}, its type " + "/".join(f"{k} {v:.2f}" for k, v in o["nonwriter_top_type"].items() if v >= 0.05) + f", its max cos with a writer {o['nonwriter_near_duplicate']['median_max_cos_with_a_writer']:.2f} (above 0.5: {o['nonwriter_near_duplicate']['share_above_0_5']:.2f}); W64's argmax is a writer {o['prediction_from_ledger']['w64_argmax_is_writer']:.2f} (accuracy {o['prediction_from_ledger']['w64_accuracy']:.2f}), Wall's {o['prediction_from_ledger']['wall_argmax_is_writer']:.2f} (accuracy {o['prediction_from_ledger']['wall_accuracy']:.2f}, same atom {o['prediction_from_ledger']['wall_argmax_matches']:.2f})" for b, o in Bk.items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e505_anatomy_{name}{'_' + rev if rev else ''}", res, summ)
