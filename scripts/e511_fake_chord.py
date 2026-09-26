"""e511: is the chord's sparing specific to the writes that made the state? e509 found the small writes contract the
position's actual chord a third to a half as much as a random direction of the same size, and e510 found the linear
contraction blind to the atoms, so the sparing is a nonlinear, chord-amplitude, pattern-level effect. Two patterns
remain to separate: any chord of native rows with plausible coefficients (a property of the vocabulary's manifold),
or the rows that actually wrote at this position (a property of the write history). Two fake chords decide it: the
position's own coefficients on random rows of the same blocks, and on rows drawn from the block's most-used rows as
native words, each rescaled to the real chord-so-far's norm at every block, with the embeddings kept.
Per position and block b' up to the state's block, as e509: the perturbation (real chord so far, fake chord of
random rows, fake chord of used rows, random direction) is subtracted from the input of block b''s second layer
norm; the small writes' response is the change of the rows outside the perturbation's own rows, times their write
directions, summed over blocks, projected on the perturbation's direction over the real centred chord's norm.
Reported per block, medians over typical positions: the four response gains, the fake gains as a share of the way
from the real chord's to the random direction's, and the response norms over the crowd's norm.
Setup: a model at a checkpoint (arguments), blocks NB/4, NB/2, 3NB/4; 8 x 256 evaluation tokens, typical positions.
Pre-registered (honest guesses):
- the fake chord of random rows is contracted like a random direction (its gain within 25% of the random
  direction's at the middle block) (0.5);
- the fake chord of used rows sits between the real chord and the random direction (0.5);
- the real chord is contracted least of the four at every block of every model (0.6).
Arguments: name [revision]."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; NW = 64; NUSED = 256; K = 16
blocks = sorted({NB // 4, NB // 2, (3 * NB) // 4}); LB = max(blocks); ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape; CH = 4
def ln2(b): return arch.layers[b].ln_2 if fam == "gpt2" else arch.layers[b].post_attention_layernorm
acts = {b: [] for b in range(LB + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)]
try: S_ = block_states(model, arch, ids, blocks, chunk=CH)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(LB + 1)}; del acts; rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}
tokens = ids[:, 1:].reshape(-1); positions = torch.arange(1, T_, device=DEV).repeat(B_); has_pos = len(arch.emb) > 1; Nall = B_ * (T_ - 1)
def acts_with_delta(bp, delta):
    out = []
    for s0 in range(0, B_, CH):
        cap = {}
        def pre_ln(m, args):
            h = args[0]; h2 = h.clone(); h2[:, 1:] = h2[:, 1:] - delta[s0:s0 + h.shape[0]].to(h.dtype); return (h2,)
        def pre_lin(m, a): cap["a"] = a[0].detach().float()[:, 1:].reshape(-1, DFF); return None
        def stop(m, i, o): raise Stop
        hh = [ln2(bp).register_forward_pre_hook(pre_ln), arch.mlp_lin(bp).register_forward_pre_hook(pre_lin), arch.layers[bp].register_forward_hook(stop)]
        try:
            with torch.no_grad(): model(ids[s0:s0 + CH])
        except Stop: pass
        finally: [h.remove() for h in hh]
        out.append(cap["a"])
    return torch.cat(out)
g = torch.Generator(device=DEV).manual_seed(0)
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0)
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); anorm = A.float().norm(dim=-1); Au = unitr(A); del A; m = Au.shape[0]
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV)
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    Cled = torch.cat([A_all[bb] * rown[bb][None] for bb in range(b + 1)], 1); top = Cled.abs().topk(NW, dim=1); writers = gidx.reshape(-1)[top.indices]; wcoef = Cled.gather(1, top.indices); wblk = top.indices // DFF; widx = top.indices % DFF; del Cled
    # the rows used as native words at this block (16-word OMP over the centred typical states), per block of origin
    sel, _, _ = omp(Xc, Au, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m)
    used_rows = {}
    for bb in range(b + 1):
        cand = gidx[bb]; u = usage[cand]; order = u.argsort(descending=True); nz = int((u > 0).sum()); used_rows[bb] = idx[cand[order[:max(min(NUSED, nz), 1)]]]
    # fake writers: the same blocks and coefficients, rows drawn at random from the block (fake_random) or from its used rows (fake_used)
    f_rand = torch.randint(0, DFF, (Nall, NW), generator=g, device=DEV)
    f_used = torch.empty_like(f_rand)
    for bb in range(b + 1):
        here = wblk == bb; pool = used_rows[bb]; f_used[here] = pool[torch.randint(0, pool.numel(), (int(here.sum()),), generator=g, device=DEV)]
    W_rand = gidx.reshape(-1)[wblk * DFF + f_rand]; W_used = gidx.reshape(-1)[wblk * DFF + f_used]
    mt = typ == T_TOK; tok_lut = torch.full((int(idx[mt].max()) + 1,), -1, device=DEV, dtype=torch.long); tok_lut[idx[mt]] = torch.nonzero(mt)[:, 0]; tatom = tok_lut[tokens]
    emb = anorm[tatom][:, None] * Au[tatom]
    if has_pos:
        mp = typ == T_POS; pos_lut = torch.full((int(idx[mp].max()) + 1,), -1, device=DEV, dtype=torch.long); pos_lut[idx[mp]] = torch.nonzero(mp)[:, 0]; patom = pos_lut[positions]; emb = emb + anorm[patom][:, None] * Au[patom]
    assert bool((writers >= 0).all()) and bool((W_rand >= 0).all()) and bool((W_used >= 0).all())
    chord = torch.einsum("nk,nkd->nd", wcoef, Au[writers]) + emb; rdir = unitr(torch.randn(Nall, D, generator=g, device=DEV))
    kinds = {"real": (writers, widx), "fake_random": (W_rand, f_rand), "fake_used": (W_used, f_used), "random": (None, None)}
    S = torch.zeros(Nall, D, device=DEV); dS = {k: torch.zeros(Nall, D, device=DEV) for k in kinds}; fdir = {}
    for bp in range(b + 1):
        present = (wblk < bp).float(); so_far = torch.einsum("nk,nkd->nd", wcoef * present, Au[writers]) + emb; nsf = so_far.norm(dim=-1, keepdim=True)
        deltas = {"real": so_far}
        for k in ("fake_random", "fake_used"):
            fk = torch.einsum("nk,nkd->nd", wcoef * present, Au[kinds[k][0]]) + emb; deltas[k] = fk * (nsf / fk.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        deltas["random"] = rdir * nsf
        a_base = A_all[bp]; Wd = arch.wdir(bp); S += a_base @ Wd
        for k, delta in deltas.items():
            a_mod = acts_with_delta(bp, delta.view(B_, T_ - 1, D)); da = a_base - a_mod
            if kinds[k][0] is not None:
                mask = torch.zeros(Nall, DFF, dtype=torch.bool, device=DEV); here = wblk == bp; mask[torch.nonzero(here)[:, 0], kinds[k][1][here]] = True; da = da * (~mask)
            dS[k] += da @ Wd
        if bp == b: fdir = {k: unitr(v) for k, v in deltas.items()}
    k_ = keep; ch = chord[k_]; chc = ch - ch.mean(0); cn = chc.norm(dim=-1).clamp_min(1e-9); Sk = S[k_]
    gains = {"real": float(((dS["real"][k_] * (chc / cn[:, None])).sum(1) / cn).median())}
    for k in ("fake_random", "fake_used", "random"): gains[k] = float(((dS[k][k_] * fdir[k][k_]).sum(1) / cn).median())
    norms = {k: float((dS[k][k_].norm(dim=-1) / Sk.norm(dim=-1).clamp_min(1e-9)).median()) for k in kinds}
    span = gains["random"] - gains["real"]; place = {k: (gains[k] - gains["real"]) / span if abs(span) > 1e-6 else None for k in ("fake_random", "fake_used")}
    o = dict(n_positions=int(k_.sum()), gains=gains, response_norm_over_crowd_norm=norms, fake_place_between_real_and_random=place, n_used_rows_per_block=[int(v.numel()) for v in used_rows.values()])
    res["by_block"][b] = o
    log(f"{name}{' ' + rev if rev else ''} block {b} ({o['n_positions']} positions): small writes' response gain to the real chord {gains['real']:+.2f}, to a fake chord of random rows {gains['fake_random']:+.2f}, of used rows {gains['fake_used']:+.2f}, to a random direction {gains['random']:+.2f}; the fakes' place between real (0) and random (1): random rows {place['fake_random'] if place['fake_random'] is None else round(place['fake_random'], 2)}, used rows {place['fake_used'] if place['fake_used'] is None else round(place['fake_used'], 2)}; response norms over the crowd's " + ", ".join(f"{k} {v:.2f}" for k, v in norms.items()))
    del Au, chord, S, dS; torch.cuda.empty_cache()
mid = res["by_block"][NB // 2]; Bk = res["by_block"]; gm = mid["gains"]
res["checks"] = dict(fake_random_like_random=abs(gm["fake_random"] - gm["random"]) <= 0.25 * abs(gm["random"]), fake_used_between=gm["random"] < gm["fake_used"] < gm["real"] or gm["real"] < gm["fake_used"] < gm["random"], real_least_contracted=all(o["gains"]["real"] > max(o["gains"]["fake_random"], o["gains"]["fake_used"], o["gains"]["random"]) for o in Bk.values()))
summ = (f"{name}{' ' + rev if rev else ''}: by block " + " | ".join(f"{b}: response gain to the real chord {o['gains']['real']:+.2f}, fake chord of random rows {o['gains']['fake_random']:+.2f}, of used rows {o['gains']['fake_used']:+.2f}, random direction {o['gains']['random']:+.2f}" for b, o in Bk.items()) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e511_fakechord_{name}{'_' + rev if rev else ''}", res, summ)
