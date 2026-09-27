"""e521b: is the alignment's source genuinely diffuse, or heavy-tailed behind a large participation ratio? e521 found
that a future word's projection at its best position is attributed, by first-order transport, to MLP writes whose
participation ratio is 30,000-120,000. The relayed caution is right that a participation ratio does not settle the
matter: the ratio is computed on absolute attributions, and if the signed attributions cancel heavily, a few hundred
writes could still account for most of the signed projection. This run draws the retention curve. For each row (the
same entrants and matched non-entrants as e521, the same forward pass with the graph kept), the attributions of every
MLP write at every earlier block and position are sorted by size; the signed cumulative attribution of the top k, as
a share of the projection, is recorded at k = 1, 10, 100, 1000, 10^4, 10^5, 10^6, with the cancellation ratio (the
absolute mass over the projection). Then the causal version: the top k writes are ablated together (their activations
zeroed at their positions and blocks, the projection recomputed in a forward pass) for k = 10, 100, 1000, 10^4, and
the share of the projection removed is compared with the first-order prediction; two controls ablate k random writes,
anywhere and at the row's best position. (Retaining only the top k, the other reading of the caution, zeroes 99.99%
of the writes and changes the state entirely; the removal curve is the meaningful one.)
Setup: Pythia-410m at the origin checkpoint (argument), blocks 12 and 6; 8 x 256 evaluation tokens; entrants and
S-matched non-entrants for the next checkpoint's word set from e519's records.
Pre-registered (honest guesses), block 12 entrants:
- the top 1000 writes account for under a third of the signed projection (0.6);
- the top 10^4 writes account for under two thirds (0.5);
- ablating the top 10^4 removes within 0.15 of the first-order prediction (0.6);
- the cancellation ratio (absolute attribution mass over the projection) is above 3 (0.5).
Arguments: name revision (the origin checkpoint)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2]; STEPS = ["step0", "step64", "step256", "step512", "step1000", "step2000", "step3000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
assert rev in STEPS[:-1], rev; NEXTS = STEPS[STEPS.index(rev) + 1]; CDIR = f"/workspace/wdd/cache/e519_{name}"; NWORD = 256; NR = 120; BL = [12, 6]
KS = [1, 10, 100, 1000, 10000, 100000, 1000000]; KABL = [10, 100, 1000, 10000]
ref = {s: torch.load(f"{CDIR}/{s}.pt") for s in (rev, NEXTS)}
model, tok, fam = load_model(name, revision=None if rev == "main" else rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
for p in model.parameters(): p.requires_grad_(False)
gen = torch.Generator(device=DEV).manual_seed(0)
res = dict(model=name, origin=rev, horizon=NEXTS, ks=KS, k_ablation=KABL, by_block={})
for b in BL:
    u0, u1 = ref[rev][b]["usage"], ref[NEXTS][b]["usage"]; S0 = ref[rev][b]["max_over_floor"]["real"].float()
    W0 = set(torch.nonzero(u0 > 0)[:, 0][u0[u0 > 0].argsort(descending=True)[:NWORD]].tolist()); W1 = set(torch.nonzero(u1 > 0)[:, 0][u1[u1 > 0].argsort(descending=True)[:NWORD]].tolist())
    R = u0.numel(); in0 = torch.zeros(R, dtype=torch.bool); in0[list(W0)] = True; in1 = torch.zeros(R, dtype=torch.bool); in1[list(W1)] = True
    ent = torch.nonzero(~in0 & in1)[:, 0]; ent = ent[torch.randperm(ent.numel(), generator=torch.Generator().manual_seed(0))[:NR]]; pool = torch.nonzero(~in0 & ~in1)[:, 0]; taken = torch.zeros(R, dtype=torch.bool); matched = []
    for i in ent.tolist():
        cand = pool[~taken[pool]]; j = cand[(S0[cand] - S0[i]).abs().argmin()]; taken[j] = True; matched.append(int(j))
    matched = torch.tensor(matched); rows_sel = torch.cat([ent, matched]).to(DEV); is_ent = torch.cat([torch.ones(ent.numel()), torch.zeros(matched.numel())]).bool()
    cap = dict(out={}, act={})
    def h_out(bb):
        def f(m, i, o): cap["out"][bb] = o[0] if isinstance(o, tuple) else o
        return f
    def h_act(bb):
        def f(m, a): cap["act"][bb] = a[0].detach().float(); return None
        return f
    def h_emb(m, i, o): o.requires_grad_(True); cap["emb"] = o
    def h_stop(m, i, o): raise Stop
    layers = arch.layers; emb_mod = model.gpt_neox.embed_in if fam == "neox" else (model.transformer.wte if fam == "gpt2" else model.model.embed_tokens)
    hs = [layers[bb].register_forward_hook(h_out(bb)) for bb in range(b + 1)] + [arch.mlp_lin(bb).register_forward_pre_hook(h_act(bb)) for bb in range(b + 1)] + [emb_mod.register_forward_hook(h_emb), layers[b].register_forward_hook(h_stop)]
    with torch.enable_grad():
        try: model(ids)
        except Stop: pass
        finally: [h.remove() for h in hs]
    xb = cap["out"][b].detach().float(); Xall = xb.reshape(-1, D); keep = ~sinkmask(Xall); mu = Xall[keep].mean(0)
    Wd = {bb: arch.wdir(bb).float() for bb in range(b)}
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); del A; typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV)
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    Wsel = Au[gidx.reshape(-1)[rows_sel]]; del Au
    Xc = Xall - mu; xn = Xc.norm(dim=-1).clamp_min(1e-9); P = (Xc @ Wsel.T) / xn[:, None]; P[~keep] = 0; best = P.abs().argmax(0); bi, ti = best // T_, best % T_; del P
    NW = b * B_ * T_ * DFF                                                                                                  # writes at blocks below b, all positions
    def projection_after(abl):
        """the projection at the row's best position after zeroing the given writes: abl = list of (block, flat position, neuron) tensors"""
        by = {}
        for bb, pos, j in abl: by.setdefault(bb, []).append((pos, j))
        def hk(bb, pj):
            pos = torch.tensor([p for p, _ in pj], device=DEV); jj = torch.tensor([j for _, j in pj], device=DEV)
            def f(m, a):
                a0 = a[0].clone(); a0.view(-1, DFF)[pos, jj] = 0; return (a0,)
            return f
        cap2 = {}
        def cap_stop(m, i, o): cap2["o"] = (o[0] if isinstance(o, tuple) else o).detach(); raise Stop
        hh = [arch.mlp_lin(bb).register_forward_pre_hook(hk(bb, pj)) for bb, pj in by.items()] + [layers[b].register_forward_hook(cap_stop)]
        try:
            with torch.no_grad(): model(ids)
        except Stop: pass
        finally: [h.remove() for h in hh]
        return cap2["o"]
    quant = dict(projection=[], total=[], absmass=[], participation=[], cum={k: [] for k in KS}, abl_top={k: [] for k in KABL}, pred_top={k: [] for k in KABL}, abl_rand={k: [] for k in KABL}, abl_rand_pos={k: [] for k in KABL}, top_same_pos={k: [] for k in KABL})
    for r in range(rows_sel.numel()):
        w = Wsel[r]
        with torch.enable_grad(): f = (cap["out"][b][bi[r], ti[r]].float() - mu) @ w; grads = torch.autograd.grad(f, [cap["out"][bb] for bb in range(b)], retain_graph=True)
        M = torch.cat([(cap["act"][bb] * (grads[bb].float() @ Wd[bb].T)).reshape(-1) for bb in range(b)])                     # [b * B * T * DFF] signed attribution of every write
        fv = float(f); sg = 1.0 if fv >= 0 else -1.0; tot = float(M.sum()); am = float(M.abs().sum()); sq = float(M.pow(2).sum())
        v, ix = M.abs().topk(KS[-1]); Ms = M[ix]; cs = Ms.cumsum(0)
        quant["projection"].append(fv); quant["total"].append(tot * sg / abs(fv)); quant["absmass"].append(am / abs(fv)); quant["participation"].append(am * am / sq if sq > 0 else float("nan"))
        for k in KS: quant["cum"][k].append(float(cs[k - 1]) * sg / abs(fv))
        per_block = B_ * T_ * DFF
        for k in KABL:
            sel = ix[:k]; bb_ = sel // per_block; rem = sel % per_block; pos_ = rem // DFF; j_ = rem % DFF
            abl = [(int(bb), int(p), int(j)) for bb, p, j in zip(bb_.tolist(), pos_.tolist(), j_.tolist())]
            fa = float((projection_after(abl)[bi[r], ti[r]].float() - mu) @ w); quant["abl_top"][k].append((fv - fa) * sg / abs(fv)); quant["pred_top"][k].append(float(cs[k - 1]) * sg / abs(fv))
            quant["top_same_pos"][k].append(float((pos_ == int(best[r])).float().mean()))
            rs = torch.randint(0, NW, (k,), device=DEV, generator=gen); bb_ = rs // per_block; rem = rs % per_block; pos_ = rem // DFF; j_ = rem % DFF
            fa = float((projection_after([(int(bb), int(p), int(j)) for bb, p, j in zip(bb_.tolist(), pos_.tolist(), j_.tolist())])[bi[r], ti[r]].float() - mu) @ w); quant["abl_rand"][k].append((fv - fa) * sg / abs(fv))
            rs = torch.randint(0, b * DFF, (k,), device=DEV, generator=gen); bb_ = rs // DFF; j_ = rs % DFF
            fa = float((projection_after([(int(bb), int(best[r]), int(j)) for bb, j in zip(bb_.tolist(), j_.tolist())])[bi[r], ti[r]].float() - mu) @ w); quant["abl_rand_pos"][k].append((fv - fa) * sg / abs(fv))
        del M, v, ix, Ms, cs
    T = lambda v: torch.tensor(v)
    out = dict(n_entrants=int(is_ent.sum()), n_matched=int((~is_ent).sum()), groups={})
    for gname, gm in (("entrants", is_ent), ("matched", ~is_ent)):
        out["groups"][gname] = dict(attribution_over_projection=float(T(quant["total"])[gm].median()), cancellation_ratio=float(T(quant["absmass"])[gm].median()), participation_ratio=float(T(quant["participation"])[gm].nanmedian()),
                                    cumulative_share={k: float(T(quant["cum"][k])[gm].median()) for k in KS}, ablated_top={k: float(T(quant["abl_top"][k])[gm].median()) for k in KABL}, predicted_top={k: float(T(quant["pred_top"][k])[gm].median()) for k in KABL},
                                    ablated_random={k: float(T(quant["abl_rand"][k])[gm].median()) for k in KABL}, ablated_random_same_position={k: float(T(quant["abl_rand_pos"][k])[gm].median()) for k in KABL}, top_same_position={k: float(T(quant["top_same_pos"][k])[gm].mean()) for k in KABL})
    res["by_block"][b] = out; e_ = out["groups"]["entrants"]; m_ = out["groups"]["matched"]
    log(f"{name} {rev} block {b} ({out['n_entrants']} entrants, {out['n_matched']} matched): signed cumulative share of the projection from the top k writes, entrants: " + ", ".join(f"k={k} {e_['cumulative_share'][k]:.2f}" for k in KS) + f" (matched: " + ", ".join(f"{m_['cumulative_share'][k]:.2f}" for k in KS) + f"); cancellation ratio {e_['cancellation_ratio']:.1f}/{m_['cancellation_ratio']:.1f}, participation {e_['participation_ratio']:.0f}/{m_['participation_ratio']:.0f}; ablating the top k removes / first-order predicts, entrants: " + ", ".join(f"k={k} {e_['ablated_top'][k]:.2f}/{e_['predicted_top'][k]:.2f}" for k in KABL) + "; random k anywhere / at the position: " + ", ".join(f"k={k} {e_['ablated_random'][k]:.2f}/{e_['ablated_random_same_position'][k]:.2f}" for k in KABL) + "; top-k at the same position: " + ", ".join(f"k={k} {e_['top_same_position'][k]:.2f}" for k in KABL))
    del cap, Wsel; torch.cuda.empty_cache()
e = res["by_block"][12]["groups"]["entrants"]
res["checks"] = dict(top1000_under_third=e["cumulative_share"][1000] < 1 / 3, top1e4_under_two_thirds=e["cumulative_share"][10000] < 2 / 3, ablation_within_0_15_at_1e4=abs(e["ablated_top"][10000] - e["predicted_top"][10000]) < 0.15, cancellation_over_3=e["cancellation_ratio"] > 3)
summ = f"{name} {rev} to {NEXTS}: " + " | ".join(f"block {b}: entrants' signed share from the top k writes " + ", ".join(f"{k}: {o['groups']['entrants']['cumulative_share'][k]:.2f}" for k in KS) + f"; cancellation {o['groups']['entrants']['cancellation_ratio']:.1f}; ablating the top k removes (predicted) " + ", ".join(f"{k}: {o['groups']['entrants']['ablated_top'][k]:.2f} ({o['groups']['entrants']['predicted_top'][k]:.2f})" for k in KABL) + "; random k at the position " + ", ".join(f"{k}: {o['groups']['entrants']['ablated_random_same_position'][k]:.2f}" for k in KABL) for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e521b_retention_{name}_{rev}", res, summ)
