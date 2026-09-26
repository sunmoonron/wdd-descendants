"""e514: the lineage of the future words, and the winner split by type. e513 found that sparsification is every row
turning selective while the gradient spreads, not a few rows being reinforced. The question that turns the
descriptive sequence into a mechanism, or refutes one, is whether the rows that end as native words can be told
apart from the others before they are words, and by what. This run fixes the rows that become words at the end of
training and follows them back through the checkpoints, against non-word rows matched on their eventual write
magnitude (so that "large rows become words" cannot masquerade as a precursor) and against random non-word rows.
Reference (the run at the end of training writes it, the others wait for it): at blocks 6 and 12, the 256 MLP rows
of blocks up to the block most used as native words (16-word OMP over the centred typical states); for each word a
non-word row of the same block with the closest mean absolute write at the end, taken without replacement; and 256
random non-word rows; with the rows' write directions at the end.
Per checkpoint, block and row set: the rows' activity (share of positions at which the row is active), the excess
kurtosis of the row's write coefficient across positions, the row's share of the block's loss gradient (one backward
pass, per-row norms of the down-projection's gradient), the cosine of the row's write direction with its direction
at the end, the row's mean absolute write, and its usage as a native word at that checkpoint; medians per set, and
the AUC of each quantity for words against matched non-words (0.5 is no separation). The checkpoint at which a
quantity's AUC first exceeds 0.6 is when it separates the future words; the usage AUC is when they are words.
Also, per block, the winner split by type: the share of states whose largest projection is on an MLP row among the
position's writers, on the token embedding, on an MLP row that is not a writer, or elsewhere.
Setup: Pythia-410m; 8 x 256 evaluation tokens; typical positions.
Pre-registered (honest guesses):
- future words are more selective than matched non-words before they are words: the selectivity AUC exceeds 0.6 at
  an earlier checkpoint than the usage AUC does, at block 12 (0.5);
- their directions settle earlier: the cosine-with-the-end AUC exceeds 0.6 by step 4000 (0.5);
- the gradient share does not separate them before wordhood (its AUC stays under 0.6 through step 2000) (0.6);
- matched on eventual magnitude, the selectivity separation is still present at the end (AUC above 0.6) (0.6).
Arguments: name [revision]."""
import sys, os, json as _json, math, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
BL = [6, 12]; LB = max(BL); NWORD = 256; K = 16; NW = 64; CACHE = f"/workspace/wdd/cache/e514_words_{name}.pt"
if rev is not None:
    t0 = time.time()
    while not os.path.exists(CACHE):
        if time.time() - t0 > 1800: raise SystemExit("reference not found")
        time.sleep(20)
    time.sleep(5)
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
# the gradient's share per row, from one backward pass
for p in model.parameters(): p.requires_grad_(False)
lins = {b: arch.mlp_lin(b) for b in range(LB + 1)}
for lin in lins.values(): lin.weight.requires_grad_(True)
with torch.enable_grad():
    loss = model(ids, labels=ids).loss; loss.backward()
gshare = {}
for b, lin in lins.items():
    gr = lin.weight.grad.detach().float(); rn = gr.norm(dim=1) if gr.shape[0] == DFF else gr.norm(dim=0); e = rn.pow(2); gshare[b] = e / e.sum().clamp_min(1e-30); lin.weight.grad = None; lin.weight.requires_grad_(False)
lm_loss = float(loss.detach())
acts = {b: [] for b in range(LB + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)]
try: S_ = block_states(model, arch, ids, BL, chunk=4)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(LB + 1)}; del acts; rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}; tokens = ids[:, 1:].reshape(-1)
g = torch.Generator(device=DEV).manual_seed(0)
def auc(pos, neg):
    """Mann-Whitney AUC of pos against neg"""
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def kurt_cols(M):
    z = (M - M.mean(0, keepdim=True)) / M.std(0, keepdim=True).clamp_min(1e-9); return z.pow(4).mean(0) - 3
per_block = {}
for b in BL:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); U = unitr(Xc); N = U.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); del A; typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); m = Au.shape[0]
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    sel, _, _ = omp(Xc, Au, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m)
    mag = torch.stack([(A_all[bb][keep].abs() * rown[bb][None]).mean(0) for bb in range(b + 1)])                       # [b+1, DFF] mean absolute write per row
    per_block[b] = dict(keep=keep, U=U, Au=Au, typ=typ, idx=idx, gidx=gidx, usage=usage, mag=mag, m=m)
if rev is None:
    ref = {}
    for b in BL:
        pb = per_block[b]; u = pb["usage"][pb["gidx"]]                                                                     # [b+1, DFF] usage per row
        flat_u = u.reshape(-1); order = flat_u.argsort(descending=True); words = order[:NWORD]; assert bool((flat_u[words] > 0).all())
        wb, wi = words // DFF, words % DFF; pool = torch.nonzero(flat_u == 0)[:, 0]; taken = torch.zeros(flat_u.numel(), dtype=torch.bool, device=DEV); matched = []
        for k in range(NWORD):
            cand = pool[(pool // DFF == wb[k]) & (~taken[pool])]; j = cand[(pb["mag"].reshape(-1)[cand] - pb["mag"][wb[k], wi[k]]).abs().argmin()]; taken[j] = True; matched.append(j)
        matched = torch.stack(matched); rnd = pool[torch.randperm(pool.numel(), generator=g, device=DEV)[:NWORD]]
        sets = {"words": words, "matched": matched, "random": rnd}
        ref[b] = {nm: dict(rows=v.cpu(), dirs=torch.stack([unitr(arch.wdir(int(r // DFF)).float())[int(r % DFF)] for r in v]).cpu()) for nm, v in sets.items()}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True); torch.save(ref, CACHE); log(f"reference written: {NWORD} words per block, matched and random non-words")
ref = torch.load(CACHE)
res = dict(model=name, revision=rev, blocks=BL, lm_loss=lm_loss, by_block={})
for b in BL:
    pb = per_block[b]; keep, U, Au, typ, gidx, usage, mag, m = pb["keep"], pb["U"], pb["Au"], pb["typ"], pb["gidx"], pb["usage"], pb["mag"], pb["m"]; N = U.shape[0]
    # the winner split by type
    Cled = torch.cat([A_all[bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); top = Cled.abs().topk(NW, dim=1).indices; writers = gidx.reshape(-1)[top]; del Cled
    mt = typ == T_TOK; ta = torch.nonzero(mt)[:, 0]; lut = torch.full((int(pb["idx"][ta].max()) + 1,), -1, device=DEV, dtype=torch.long); lut[pb["idx"][ta]] = ta; tatom = lut[tokens[keep]]; assert bool((tatom >= 0).all())
    arg = torch.cat([(U[s:s + 128] @ Au.T).abs().argmax(1) for s in range(0, N, 128)]); at = typ[arg]
    is_w = (writers == arg[:, None]).any(1); split = dict(mlp_writer=float(((at == T_MLP) & is_w).float().mean()), embedding=float((arg == tatom).float().mean()), mlp_nonwriter=float(((at == T_MLP) & ~is_w).float().mean()))
    split["other"] = 1 - split["mlp_writer"] - split["embedding"] - split["mlp_nonwriter"]
    out = dict(n_states=N, winner_split=split, sets={}, auc={})
    q = {}
    for nm, d in ref[b].items():
        rows = d["rows"].to(DEV); dirs_end = d["dirs"].to(DEV); rb, ri = rows // DFF, rows % DFF; n = rows.numel()
        act = torch.zeros(N, n, device=DEV); coef = torch.zeros(N, n, device=DEV); gs = torch.zeros(n, device=DEV); cosv = torch.zeros(n, device=DEV); use = torch.zeros(n, device=DEV); mg = torch.zeros(n, device=DEV)
        for bb in range(b + 1):
            here = torch.nonzero(rb == bb)[:, 0]
            if here.numel() == 0: continue
            cols = ri[here]; a = A_all[bb][keep][:, cols]; act[:, here] = a; coef[:, here] = a * rown[bb][cols][None]; gs[here] = gshare[bb][cols]; cosv[here] = (unitr(arch.wdir(bb).float())[cols] * dirs_end[here]).sum(1); use[here] = usage[gidx[bb, cols]].float(); mg[here] = mag[bb, cols]
        q[nm] = dict(activity=(act > 0).float().mean(0), kurtosis=kurt_cols(coef), gradient_share=gs, cosine_with_end=cosv, magnitude=mg, usage=use)
        out["sets"][nm] = {k: float(v.median()) for k, v in q[nm].items()} | dict(share_used=float((use > 0).float().mean()))
    for k in ("activity", "kurtosis", "gradient_share", "cosine_with_end", "magnitude", "usage"):
        out["auc"][k] = dict(words_vs_matched=auc(q["words"][k], q["matched"][k]), words_vs_random=auc(q["words"][k], q["random"][k]))
    out["auc"]["selectivity"] = dict(words_vs_matched=1 - out["auc"]["activity"]["words_vs_matched"], words_vs_random=1 - out["auc"]["activity"]["words_vs_random"])
    res["by_block"][b] = out; w, mt_, r_ = out["sets"]["words"], out["sets"]["matched"], out["sets"]["random"]; a_ = out["auc"]
    log(f"{name}{' ' + rev if rev else ''} block {b}: winners on MLP writers {split['mlp_writer']:.2f}, the embedding {split['embedding']:.2f}, MLP non-writers {split['mlp_nonwriter']:.2f}, other {split['other']:.2f} | words / matched / random: activity {w['activity']:.2f}/{mt_['activity']:.2f}/{r_['activity']:.2f}, kurtosis {w['kurtosis']:.1f}/{mt_['kurtosis']:.1f}/{r_['kurtosis']:.1f}, gradient share x1e4 {1e4 * w['gradient_share']:.1f}/{1e4 * mt_['gradient_share']:.1f}/{1e4 * r_['gradient_share']:.1f}, cosine with the end {w['cosine_with_end']:.2f}/{mt_['cosine_with_end']:.2f}/{r_['cosine_with_end']:.2f}, magnitude {w['magnitude']:.3f}/{mt_['magnitude']:.3f}/{r_['magnitude']:.3f}, share used as words {w['share_used']:.2f}/{mt_['share_used']:.2f}/{r_['share_used']:.2f} | AUC words vs matched: selectivity {a_['selectivity']['words_vs_matched']:.2f}, kurtosis {a_['kurtosis']['words_vs_matched']:.2f}, gradient {a_['gradient_share']['words_vs_matched']:.2f}, cosine {a_['cosine_with_end']['words_vs_matched']:.2f}, magnitude {a_['magnitude']['words_vs_matched']:.2f}, usage {a_['usage']['words_vs_matched']:.2f}")
    del Au; torch.cuda.empty_cache()
summ = (f"{name}{' ' + rev if rev else ''} (loss {lm_loss:.2f}): by block " + " | ".join(f"{b}: winners MLP-writer/embedding/MLP-non-writer/other {o['winner_split']['mlp_writer']:.2f}/{o['winner_split']['embedding']:.2f}/{o['winner_split']['mlp_nonwriter']:.2f}/{o['winner_split']['other']:.2f}; words/matched activity {o['sets']['words']['activity']:.2f}/{o['sets']['matched']['activity']:.2f}, cosine with the end {o['sets']['words']['cosine_with_end']:.2f}/{o['sets']['matched']['cosine_with_end']:.2f}, used {o['sets']['words']['share_used']:.2f}/{o['sets']['matched']['share_used']:.2f}; AUC words vs matched selectivity/kurtosis/gradient/cosine/magnitude/usage {o['auc']['selectivity']['words_vs_matched']:.2f}/{o['auc']['kurtosis']['words_vs_matched']:.2f}/{o['auc']['gradient_share']['words_vs_matched']:.2f}/{o['auc']['cosine_with_end']['words_vs_matched']:.2f}/{o['auc']['magnitude']['words_vs_matched']:.2f}/{o['auc']['usage']['words_vs_matched']:.2f}" for b, o in res["by_block"].items()))
log(summ); record(f"e514_lineage_{name}{'_' + rev if rev else ''}", res, summ)
