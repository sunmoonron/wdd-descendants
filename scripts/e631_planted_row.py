"""e631 (session 125): the removed speaker with the budget Lo, Cohen and Barez used, and the planted row: the nearest-row
law as a causal prediction. Pythia-160m at step 8000, block 6, thirty words with classes of ten or more positions on 24
Pile sequences of 512 tokens; their write columns zeroed and frozen; training continues for 5,000 steps of 8 windows of
256 tokens (10M tokens, four times e590's budget) on one of two streams: generic Pile windows, or an enriched stream in
which half of every batch is a window containing a class bigram of a removed word (the concept's own data, as Lo et al.
retrain on it). The planted row: for every removed word a quiet non-word row of the same block (bottom fifth by usage)
is designated; in the planted conditions its write direction is turned toward the removed row's, mixed by bisection so
that its largest projection ratio over the class on the step-8000 states is 0.7 (under the floor, and nearer than any
other available row), with its norm set to half the removed row's; its read side is either the removed neuron's (plant_both,
plant_frozen) or its own (plant_dir), and plant_read copies the read side alone with the direction untouched. The smoke
test showed that a full clone is over the floor at once, because the class's coalition builds the state along the
direction without the row (sessions 64 and 109), so the plant is kept under the floor and the question is whether
training carries it over. plant_frozen holds the planted neuron fixed, so that the states' motion alone is measured.
The same rows are designated in every condition, so the unplanted conditions give the placebo rate. Conditions: generic,
enriched, plant_both, plant_dir, plant_read, plant_frozen, random (thirty block-matched non-word rows removed and frozen,
enriched stream), unfrozen (nothing removed, enriched stream). Every 1,000 steps and at the end, per class: a substitute
speaker (a row other than the removed one over the floor at half the class positions or more), its rank at 8000 by mean
ratio over the class, whether it is the designated row, the designated row's largest ratio over the class, its share of
the class over the floor, its OMP wordhood, its own write's share of the state's projection on it at the class positions,
and the loss at the class positions and elsewhere. Arguments: condition [--smoke]. Pre-registered in e631_prereg.json
(v2, revised after the smoke test and fixed before the full runs)."""
import sys, os, time, json, math, collections; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
from datasets import load_dataset
CONDS = ["generic", "enriched", "plant_both", "plant_dir", "plant_read", "plant_frozen", "random", "unfrozen"]; PLANT_S = 0.7; COND = sys.argv[1]; assert COND in CONDS; SMOKE = "--smoke" in sys.argv; t0 = time.time(); torch.set_grad_enabled(False)
name = "pythia160"; B = 6; NW = 30; TEV = 512; NSEQ = 4 if SMOKE else 24; STEPS = 10 if SMOKE else 5000; BSZ = 8; TW = 256; LR = 3e-5; SNAP = 5 if SMOKE else 1000; MAXW = 400 if SMOKE else 45000; sfx = "_smoke" if SMOKE else ""
model, tok, fam = load_model(name, revision="step8000"); arch = Arch(model, fam); DFF = arch.DFF; D = arch.D; assert fam == "neox"
sd0 = {k: v.detach().clone() for k, v in model.state_dict().items()}; ids = pile_ids(name, NSEQ, TEV)
def snapshot():
    X = block_states(model, arch, ids, [B], chunk=4)[B].reshape(-1, D); keep = ~sinkmask(X); A, norms = rows_of(arch, B); U = unitr(X[keep] - X[keep].mean(0)); st = stats(U, A, K); return dict(keep=keep.cpu(), st=st, words=wordset(st["usage"]), A=A, norms=norms)
def lossmap():
    out = []
    for s0 in range(0, ids.shape[0], 4):
        x = ids[s0:s0 + 4]; lg = model(x).logits.float(); out.append(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="none"))
    return torch.cat(out)
S0 = snapshot(); keep0 = S0["keep"]; R0 = S0["st"]["ratio"].float(); words0 = S0["words"]; usage0 = S0["st"]["usage"]; m = R0.shape[1]; w0 = torch.nonzero(words0)[:, 0]; csize = (R0[:, w0] > 1).sum(0); g = torch.Generator().manual_seed(0)
elig = w0[csize >= 10]; targets = elig[torch.randperm(elig.numel(), generator=g)[:NW]].tolist(); CLS = {w: torch.nonzero(R0[:, w] > 1)[:, 0] for w in targets}; kidx0 = torch.nonzero(keep0)[:, 0]
blk = torch.arange(m) // DFF; nonw = torch.nonzero(~words0)[:, 0]
rnd = [int(nonw[blk[nonw] == int(blk[w])][torch.randint(0, int((blk[nonw] == int(blk[w])).sum()), (1,), generator=g)]) for w in targets]
gq = torch.Generator().manual_seed(3); used = set(targets) | set(rnd); quiet = {}
def pick_quiet(bw):
    rows_b = torch.nonzero((blk == bw) & ~words0)[:, 0]; u = usage0[rows_b]; cand = [r for r in rows_b[u <= u.quantile(0.2)].tolist() if r not in used]; q = cand[int(torch.randint(0, len(cand), (1,), generator=gq))]; used.add(q); return q
for w in targets: quiet[w] = pick_quiet(w // DFF)
S_w = {w: float(R0[CLS[w], w].max()) for w in targets}; L0 = lossmap()
log(f"{NW} targets (classes of {int(csize[csize >= 10].float().median())} positions at the median, S {med(list(S_w.values())):.2f}); designated quiet rows chosen; loss {float(L0.mean()):.3f} ({time.time() - t0:.0f}s)")
def plant(mode):
    """turn the designated row's direction toward the removed row's so that its largest ratio over the class on the step-8000 states is PLANT_S; returns per-target records"""
    X0 = block_states(model, arch, ids, [B], chunk=4)[B].reshape(-1, D); U0 = unitr(X0[keep0.to(DEV)] - X0[keep0.to(DEV)].mean(0)); A0, N0 = rows_of(arch, B); L0f = floor_of(U0, floor_calibration(U0, A0)); rec = {}
    with torch.no_grad():
        for w in targets:
            q = quiet[w]; bw, bq = w // DFF, q // DFF; jw, jq = w % DFF, q % DFF; lin_in_w, lin_out_w = arch.layers[bw].mlp.dense_h_to_4h, arch.mlp_lin(bw); lin_in_q, lin_out_q = arch.layers[bq].mlp.dense_h_to_4h, arch.mlp_lin(bq)
            pos = CLS[w].to(DEV); Up, Lp = U0[pos], L0f[pos]; aw, aq = A0[w], A0[q]; Sof = lambda d: float(((Up @ d).abs() / Lp).max())
            alpha, lo, hi = 1.0, 0.0, 1.0
            if mode in ("plant_both", "plant_dir", "plant_frozen"):
                for _ in range(25):
                    mid_ = (lo + hi) / 2; d = unitr((mid_ * aw + (1 - mid_) * aq)[None])[0]
                    if Sof(d) > PLANT_S: hi = mid_
                    else: lo = mid_
                alpha = (lo + hi) / 2; d = unitr((alpha * aw + (1 - alpha) * aq)[None])[0]; lin_out_q.weight[:, jq] = d * (0.5 * float(N0[w]))
            if mode in ("plant_both", "plant_read", "plant_frozen"): lin_in_q.weight[jq] = lin_in_w.weight[jw].clone(); lin_in_q.bias[jq] = lin_in_w.bias[jw].clone()
            dq = unitr(lin_out_q.weight[:, jq][None])[0]; rec[str(w)] = dict(alpha=alpha, S_on_8000_states=Sof(dq), cos_with_removed=float(dq @ aw), cos_quiet_original=float(dq @ aq), S_quiet_original=Sof(aq), S_removed=Sof(aw))
    del X0, U0; torch.cuda.empty_cache(); return rec
def remove(rows):
    with torch.no_grad():
        for r in rows: arch.mlp_lin(r // DFF).weight[:, r % DFF] = 0
HOLD = []
def hold_snapshot(rows):
    """save the full planted neurons (read row, bias, write column) so that they can be restored after every step"""
    out = []
    for r in rows:
        b_, j_ = r // DFF, r % DFF; li, lo = arch.layers[b_].mlp.dense_h_to_4h, arch.mlp_lin(b_); out.append((li, lo, j_, li.weight[j_].detach().clone(), li.bias[j_].detach().clone(), lo.weight[:, j_].detach().clone()))
    return out
def hold_restore():
    with torch.no_grad():
        for li, lo, j_, wr, br, wc in HOLD: li.weight[j_] = wr; li.bias[j_] = br; lo.weight[:, j_] = wc
def freeze_hooks(rows):
    hooks = []
    for b in range(B + 1):
        js = torch.tensor([r % DFF for r in rows if r // DFF == b], device=DEV)
        if js.numel() == 0: continue
        def mk(js_):
            def hk(grad): g2 = grad.clone(); g2[:, js_] = 0; return g2
            return hk
        hooks.append(arch.mlp_lin(b).weight.register_hook(mk(js)))
    return hooks
def rank_at_8000(w, row):
    mr = R0[CLS[w]].mean(0); return int((mr > mr[row]).sum()) + 1
def designated_acts():
    """activations of the designated neurons on the eval ids, [NSEQ*(TEV-1), NW] at the state positions (position t's state <- activation at token t)"""
    qs = [quiet[w] for w in targets]; cap = {}; hs = []
    for b_ in sorted(set(q // DFF for q in qs)):
        js = torch.tensor([q % DFF for q in qs if q // DFF == b_], device=DEV)
        hs.append(arch.mlp_lin(b_).register_forward_pre_hook((lambda b__, js_: lambda mm, a: cap.setdefault(b__, []).append(a[0].detach().float()[:, 1:, js_].reshape(-1, js_.numel())))(b_, js)))
    for s0 in range(0, ids.shape[0], 4): model(ids[s0:s0 + 4])
    [h.remove() for h in hs]; out = torch.zeros(ids.shape[0] * (TEV - 1), len(qs), device=DEV)
    for b_ in cap:
        js = [i for i, q in enumerate(qs) if q // DFF == b_]; out[:, js] = torch.cat(cap[b_])
    return out
def evaluate(tag):
    S = snapshot(); keep = S["keep"]; R = S["st"]["ratio"].float(); L = lossmap(); ACT = designated_acts().cpu(); Xraw = block_states(model, arch, ids, [B], chunk=4)[B].reshape(-1, D); Xc = (Xraw[keep.to(DEV)] - Xraw[keep.to(DEV)].mean(0)).cpu(); Aq = S["A"].cpu(); Nq = S["norms"].cpu(); pos_now = torch.full((keep.numel(),), -1, dtype=torch.long); pos_now[torch.nonzero(keep)[:, 0]] = torch.arange(int(keep.sum())); out = []
    for w in targets:
        full = kidx0[CLS[w]]; idx_all = pos_now[full]; okp = idx_all >= 0; idx_now = idx_all[okp]; fullk = full[okp]; k = idx_now.numel(); r = R[idx_now]; over = (r > 1).sum(0); over[w] = 0; sub = int(over.argmax()); has = bool(k > 0 and over[sub] >= k / 2); qs = quiet[w]; proj = Xc[idx_now] @ Aq[qs]; own = ACT[fullk, targets.index(w)] * float(Nq[qs]); own_share = float((own / proj.abs().clamp_min(1e-6) * proj.sign()).median()) if k else None
        out.append(dict(word=w, substitute=sub if has else None, substitute_rank_8000=rank_at_8000(w, sub) if has else None, substitute_is_designated=bool(has and sub == qs), own_over=int((r[:, w] > 1).sum()), k=k,
                        q_over_share=float(over[qs]) / max(k, 1), q_S_class=float(r[:, qs].max()) if k else None, q_S=float(R[:, qs].max()), q_class_ratio=float(r[:, qs].mean()) if k else None, q_is_word=bool(S["words"][qs]), q_own_share=own_share, q_usage=float(S["st"]["usage"][qs]),
                        loss_class=float(L[full].mean()), loss_class_0=float(L0[full].mean())))
    subs = [x for x in out if x["substitute"] is not None]
    o = dict(share_with_substitute=mean([float(x["substitute"] is not None) for x in out]), substitute_rank_median=med([x["substitute_rank_8000"] for x in subs]), share_substitute_top10=mean([float(x["substitute_rank_8000"] <= 10) for x in subs]) if subs else None,
             share_designated_is_substitute=mean([float(x["substitute_is_designated"]) for x in out]), q_over_share_mean=mean([x["q_over_share"] for x in out]), q_over_half_share=mean([float(x["q_over_share"] >= 0.5) for x in out]), q_S_class_median=med([x["q_S_class"] for x in out if x["q_S_class"] is not None]), q_S_median=med([x["q_S"] for x in out]), q_class_ratio_median=med([x["q_class_ratio"] for x in out if x["q_class_ratio"] is not None]), q_word_share=mean([float(x["q_is_word"]) for x in out]), q_own_share_median=med([x["q_own_share"] for x in out if x["q_own_share"] is not None]), q_usage_median=med([x["q_usage"] for x in out]),
             own_still_over=mean([float(x["own_over"] >= x["k"] / 2) for x in out]), targets_still_words=float(S["words"][torch.tensor(targets)].float().mean()),
             loss_class=mean([x["loss_class"] for x in out]), loss_class_0=mean([x["loss_class_0"] for x in out]), loss_all=float(L.mean()), loss_all_0=float(L0.mean()), n_words_now=int(S["words"].sum()), words_kept=int((S["words"] & words0).sum()), per=out)
    log(f"{tag}: substitute speakers {o['share_with_substitute']:.2f} (rank at 8000 {o['substitute_rank_median']}, top 10 {o['share_substitute_top10']}); the designated row is the substitute {o['share_designated_is_substitute']:.2f}, over the floor at {o['q_over_share_mean']:.2f} of its class (half or more for {o['q_over_half_share']:.2f}), its largest ratio over the class {o['q_S_class_median']}, S anywhere {o['q_S_median']:.2f}, a word for {o['q_word_share']:.2f}, own share of the projection {o['q_own_share_median']}; removed row still over at half its class {o['own_still_over']:.2f}, targets still words {o['targets_still_words']:.2f}; loss at the classes {o['loss_class_0']:.3f} -> {o['loss_class']:.3f}, everywhere {o['loss_all_0']:.3f} -> {o['loss_all']:.3f}; words kept {o['words_kept']} of {int(words0.sum())} ({time.time() - t0:.0f}s)"); return o
# ---------------- streams
ds = load_dataset("NeelNanda/pile-10k", split="train"); buf, wins = [], []
for i in range(24, 10000):
    buf += tok(ds[i]["text"])["input_ids"] + [tok.eos_token_id]
    while len(buf) >= TW + 1 and len(wins) < MAXW: wins.append(buf[:TW + 1]); buf = buf[TW + 1:]
    if len(wins) >= MAXW: break
TR = torch.tensor(wins, device=DEV); trn = TR[:, :TW]; prev, cur = trn[:, :-1].reshape(-1), trn[:, 1:].reshape(-1); key = prev * 100000 + cur; win_of = torch.arange(trn.shape[0], device=DEV).repeat_interleave(TW - 1)
def bigram_at(i):
    f = int(kidx0[i]); seq, t = f // (TEV - 1), f % (TEV - 1) + 1; return (int(ids[seq, t - 1]), int(ids[seq, t]))
SIG = {}
for w in targets:
    c = collections.Counter(bigram_at(int(i)) for i in CLS[w]); keep_bg = []
    for bg, n in c.most_common(12):
        share = float(torch.unique(win_of[key == bg[0] * 100000 + bg[1]]).numel()) / trn.shape[0]
        if share <= 0.05: keep_bg.append(bg)
    SIG[w] = keep_bg
allbg = torch.tensor([p * 100000 + c for w in targets for p, c in SIG[w]], device=DEV, dtype=torch.long); hit = torch.isin(key, allbg); hits_per_window = torch.bincount(win_of[hit], minlength=trn.shape[0]); class_windows = torch.nonzero(hits_per_window > 0)[:, 0]
log(f"{trn.shape[0]} training windows; class bigrams {int(allbg.numel())} over {sum(1 for w in targets if SIG[w])} targets; class windows {class_windows.numel()} ({class_windows.numel() / trn.shape[0]:.2f}); hits per window {float(hits_per_window.float().mean()):.2f} overall, {float(hits_per_window[class_windows].float().mean()):.2f} in class windows ({time.time() - t0:.0f}s)")
gg = torch.Generator().manual_seed(0); order = torch.randperm(trn.shape[0], generator=gg); corder = class_windows[torch.randperm(class_windows.numel(), generator=gg).to(DEV)]
def batch(s, enriched):
    if not enriched: return TR[order[(s * BSZ) % trn.shape[0]:(s * BSZ) % trn.shape[0] + BSZ]] if (s * BSZ) % trn.shape[0] + BSZ <= trn.shape[0] else TR[order[:BSZ]]
    h = BSZ // 2; gi = order[(s * h) % trn.shape[0]:(s * h) % trn.shape[0] + h]; ci = corder[(s * h) % corder.numel():(s * h) % corder.numel() + h]
    if gi.numel() < h: gi = order[:h]
    if ci.numel() < h: ci = corder[:h]
    return torch.cat([TR[gi], TR[ci]])
def continue_training(frozen_rows, enriched):
    for p_ in model.parameters(): p_.requires_grad_(True)
    hooks = freeze_hooks(frozen_rows) if frozen_rows else []; opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0); model.train(); snaps = {}; hits = []
    with torch.enable_grad():
        for s in range(STEPS):
            b = batch(s, enriched); x = b[:, :TW]; lg = model(x).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), b[:, 1:TW].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if frozen_rows: remove(frozen_rows)
            if HOLD: hold_restore()
            if s < 50: hits.append(float(torch.isin(x[:, :-1] * 100000 + x[:, 1:], allbg).float().sum(1).mean()))
            if (s + 1) % SNAP == 0 and s + 1 < STEPS:
                model.eval(); torch.set_grad_enabled(False); snaps[s + 1] = evaluate(f"{COND} step {s + 1}"); torch.set_grad_enabled(True); model.train(); log(f"  step {s + 1}: train loss {float(loss):.3f} ({time.time() - t0:.0f}s)")
    [h.remove() for h in hooks]; model.eval(); torch.set_grad_enabled(False)
    for p_ in model.parameters(): p_.requires_grad_(False)
    return snaps, mean(hits)
res = dict(condition=COND, block=B, steps=STEPS, window=TW, batch=BSZ, lr=LR, plant_S=PLANT_S, n_targets=NW, targets=targets, quiet={str(w): q for w, q in quiet.items()}, S_w={str(w): v for w, v in S_w.items()}, random_rows=rnd, signatures={str(w): SIG[w] for w in targets}, n_class_windows=int(class_windows.numel()), n_windows=int(trn.shape[0]), readouts={})
enriched = COND != "generic"; frozen = rnd if COND == "random" else ([] if COND == "unfrozen" else targets)
if COND.startswith("plant"): res["plant"] = plant(COND); log(f"planted: alpha {med([v['alpha'] for v in res['plant'].values()]):.2f}, S on the 8000 states {med([v['S_on_8000_states'] for v in res['plant'].values()]):.2f} (removed row {med([v['S_removed'] for v in res['plant'].values()]):.2f}, quiet row before {med([v['S_quiet_original'] for v in res['plant'].values()]):.2f}), cosine with the removed row {med([v['cos_with_removed'] for v in res['plant'].values()]):.2f}")
if COND == "plant_frozen": HOLD.extend(hold_snapshot([quiet[w] for w in targets]))
if frozen: remove(frozen)
res["readouts"]["removed_no_training"] = evaluate(f"{COND}: removed, no training" if frozen else f"{COND}: intact")
snaps, hits = continue_training(frozen, enriched); res["class_hits_per_window_in_batches"] = hits
for s_, o in snaps.items(): res["readouts"][f"step_{s_}"] = o
res["readouts"]["final"] = evaluate(f"{COND} final ({STEPS} steps)")
rn, rt = res["readouts"]["removed_no_training"], res["readouts"]["final"]; rise = rn["loss_class"] - rn["loss_class_0"]; res["recovery"] = (rn["loss_class"] - rt["loss_class"]) / rise if rise > 0.001 else None
summ = (f"planted row / removed speaker ({COND}, {STEPS} steps of {BSZ} x {TW}, class windows {class_windows.numel()} of {trn.shape[0]}, class hits per window in batches {hits:.2f}): substitute speakers after removal {rn['share_with_substitute']:.2f} -> final {rt['share_with_substitute']:.2f} (rank at 8000 {rt['substitute_rank_median']}, top 10 {rt['share_substitute_top10']}); the designated row is the substitute {rn['share_designated_is_substitute']:.2f} -> {rt['share_designated_is_substitute']:.2f}, over the floor at {rn['q_over_share_mean']:.2f} -> {rt['q_over_share_mean']:.2f} of its class (half or more {rn['q_over_half_share']:.2f} -> {rt['q_over_half_share']:.2f}), largest ratio over the class {rn['q_S_class_median']} -> {rt['q_S_class_median']}, a word for {rn['q_word_share']:.2f} -> {rt['q_word_share']:.2f}, own share of the projection {rn['q_own_share_median']} -> {rt['q_own_share_median']}; loss at the classes {rn['loss_class_0']:.3f} -> {rn['loss_class']:.3f} -> {rt['loss_class']:.3f} (recovery {res['recovery']}), everywhere {rn['loss_all_0']:.3f} -> {rt['loss_all']:.3f}; targets still words {rt['targets_still_words']:.2f}; words kept {rt['words_kept']} | {time.time() - t0:.0f}s")
log(summ); record(f"e631_planted_{COND}{sfx}", res, summ)
