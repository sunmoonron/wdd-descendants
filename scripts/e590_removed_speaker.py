"""e590 (session 109): the blocked recruit at scale, as one decisive experiment. The real training interval between
checkpoints is two billion tokens, so a proxy continuation cannot reproduce natural recruitments; the tractable causal
question is the removed speaker: at step 8000, thirty words (classes of ten or more positions) have their write
columns zeroed and frozen, and training continues on Pile text (pile-10k windows of 512 tokens, batch 8, AdamW at
3e-5, 600 steps, 2.5M tokens). Do their classes recruit substitutes, and does the network's loss at the class positions
recover? Conditions: the thirty words removed and frozen; thirty random non-word rows removed and frozen (block-matched);
nothing removed (the same continuation); and the removal without training. Measures per class: a substitute speaker
(a row other than the removed one over the floor at half the class's positions or more), its rank at 8000 by mean
ratio over the class (the next nearest?), the class's loss against the loss elsewhere. Pre-registered (probabilities
are honest guesses):
 R1 (0.5) after 600 steps at least half of the removed words' classes have a substitute speaker, against under a
    fifth immediately after removal and under a fifth in the unfrozen continuation;
 R2 (0.6) the substitute is among the ten rows nearest to the class at 8000 by ratio for most classes;
 R3 (0.6) the loss at the class positions rises on removal and recovers by more than half of the rise with training."""
from s101_common import *
from datasets import load_dataset
t0 = time.time(); B = 12; NW = 30; STEPS, BS, T = 600, 8, 512; LR = 3e-5; ids = eval_ids("pythia410")[:24, :512].to(DEV)
model, tok, fam = load_model("pythia410", revision="step8000"); arch = Arch(model, fam); DFF = arch.DFF
sd0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
def snapshot():
    X = block_states(model, arch, ids, [B], chunk=4)[B].reshape(-1, arch.D); keep = ~sinkmask(X); A, norms = rows_of(arch, B); U = unitr(X[keep] - X[keep].mean(0)); st = stats(U, A, K); return dict(keep=keep, st=st, words=wordset(st["usage"]), A=A)
def lossmap():
    with torch.no_grad():
        lg = model(ids).logits.float(); l = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="none")
    return l   # per position 1:
S0 = snapshot(); keep0 = S0["keep"]; R0 = S0["st"]["ratio"].float(); w0 = torch.nonzero(S0["words"])[:, 0]; csize = (R0[:, w0] > 1).sum(0); g = torch.Generator().manual_seed(0)
elig = w0[csize >= 10]; targets = elig[torch.randperm(elig.numel(), generator=g)[:NW]]; CLS = {int(w): torch.nonzero(R0[:, w] > 1)[:, 0] for w in targets.tolist()}
kidx0 = torch.nonzero(keep0.cpu())[:, 0]; blk = torch.arange(S0["A"].shape[0]) // DFF; nonw = torch.nonzero(~S0["words"])[:, 0]
rnd = torch.cat([nonw[blk[nonw] == int(blk[w])][torch.randint(0, int((blk[nonw] == int(blk[w])).sum()), (1,), generator=g)] for w in targets.tolist()])
L0 = lossmap(); log(f"{NW} target words with classes of {int(csize[csize >= 10].float().median())} positions at the median; loss {float(L0.mean()):.3f}")
def rank_at_8000(w, row):
    pos = CLS[w]; mr = R0[pos].mean(0); return int((mr > mr[row]).sum()) + 1
def evaluate(tag):
    S = snapshot(); keep = S["keep"]; R = S["st"]["ratio"].float(); L = lossmap(); out = []
    for w, pos in CLS.items():
        full = kidx0[pos]; kept_now = keep.cpu()[full]; posn = torch.nonzero(keep.cpu())[:, 0]; idx_now = torch.tensor([int((posn == p).nonzero()[0]) for p in full[kept_now].tolist()]) if kept_now.any() else torch.tensor([], dtype=torch.long)
        r = R[idx_now]; over = (r > 1).sum(0); over[w] = 0; k = idx_now.numel(); sub = int(over.argmax()); has = bool(over[sub] >= k / 2)
        lw = float(L[full].mean()); out.append(dict(word=w, substitute=sub if has else None, substitute_rank_8000=rank_at_8000(w, sub) if has else None, own_over=int((r[:, w] > 1).sum()), loss_class=lw, loss_class_0=float(L0[full].mean())))
    o = dict(share_with_substitute=mean([float(x["substitute"] is not None) for x in out]), substitute_rank_median=med([x["substitute_rank_8000"] for x in out if x["substitute"] is not None]), share_substitute_top10=mean([float(x["substitute_rank_8000"] <= 10) for x in out if x["substitute"] is not None]) if any(x["substitute"] is not None for x in out) else None,
             own_still_over=mean([float(x["own_over"] >= len(CLS[x["word"]]) / 2) for x in out]), loss_class=mean([x["loss_class"] for x in out]), loss_class_0=mean([x["loss_class_0"] for x in out]), loss_all=float(L.mean()), loss_all_0=float(L0.mean()), n_words_now=int(S["words"].sum()), targets_still_words=float(S["words"][targets].float().mean()), per=out)
    log(f"{tag}: classes with a substitute speaker {o['share_with_substitute']:.2f} (its rank at 8000 {o['substitute_rank_median']}, top 10 for {o['share_substitute_top10']}), the removed row still over the floor at half its class {o['own_still_over']:.2f}, targets still words {o['targets_still_words']:.2f}; loss at the classes {o['loss_class_0']:.3f} -> {o['loss_class']:.3f}, everywhere {o['loss_all_0']:.3f} -> {o['loss_all']:.3f}"); return o
def remove(rows):
    with torch.no_grad():
        for r in rows.tolist(): arch.mlp_lin(r // DFF).weight[:, r % DFF] = 0
def freeze_hooks(rows):
    hooks = []
    for b in range(B + 1):
        js = torch.tensor([r % DFF for r in rows.tolist() if r // DFF == b], device=DEV)
        if js.numel() == 0: continue
        def mk(js_):
            def hk(grad): g2 = grad.clone(); g2[:, js_] = 0; return g2
            return hk
        hooks.append(arch.mlp_lin(b).weight.register_hook(mk(js)))
    return hooks
ds = load_dataset("NeelNanda/pile-10k", split="train"); buf, wins = [], []
for i in range(24, 10000):
    buf += tok(ds[i]["text"])["input_ids"] + [tok.eos_token_id]
    while len(buf) >= T + 1 and len(wins) < STEPS * BS: wins.append(buf[:T + 1]); buf = buf[T + 1:]
    if len(wins) >= STEPS * BS: break
train = torch.tensor(wins, device=DEV); log(f"{train.shape[0]} training windows ({time.time() - t0:.0f}s)")
def continue_training(frozen_rows):
    for p_ in model.parameters(): p_.requires_grad_(True)
    hooks = freeze_hooks(frozen_rows) if frozen_rows is not None else []; opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0); model.train()
    with torch.enable_grad():
        for s in range(STEPS):
            b = train[s * BS:(s + 1) * BS]; lg = model(b[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), b[:, 1:T].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if frozen_rows is not None: remove(frozen_rows)
            if s % 200 == 199: log(f"  step {s + 1}: train loss {float(loss):.3f} ({time.time() - t0:.0f}s)")
    [h.remove() for h in hooks]; model.eval()
    for p_ in model.parameters(): p_.requires_grad_(False)
res = dict(n_targets=NW, conditions={})
remove(targets); res["conditions"]["removed_no_training"] = evaluate("removed, no training")
continue_training(targets); res["conditions"]["removed_trained"] = evaluate("removed and frozen, 600 steps"); model.load_state_dict(sd0)
remove(rnd); continue_training(rnd); res["conditions"]["random_removed_trained"] = evaluate("random rows removed and frozen, 600 steps"); model.load_state_dict(sd0)
continue_training(None); res["conditions"]["unfrozen_trained"] = evaluate("nothing removed, 600 steps"); model.load_state_dict(sd0)
C = res["conditions"]; rt, rn, ut, rr = C["removed_trained"], C["removed_no_training"], C["unfrozen_trained"], C["random_removed_trained"]
rise = rn["loss_class"] - rn["loss_class_0"]; rec = (rn["loss_class"] - rt["loss_class"]) / rise if rise > 0 else None
summ = (f"removed speaker at scale (Pythia-410m from step 8000, {NW} words, 600 proxy steps): classes with a substitute speaker: removed without training {rn['share_with_substitute']:.2f}, removed and trained {rt['share_with_substitute']:.2f} (substitute's rank at 8000 {rt['substitute_rank_median']}, top 10 for {rt['share_substitute_top10']}), unfrozen continuation {ut['share_with_substitute']:.2f}, random rows removed {rr['share_with_substitute']:.2f}; loss at the classes {rn['loss_class_0']:.3f} -> {rn['loss_class']:.3f} on removal -> {rt['loss_class']:.3f} after training (elsewhere {rn['loss_all_0']:.3f} -> {rn['loss_all']:.3f} -> {rt['loss_all']:.3f}; unfrozen {ut['loss_class']:.3f} / {ut['loss_all']:.3f}); recovery of the rise {rec if rec is None else round(rec, 2)}; targets still words after removal and training {rt['targets_still_words']:.2f} | {time.time() - t0:.0f}s")
res["recovery"] = rec; log(summ); record("e590_removed_speaker", res, summ)
