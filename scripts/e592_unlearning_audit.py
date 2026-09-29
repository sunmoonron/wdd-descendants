"""e592 (session 110): an unlearning audit with parameter provenance. After a forget set is unlearned, do the parameters
that wrote its classes still write them? Pythia-160m, block 6, the MLP rows of blocks 0-6. Forget set: PubMed
abstracts from pile-10k (96 training windows of 256 tokens, 16 held out); retain set: other domains (96 and 16).
Words on the held-out states; forget words are those whose class (over-the-floor positions) is at least 70% forget
positions, retain words at least 70% retain. Two methods, gradient ascent on the forget set and training the forget
set toward the uniform distribution, each with a KL-to-original term on the retain set, at 50, 100, 200 and 400 steps
(lr 1e-5): eight conditions from the same original. After each: the forget and retain held-out losses; the audit (the
forget words' rows' cosine with their originals, the share still over the floor at half their class positions on the
unlearned model's states, the same for retain words); where the change lives (the original's blocks 7-11 on the
unlearned model's block-6 states, and the reverse); and relearning, 30 steps on the forget training windows, the
share of the unlearning rise recovered. Pre-registered (probabilities are honest guesses):
 U1 (0.6) after gradient ascent the forget words' rows keep cosine 0.95 or more with their originals and at least
    0.6 of them still write over the floor at their classes: the unlearning acts elsewhere;
 U2 (0.6) across the eight conditions the share of forget words still writing correlates with the relearning
    recovery at Spearman 0.5 or more (provenance predicts shallow unlearning);
 U3 (0.7) retain words are unchanged (cosine 0.99 or more, the writing share within 0.1)."""
from s101_common import *
from datasets import load_dataset
import copy
t0 = time.time(); B = 6; T = 256; NTR, NEV = 96, 16; LR, LAM = 1e-5, 1.0; torch.set_grad_enabled(False)
model, tok, fam = load_model("pythia160"); arch = Arch(model, fam); orig = copy.deepcopy(model).eval()
for p_ in orig.parameters(): p_.requires_grad_(False)
ds = load_dataset("NeelNanda/pile-10k", split="train")
def windows(pred, nwin, skipdocs=0):
    wins, buf, seen = [], [], 0
    for ex in ds:
        if not pred(ex["meta"]["pile_set_name"]): continue
        seen += 1
        if seen <= skipdocs: continue
        buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
        while len(buf) >= T + 1 and len(wins) < nwin: wins.append(buf[:T + 1]); buf = buf[T + 1:]
        if len(wins) >= nwin: break
    return torch.tensor(wins, device=DEV)
FORGET = lambda s: s == "PubMed Abstracts"; RETAIN = lambda s: s not in ("PubMed Abstracts", "PubMed Central")
f_tr = windows(FORGET, NTR); f_ev = windows(FORGET, NEV, skipdocs=400); r_tr = windows(RETAIN, NTR); r_ev = windows(RETAIN, NEV, skipdocs=600); log(f"windows: forget {f_tr.shape[0]}/{f_ev.shape[0]}, retain {r_tr.shape[0]}/{r_ev.shape[0]} ({time.time() - t0:.0f}s)")
def lossof(m, ids):
    lg = m(ids).logits.float(); return float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)))
EV = torch.cat([f_ev, r_ev])[:, :T]; is_forget_seq = torch.cat([torch.ones(NEV), torch.zeros(NEV)]).bool()
def states(m):
    X = block_states(m, Arch(m, fam), EV, [B])[B].reshape(-1, arch.D); keep = ~sinkmask(X); A, norms = rows_of(Arch(m, fam), B); return X, keep, A
X0, keep0, A0 = states(orig); U0 = unitr(X0[keep0] - X0[keep0].mean(0)); st0 = stats(U0, A0, K); w0 = torch.nonzero(wordset(st0["usage"]))[:, 0]; R0 = st0["ratio"].float(); pos_forget = is_forget_seq.repeat_interleave(T - 1)[keep0.cpu()]
CLS = {int(w): torch.nonzero(R0[:, w] > 1)[:, 0] for w in w0.tolist()}; fshare = {w: float(pos_forget[c].float().mean()) if c.numel() else 0.0 for w, c in CLS.items()}
fw = [w for w, s in fshare.items() if s >= 0.7 and CLS[w].numel() >= 5]; rw = [w for w, s in fshare.items() if s <= 0.3 and CLS[w].numel() >= 5]; log(f"{len(w0)} words: {len(fw)} forget words, {len(rw)} retain words (forget positions {float(pos_forget.float().mean()):.2f} of kept)")
L0f, L0r = lossof(orig, f_ev[:, :T]), lossof(orig, r_ev[:, :T]); log(f"original loss forget {L0f:.3f}, retain {L0r:.3f}")
def audit(m):
    X, keep, A = states(m); U = unitr(X[keep] - X[keep].mean(0)); st = stats(U, A, K); R = st["ratio"].float(); out = {}
    kidx0 = torch.nonzero(keep0.cpu())[:, 0]; posn = torch.nonzero(keep.cpu())[:, 0]; pmap = {int(p): i for i, p in enumerate(posn.tolist())}
    for tag, ws in (("forget", fw), ("retain", rw)):
        cos = [float(A0[w] @ A[w]) for w in ws]; writes = []
        for w in ws:
            full = kidx0[CLS[w]]; idx = torch.tensor([pmap[int(p)] for p in full.tolist() if int(p) in pmap]); writes.append(float((R[idx, w] > 1).float().mean() >= 0.5) if idx.numel() else 0.0)
        out[tag] = dict(row_cos=med(cos), row_cos_min=min(cos) if cos else None, share_still_writing=mean(writes), S=med([float(st["S"][w]) for w in ws]), still_word=mean([float(wordset(st["usage"])[w]) for w in ws]))
    return out
base_audit = audit(orig)
def where(m):
    """KL between the original and the unlearned model at forget-eval positions, with the unlearned model's blocks 0-6 under the original's 7-11 and the reverse"""
    ids = f_ev[:, :T]
    def logp(mod, xb):
        cap = {}
        def hk(mm, i, o):
            if hk.x is not None: return (hk.x,) + tuple(o[1:]) if isinstance(o, tuple) else hk.x
            cap["x"] = (o[0] if isinstance(o, tuple) else o).detach()
        hk.x = xb; h = Arch(mod, fam).layers[B].register_forward_hook(hk)
        try: lg = mod(ids).logits.float().log_softmax(-1)
        finally: h.remove()
        return lg, cap.get("x")
    lo, xo = logp(orig, None); lu, xu = logp(m, None); l_early, _ = logp(orig, xu); l_late, _ = logp(m, xo)
    kl = lambda a, b: float((a.exp() * (a - b)).sum(-1).mean())
    return dict(total=kl(lo, lu), early_blocks_only=kl(lo, l_early), late_blocks_only=kl(lo, l_late))
def unlearn(method, steps):
    m = copy.deepcopy(orig); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(steps)
    for p_ in m.parameters(): p_.requires_grad_(True)
    m.train()
    with torch.enable_grad():
        for s in range(steps):
            fb = f_tr[torch.randint(0, NTR, (8,), generator=g)]; rb = r_tr[torch.randint(0, NTR, (8,), generator=g)]
            lgf = m(fb[:, :T]).logits.float()[:, :-1].reshape(-1, model.config.vocab_size); tgt = fb[:, 1:T].reshape(-1)
            if method == "ga": lf = -torch.nn.functional.cross_entropy(lgf, tgt)
            else: lf = -(lgf.log_softmax(-1).mean(-1)).mean()   # toward the uniform distribution
            with torch.no_grad(): lo = orig(rb[:, :T]).logits.float()[:, :-1].log_softmax(-1)
            lr_ = m(rb[:, :T]).logits.float()[:, :-1].log_softmax(-1); kl = (lo.exp() * (lo - lr_)).sum(-1).mean(); loss = lf + LAM * kl
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    return m
def relearn(m, steps=30):
    m2 = copy.deepcopy(m); opt = torch.optim.AdamW(m2.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(7)
    for p_ in m2.parameters(): p_.requires_grad_(True)
    m2.train()
    with torch.enable_grad():
        for s in range(steps):
            fb = f_tr[torch.randint(0, NTR, (8,), generator=g)]; lg = m2(fb[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), fb[:, 1:T].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m2.eval()
    for p_ in m2.parameters(): p_.requires_grad_(False)
    return lossof(m2, f_ev[:, :T])
res = dict(n_words=len(w0), n_forget_words=len(fw), n_retain_words=len(rw), loss0=dict(forget=L0f, retain=L0r), base_audit=base_audit, conditions={})
for method in ("ga", "uniform"):
    for steps in (50, 100, 200, 400):
        m = unlearn(method, steps); Lf, Lr = lossof(m, f_ev[:, :T]), lossof(m, r_ev[:, :T]); au = audit(m); wh = where(m); Lre = relearn(m); rise = Lf - L0f; rec = (Lf - Lre) / rise if rise > 0.05 else None
        res["conditions"][f"{method}_{steps}"] = dict(method=method, steps=steps, loss_forget=Lf, loss_retain=Lr, rise_forget=rise, rise_retain=Lr - L0r, audit=au, where=wh, loss_after_relearn=Lre, recovery=rec)
        log(f"{method} {steps}: forget loss {L0f:.3f} -> {Lf:.3f} (retain {L0r:.3f} -> {Lr:.3f}); forget words' row cosine {au['forget']['row_cos']:.3f}, still writing {au['forget']['share_still_writing']:.2f}, still words {au['forget']['still_word']:.2f}; retain words cosine {au['retain']['row_cos']:.3f}, writing {au['retain']['share_still_writing']:.2f} (before {base_audit['retain']['share_still_writing']:.2f}); change lives in blocks 0-6 {wh['early_blocks_only']:.3f} vs 7-11 {wh['late_blocks_only']:.3f} of total {wh['total']:.3f}; relearn 30 steps -> {Lre:.3f} (recovery {rec}) | {time.time() - t0:.0f}s")
        del m; torch.cuda.empty_cache()
C = res["conditions"]; ok = [c for c in C.values() if c["recovery"] is not None]
def spear(x, y):
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64); return float(torch.corrcoef(torch.stack([x.argsort().argsort().double(), y.argsort().argsort().double()]))[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else None
res["summary"] = dict(n_conditions=len(ok), spearman_writing_recovery=spear([c["audit"]["forget"]["share_still_writing"] for c in ok], [c["recovery"] for c in ok]), spearman_cos_recovery=spear([c["audit"]["forget"]["row_cos"] for c in ok], [c["recovery"] for c in ok]), spearman_rise_recovery=spear([c["rise_forget"] for c in ok], [c["recovery"] for c in ok]))
Sm = res["summary"]; ga = C["ga_200"]
summ = (f"unlearning audit (Pythia-160m, {len(fw)} forget words / {len(rw)} retain words; before: forget words writing {base_audit['forget']['share_still_writing']:.2f}, retain {base_audit['retain']['share_still_writing']:.2f}): gradient ascent 200 steps raises the forget loss {L0f:.3f} -> {ga['loss_forget']:.3f} (retain {ga['rise_retain']:+.3f}); forget words' rows at cosine {ga['audit']['forget']['row_cos']:.3f}, still writing {ga['audit']['forget']['share_still_writing']:.2f}, still words {ga['audit']['forget']['still_word']:.2f}; retain words cosine {ga['audit']['retain']['row_cos']:.3f}, writing {ga['audit']['retain']['share_still_writing']:.2f}; the change lives in blocks 7-11 for {ga['where']['late_blocks_only'] / max(ga['where']['total'], 1e-9):.2f} of the KL and in 0-6 for {ga['where']['early_blocks_only'] / max(ga['where']['total'], 1e-9):.2f}; relearning 30 steps recovers {ga['recovery']}; "
        f"across {Sm['n_conditions']} conditions Spearman of still-writing with recovery {Sm['spearman_writing_recovery']}, of row cosine with recovery {Sm['spearman_cos_recovery']}, of the rise with recovery {Sm['spearman_rise_recovery']} | {time.time() - t0:.0f}s")
log(summ); record("e592_unlearning_audit", res, summ)
