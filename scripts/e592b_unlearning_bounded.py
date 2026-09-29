"""e592b (session 110): the unlearning audit of e592 redone with a bounded protocol. e592's gradient ascent ran away
(forget loss 2.8 -> 137-222 nats, retain 3.1 -> 31-65, in fifty steps at lr 1e-5): a destroyed model, whose audit means
nothing. Here the forget term is switched off on any batch whose forget loss already exceeds the target, the retain
term is a KL to the original with weight 5, the learning rate is 2e-6, and each condition stops when the held-out
forget loss reaches its target rise (0.5, 1, 2 or 4 nats above the original; 600 steps at most), so that conditions are
matched on how much was forgotten and the retain loss is reported as the price. Methods: capped gradient ascent and
training toward the uniform distribution (eight conditions); a drift control fine-tunes 200 steps on the retain windows
alone (no forget term) at the same learning rate. The audit as e592 (the forget words' rows' cosine with their
originals, the share still over the floor at half their class positions on the unlearned model's states, the same for
retain words; the KL split between the unlearned model's blocks 0-6 under the original's 7-11 and the reverse) plus
where the change lives in parameter space: the relative change of the forget words' rows against every MLP row of
blocks 0-6, and per parameter group. Relearning: 20 steps at the same learning rate on the forget training windows,
the share of the rise recovered, and the steps until the held-out forget loss is back within 0.1 nats (100 at most).
Pre-registered (probabilities are honest guesses):
 U1 (0.6) at the 2-nat target the forget words' rows keep cosine 0.99 or more and at least 0.6 still write over the
    floor at their classes: the unlearning acts elsewhere;
 U2 (0.5) across the eight conditions the share of forget words still writing correlates with the recovery at
    Spearman 0.5 or more;
 U3 (0.7) retain words are unchanged at the 2-nat target (cosine 0.99 or more, the writing share within 0.1);
 U4 (0.6) the forget words' rows change no more than MLP rows of the same norm (norm-matched ratio within 0.8-1.25; the
    raw ratio is confounded, since Adam moves every element by about the same amount and the words are high-norm rows);
 U5 (0.5) the change lives in blocks 0-6 for more than half of the KL split at the 2-nat target."""
from s101_common import *
from datasets import load_dataset
import copy, re, collections
t0 = time.time(); B = 6; T = 256; NTR, NEV = 96, 16; LR, LAM, MAXS, CHK = 2e-6, 5.0, 600, 5; torch.set_grad_enabled(False)
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

def group(n):
    if "embed_in" in n: return "embeddings"
    if "embed_out" in n: return "unembedding"
    m_ = re.search(r"layers\.(\d+)\.", n)
    if m_ is None: return "other"
    half = "0-6" if int(m_.group(1)) <= B else "7-11"
    return ("attention " if "attention" in n else "mlp " if "mlp" in n else "layernorm ") + half
def rel_change(m):
    num, den = collections.defaultdict(float), collections.defaultdict(float); po = dict(orig.named_parameters())
    for n, p_ in m.named_parameters(): g_ = group(n); num[g_] += float((p_.float() - po[n].float()).pow(2).sum()); den[g_] += float(po[n].float().pow(2).sum())
    Ro = torch.cat([arch.wdir(b) for b in range(B + 1)]).float(); Ru = torch.cat([Arch(m, fam).wdir(b) for b in range(B + 1)]).float(); rc = (Ru - Ro).norm(dim=1) / Ro.norm(dim=1).clamp_min(1e-8)
    rank = rc.argsort().argsort().float() / (rc.numel() - 1); fwt, rwt = torch.tensor(fw, device=rc.device), torch.tensor(rw, device=rc.device); fr, rr = rc[fwt], rc[rwt]
    nrm = Ro.norm(dim=1); ac = (Ru - Ro).norm(dim=1); dec = (nrm.argsort().argsort().float() * 10 / nrm.numel()).long().clamp(max=9); dmed = torch.stack([rc[dec == d_].median() for d_ in range(10)]); nrank = nrm.argsort().argsort().float() / (nrm.numel() - 1)
    return dict(groups={g_: math.sqrt(num[g_] / max(den[g_], 1e-12)) for g_ in num}, row_rel_change=dict(forget_words=med(fr.tolist()), retain_words=med(rr.tolist()), all_rows=med(rc.tolist()), forget_over_all=med(fr.tolist()) / max(med(rc.tolist()), 1e-12), forget_percentile=med(rank[fwt].tolist()),
                forget_over_norm_matched=med((rc[fwt] / dmed[dec[fwt]].clamp_min(1e-12)).tolist()), retain_over_norm_matched=med((rc[rwt] / dmed[dec[rwt]].clamp_min(1e-12)).tolist()), forget_abs_over_all=med(ac[fwt].tolist()) / max(med(ac.tolist()), 1e-12), forget_norm_percentile=med(nrank[fwt].tolist()), retain_over_all=med(rr.tolist()) / max(med(rc.tolist()), 1e-12)))
def unlearn(method, target):
    m = copy.deepcopy(orig); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(int(target * 10) + 3); cap = L0f + target; steps = 0; Lf = L0f
    for p_ in m.parameters(): p_.requires_grad_(True)
    m.train()
    with torch.enable_grad():
        for s in range(MAXS if method != "retain_only" else 200):
            fb = f_tr[torch.randint(0, NTR, (8,), generator=g)]; rb = r_tr[torch.randint(0, NTR, (8,), generator=g)]
            if method == "retain_only":
                lg = m(rb[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), rb[:, 1:T].reshape(-1))
            else:
                lgf = m(fb[:, :T]).logits.float()[:, :-1].reshape(-1, model.config.vocab_size); tgt = fb[:, 1:T].reshape(-1); ce = torch.nn.functional.cross_entropy(lgf, tgt)
                if float(ce) >= cap + 0.5: lf = 0 * ce
                elif method == "ga": lf = -ce
                else: lf = -(lgf.log_softmax(-1).mean(-1)).mean()
                with torch.no_grad(): lo = orig(rb[:, :T]).logits.float()[:, :-1].log_softmax(-1)
                lr_ = m(rb[:, :T]).logits.float()[:, :-1].log_softmax(-1); kl = (lo.exp() * (lo - lr_)).sum(-1).mean(); loss = lf + LAM * kl
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); steps = s + 1
            if method != "retain_only" and steps % CHK == 0:
                with torch.no_grad(): Lf = lossof(m, f_ev[:, :T])
                if Lf >= cap: break
    m.eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    return m, steps
def relearn(m, Lf):
    m2 = copy.deepcopy(m); opt = torch.optim.AdamW(m2.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(7); back = None; L20 = None
    for p_ in m2.parameters(): p_.requires_grad_(True)
    m2.train()
    with torch.enable_grad():
        for s in range(100):
            fb = f_tr[torch.randint(0, NTR, (8,), generator=g)]; lg = m2(fb[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), fb[:, 1:T].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if (s + 1) % CHK == 0:
                with torch.no_grad(): L = lossof(m2, f_ev[:, :T])
                if s + 1 == 20: L20 = L
                if back is None and L <= L0f + 0.1: back = s + 1
                if back is not None and L20 is not None: break
    return L20, back
res = dict(n_words=len(w0), n_forget_words=len(fw), n_retain_words=len(rw), loss0=dict(forget=L0f, retain=L0r), base_audit=base_audit, lr=LR, kl_weight=LAM, conditions={})
CONDS = [(mth, tg) for mth in ("ga", "uniform") for tg in (0.5, 1.0, 2.0, 4.0)] + [("retain_only", 0.0)]
for method, target in CONDS:
    m, steps = unlearn(method, target); Lf, Lr = lossof(m, f_ev[:, :T]), lossof(m, r_ev[:, :T]); au = audit(m); wh = where(m); rcg = rel_change(m); rise = Lf - L0f
    L20, back = relearn(m, Lf) if method != "retain_only" else (None, None); rec = (Lf - L20) / rise if (L20 is not None and rise > 0.05) else None; sp_ = wh["early_blocks_only"] / max(wh["early_blocks_only"] + wh["late_blocks_only"], 1e-9)
    name = f"{method}_{target:g}" if method != "retain_only" else "retain_only"
    res["conditions"][name] = dict(method=method, target=target, steps=steps, loss_forget=Lf, loss_retain=Lr, rise_forget=rise, rise_retain=Lr - L0r, audit=au, where=wh, early_share=sp_, param_change=rcg, loss_after_relearn20=L20, recovery=rec, steps_to_relearn=back)
    rr_ = rcg["row_rel_change"]; gg = rcg["groups"]
    log(f"{name} ({steps} steps): forget loss {L0f:.3f} -> {Lf:.3f} (retain {L0r:.3f} -> {Lr:.3f}); forget words' row cosine {au['forget']['row_cos']:.4f}, still writing {au['forget']['share_still_writing']:.2f}, still words {au['forget']['still_word']:.2f}; retain words cosine {au['retain']['row_cos']:.4f}, writing {au['retain']['share_still_writing']:.2f} (before {base_audit['retain']['share_still_writing']:.2f}); KL split early {sp_:.2f} (0-6 {wh['early_blocks_only']:.3f}, 7-11 {wh['late_blocks_only']:.3f}, total {wh['total']:.3f}); row change forget {rr_['forget_words']:.2e} / all {rr_['all_rows']:.2e} (ratio {rr_['forget_over_all']:.2f}, percentile {rr_['forget_percentile']:.2f}; norm-matched ratio {rr_['forget_over_norm_matched']:.2f}, retain words {rr_['retain_over_norm_matched']:.2f}; absolute change ratio {rr_['forget_abs_over_all']:.2f}; the forget rows' norm percentile {rr_['forget_norm_percentile']:.2f}); groups " + ", ".join(f"{k} {v:.2e}" for k, v in sorted(gg.items())) + f"; relearn 20 steps -> {L20} (recovery {rec}), back within 0.1 after {back} steps | {time.time() - t0:.0f}s")
    del m; torch.cuda.empty_cache()
C = res["conditions"]; ok = [c for c in C.values() if c["recovery"] is not None]
def spear(x, y):
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64); return float(torch.corrcoef(torch.stack([x.argsort().argsort().double(), y.argsort().argsort().double()]))[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else None
res["summary"] = dict(n_conditions=len(ok), spearman_writing_recovery=spear([c["audit"]["forget"]["share_still_writing"] for c in ok], [c["recovery"] for c in ok]), spearman_cos_recovery=spear([c["audit"]["forget"]["row_cos"] for c in ok], [c["recovery"] for c in ok]), spearman_rise_recovery=spear([c["rise_forget"] for c in ok], [c["recovery"] for c in ok]),
                      spearman_writing_back=spear([c["audit"]["forget"]["share_still_writing"] for c in ok if c["steps_to_relearn"] is not None], [c["steps_to_relearn"] for c in ok if c["steps_to_relearn"] is not None]))
Sm = res["summary"]; ga = C["ga_2"]; rr_ = ga["param_change"]["row_rel_change"]
summ = (f"unlearning audit, bounded (Pythia-160m, {len(fw)} forget words / {len(rw)} retain words; lr {LR:g}, KL weight {LAM:g}): capped gradient ascent to +2 nats takes {ga['steps']} steps (forget {L0f:.3f} -> {ga['loss_forget']:.3f}, retain {ga['rise_retain']:+.3f}); forget words' rows at cosine {ga['audit']['forget']['row_cos']:.4f}, still writing {ga['audit']['forget']['share_still_writing']:.2f} (before {base_audit['forget']['share_still_writing']:.2f}), still words {ga['audit']['forget']['still_word']:.2f}; retain words cosine {ga['audit']['retain']['row_cos']:.4f}, writing {ga['audit']['retain']['share_still_writing']:.2f} (before {base_audit['retain']['share_still_writing']:.2f}); the forget rows change {rr_['forget_over_all']:.2f} times the median row (percentile {rr_['forget_percentile']:.2f}; norm-matched {rr_['forget_over_norm_matched']:.2f}, absolute {rr_['forget_abs_over_all']:.2f}); KL split early {ga['early_share']:.2f}; relearn recovery {ga['recovery']}, back after {ga['steps_to_relearn']} steps; drift control: forget words writing {C['retain_only']['audit']['forget']['share_still_writing']:.2f}, retain {C['retain_only']['audit']['retain']['share_still_writing']:.2f}; "
        f"across {Sm['n_conditions']} conditions Spearman of still-writing with recovery {Sm['spearman_writing_recovery']}, of row cosine {Sm['spearman_cos_recovery']}, of the rise {Sm['spearman_rise_recovery']}, of still-writing with steps back {Sm['spearman_writing_back']} | {time.time() - t0:.0f}s")
log(summ); record("e592b_unlearning_bounded", res, summ)
