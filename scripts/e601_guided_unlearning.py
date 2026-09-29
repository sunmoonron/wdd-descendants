"""e601 (session 111): WDD-guided unlearning, the question a reader of the audit asks next: if the audit names the rows
that write a domain's classes, can removing those rows unlearn the domain, and is that deeper than the training
methods? Pythia-160m block-6 words on PubMed Abstracts or Github as in e597. Conditions, each one shot: the forget
words' write columns zeroed (the rows the audit names); the same number of random non-word rows of blocks 0-6
zeroed; the retain words' rows zeroed (the collateral control); every row whose class is at least half forget
positions zeroed (the wider net); the forget words' rows zeroed and then 50 steps of retain fine-tuning (repair);
and capped gradient ascent to +2 nats followed by zeroing the forget words' rows (the hybrid). Measures as e597: the
losses, the audit, the probe, and the three attacks. The removed-speaker result (e590: thirty writers removed cost
their classes +0.005 nats) predicts the rows are not load-bearing enough to unlearn by removal.
Pre-registered (honest guesses):
 G1 (0.7) zeroing the forget words' rows raises the forget loss by less than 0.3 nats (the coalition carries the
    classes), against a retain change under 0.05;
 G2 (0.5) the wider net raises it by less than 1 nat;
 G3 (0.6) the hybrid relearns as fast as capped ascent alone (steps back within 20 of each other): removing the
    writers after silencing them buys no depth."""
from s101_common import *
from ma_common import Stop
from datasets import load_dataset
import copy, re, collections
name, DOMAIN, METHOD = sys.argv[1], sys.argv[2], "guided"
t0 = time.time(); B = MID[name]; T = 256; NTR, NEV = 96, 16; LR, LAM, MAXS, CHK = 2e-6, 5.0, 600, 5; TARGETS = (1.0, 2.0, 4.0); torch.set_grad_enabled(False)
model, tok, fam = load_model(name); arch = Arch(model, fam); orig = copy.deepcopy(model).eval(); DFF = arch.DFF
for p_ in orig.parameters(): p_.requires_grad_(False)
ds = load_dataset("NeelNanda/pile-10k", split="train")
FSET = {"pubmed": ("PubMed Abstracts", ("PubMed Abstracts", "PubMed Central")), "github": ("Github", ("Github",))}[DOMAIN]
FORGET = lambda s: s == FSET[0]; RETAIN = lambda s: s not in FSET[1]
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
f_tr = windows(FORGET, NTR); f_ev = windows(FORGET, NEV, skipdocs=300); r_tr = windows(RETAIN, NTR); r_ev = windows(RETAIN, NEV, skipdocs=600); log(f"{name} {DOMAIN} {METHOD}: windows forget {f_tr.shape[0]}/{f_ev.shape[0]}, retain {r_tr.shape[0]}/{r_ev.shape[0]} ({time.time() - t0:.0f}s)")
def lossof(m, ids, chunk=8):
    tot = 0.0
    for s0 in range(0, ids.shape[0], chunk):
        x = ids[s0:s0 + chunk]; lg = m(x).logits.float(); tot += float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))) * x.shape[0]; del lg
    return tot / ids.shape[0]
EV = torch.cat([f_ev, r_ev])[:, :T]; is_forget_seq = torch.cat([torch.ones(NEV), torch.zeros(NEV)]).bool()
def states(m):
    X = block_states(m, Arch(m, fam), EV, [B])[B].reshape(-1, arch.D); keep = ~sinkmask(X); A, norms = rows_of(Arch(m, fam), B); return X, keep, A
X0, keep0, A0 = states(orig); U0 = unitr(X0[keep0] - X0[keep0].mean(0)); st0 = stats(U0, A0, K); w0 = torch.nonzero(wordset(st0["usage"]))[:, 0]; R0 = st0["ratio"].float(); pos_forget = is_forget_seq.repeat_interleave(T - 1)[keep0.cpu()]
CLS = {int(w): torch.nonzero(R0[:, w] > 1)[:, 0] for w in w0.tolist()}; fshare = {w: float(pos_forget[c].float().mean()) if c.numel() else 0.0 for w, c in CLS.items()}
fw = [w for w, s in fshare.items() if s >= 0.7 and CLS[w].numel() >= 5]; rw = [w for w, s in fshare.items() if s <= 0.3 and CLS[w].numel() >= 5]; kidx0 = torch.nonzero(keep0.cpu())[:, 0]
log(f"{len(w0)} words: {len(fw)} forget words, {len(rw)} retain words (forget positions {float(pos_forget.float().mean()):.2f} of kept)")
L0f, L0r = lossof(orig, f_ev[:, :T]), lossof(orig, r_ev[:, :T]); log(f"original loss forget {L0f:.3f}, retain {L0r:.3f}")
# ---- the probe baseline: forget-against-retain on the original's block-B states of the training windows
TRW = torch.cat([f_tr[:48], r_tr[:48]])[:, :T]; Xtr_ = block_states(orig, Arch(orig, fam), TRW, [B])[B].reshape(-1, arch.D); ytr_ = torch.cat([torch.ones(48), torch.zeros(48)]).bool().repeat_interleave(T - 1).to(DEV); ktr = ~sinkmask(Xtr_)
gsub = torch.Generator().manual_seed(3); sub = torch.nonzero(ktr.cpu())[:, 0]; sub = sub[torch.randperm(sub.numel(), generator=gsub)[:12000]].to(DEV)
Xp, yp = Xtr_[sub], ytr_[sub]; del Xtr_
def probe(m):
    X, keep, _ = states(m); sc = logreg(Xp, yp, X[keep]); pred = sc > 0; pf = is_forget_seq.repeat_interleave(T - 1)[keep.cpu()].to(DEV)
    return dict(forget_acc=float(pred[pf].float().mean()), retain_acc=float((~pred[~pf]).float().mean()), forget_logit=float(sc[pf].mean()))
pr0 = probe(orig); log(f"probe on the original: forget positions classified forget {pr0['forget_acc']:.3f}, retain classified retain {pr0['retain_acc']:.3f}")
# ---- the direct activation of the words' neurons at their class positions
WS = fw + rw; byblock = {}
for w in WS: byblock.setdefault(w // DFF, []).append(w % DFF)
def acts(m):
    a2 = Arch(m, fam); cap = {}; hs = []
    for b, cols in byblock.items():
        colt = torch.tensor(cols, device=DEV)
        def mk(b, colt):
            def hk(mod, inp): cap.setdefault(b, []).append(inp[0][:, 1:, colt].detach().float())
            return hk
        lin = a2.layers[b].mlp.dense_4h_to_h if fam == "neox" else (a2.layers[b].mlp.c_proj if fam == "gpt2" else a2.layers[b].mlp.down_proj)
        hs.append(lin.register_forward_pre_hook(mk(b, colt)))
    try:
        for s0 in range(0, EV.shape[0], 8): m(EV[s0:s0 + 8])
    finally: [h.remove() for h in hs]
    out = {}
    for b, cols in byblock.items():
        Ab = torch.cat(cap[b]).reshape(-1, len(cols))
        for j, c in enumerate(cols):
            w = b * DFF + c; full = kidx0[CLS[w]].to(DEV); out[w] = float(Ab[full, j].mean())
    return out
act0 = acts(orig)
def audit(m):
    X, keep, A = states(m); U = unitr(X[keep] - X[keep].mean(0)); st = stats(U, A, K); R = st["ratio"].float(); out = {}
    posn = torch.nonzero(keep.cpu())[:, 0]; pmap = {int(p): i for i, p in enumerate(posn.tolist())}; am = acts(m)
    for tag, ws in (("forget", fw), ("retain", rw)):
        cos = [float(A0[w] @ A[w]) for w in ws]; writes = []; ratios = []
        for w in ws:
            full = kidx0[CLS[w]]; idx = torch.tensor([pmap[int(p)] for p in full.tolist() if int(p) in pmap]); writes.append(float((R[idx, w] > 1).float().mean() >= 0.5) if idx.numel() else 0.0)
            ratios.append(am[w] / act0[w] if abs(act0[w]) > 1e-6 else None)
        rr = [x for x in ratios if x is not None]
        out[tag] = dict(row_cos=med(cos), row_cos_min=min(cos) if cos else None, share_still_writing=mean(writes), S=med([float(st["S"][w]) for w in ws]), still_word=mean([float(wordset(st["usage"])[w]) for w in ws]), act_ratio=med(rr) if rr else None, share_act_halved=mean([float(x < 0.5) for x in rr]) if rr else None)
    return out
base_audit = audit(orig)
def where(m):
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
    half = f"0-{B}" if int(m_.group(1)) <= B else f"{B + 1}-{arch.NB - 1}"
    return ("attention " if "attention" in n else "mlp " if "mlp" in n else "layernorm ") + half
def rel_change(m):
    num, den = collections.defaultdict(float), collections.defaultdict(float); po = dict(orig.named_parameters())
    for n, p_ in m.named_parameters(): g_ = group(n); num[g_] += float((p_.float() - po[n].float()).pow(2).sum()); den[g_] += float(po[n].float().pow(2).sum())
    Ro = torch.cat([arch.wdir(b) for b in range(B + 1)]).float(); Ru = torch.cat([Arch(m, fam).wdir(b) for b in range(B + 1)]).float(); rc = (Ru - Ro).norm(dim=1) / Ro.norm(dim=1).clamp_min(1e-8)
    fwt, rwt = torch.tensor(fw, device=rc.device), torch.tensor(rw, device=rc.device); nrm = Ro.norm(dim=1); dec = (nrm.argsort().argsort().float() * 10 / nrm.numel()).long().clamp(max=9); dmed = torch.stack([rc[dec == d_].median() for d_ in range(10)])
    tot = math.sqrt(sum(num.values()) / max(sum(den.values()), 1e-12))
    return dict(total=tot, groups={g_: math.sqrt(num[g_] / max(den[g_], 1e-12)) for g_ in num}, row_rel_change=dict(forget_words=med(rc[fwt].tolist()), retain_words=med(rc[rwt].tolist()), all_rows=med(rc.tolist()), forget_over_norm_matched=med((rc[fwt] / dmed[dec[fwt]].clamp_min(1e-12)).tolist()), retain_over_norm_matched=med((rc[rwt] / dmed[dec[rwt]].clamp_min(1e-12)).tolist())))
def state_at_B(m, ids):
    cap = {}
    def hk(mm, i, o): cap["x"] = (o[0] if isinstance(o, tuple) else o); raise Stop
    h = Arch(m, fam).layers[B].register_forward_hook(hk)
    try: m(ids)
    except Stop: pass
    finally: h.remove()
    return cap["x"]
def seqlogp(m, ids):
    lg = m(ids[:, :T]).logits.float()[:, :-1].log_softmax(-1); return lg.gather(-1, ids[:, 1:T, None])[..., 0].sum(-1)
def prep(m, params):
    for p_ in m.parameters(): p_.requires_grad_(False)
    for p_ in params: p_.requires_grad_(True)
    m.train()
def unlearn(method, target):
    m = copy.deepcopy(orig); g = torch.Generator().manual_seed(int(target * 10) + 3); cap = L0f + target; steps = 0
    if method == "rmu":
        params = [Arch(m, fam).layers[b].mlp.dense_4h_to_h.weight for b in range(B - 2, B + 1)]; lr_ = 5e-5
        with torch.no_grad(): c = float(state_at_B(orig, f_tr[:8, :T]).norm(dim=-1).mean()); u = unitr(torch.randn(1, arch.D, generator=torch.Generator().manual_seed(7)).to(DEV))[0] * c
    else: params = list(m.parameters()); lr_ = LR
    prep(m, params); opt = torch.optim.AdamW(params, lr=lr_, weight_decay=0.0)
    with torch.enable_grad():
        for s in range(MAXS if method != "drift" else 200):
            fb = f_tr[torch.randint(0, NTR, (8,), generator=g)]; rb = r_tr[torch.randint(0, NTR, (8,), generator=g)]
            if method == "drift":
                lg = m(rb[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), rb[:, 1:T].reshape(-1))
            elif method == "rmu":
                hf_ = state_at_B(m, fb[:, :T]); hr_ = state_at_B(m, rb[:, :T])
                with torch.no_grad(): hr0 = state_at_B(orig, rb[:, :T])
                loss = ((hf_ - u) ** 2).sum(-1).mean() / c ** 2 + 20.0 * ((hr_ - hr0) ** 2).sum(-1).mean() / c ** 2
            else:
                lgf = m(fb[:, :T]).logits.float()[:, :-1].reshape(-1, model.config.vocab_size); tgt = fb[:, 1:T].reshape(-1); ce = torch.nn.functional.cross_entropy(lgf, tgt)
                if float(ce) >= cap + 0.5: lf = 0 * ce
                elif method in ("ga", "gd"): lf = -ce
                else:   # npo
                    lp = lgf.log_softmax(-1).gather(-1, tgt[:, None])[:, 0].reshape(8, -1).sum(-1)
                    with torch.no_grad(): lp0 = seqlogp(orig, fb)
                    lf = -(2 / 0.1) * torch.nn.functional.logsigmoid(-0.1 * (lp - lp0)).mean()
                if method == "gd":
                    lg = m(rb[:, :T]).logits.float(); loss = lf + torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), rb[:, 1:T].reshape(-1))
                else:
                    with torch.no_grad(): lo = orig(rb[:, :T]).logits.float()[:, :-1].log_softmax(-1)
                    lr2 = m(rb[:, :T]).logits.float()[:, :-1].log_softmax(-1); kl = (lo.exp() * (lo - lr2)).sum(-1).mean(); loss = lf + LAM * kl
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); steps = s + 1
            if method != "drift" and steps % CHK == 0:
                m.eval()
                with torch.no_grad(): Lf = lossof(m, f_ev[:, :T])
                m.train()
                if Lf >= cap: break
    m.eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    return m, steps
def importance(ids_tr, nb=24):
    m = copy.deepcopy(orig); prep(m, list(m.parameters())); D_ = {n: torch.zeros_like(p_) for n, p_ in m.named_parameters()}; g = torch.Generator().manual_seed(5)
    with torch.enable_grad():
        for i in range(nb):
            b_ = ids_tr[torch.randint(0, NTR, (8,), generator=g)]; lg = m(b_[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), b_[:, 1:T].reshape(-1)); m.zero_grad(set_to_none=True); loss.backward()
            for n, p_ in m.named_parameters(): D_[n] += p_.grad.float().pow(2) / nb
    del m; torch.cuda.empty_cache(); return D_
def ssd_models():
    Df, Dr = importance(f_tr), importance(r_tr); out = {}; po = dict(orig.named_parameters()); todo = list(TARGETS)
    for alpha in (300, 100, 50, 30, 20, 15, 10, 7, 5, 4, 3, 2.5, 2, 1.7, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8):
        m = copy.deepcopy(orig); nsel = 0; ntot = 0
        with torch.no_grad():
            for n, p_ in m.named_parameters():
                mask = Df[n] > alpha * Dr[n]; fac = torch.where(mask, (Dr[n] / Df[n].clamp_min(1e-30)).clamp(max=1.0), torch.ones_like(p_)); p_.mul_(fac); nsel += int(mask.sum()); ntot += mask.numel()
        Lf = lossof(m, f_ev[:, :T]); log(f"ssd alpha {alpha}: {nsel / ntot:.2e} of parameters dampened, forget loss {Lf:.3f} ({time.time() - t0:.0f}s)")
        while todo and Lf >= L0f + todo[0]: out[todo[0]] = (m, alpha, nsel / ntot); todo = todo[1:]
        if not todo: break
    del Df, Dr; torch.cuda.empty_cache(); return out
def relearn(m):
    m2 = copy.deepcopy(m); prep(m2, list(m2.parameters())); opt = torch.optim.AdamW(m2.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(7); back = None; L20 = None
    with torch.enable_grad():
        for s in range(100):
            fb = f_tr[torch.randint(0, NTR, (8,), generator=g)]; lg = m2(fb[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), fb[:, 1:T].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if (s + 1) % CHK == 0:
                m2.eval()
                with torch.no_grad(): L = lossof(m2, f_ev[:, :T])
                m2.train()
                if s + 1 == 20: L20 = L
                if back is None and L <= L0f + 0.1: back = s + 1
                if back is not None and L20 is not None: break
    del m2; torch.cuda.empty_cache(); return L20, back
def benign(m, steps=50):
    m2 = copy.deepcopy(m); prep(m2, list(m2.parameters())); opt = torch.optim.AdamW(m2.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(9)
    with torch.enable_grad():
        for s in range(steps):
            rb = r_tr[torch.randint(0, NTR, (8,), generator=g)]; lg = m2(rb[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), rb[:, 1:T].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m2.eval(); L = lossof(m2, f_ev[:, :T]); Lr = lossof(m2, r_ev[:, :T]); del m2; torch.cuda.empty_cache(); return L, Lr
def quantize(m, bits, group=64):
    """round-to-nearest symmetric quantisation of every linear weight: 4-bit in groups of 64 along the input dimension, 8-bit per output channel"""
    m2 = copy.deepcopy(m); qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for n, mod in m2.named_modules():
            if isinstance(mod, torch.nn.Linear):
                W = mod.weight; o_, i_ = W.shape
                if bits <= 4 and i_ % group == 0:
                    Wg = W.reshape(o_, i_ // group, group); sc = Wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / qmax; mod.weight.copy_(((Wg / sc).round().clamp(-qmax - 1, qmax) * sc).reshape(o_, i_))
                else:
                    sc = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax; mod.weight.copy_((W / sc).round().clamp(-qmax - 1, qmax) * sc)
    return m2
Q0 = {bits: lossof(quantize(orig, bits), f_ev[:, :T]) for bits in (4, 8)}; log(f"the original quantised: forget loss 4-bit {Q0[4]:.3f}, 8-bit {Q0[8]:.3f}")

res = dict(model=name, domain=DOMAIN, method="guided", n_words=len(w0), n_forget_words=len(fw), n_retain_words=len(rw), loss0=dict(forget=L0f, retain=L0r, forget_q4=Q0[4], forget_q8=Q0[8]), base_audit=base_audit, probe0=pr0, conditions={})
wide = [w for w, s_ in fshare.items() if s_ >= 0.5 and CLS[w].numel() >= 5]
def zero_rows(m, rows_, scale=0.0):
    a2 = Arch(m, fam)
    with torch.no_grad():
        for w in rows_:
            b_, j = w // DFF, w % DFF; lin = a2.layers[b_].mlp.dense_4h_to_h if fam == "neox" else a2.layers[b_].mlp.c_proj
            if fam == "neox": lin.weight[:, j] *= scale
            else: lin.weight[j, :] *= scale
    return m
def retain_ft(m, steps=50):
    prep(m, list(m.parameters())); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(11)
    with torch.enable_grad():
        for s in range(steps):
            rb = r_tr[torch.randint(0, NTR, (8,), generator=g)]; lg = m(rb[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), rb[:, 1:T].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    return m
gr = torch.Generator().manual_seed(21); nonword = [i for i in range(A0.shape[0]) if i not in set(w0.tolist())]; rnd = [nonword[int(i)] for i in torch.randperm(len(nonword), generator=gr)[:len(fw)]]
def measure(m, key, extra):
    Lf, Lr = lossof(m, f_ev[:, :T]), lossof(m, r_ev[:, :T]); rise = Lf - L0f; au = audit(m); wh = where(m); rcg = rel_change(m); pb = probe(m)
    L20, back = relearn(m); rec = (Lf - L20) / rise if (L20 is not None and rise > 0.05) else None; Lb, Lbr = benign(m); rec_b = (Lf - Lb) / rise if rise > 0.05 else None
    q = {}
    for bits in (4, 8):
        Lq = lossof(quantize(m, bits), f_ev[:, :T]); q[f"forget_q{bits}"] = Lq; q[f"recovery_q{bits}"] = (1 - (Lq - Q0[bits]) / rise) if rise > 0.05 else None
    sp_ = wh["early_blocks_only"] / max(wh["early_blocks_only"] + wh["late_blocks_only"], 1e-9)
    res["conditions"][key] = dict(method=key, loss_forget=Lf, loss_retain=Lr, rise_forget=rise, rise_retain=Lr - L0r, audit=au, where=wh, early_share=sp_, param_change=rcg, probe=pb, loss_after_relearn20=L20, recovery=rec, steps_to_relearn=back, benign_forget=Lb, benign_retain=Lbr, recovery_benign=rec_b, **q, **extra)
    a_ = au["forget"]; log(f"{key} ({extra}): forget {L0f:.3f} -> {Lf:.3f} ({rise:+.3f}; retain {Lr - L0r:+.3f}); forget words still writing {a_['share_still_writing']:.2f}, activation ratio {a_['act_ratio'] if a_['act_ratio'] is None else round(a_['act_ratio'], 2)}; retain words writing {au['retain']['share_still_writing']:.2f}; probe {pb['forget_acc']:.3f}; relearn 20 -> {L20} (recovery {rec}), back after {back}; benign -> {Lb:.3f} (recovery {rec_b}); q4/q8 recovery {q['recovery_q4']}/{q['recovery_q8']} | {time.time() - t0:.0f}s")
    del m; torch.cuda.empty_cache()
measure(zero_rows(copy.deepcopy(orig), fw), "forget_rows_zero", dict(n_rows=len(fw)))
measure(zero_rows(copy.deepcopy(orig), rnd), "random_rows_zero", dict(n_rows=len(rnd)))
measure(zero_rows(copy.deepcopy(orig), rw), "retain_rows_zero", dict(n_rows=len(rw)))
measure(zero_rows(copy.deepcopy(orig), wide), "wide_net_zero", dict(n_rows=len(wide)))
measure(retain_ft(zero_rows(copy.deepcopy(orig), fw)), "forget_rows_zero_repaired", dict(n_rows=len(fw)))
mga, steps = unlearn("ga", 2.0); Lga = lossof(mga, f_ev[:, :T]); L20g, backg = relearn(mga); res["ga_2_alone"] = dict(steps=steps, loss_forget=Lga, loss_after_relearn20=L20g, steps_to_relearn=backg, recovery=(Lga - L20g) / (Lga - L0f) if L20g is not None else None)
log(f"capped ascent alone: forget {Lga:.3f} after {steps} steps, relearn 20 -> {L20g}, back after {backg}")
measure(zero_rows(mga, fw), "ga_2_then_forget_rows_zero", dict(n_rows=len(fw), ga_steps=steps))
C = res["conditions"]; summ = f"guided unlearning ({name} {DOMAIN}, {len(fw)} forget words, wide net {len(wide)} rows): " + "; ".join(f"{k}: rise {c['rise_forget']:+.3f} (retain {c['rise_retain']:+.3f}), recovery {None if c['recovery'] is None else round(c['recovery'], 2)}, back {c['steps_to_relearn']}" for k, c in C.items()) + f"; capped ascent alone: back after {backg} | {time.time() - t0:.0f}s"
log(summ); record(f"e601_guided_unlearning_{name}_{DOMAIN}", res, summ)
