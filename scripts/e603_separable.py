"""e603 (session 112): separable interventions. Session 111's hybrid (capped ascent, then the WDD writers' rows
zeroed) relearned slower than ascent alone on one seed per domain. Here, with three seeds and both domains on
Pythia-160m: capped ascent to +2 nats and RMU to +2 nats, each alone and each followed by (a) the forget words' write
columns zeroed (output side), (b) the forget words' input weights and biases zeroed, so the neurons never fire
(upstream suppression), (c) as many random non-word rows zeroed (control), (d) the difference-selected neurons'
inputs zeroed (the naive set, the selection control); plus the input-side silencing alone with no unlearning. For
each: the losses, the writers' activation ratio at their classes and the still-writing share, the probe, and the
attacks (20-step relearning recovery and steps back within 0.1 nats, 100 at most; benign relearning, 50 steps).
The prediction from session 111: silencing after RMU, which leaves the writers firing, buys more than silencing
after ascent, which already silences them in part. Arguments: domain seed.
Pre-registered (honest guesses):
 S1 (0.6) after RMU, input-side silencing of the writers adds 30 or more steps to the relearning against RMU alone
    (median over seeds), and after ascent 20 or more;
 S2 (0.5) the difference-selected set silenced after RMU adds less than the WDD writers silenced (the selection
    matters);
 S3 (0.7) random rows zeroed add fewer than 10 steps."""
from s101_common import *
from ma_common import Stop
from datasets import load_dataset
import copy, re, collections
name, DOMAIN, METHOD, SEED = "pythia160", sys.argv[1], "separable", int(sys.argv[2]); SMOKE = "--smoke" in sys.argv
t0 = time.time(); B = MID[name]; T = 256; NTR, NEV = (16, 8) if SMOKE else (96, 16); LR, LAM, MAXS, CHK = 2e-6, 5.0, (6 if SMOKE else 600), (3 if SMOKE else 5); TARGETS = (0.2,) if SMOKE else (1.0, 2.0, 4.0); BETA = 0.1 * 20 / (T - 1); RL, BN = (10, 5) if SMOKE else (100, 50); torch.set_grad_enabled(False)   # npo's beta scaled from the 20-token answers it was tuned on to these 255-token windows
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
    undefined = float(dmed[dec[fwt]].median()) < 1e-9
    tot = math.sqrt(sum(num.values()) / max(sum(den.values()), 1e-12))
    return dict(total=tot, groups={g_: math.sqrt(num[g_] / max(den[g_], 1e-12)) for g_ in num}, row_rel_change=dict(forget_words=med(rc[fwt].tolist()), retain_words=med(rc[rwt].tolist()), all_rows=med(rc.tolist()), forget_over_norm_matched=(None if undefined else med((rc[fwt] / dmed[dec[fwt]].clamp_min(1e-12)).tolist())), retain_over_norm_matched=(None if undefined else med((rc[rwt] / dmed[dec[rwt]].clamp_min(1e-12)).tolist()))))

# ---- the probe baseline (as e597)
TRW = torch.cat([f_tr[:NTR // 2], r_tr[:NTR // 2]])[:, :T]; Xtr_ = block_states(orig, Arch(orig, fam), TRW, [B])[B].reshape(-1, arch.D); ytr_ = torch.cat([torch.ones(NTR // 2), torch.zeros(NTR // 2)]).bool().repeat_interleave(T - 1).to(DEV); ktr = ~sinkmask(Xtr_)
gsub = torch.Generator().manual_seed(3); sub = torch.nonzero(ktr.cpu())[:, 0]; sub = sub[torch.randperm(sub.numel(), generator=gsub)[:12000]].to(DEV)
Xp, yp = Xtr_[sub], ytr_[sub]; del Xtr_
def probe(m):
    X, keep, _ = states(m); sc = logreg(Xp, yp, X[keep]); pred = sc > 0; pf = is_forget_seq.repeat_interleave(T - 1)[keep.cpu()].to(DEV)
    return dict(forget_acc=float(pred[pf].float().mean()), retain_acc=float((~pred[~pf]).float().mean()))
pr0 = probe(orig)
# ---- every neuron's mean activation at the forget and retain positions (the original's kept positions)
NB1 = B + 1; kept_full = keep0.cpu(); fpos = is_forget_seq.repeat_interleave(T - 1) & kept_full; rpos = (~is_forget_seq.repeat_interleave(T - 1)) & kept_full
def neuron_means(m):
    a2 = Arch(m, fam); MF = torch.zeros(NB1 * DFF, device=DEV); MR = torch.zeros(NB1 * DFF, device=DEV); hs = []; state = {"s0": 0}
    def mk(b):
        def hk(mod, inp):
            x = inp[0][:, 1:, :].reshape(-1, DFF).float(); n = x.shape[0]; s0 = state["s0"] * (T - 1); fm = fpos[s0:s0 + n].to(DEV); rm = rpos[s0:s0 + n].to(DEV)
            MF[b * DFF:(b + 1) * DFF] += x[fm].sum(0); MR[b * DFF:(b + 1) * DFF] += x[rm].sum(0)
        return hk
    for b in range(NB1):
        lin = a2.layers[b].mlp.dense_4h_to_h if fam == "neox" else (a2.layers[b].mlp.c_proj if fam == "gpt2" else a2.layers[b].mlp.down_proj); hs.append(lin.register_forward_pre_hook(mk(b)))
    try:
        for s0 in range(0, EV.shape[0], 8): state["s0"] = s0; m(EV[s0:s0 + 8])
    finally: [h.remove() for h in hs]
    return MF / max(int(fpos.sum()), 1), MR / max(int(rpos.sum()), 1)
MF0, MR0 = neuron_means(orig); nf = len(fw)
allw = set(w0.tolist()); gsel = torch.Generator().manual_seed(21 + SEED)
diff_sel = (MF0 - MR0).argsort(descending=True)[:nf].tolist(); ratio_sel = (MF0 / (MR0.abs() + 1e-3)).argsort(descending=True)[:nf].tolist(); mag_sel = MF0.argsort(descending=True)[:nf].tolist(); rand_sel = torch.randperm(NB1 * DFF, generator=gsel)[:nf].tolist()
SETS = {"wdd_forget": fw, "wdd_retain": rw, "diff_selected": diff_sel, "ratio_selected": ratio_sel, "magnitude_selected": mag_sel, "random": rand_sel}
jac = lambda a_, b_: len(set(a_) & set(b_)) / max(len(set(a_) | set(b_)), 1)
OVER = {k: jac(fw, v) for k, v in SETS.items() if k != "wdd_forget"}; OVER["diff_vs_magnitude"] = jac(diff_sel, mag_sel); OVER["diff_selected_among_words"] = mean([float(j in allw) for j in diff_sel]); OVER["magnitude_selected_among_words"] = mean([float(j in allw) for j in mag_sel])
log(f"neuron sets of {nf}: Jaccard with the WDD forget words: " + ", ".join(f"{k} {v:.2f}" for k, v in OVER.items()) + f"; forget words' mean activation on forget positions {float(MF0[fw].mean()):.3f} (retain positions {float(MR0[fw].mean()):.3f}), difference-selected {float(MF0[diff_sel].mean()):.3f} ({float(MR0[diff_sel].mean()):.3f})")
def set_ratios(m):
    MF, MR = neuron_means(m); out = {}
    for k, idx in SETS.items():
        it = torch.tensor(idx, device=DEV); r_ = MF[it] / MF0[it].abs().clamp_min(1e-4) * torch.sign(MF0[it]); out[k] = med(r_.tolist()); out[k + "_retain_positions"] = med((MR[it] / MR0[it].abs().clamp_min(1e-4) * torch.sign(MR0[it])).tolist())
    live = MF0.abs() > 1e-3; out["all_neurons"] = med((MF[live] / MF0[live].abs() * torch.sign(MF0[live])).tolist()); return out
# ---- the unlearning methods (as e597, with the seed in the batch order and RMU's direction)
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
    m = copy.deepcopy(orig); g = torch.Generator().manual_seed(int(target * 10) + 3 + 1000 * SEED); cap = L0f + target; steps = 0
    if method == "rmu":
        params = [Arch(m, fam).layers[b].mlp.dense_4h_to_h.weight for b in range(B - 2, B + 1)]; lr_ = 5e-5
        with torch.no_grad(): c = float(state_at_B(orig, f_tr[:8, :T]).norm(dim=-1).mean()); u = unitr(torch.randn(1, arch.D, generator=torch.Generator().manual_seed(7 + SEED)).to(DEV))[0] * c
    else: params = list(m.parameters()); lr_ = LR
    prep(m, params); opt = torch.optim.AdamW(params, lr=lr_, weight_decay=0.0)
    with torch.enable_grad():
        for s in range(MAXS if method != "drift" else (6 if SMOKE else 200)):
            fb = f_tr[torch.randint(0, NTR, (8,), generator=g)]; rb = r_tr[torch.randint(0, NTR, (8,), generator=g)]
            if method == "drift":
                lg = m(rb[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), rb[:, 1:T].reshape(-1))
            elif method == "rmu":
                hf_ = state_at_B(m, fb[:, :T]); hr_ = state_at_B(m, rb[:, :T])
                with torch.no_grad(): hr0 = state_at_B(orig, rb[:, :T])
                loss = ((hf_ - u) ** 2).sum(-1).mean() / c ** 2 + 20.0 * ((hr_ - hr0) ** 2).sum(-1).mean() / c ** 2
            else:
                lgf = m(fb[:, :T]).logits.float()[:, :-1].reshape(-1, model.config.vocab_size); tgt = fb[:, 1:T].reshape(-1); ce = torch.nn.functional.cross_entropy(lgf, tgt)
                if float(ce.detach()) >= cap + 0.5: lf = 0 * ce
                elif method in ("ga", "gd"): lf = -ce
                else:
                    lp = lgf.log_softmax(-1).gather(-1, tgt[:, None])[:, 0].reshape(8, -1).sum(-1)
                    with torch.no_grad(): lp0 = seqlogp(orig, fb)
                    lf = -(2 / BETA) * torch.nn.functional.logsigmoid(-BETA * (lp - lp0)).mean()
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
def importance(ids_tr, nb=2 if SMOKE else 24):
    m = copy.deepcopy(orig); prep(m, list(m.parameters())); D_ = {n: torch.zeros_like(p_) for n, p_ in m.named_parameters()}; g = torch.Generator().manual_seed(5 + SEED)
    with torch.enable_grad():
        for i in range(nb):
            b_ = ids_tr[torch.randint(0, NTR, (8,), generator=g)]; lg = m(b_[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), b_[:, 1:T].reshape(-1)); m.zero_grad(set_to_none=True); loss.backward()
            for n, p_ in m.named_parameters(): D_[n] += p_.grad.float().pow(2) / nb
    del m; torch.cuda.empty_cache(); return D_
def ssd_models():
    Df, Dr = importance(f_tr), importance(r_tr); out = {}; todo = list(TARGETS)
    for alpha in (300, 100, 50, 30, 20, 15, 10, 7, 5, 4, 3, 2.5, 2, 1.7, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8):
        m = copy.deepcopy(orig); nsel = 0; ntot = 0
        with torch.no_grad():
            for n, p_ in m.named_parameters():
                if p_.dim() != 2 or not ("attention" in n or "mlp" in n): continue
                eps = 0.01 * float(Dr[n].mean()); mask = Df[n] > alpha * (Dr[n] + eps); fac = torch.where(mask, ((Dr[n] + eps) / Df[n].clamp_min(1e-30)).clamp(min=0.1, max=1.0), torch.ones_like(p_)); p_.mul_(fac); nsel += int(mask.sum()); ntot += mask.numel()
        Lf = lossof(m, f_ev[:, :T]); Lr_ = lossof(m, r_ev[:, :T]); log(f"ssd alpha {alpha}: {nsel / ntot:.2e} of the weights dampened, forget loss {Lf:.3f}, retain {Lr_:.3f} ({time.time() - t0:.0f}s)")
        if Lr_ - L0r > 3.0: break
        while todo and Lf >= L0f + todo[0]: out[todo[0]] = (m, alpha, nsel / ntot); todo = todo[1:]
        if not todo: break
    del Df, Dr; torch.cuda.empty_cache(); return out
def relearn(m):
    m2 = copy.deepcopy(m); prep(m2, list(m2.parameters())); opt = torch.optim.AdamW(m2.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(7); back = None; L20 = None
    with torch.enable_grad():
        for s in range(RL):
            fb = f_tr[torch.randint(0, NTR, (8,), generator=g)]; lg = m2(fb[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), fb[:, 1:T].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if (s + 1) % CHK == 0:
                m2.eval()
                with torch.no_grad(): L = lossof(m2, f_ev[:, :T])
                m2.train()
                if s + 1 == 20 or (SMOKE and L20 is None): L20 = L
                if back is None and L <= L0f + 0.1: back = s + 1
                if back is not None and L20 is not None: break
    del m2; torch.cuda.empty_cache(); return L20, back
def benign(m, steps=BN):
    m2 = copy.deepcopy(m); prep(m2, list(m2.parameters())); opt = torch.optim.AdamW(m2.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(9)
    with torch.enable_grad():
        for s in range(steps):
            rb = r_tr[torch.randint(0, NTR, (8,), generator=g)]; lg = m2(rb[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), rb[:, 1:T].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m2.eval(); L = lossof(m2, f_ev[:, :T]); del m2; torch.cuda.empty_cache(); return L

def zero_rows(m, rows_, side="output"):
    a2 = Arch(m, fam)
    with torch.no_grad():
        for w in rows_:
            b_, j = w // DFF, w % DFF; l = a2.layers[b_].mlp
            if side == "output": l.dense_4h_to_h.weight[:, j] = 0
            else: l.dense_h_to_4h.weight[j, :] = 0; l.dense_h_to_4h.bias[j] = 0
    return m
nonword = [i for i in range(NB1 * DFF) if i not in allw]; grnd = torch.Generator().manual_seed(31 + SEED); rnd_rows = [nonword[int(i)] for i in torch.randperm(len(nonword), generator=grnd)[:len(fw)]]
res = dict(model=name, domain=DOMAIN, seed=SEED, n_forget_words=len(fw), n_retain_words=len(rw), loss0=dict(forget=L0f, retain=L0r), base_audit=base_audit, probe0=pr0, overlaps=OVER, conditions={})
def measure(m, key, extra=None):
    Lf, Lr = lossof(m, f_ev[:, :T]), lossof(m, r_ev[:, :T]); rise = Lf - L0f; au = audit(m); sr = set_ratios(m); pb = probe(m)
    L20, back = relearn(m); rec = (Lf - L20) / rise if (L20 is not None and rise > 0.05) else None; Lb = benign(m); rec_b = (Lf - Lb) / rise if rise > 0.05 else None
    cnd = dict(condition=key, seed=SEED, loss_forget=Lf, loss_retain=Lr, rise_forget=rise, rise_retain=Lr - L0r, audit=au, set_ratios=sr, probe=pb, loss_after_relearn20=L20, recovery=rec, steps_to_relearn=back, benign_forget=Lb, recovery_benign=rec_b)
    if extra: cnd.update(extra)
    res["conditions"][key] = cnd; a_ = au["forget"]
    log(f"{key}: forget {L0f:.3f} -> {Lf:.3f} (retain {Lr - L0r:+.3f}); writers' activation at their classes {a_['act_ratio'] if a_['act_ratio'] is None else round(a_['act_ratio'], 2)}, still writing {a_['share_still_writing']:.2f}; difference-selected activation {sr['diff_selected']:.2f}; probe {pb['forget_acc']:.3f}; relearn 20 -> {L20} (recovery {rec}), back after {back}; benign -> {Lb:.3f} (recovery {rec_b}) | {time.time() - t0:.0f}s")
    del m; torch.cuda.empty_cache()
measure(zero_rows(copy.deepcopy(orig), fw, "input"), "inputs_zero_alone", dict(n_rows=len(fw)))
for method in ("ga", "rmu"):
    base, steps = unlearn(method, 0.2 if SMOKE else 2.0); log(f"{method} to target in {steps} steps ({time.time() - t0:.0f}s)")
    measure(copy.deepcopy(base), f"{method}_alone", dict(steps=steps))
    measure(zero_rows(copy.deepcopy(base), fw, "output"), f"{method}_rows_zero", dict(steps=steps, n_rows=len(fw)))
    measure(zero_rows(copy.deepcopy(base), fw, "input"), f"{method}_inputs_zero", dict(steps=steps, n_rows=len(fw)))
    measure(zero_rows(copy.deepcopy(base), rnd_rows, "output"), f"{method}_random_rows_zero", dict(steps=steps, n_rows=len(rnd_rows)))
    measure(zero_rows(copy.deepcopy(base), diff_sel, "input"), f"{method}_diffselected_inputs_zero", dict(steps=steps, n_rows=len(diff_sel)))
    del base; torch.cuda.empty_cache()
C = res["conditions"]; bk = lambda k: C[k]["steps_to_relearn"]
summ = f"separable interventions ({DOMAIN}, seed {SEED}): steps back within 0.1 nats: " + ", ".join(f"{k} {bk(k)}" for k in C) + "; 20-step recovery: " + ", ".join(f"{k} {None if C[k]['recovery'] is None else round(C[k]['recovery'], 2)}" for k in C) + f" | {time.time() - t0:.0f}s"
log(summ); record(f"e603_separable_{DOMAIN}_s{SEED}" + ("_smoke" if SMOKE else ""), res, summ)
