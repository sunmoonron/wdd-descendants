"""e613 (session 115): the auditor bake-off on the field's benchmark. Qwen2.5-0.5B fine-tuned on TOFU (4,000 question
and answer pairs about fictitious authors, three epochs), then the forget10 split unlearned by capped gradient
ascent, NPO and RMU (each with the retain90 split as the retain set, stopped at a forget-answer loss rise of 1 or 2
nats), two seeds. The ground truth of depth is the relearning attack (20 steps on the forget pairs, the share of the
rise recovered, and the steps until the forget loss is back within 0.1 nats, 100 at most), with benign relearning on
the retain pairs and an in-context attack (two forget pairs prepended) alongside. The auditors, each one forward pass
on the unlearned model: the WDD forget words' activation ratio at their classes (words on the fine-tuned model's
block-12 states at answer positions, forget words those whose class is at least 70% forget positions), the
difference- and ratio-selected neuron sets of the same size, a random set, the still-writing share, a logistic
forget-against-retain probe on the states, the representation drift (one minus the mean cosine between the
fine-tuned and the unlearned states at forget answer positions), the output KL at those positions, the relative
parameter change, and the forget-against-retain loss AUC (the loss-based membership test). The summary: the Spearman
of each auditor with the recovery over the conditions. Argument: --smoke.
Pre-registered (honest guesses):
 B1 (0.5) on TOFU the writers' activation correlates with the recovery at 0.7 or more over the twelve conditions,
    and the probe at under 0.5;
 B2 (0.6) RMU leaves the writers firing (ratio 0.95 or more) and is the shallowest method here as on the Pile;
 B3 (0.5) the in-context attack recovers more of RMU's rise than of ascent's."""
from s101_common import *
from ma_common import Stop
from datasets import load_dataset
import wdd_common, copy, re, collections
SMOKE = "--smoke" in sys.argv; name = "qwen05"; t0 = time.time(); B = 12; T = 48 if SMOKE else 96; NF, NR = (16, 16) if SMOKE else (400, 400); LR, LAM, MAXS, CHK = 2e-6, 5.0, (6 if SMOKE else 400), (3 if SMOKE else 5); TARGETS = (0.2,) if SMOKE else (1.0, 2.0); SEEDS = (0,) if SMOKE else (0, 1); BETA = 0.1 * 20 / 40; RL, BN = (10, 5) if SMOKE else (100, 50); FT_STEPS = 10 if SMOKE else 750; torch.set_grad_enabled(False)
model, tok, fam = load_model(name); arch = Arch(model, fam); DFF = arch.DFF; NB1 = B + 1; V = model.config.vocab_size
full = load_dataset("locuslab/TOFU", "full")["train"]; forget = load_dataset("locuslab/TOFU", "forget10")["train"]; retain = load_dataset("locuslab/TOFU", "retain90")["train"]
def enc(ds, n=None, skip=0):
    ids = torch.full((min(n or len(ds), len(ds) - skip), T), tok.pad_token_id or tok.eos_token_id, dtype=torch.long); lab = torch.full_like(ids, -100)
    for i, ex in enumerate(ds.select(range(skip, skip + ids.shape[0]))):
        q = tok(f"Question: {ex['question']}\nAnswer:")["input_ids"]; a = tok(f" {ex['answer']}")["input_ids"] + [tok.eos_token_id]; seq = (q + a)[:T]; ids[i, :len(seq)] = torch.tensor(seq); lab[i, len(q):len(seq)] = torch.tensor(seq[len(q):])
    return ids.to(DEV), lab.to(DEV)
ALL_ids, ALL_lab = enc(full, 64 if SMOKE else None); F_ids, F_lab = enc(forget, NF); R_ids, R_lab = enc(retain, NR); RT_ids, RT_lab = enc(retain, 512 if SMOKE else None, skip=NR)
log(f"TOFU: {ALL_ids.shape[0]} pairs to learn, forget {F_ids.shape[0]}, retain eval {R_ids.shape[0]}, retain train {RT_ids.shape[0]} ({time.time() - t0:.0f}s)")
def lossof(m, ids, lab, chunk=16):
    tot, n = 0.0, 0
    for s0 in range(0, ids.shape[0], chunk):
        x, y = ids[s0:s0 + chunk], lab[s0:s0 + chunk]; lg = m(x).logits.float()[:, :-1]; yy = y[:, 1:]; l = torch.nn.functional.cross_entropy(lg.reshape(-1, V), yy.reshape(-1), ignore_index=-100, reduction="sum"); tot += float(l); n += int((yy != -100).sum()); del lg
    return tot / max(n, 1)
def per_pair_loss(m, ids, lab, chunk=16):
    out = []
    for s0 in range(0, ids.shape[0], chunk):
        x, y = ids[s0:s0 + chunk], lab[s0:s0 + chunk]; lg = m(x).logits.float()[:, :-1].log_softmax(-1); yy = y[:, 1:]; mask = yy != -100; tl = -lg.gather(-1, yy.clamp_min(0)[..., None])[..., 0]; out.append((tl * mask).sum(1) / mask.sum(1).clamp_min(1)); del lg
    return torch.cat(out)
def prep(m, params):
    for p_ in m.parameters(): p_.requires_grad_(False)
    for p_ in params: p_.requires_grad_(True)
    m.train()
def train_steps(m, ids, lab, steps, lr, seed, bs=16):
    params = list(m.parameters()); prep(m, params); opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0); g = torch.Generator().manual_seed(seed)
    with torch.enable_grad():
        for s in range(steps):
            i = torch.randint(0, ids.shape[0], (bs,), generator=g); lg = m(ids[i]).logits.float()[:, :-1]; loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), lab[i][:, 1:].reshape(-1), ignore_index=-100); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    m.eval()
    for p_ in params: p_.requires_grad_(False)
    return m
L_pre = lossof(model, F_ids, F_lab); orig = train_steps(copy.deepcopy(model), ALL_ids, ALL_lab, FT_STEPS, 1e-5, 0); del model; torch.cuda.empty_cache(); arch = Arch(orig, fam)
L0f, L0r = lossof(orig, F_ids, F_lab), lossof(orig, R_ids, R_lab); log(f"fine-tuned: forget-answer loss {L_pre:.3f} -> {L0f:.3f}, retain {L0r:.3f} ({time.time() - t0:.0f}s)")
# ---- words and classes on the fine-tuned model's block-B states at answer positions
EV = torch.cat([F_ids, R_ids]); EVlab = torch.cat([F_lab, R_lab]); is_forget_seq = torch.cat([torch.ones(F_ids.shape[0]), torch.zeros(R_ids.shape[0])]).bool(); ans = (EVlab[:, 1:] != -100).reshape(-1).cpu()
def states(m):
    X = block_states(m, Arch(m, fam), EV, [B])[B].reshape(-1, arch.D); keep = (~sinkmask(X)).cpu() & ans; A, norms = rows_of(Arch(m, fam), B); return X, keep.to(DEV), A
X0, keep0, A0 = states(orig); U0 = unitr(X0[keep0] - X0[keep0].mean(0)); st0 = stats(U0, A0, K); w0 = torch.nonzero(wordset(st0["usage"]))[:, 0]; R0 = st0["ratio"].float(); pos_forget = is_forget_seq.repeat_interleave(T - 1)[keep0.cpu()]; kidx0 = torch.nonzero(keep0.cpu())[:, 0]
CLS = {int(w): torch.nonzero(R0[:, w] > 1)[:, 0] for w in w0.tolist()}; fshare = {w: float(pos_forget[c].float().mean()) if c.numel() else 0.0 for w, c in CLS.items()}
fw = [w for w, s_ in fshare.items() if s_ >= 0.7 and CLS[w].numel() >= 5]; rw = [w for w, s_ in fshare.items() if s_ <= 0.3 and CLS[w].numel() >= 5]; log(f"{len(w0)} words at answer positions: {len(fw)} forget words, {len(rw)} retain words (forget positions {float(pos_forget.float().mean()):.2f} of kept)")
fpos = is_forget_seq.repeat_interleave(T - 1) & keep0.cpu(); rpos = (~is_forget_seq.repeat_interleave(T - 1)) & keep0.cpu()
def neuron_means(m):
    a2 = Arch(m, fam); MF = torch.zeros(NB1 * DFF, device=DEV); MR = torch.zeros(NB1 * DFF, device=DEV); hs = []; state = {"s0": 0}
    def mk(b):
        def hk(mod, inp):
            x = inp[0][:, 1:, :].reshape(-1, DFF).float(); n = x.shape[0]; s0 = state["s0"] * (T - 1); MF[b * DFF:(b + 1) * DFF] += x[fpos[s0:s0 + n].to(DEV)].sum(0); MR[b * DFF:(b + 1) * DFF] += x[rpos[s0:s0 + n].to(DEV)].sum(0)
        return hk
    for b in range(NB1): hs.append(a2.layers[b].mlp.down_proj.register_forward_pre_hook(mk(b)))
    try:
        for s0 in range(0, EV.shape[0], 16): state["s0"] = s0; m(EV[s0:s0 + 16])
    finally: [h.remove() for h in hs]
    return MF / max(int(fpos.sum()), 1), MR / max(int(rpos.sum()), 1)
MF0, MR0 = neuron_means(orig); nf = max(len(fw), 1); diff_sel = (MF0 - MR0).argsort(descending=True)[:nf].tolist(); ratio_sel = (MF0 / (MR0.abs() + 1e-3)).argsort(descending=True)[:nf].tolist(); rand_sel = torch.randperm(NB1 * DFF, generator=torch.Generator().manual_seed(21))[:nf].tolist()
SETS = {"wdd_forget": fw, "diff_selected": diff_sel, "ratio_selected": ratio_sel, "random": rand_sel}
def set_ratios(m):
    MF, MR = neuron_means(m); out = {}
    for k, idx in SETS.items():
        if not idx: out[k] = None; continue
        it = torch.tensor(idx, device=DEV); out[k] = med((MF[it] / MF0[it].abs().clamp_min(1e-4) * torch.sign(MF0[it])).tolist())
    return out
byblock = {}
for w in fw: byblock.setdefault(w // DFF, []).append(w % DFF)
def acts(m):
    a2 = Arch(m, fam); cap = {}; hs = []
    for b, cols in byblock.items():
        colt = torch.tensor(cols, device=DEV)
        def mk(b, colt):
            def hk(mod, inp): cap.setdefault(b, []).append(inp[0][:, 1:, colt].detach().float())
            return hk
        hs.append(a2.layers[b].mlp.down_proj.register_forward_pre_hook(mk(b, colt)))
    try:
        for s0 in range(0, EV.shape[0], 16): m(EV[s0:s0 + 16])
    finally: [h.remove() for h in hs]
    out = {}
    for b, cols in byblock.items():
        Ab = torch.cat(cap[b]).reshape(-1, len(cols))
        for j, c in enumerate(cols): w = b * DFF + c; out[w] = float(Ab[kidx0[CLS[w]].to(DEV), j].mean())
    return out
act0 = acts(orig)
half = torch.zeros(X0.shape[0], dtype=torch.bool); half[:X0.shape[0] // 2] = True; gsub = torch.Generator().manual_seed(3); kk = torch.nonzero(keep0.cpu())[:, 0]; tr_idx = kk[(kk % 2 == 0)]; Xp, yp = X0[tr_idx.to(DEV)], is_forget_seq.repeat_interleave(T - 1)[tr_idx].to(DEV)
def audit(m):
    X, keep, A = states(m); U = unitr(X[keep] - X[keep].mean(0)); st = stats(U, A, K); R = st["ratio"].float(); posn = torch.nonzero(keep.cpu())[:, 0]; pmap = {int(p): i for i, p in enumerate(posn.tolist())}; am = acts(m)
    writes, ratios = [], []
    for w in fw:
        idx = torch.tensor([pmap[int(p)] for p in kidx0[CLS[w]].tolist() if int(p) in pmap]); writes.append(float((R[idx, w] > 1).float().mean() >= 0.5) if idx.numel() else 0.0); ratios.append(am[w] / act0[w] if abs(act0[w]) > 1e-6 else None)
    rr = [x for x in ratios if x is not None]; te_idx = kk[(kk % 2 == 1)].to(DEV); te_f = is_forget_seq.repeat_interleave(T - 1)[te_idx.cpu()].to(DEV)
    Xk = X[te_idx]; sc = logreg(Xp, yp, Xk); probe_acc = float((sc > 0)[te_f].float().mean())
    fsel = torch.nonzero(fpos)[:, 0].to(DEV); drift = 1 - float(torch.nn.functional.cosine_similarity(X0[fsel], X[fsel], dim=1).mean())
    return dict(act_ratio=med(rr) if rr else None, still_writing=mean(writes) if writes else None, set_ratios=set_ratios(m), probe=probe_acc, drift=drift)
def kl_forget(m):
    tot, n = 0.0, 0
    for s0 in range(0, F_ids.shape[0], 16):
        x, y = F_ids[s0:s0 + 16], F_lab[s0:s0 + 16]; lo = orig(x).logits.float()[:, :-1].log_softmax(-1); lu = m(x).logits.float()[:, :-1].log_softmax(-1); mask = (y[:, 1:] != -100); tot += float(((lo.exp() * (lo - lu)).sum(-1) * mask).sum()); n += int(mask.sum()); del lo, lu
    return tot / max(n, 1)
def param_change(m):
    po = dict(orig.named_parameters()); num = sum(float((p_.float() - po[n_].float()).pow(2).sum()) for n_, p_ in m.named_parameters()); den = sum(float(p_.float().pow(2).sum()) for p_ in orig.parameters()); return math.sqrt(num / max(den, 1e-12))
def mia_auc(m):
    lf, lr_ = per_pair_loss(m, F_ids, F_lab), per_pair_loss(m, R_ids, R_lab); return float((lf[:, None] > lr_[None, :]).float().mean())   # forget pairs look less memorised than retain pairs
def state_at_B(m, ids):
    cap = {}
    def hk(mm, i, o): cap["x"] = (o[0] if isinstance(o, tuple) else o); raise Stop
    h = Arch(m, fam).layers[B].register_forward_hook(hk)
    try: m(ids)
    except Stop: pass
    finally: h.remove()
    return cap["x"]
def unlearn(method, target, seed):
    m = copy.deepcopy(orig); g = torch.Generator().manual_seed(int(target * 10) + 3 + 1000 * seed); cap = L0f + target; steps = 0
    if method == "rmu":
        params = [Arch(m, fam).layers[b].mlp.down_proj.weight for b in range(B - 2, B + 1)]; lr_ = 5e-5
        with torch.no_grad(): c = float(state_at_B(orig, F_ids[:16]).norm(dim=-1).mean()); u = unitr(torch.randn(1, arch.D, generator=torch.Generator().manual_seed(7 + seed)).to(DEV))[0] * c
    else: params = list(m.parameters()); lr_ = LR
    prep(m, params); opt = torch.optim.AdamW(params, lr=lr_, weight_decay=0.0)
    with torch.enable_grad():
        for s in range(MAXS):
            fi = torch.randint(0, F_ids.shape[0], (8,), generator=g); ri = torch.randint(0, RT_ids.shape[0], (8,), generator=g); fb, flb, rb, rlb = F_ids[fi], F_lab[fi], RT_ids[ri], RT_lab[ri]
            if method == "rmu":
                hf_ = state_at_B(m, fb); hr_ = state_at_B(m, rb)
                with torch.no_grad(): hr0 = state_at_B(orig, rb)
                amask = (flb != -100).float()[..., None]; loss = (((hf_ - u) ** 2).sum(-1, keepdim=True) * amask).sum() / amask.sum().clamp_min(1) / c ** 2 + 20.0 * ((hr_ - hr0) ** 2).sum(-1).mean() / c ** 2
            else:
                lgf = m(fb).logits.float()[:, :-1]; yy = flb[:, 1:]; ce = torch.nn.functional.cross_entropy(lgf.reshape(-1, V), yy.reshape(-1), ignore_index=-100)
                if float(ce.detach()) >= cap + 0.5: lf = 0 * ce
                elif method == "ga": lf = -ce
                else:
                    mask = (yy != -100); lp = (lgf.log_softmax(-1).gather(-1, yy.clamp_min(0)[..., None])[..., 0] * mask).sum(1)
                    with torch.no_grad(): lo0 = orig(fb).logits.float()[:, :-1].log_softmax(-1); lp0 = (lo0.gather(-1, yy.clamp_min(0)[..., None])[..., 0] * mask).sum(1)
                    lf = -(2 / BETA) * torch.nn.functional.logsigmoid(-BETA * (lp - lp0)).mean()
                with torch.no_grad(): lo = orig(rb).logits.float()[:, :-1].log_softmax(-1)
                lr2 = m(rb).logits.float()[:, :-1].log_softmax(-1); rmask = (rlb[:, 1:] != -100).float(); kl = ((lo.exp() * (lo - lr2)).sum(-1) * rmask).sum() / rmask.sum().clamp_min(1); loss = lf + LAM * kl
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); steps = s + 1
            if steps % CHK == 0:
                m.eval()
                with torch.no_grad(): Lf = lossof(m, F_ids, F_lab)
                m.train()
                if Lf >= cap: break
    m.eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    return m, steps
def relearn(m):
    m2 = copy.deepcopy(m); params = list(m2.parameters()); prep(m2, params); opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(7); back = None; L20 = None
    with torch.enable_grad():
        for s in range(RL):
            i = torch.randint(0, F_ids.shape[0], (8,), generator=g); lg = m2(F_ids[i]).logits.float()[:, :-1]; loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), F_lab[i][:, 1:].reshape(-1), ignore_index=-100); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if (s + 1) % CHK == 0:
                m2.eval()
                with torch.no_grad(): L = lossof(m2, F_ids, F_lab)
                m2.train()
                if s + 1 == 20 or (SMOKE and L20 is None): L20 = L
                if back is None and L <= L0f + 0.1: back = s + 1
                if back is not None and L20 is not None: break
    del m2; torch.cuda.empty_cache(); return L20, back
def benign(m):
    m2 = train_steps(copy.deepcopy(m), RT_ids, RT_lab, BN, LR, 9, bs=8); L = lossof(m2, F_ids, F_lab); del m2; torch.cuda.empty_cache(); return L
PRE = 2 if SMOKE else 2
def incontext_loss(m):
    """the forget-answer loss with two other forget pairs prepended"""
    g = torch.Generator().manual_seed(23); tot, n = 0.0, 0
    for s0 in range(0, F_ids.shape[0], 8):
        x, y = F_ids[s0:s0 + 8], F_lab[s0:s0 + 8]; j = torch.randint(0, F_ids.shape[0], (x.shape[0], PRE), generator=g); pref = torch.cat([F_ids[j[:, k]] for k in range(PRE)], 1); full_ = torch.cat([pref, x], 1); lg = m(full_).logits.float()[:, pref.shape[1]:-1]; yy = y[:, 1:]; l = torch.nn.functional.cross_entropy(lg.reshape(-1, V), yy.reshape(-1), ignore_index=-100, reduction="sum"); tot += float(l); n += int((yy != -100).sum()); del lg
    return tot / max(n, 1)
L0_ctx = incontext_loss(orig); base_audit = audit(orig); log(f"fine-tuned model: writers at 1.00 by definition, probe {base_audit['probe']:.3f}, in-context forget loss {L0_ctx:.3f}, MIA AUC {mia_auc(orig):.3f} ({time.time() - t0:.0f}s)")
res = dict(model=name, benchmark="TOFU forget10", n_forget_words=len(fw), n_retain_words=len(rw), loss0=dict(forget_before_finetune=L_pre, forget=L0f, retain=L0r, forget_in_context=L0_ctx), probe0=base_audit["probe"], mia0=mia_auc(orig), conditions={})
for seed in SEEDS:
    for method in ("ga", "npo", "rmu"):
        for target in TARGETS:
            m, steps = unlearn(method, target, seed); Lf, Lr = lossof(m, F_ids, F_lab), lossof(m, R_ids, R_lab); rise = Lf - L0f; au = audit(m); klv = kl_forget(m); pc = param_change(m); mia = mia_auc(m)
            L20, back = relearn(m); rec = (Lf - L20) / rise if (L20 is not None and rise > 0.05) else None; Lb = benign(m); rec_b = (Lf - Lb) / rise if rise > 0.05 else None; Lc = incontext_loss(m); rec_c = 1 - (Lc - L0_ctx) / rise if rise > 0.05 else None
            key = f"{method}_{target:g}_s{seed}"; res["conditions"][key] = dict(method=method, target=target, seed=seed, steps=steps, loss_forget=Lf, rise_forget=rise, rise_retain=Lr - L0r, act_ratio=au["act_ratio"], still_writing=au["still_writing"], set_ratios=au["set_ratios"], probe=au["probe"], drift=au["drift"], kl=klv, param_change=pc, mia_auc=mia, recovery=rec, steps_to_relearn=back, recovery_benign=rec_b, recovery_in_context=rec_c)
            log(f"{key} ({steps} steps): forget {L0f:.3f} -> {Lf:.3f} (retain {Lr - L0r:+.3f}); writers {au['act_ratio'] if au['act_ratio'] is None else round(au['act_ratio'], 2)}, difference-selected {au['set_ratios']['diff_selected'] if au['set_ratios']['diff_selected'] is None else round(au['set_ratios']['diff_selected'], 2)}, random {au['set_ratios']['random'] if au['set_ratios']['random'] is None else round(au['set_ratios']['random'], 2)}, still writing {au['still_writing']}, probe {au['probe']:.3f}, drift {au['drift']:.3f}, KL {klv:.3f}, params {pc:.2e}, MIA {mia:.3f}; relearn recovery {rec}, back {back}; benign {rec_b}; in-context {rec_c} | {time.time() - t0:.0f}s")
            del m; torch.cuda.empty_cache()
def spear(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 4: return None
    xx, yy = torch.tensor([p[0] for p in pairs], dtype=torch.float64), torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    return None if xx.std() == 0 or yy.std() == 0 else float(torch.corrcoef(torch.stack([xx.argsort().argsort().double(), yy.argsort().argsort().double()]))[0, 1])
C = list(res["conditions"].values()); AUD = {"writers": lambda c: c["act_ratio"], "diff_selected": lambda c: c["set_ratios"]["diff_selected"], "ratio_selected": lambda c: c["set_ratios"]["ratio_selected"], "random": lambda c: c["set_ratios"]["random"], "still_writing": lambda c: c["still_writing"], "probe": lambda c: c["probe"], "drift": lambda c: c["drift"], "kl": lambda c: c["kl"], "param_change": lambda c: c["param_change"], "mia_auc": lambda c: c["mia_auc"], "rise_forget": lambda c: c["rise_forget"]}
res["spearman"] = {o: {k: spear([f(c) for c in C], [c[o] for c in C]) for k, f in AUD.items()} for o in ("recovery", "steps_to_relearn", "recovery_benign", "recovery_in_context")}
res["by_method"] = {m: {k: (med([f(c) for c in C if c["method"] == m and f(c) is not None]) if any(f(c) is not None for c in C if c["method"] == m) else None) for k, f in AUD.items()} | {o: (med([c[o] for c in C if c["method"] == m and c[o] is not None]) if any(c[o] is not None for c in C if c["method"] == m) else None) for o in ("recovery", "steps_to_relearn", "recovery_benign", "recovery_in_context")} for m in ("ga", "npo", "rmu")}
S = res["spearman"]; summ = (f"TOFU bake-off (Qwen2.5-0.5B, forget10, {len(C)} conditions, {len(fw)} forget words): Spearman with the 20-step recovery: " + ", ".join(f"{k} {None if S['recovery'][k] is None else round(S['recovery'][k], 2)}" for k in AUD) + "; with the in-context recovery: " + ", ".join(f"{k} {None if S['recovery_in_context'][k] is None else round(S['recovery_in_context'][k], 2)}" for k in ("writers", "diff_selected", "probe", "drift", "kl")) + "; by method: " + "; ".join(f"{m}: writers {None if b['writers'] is None else round(b['writers'], 2)}, probe {None if b['probe'] is None else round(b['probe'], 2)}, recovery {None if b['recovery'] is None else round(b['recovery'], 2)}, in-context {None if b['recovery_in_context'] is None else round(b['recovery_in_context'], 2)}" for m, b in res["by_method"].items()) + f" | {time.time() - t0:.0f}s")
log(summ); record("e613_tofu_bakeoff" + ("_smoke" if SMOKE else ""), res, summ)
