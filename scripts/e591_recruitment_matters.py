"""e591 (session 110): does recruitment matter for performance? Every class is unspoken for checkpoints before its row
arrives (e585). If the model's loss at the class's positions improves when the row is recruited, beyond what
token-matched positions do at the same time, recruitment is a performance event; if not, the word is a symptom. From
e582's cache (Pythia-410m block 12, steps 1000-16000, 24 x 512 tokens): the eventual words at 16000 with classes of
ten or more positions and not words at 1000, their recruitment checkpoint (the first from which the row is a word
over the floor and stays one); at every checkpoint the per-position next-token loss of the checkpoint model. Aligned
on recruitment (k = -4..+4): the mean loss at the class's positions against (a) token-matched control positions
outside every tracked class, (b) the same positions aligned on a random pseudo-recruitment (time-shuffled), (c) all
positions. Statistics per word: the drop across recruitment (k = -1 to +1) at the class minus at its controls, and
the acceleration at recruitment (that drop minus the mean drop of the neighbouring intervals); medians with bootstrap
intervals over words. The functional test: at k = -2..+2 the row's direction at that checkpoint removed from the
block-12 state at the class's positions, the KL of the change there, against a random unit direction removed at the
same positions. Argument: pythia410 (the full test) or olmo1b (the loss-aligned part from e550's cache, which stores
the per-position loss). Pre-registered (probabilities are honest guesses):
 P1 (0.4) the class's loss drops across recruitment by 0.02 nats or more beyond its token-matched controls;
 P2 (0.5) the removal effect of the row's direction at the class rises across recruitment by a factor of two or more
    while the random direction's stays flat;
 P3 (0.5) the acceleration at recruitment is positive for most words."""
from s101_common import *
name = sys.argv[1] if len(sys.argv) > 1 else "pythia410"; t0 = time.time(); steps = list(range(1000, 16001, 1000)); T = len(steps); g = torch.Generator().manual_seed(0)
if name == "pythia410":
    CD = "/workspace/wdd/cache/e582_pythia410"; B = 12; ids = eval_ids("pythia410")[:24, :512].to(DEV); ck = {n: torch.load(f"{CD}/step{n}.pt", map_location="cpu") for n in steps}
    keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0); U = {n: unitr((ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV)) for n in steps}; A = {n: ck[n]["rows"].float().to(DEV) for n in steps}
    tok_all = ids[:, 1:].reshape(-1).cpu(); tok = tok_all[keepc]
    LOSS = {}
    for n in steps:
        m, _, fam = load_model("pythia410", revision=f"step{n}")
        with torch.no_grad(): lg = m(ids).logits.float(); l = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="none")
        LOSS[n] = l.cpu()[keepc]; del m, lg; torch.cuda.empty_cache()
    log(f"losses at {T} checkpoints ({time.time() - t0:.0f}s)")
else:
    CD = "/workspace/wdd/cache/e550_olmo1b"; B = 8; ck = {n: torch.load(f"{CD}/step{n}.pt", map_location="cpu") for n in steps}; keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0)
    U = {n: unitr((ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV)) for n in steps}; A = {n: ck[n]["rows"].float().to(DEV) for n in steps}; LOSS = {n: ck[n]["loss_next"].float()[keepc] for n in steps}
    tok = eval_ids("olmo1b")[:8, 1:256].reshape(-1)[keepc]
N = int(keepc.sum()); kidx = torch.nonzero(keepc)[:, 0]
ST = {}; WORDS = {}
for n in steps:
    st = stats(U[n], A[n], K); ST[n] = dict(S=st["S"], usage=st["usage"]); WORDS[n] = wordset(st["usage"])
    if n == 16000: R16 = st["ratio"].float()
    del st; torch.cuda.empty_cache()
w16 = torch.nonzero(WORDS[16000])[:, 0]; csize = (R16[:, w16] > 1).sum(0); cand = w16[(csize >= 10) & ~WORDS[1000][w16]]
def first_hold(flags):
    for i in range(len(flags)):
        if flags[i] and all(flags[i:]): return i
    return None
TR = {}
for w in cand.tolist():
    i = first_hold([bool(WORDS[n][w]) and float(ST[n]["S"][w]) >= 1 for n in steps])
    if i is not None and 1 <= i <= T - 2: TR[w] = i
CLS = {w: torch.nonzero(R16[:, w] > 1)[:, 0] for w in TR}; in_class = torch.zeros(N, dtype=torch.bool)
for c in CLS.values(): in_class[c] = True
log(f"{len(TR)} tracked words (recruited at index 1-{T - 2}); {int(in_class.sum())} class positions of {N}")
# token-matched controls outside every tracked class
by_tok = {}
for p in torch.nonzero(~in_class)[:, 0].tolist(): by_tok.setdefault(int(tok[p]), []).append(p)
CTRL = {}
for w, c in CLS.items():
    q = []; used = set()
    for p in c.tolist():
        pool = [x for x in by_tok.get(int(tok[p]), []) if x not in used]
        if pool: x = pool[int(torch.randint(0, len(pool), (1,), generator=g))]; used.add(x); q.append(x)
    CTRL[w] = torch.tensor(q, dtype=torch.long)
Lmat = torch.stack([LOSS[n] for n in steps])   # [T, N]
gmean = Lmat.mean(1)
def curve(pos, tr, KR=range(-4, 5)): return {k: float(Lmat[tr + k][pos].mean()) if 0 <= tr + k < T and pos.numel() else None for k in KR}
rows = []
for w, tr in TR.items():
    c, q = CLS[w], CTRL[w]; cc, cq = curve(c, tr), curve(q, tr); cg = {k: float(gmean[tr + k]) if 0 <= tr + k < T else None for k in range(-4, 5)}
    tsh = int(torch.randint(1, T - 1, (1,), generator=g)); cs = curve(c, tsh)
    def drop(cv, a, b): return (cv[b] - cv[a]) if (cv.get(a) is not None and cv.get(b) is not None) else None
    def accel(cv):
        d0, dm, dp = drop(cv, -1, 1), drop(cv, -3, -1), drop(cv, 1, 3)
        return (d0 - (dm + dp) / 2) if None not in (d0, dm, dp) else None
    rows.append(dict(word=w, tr=tr, n_class=int(c.numel()), n_ctrl=int(q.numel()), class_curve=cc, ctrl_curve=cq, global_curve=cg, shuffled_curve=cs, drop_class=drop(cc, -1, 1), drop_ctrl=drop(cq, -1, 1), drop_global=drop(cg, -1, 1), drop_shuffled=drop(cs, -1, 1), accel_class=accel(cc), accel_ctrl=accel(cq), accel_shuffled=accel(cs)))
def boot(vals, reps=500):
    v = torch.tensor([x for x in vals if x is not None], dtype=torch.float64)
    if v.numel() < 5: return dict(median=None, lo=None, hi=None, n=int(v.numel()))
    meds = torch.stack([v[torch.randint(0, v.numel(), (v.numel(),), generator=g)].median() for _ in range(reps)]); return dict(median=float(v.median()), lo=float(meds.quantile(0.025)), hi=float(meds.quantile(0.975)), n=int(v.numel()))
diff = [(r["drop_class"] - r["drop_ctrl"]) if None not in (r["drop_class"], r["drop_ctrl"]) else None for r in rows]; diff_sh = [(r["drop_class"] - r["drop_shuffled"]) if None not in (r["drop_class"], r["drop_shuffled"]) else None for r in rows]
res = dict(model=name, n_tracked=len(TR), n_positions=N, drop_class=boot([r["drop_class"] for r in rows]), drop_ctrl=boot([r["drop_ctrl"] for r in rows]), drop_global=boot([r["drop_global"] for r in rows]), drop_shuffled=boot([r["drop_shuffled"] for r in rows]), diff_class_minus_ctrl=boot(diff), diff_class_minus_shuffled=boot(diff_sh),
           accel_class=boot([r["accel_class"] for r in rows]), accel_ctrl=boot([r["accel_ctrl"] for r in rows]), accel_shuffled=boot([r["accel_shuffled"] for r in rows]), share_class_beats_ctrl=mean([float(x < 0) for x in diff if x is not None]), share_accel_positive=mean([float(r["accel_class"] < 0) for r in rows if r["accel_class"] is not None]),
           aligned_curves={k: dict(cls=med([r["class_curve"][k] for r in rows if r["class_curve"][k] is not None]), ctrl=med([r["ctrl_curve"][k] for r in rows if r["ctrl_curve"][k] is not None]), glob=med([r["global_curve"][k] for r in rows if r["global_curve"][k] is not None]), shuf=med([r["shuffled_curve"][k] for r in rows if r["shuffled_curve"][k] is not None])) for k in range(-4, 5)})
for k in range(-4, 5): a = res["aligned_curves"][k]; log(f"k={k:+d}: loss at the class {a['cls']:.3f}, token-matched controls {a['ctrl']:.3f}, all positions {a['glob']:.3f}, time-shuffled {a['shuf']:.3f}")
D = res
log(f"drop across recruitment (-1 -> +1): class {D['drop_class']['median']:+.4f} [{D['drop_class']['lo']:+.4f}, {D['drop_class']['hi']:+.4f}], token-matched {D['drop_ctrl']['median']:+.4f}, all positions {D['drop_global']['median']:+.4f}, time-shuffled {D['drop_shuffled']['median']:+.4f}; class minus control {D['diff_class_minus_ctrl']['median']:+.4f} [{D['diff_class_minus_ctrl']['lo']:+.4f}, {D['diff_class_minus_ctrl']['hi']:+.4f}] (class beats control for {D['share_class_beats_ctrl']:.2f}); acceleration at recruitment: class {D['accel_class']['median']:+.4f} [{D['accel_class']['lo']:+.4f}, {D['accel_class']['hi']:+.4f}], control {D['accel_ctrl']['median']:+.4f}, shuffled {D['accel_shuffled']['median']:+.4f} (negative = concentrated drop; class negative for {D['share_accel_positive']:.2f})")
# the functional test: the direction removed at the class across recruitment (Pythia only)
if name == "pythia410":
    sel = [w for w, tr in TR.items() if 2 <= tr <= T - 3]; sel = sel[:120]; by_ck = {}
    for w in sel:
        for k in range(-2, 3): by_ck.setdefault(steps[TR[w] + k], []).append((w, k))
    EFF = {w: {} for w in sel}; gg = torch.Generator(device=DEV).manual_seed(1)
    for n, pairs in sorted(by_ck.items()):
        m, _, fam = load_model("pythia410", revision=f"step{n}"); arch = Arch(m, fam)
        with torch.no_grad(): base = m(ids).logits.float().log_softmax(-1)
        for w, k in pairs:
            pm = torch.zeros(ids.shape, dtype=torch.bool); full = torch.zeros(keepc.numel(), dtype=torch.bool); full[kidx[CLS[w]]] = True; pm[:, 1:] = full.reshape(ids.shape[0], -1); pm = pm.to(DEV)
            out = {}
            for tag, d in (("row", A[n][w]), ("random", unitr(torch.randn(1, arch.D, device=DEV, generator=gg))[0])):
                def hk(mm, i, o):
                    x = o[0] if isinstance(o, tuple) else o; y = torch.where(pm[..., None], x - (x @ d)[..., None] * d[None, None], x); return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
                h = arch.layers[B].register_forward_hook(hk)
                try:
                    with torch.no_grad(): lp = m(ids).logits.float().log_softmax(-1)
                finally: h.remove()
                out[tag] = float(((base.exp() * (base - lp)).sum(-1))[pm].mean())
            EFF[w][k] = out
        del m, base; torch.cuda.empty_cache(); log(f"removal effects at step {n}: {len(pairs)} pairs ({time.time() - t0:.0f}s)")
    res["removal"] = {k: dict(row=med([EFF[w][k]["row"] for w in sel if k in EFF[w]]), random=med([EFF[w][k]["random"] for w in sel if k in EFF[w]])) for k in range(-2, 3)}
    ratio = [EFF[w][2]["row"] / max(EFF[w][-2]["row"], 1e-6) for w in sel if 2 in EFF[w] and -2 in EFF[w]]; res["removal_rise"] = dict(median_ratio=med(ratio), share_over_2=mean([float(x >= 2) for x in ratio]), n=len(ratio))
    for k in range(-2, 3): log(f"removal effect at k={k:+d}: the row's direction {res['removal'][k]['row']:.4f}, a random direction {res['removal'][k]['random']:.4f}")
    log(f"rise of the row's removal effect from k=-2 to +2: median ratio {res['removal_rise']['median_ratio']:.2f}, at least doubled for {res['removal_rise']['share_over_2']:.2f} of {res['removal_rise']['n']}")
summ = (f"{name}: does recruitment matter ({len(TR)} tracked words): across recruitment (-1 -> +1) the class's loss changes {D['drop_class']['median']:+.4f} nats [{D['drop_class']['lo']:+.4f}, {D['drop_class']['hi']:+.4f}] against {D['drop_ctrl']['median']:+.4f} for token-matched controls and {D['drop_shuffled']['median']:+.4f} time-shuffled; class minus control {D['diff_class_minus_ctrl']['median']:+.4f} [{D['diff_class_minus_ctrl']['lo']:+.4f}, {D['diff_class_minus_ctrl']['hi']:+.4f}], class beats control for {D['share_class_beats_ctrl']:.2f}; acceleration at recruitment class {D['accel_class']['median']:+.4f} [{D['accel_class']['lo']:+.4f}, {D['accel_class']['hi']:+.4f}] vs control {D['accel_ctrl']['median']:+.4f}"
        + (f"; the row's removal effect at the class k=-2/0/+2 {res['removal'][-2]['row']:.4f}/{res['removal'][0]['row']:.4f}/{res['removal'][2]['row']:.4f} (random direction {res['removal'][-2]['random']:.4f}/{res['removal'][0]['random']:.4f}/{res['removal'][2]['random']:.4f}), rise ratio {res['removal_rise']['median_ratio']:.2f} (doubled for {res['removal_rise']['share_over_2']:.2f})" if "removal" in res else "") + f" | {time.time() - t0:.0f}s")
log(summ); record(f"e591_recruitment_matters_{name}", res, summ)
