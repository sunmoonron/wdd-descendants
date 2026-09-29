"""e591c (session 110): the loss-aligned test of e591 with controls matched on difficulty as well as token. e591's
token-matched controls sit at a lower loss than the class's positions (2.7 against 3.6 nats at recruitment), so a
difference in their drops could be a difference in loss level. Here, for the same tracked words, the controls are
(a) token-matched positions outside every tracked class whose loss one checkpoint before recruitment is within 0.25
nats of the class position's (nearest available; the class positions that find a partner are the easier ones, so the
comparison is between those matched class positions and their partners), and (b) difficulty-only controls: positions outside every class of
any token with the nearest loss one checkpoint before. Measures as e591: the aligned curves, the drop across
recruitment, the class-minus-control difference with bootstrap intervals. Argument pythia410 or olmo1b.
Pre-registered (honest guess): P7 (0.7) the matched class positions' drop minus their token-and-difficulty
controls' has an interval containing zero on both models, and so does the whole class's against the difficulty-only
controls."""
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
# controls matched on token and on the loss one checkpoint before recruitment, and on the loss alone
by_tok = {}
for p in torch.nonzero(~in_class)[:, 0].tolist(): by_tok.setdefault(int(tok[p]), []).append(p)
outside = torch.nonzero(~in_class)[:, 0]
CTRL, CTRL_D, CLSM = {}, {}, {}
for w, c in CLS.items():
    tr = TR[w]; Lb = LOSS[steps[tr - 1]]; q, qd, cm, used, usedd = [], [], [], set(), set()
    for p in c.tolist():
        lp = float(Lb[p]); pool = [x for x in by_tok.get(int(tok[p]), []) if x not in used]
        if pool:
            pl = torch.tensor([float(Lb[x]) for x in pool]); j = int((pl - lp).abs().argmin())
            if abs(float(pl[j]) - lp) <= 0.25: used.add(pool[j]); q.append(pool[j]); cm.append(p)
        d = (Lb[outside] - lp).abs(); d[torch.tensor([x in usedd for x in outside.tolist()], dtype=torch.bool)] = 1e9; j = int(d.argmin()); usedd.add(int(outside[j])); qd.append(int(outside[j]))
    CTRL[w] = torch.tensor(q, dtype=torch.long); CTRL_D[w] = torch.tensor(qd, dtype=torch.long); CLSM[w] = torch.tensor(cm, dtype=torch.long)
log(f"controls: token-and-difficulty matched cover {mean([CTRL[w].numel() / max(CLS[w].numel(), 1) for w in TR]):.2f} of class positions at the median word; difficulty-only {mean([CTRL_D[w].numel() / max(CLS[w].numel(), 1) for w in TR]):.2f}")
Lmat = torch.stack([LOSS[n] for n in steps])   # [T, N]
gmean = Lmat.mean(1)
def curve(pos, tr, KR=range(-4, 5)): return {k: float(Lmat[tr + k][pos].mean()) if 0 <= tr + k < T and pos.numel() else None for k in KR}
rows = []
for w, tr in TR.items():
    c, q, qd, cmm = CLS[w], CTRL[w], CTRL_D[w], CLSM[w]; cc, cq, cd, cmc = curve(c, tr), curve(q, tr), curve(qd, tr), curve(cmm, tr); cg = {k: float(gmean[tr + k]) if 0 <= tr + k < T else None for k in range(-4, 5)}
    tsh = int(torch.randint(1, T - 1, (1,), generator=g)); cs = curve(c, tsh)
    def drop(cv, a, b): return (cv[b] - cv[a]) if (cv.get(a) is not None and cv.get(b) is not None) else None
    def accel(cv):
        d0, dm, dp = drop(cv, -1, 1), drop(cv, -3, -1), drop(cv, 1, 3)
        return (d0 - (dm + dp) / 2) if None not in (d0, dm, dp) else None
    rows.append(dict(word=w, tr=tr, n_class=int(c.numel()), n_ctrl=int(q.numel()), n_ctrl_d=int(qd.numel()), class_curve=cc, ctrl_curve=cq, ctrld_curve=cd, classm_curve=cmc, global_curve=cg, shuffled_curve=cs, drop_class=drop(cc, -1, 1), drop_classm=drop(cmc, -1, 1), drop_ctrl=drop(cq, -1, 1), drop_ctrld=drop(cd, -1, 1), drop_global=drop(cg, -1, 1), drop_shuffled=drop(cs, -1, 1), accel_class=accel(cc), accel_ctrl=accel(cq), accel_ctrld=accel(cd), accel_shuffled=accel(cs)))
def boot(vals, nb=2000):
    v = torch.tensor([x for x in vals if x is not None], dtype=torch.float64)
    if v.numel() < 3: return dict(median=None, lo=None, hi=None, n=int(v.numel()))
    idx = torch.randint(0, v.numel(), (nb, v.numel()), generator=g); meds = v[idx].median(1).values; return dict(median=float(v.median()), lo=float(meds.quantile(0.025)), hi=float(meds.quantile(0.975)), n=int(v.numel()))
diff = [(r["drop_class"] - r["drop_ctrl"]) if None not in (r["drop_class"], r["drop_ctrl"]) else None for r in rows]; diffd = [(r["drop_class"] - r["drop_ctrld"]) if None not in (r["drop_class"], r["drop_ctrld"]) else None for r in rows]; diffm = [(r["drop_classm"] - r["drop_ctrl"]) if None not in (r["drop_classm"], r["drop_ctrl"]) else None for r in rows]
res = dict(model=name, n_tracked=len(TR), n_positions=N, coverage_token_difficulty=mean([r["n_ctrl"] / max(r["n_class"], 1) for r in rows]), drop_class=boot([r["drop_class"] for r in rows]), drop_ctrl_token_difficulty=boot([r["drop_ctrl"] for r in rows]), drop_class_matched=boot([r["drop_classm"] for r in rows]), diff_matched_class_minus_token_difficulty=boot(diffm), share_matched_class_beats_token_difficulty=mean([float(x < 0) for x in diffm if x is not None]), drop_ctrl_difficulty=boot([r["drop_ctrld"] for r in rows]), drop_global=boot([r["drop_global"] for r in rows]), drop_shuffled=boot([r["drop_shuffled"] for r in rows]),
           diff_class_minus_token_difficulty=boot(diff), diff_class_minus_difficulty=boot(diffd), share_class_beats_token_difficulty=mean([float(x < 0) for x in diff if x is not None]), share_class_beats_difficulty=mean([float(x < 0) for x in diffd if x is not None]),
           accel_class=boot([r["accel_class"] for r in rows]), accel_ctrl_token_difficulty=boot([r["accel_ctrl"] for r in rows]), accel_ctrl_difficulty=boot([r["accel_ctrld"] for r in rows]),
           aligned_curves={k: dict(cls=med([r["class_curve"][k] for r in rows if r["class_curve"][k] is not None]), clsm=med([r["classm_curve"][k] for r in rows if r["classm_curve"][k] is not None]), ctrl=med([r["ctrl_curve"][k] for r in rows if r["ctrl_curve"][k] is not None]), ctrld=med([r["ctrld_curve"][k] for r in rows if r["ctrld_curve"][k] is not None]), glob=med([r["global_curve"][k] for r in rows if r["global_curve"][k] is not None])) for k in range(-4, 5)})
for k in range(-4, 5): a = res["aligned_curves"][k]; log(f"k={k:+d}: loss at the class {a['cls']:.3f} (its matched positions {a['clsm']:.3f}), token-and-difficulty matched {a['ctrl']:.3f}, difficulty matched {a['ctrld']:.3f}, all positions {a['glob']:.3f}")
D = res
summ = (f"{name}: matched controls ({len(TR)} tracked words; token-and-difficulty controls cover {D['coverage_token_difficulty']:.2f} of class positions): drop across recruitment class {D['drop_class']['median']:+.4f} [{D['drop_class']['lo']:+.4f}, {D['drop_class']['hi']:+.4f}], token-and-difficulty matched {D['drop_ctrl_token_difficulty']['median']:+.4f}, difficulty matched {D['drop_ctrl_difficulty']['median']:+.4f}; the matched class positions themselves {D['drop_class_matched']['median']:+.4f}, matched class minus token-and-difficulty {D['diff_matched_class_minus_token_difficulty']['median']:+.4f} [{D['diff_matched_class_minus_token_difficulty']['lo']:+.4f}, {D['diff_matched_class_minus_token_difficulty']['hi']:+.4f}] (class beats for {D['share_matched_class_beats_token_difficulty']:.2f}; the whole class minus it {D['diff_class_minus_token_difficulty']['median']:+.4f}), class minus difficulty {D['diff_class_minus_difficulty']['median']:+.4f} [{D['diff_class_minus_difficulty']['lo']:+.4f}, {D['diff_class_minus_difficulty']['hi']:+.4f}] (class beats for {D['share_class_beats_difficulty']:.2f}); "
        f"loss at k=0 class {D['aligned_curves'][0]['cls']:.3f} (matched positions {D['aligned_curves'][0]['clsm']:.3f}), token-and-difficulty {D['aligned_curves'][0]['ctrl']:.3f}, difficulty {D['aligned_curves'][0]['ctrld']:.3f} | {time.time() - t0:.0f}s")
log(summ); record(f"e591c_matched_controls_{name}", res, summ)
