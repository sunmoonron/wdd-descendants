"""e591b (session 110): the removal test of e591 with stronger controls. e591 removed the eventual row's direction from
the block-12 state at the class's positions and compared it with a random unit direction, which is too weak a
control (0.0002 nats against the row's 0.005-0.011). Here, for the same words at k = -2 and k = +2 around
recruitment, the directions removed at the class are: the eventual row's; a random other row of the same
dictionary; the strongest other row at the class at that checkpoint (the row, not the eventual one, with the largest
median projection over the class's states); the row-free class direction (the unit mean of the class's states); and
the eventual row's direction removed at the token-matched control positions instead of the class (the specificity
control). Per word and checkpoint the KL of the change at the masked positions; medians over words and the share of
words for which the eventual row beats the strongest other row. Pre-registered (honest guesses):
 P4 (0.6) two checkpoints before recruitment the eventual row's direction already costs more than the strongest
    other row's for most words (the row is the nearest available direction, e587);
 P5 (0.7) the class direction costs more than the row's at both checkpoints (the geometry carries the effect);
 P6 (0.6) the row's direction removed at the control positions costs less than a fifth of its cost at the class."""
from s101_common import *
name = "pythia410"; t0 = time.time(); steps = list(range(1000, 16001, 1000)); T = len(steps); g = torch.Generator().manual_seed(0)
if name == "pythia410":
    CD = "/workspace/wdd/cache/e582_pythia410"; B = 12; ids = eval_ids("pythia410")[:24, :512].to(DEV); ck = {n: torch.load(f"{CD}/step{n}.pt", map_location="cpu") for n in steps}
    keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0); U = {n: unitr((ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV)) for n in steps}; A = {n: ck[n]["rows"].float().to(DEV) for n in steps}
    tok_all = ids[:, 1:].reshape(-1).cpu(); tok = tok_all[keepc]
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

sel = [w for w, tr in TR.items() if 2 <= tr <= T - 3][:120]; by_ck = {}
for w in sel:
    for k in (-2, 2): by_ck.setdefault(steps[TR[w] + k], []).append((w, k))
TAGS = ("row", "random_row", "best_other_row", "class_dir", "row_at_controls"); EFF = {w: {} for w in sel}; gg = torch.Generator().manual_seed(1)
def mask_of(pos):
    pm = torch.zeros(ids.shape, dtype=torch.bool); full = torch.zeros(keepc.numel(), dtype=torch.bool); full[kidx[pos]] = True; pm[:, 1:] = full.reshape(ids.shape[0], -1); return pm.to(DEV)
for n, pairs in sorted(by_ck.items()):
    m, _, fam = load_model("pythia410", revision=f"step{n}"); arch = Arch(m, fam); An = unitr(A[n])
    with torch.no_grad(): base = m(ids).logits.float().log_softmax(-1)
    for w, k in pairs:
        c = CLS[w]; pm = mask_of(c); pq = mask_of(CTRL[w]) if CTRL[w].numel() else None
        proj = (U[n][c.to(DEV)] @ An.T).median(0).values; proj[w] = -1e9; jbest = int(proj.argmax()); jr = int(torch.randint(0, An.shape[0], (1,), generator=gg)); jr = jr if jr != w else (jr + 1) % An.shape[0]
        dirs = {"row": (An[w], pm), "random_row": (An[jr], pm), "best_other_row": (An[jbest], pm), "class_dir": (unitr(U[n][c.to(DEV)].mean(0, keepdim=True))[0], pm), "row_at_controls": (An[w], pq)}
        out = {"best_other_index": jbest, "best_other_proj": float(proj[jbest]), "row_proj": float((U[n][c.to(DEV)] @ An[w]).median())}
        for tag, (d, msk) in dirs.items():
            if msk is None: out[tag] = None; continue
            def hk(mm, i, o, d=d, msk=msk):
                x = o[0] if isinstance(o, tuple) else o; y = torch.where(msk[..., None], x - (x @ d)[..., None] * d[None, None], x); return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
            h = arch.layers[B].register_forward_hook(hk)
            try:
                with torch.no_grad(): lp = m(ids).logits.float().log_softmax(-1)
            finally: h.remove()
            out[tag] = float(((base.exp() * (base - lp)).sum(-1))[msk].mean())
        EFF[w][k] = out
    del m, base; torch.cuda.empty_cache(); log(f"removal effects at step {n}: {len(pairs)} pairs ({time.time() - t0:.0f}s)")
res = dict(n_words=len(sel), removal={})
for k in (-2, 2):
    E = [EFF[w][k] for w in sel if k in EFF[w]]
    res["removal"][k] = {tag: med([e[tag] for e in E if e[tag] is not None]) for tag in TAGS}
    res["removal"][k].update(share_row_beats_best_other=mean([float(e["row"] > e["best_other_row"]) for e in E]), share_row_beats_random_row=mean([float(e["row"] > e["random_row"]) for e in E]), share_class_dir_beats_row=mean([float(e["class_dir"] > e["row"]) for e in E]),
                             ratio_row_over_best_other=med([e["row"] / max(e["best_other_row"], 1e-9) for e in E]), ratio_row_over_class_dir=med([e["row"] / max(e["class_dir"], 1e-9) for e in E]), ratio_controls_over_class=med([e["row_at_controls"] / max(e["row"], 1e-9) for e in E if e["row_at_controls"] is not None]), row_proj=med([e["row_proj"] for e in E]), best_other_proj=med([e["best_other_proj"] for e in E]), n=len(E))
    r = res["removal"][k]; log(f"k={k:+d}: removed at the class: the eventual row {r['row']:.4f}, a random row {r['random_row']:.4f}, the strongest other row {r['best_other_row']:.4f}, the class direction {r['class_dir']:.4f}; the row at the control positions {r['row_at_controls']:.4f}; row beats strongest other for {r['share_row_beats_best_other']:.2f} (ratio {r['ratio_row_over_best_other']:.2f}), class direction beats row for {r['share_class_dir_beats_row']:.2f}; controls/class {r['ratio_controls_over_class']:.2f}; projections row {r['row_proj']:.3f} vs strongest other {r['best_other_proj']:.3f}")
res["per_word"] = {str(w): {str(k): EFF[w][k] for k in EFF[w]} for w in sel}
a_, b_ = res["removal"][-2], res["removal"][2]
summ = (f"removal with stronger controls ({len(sel)} words, k = -2 / +2): the eventual row {a_['row']:.4f} / {b_['row']:.4f}, a random row {a_['random_row']:.4f} / {b_['random_row']:.4f}, the strongest other row {a_['best_other_row']:.4f} / {b_['best_other_row']:.4f}, the class direction {a_['class_dir']:.4f} / {b_['class_dir']:.4f}, the row at control positions {a_['row_at_controls']:.4f} / {b_['row_at_controls']:.4f}; "
        f"the row beats the strongest other row for {a_['share_row_beats_best_other']:.2f} / {b_['share_row_beats_best_other']:.2f} of words, the class direction beats the row for {a_['share_class_dir_beats_row']:.2f} / {b_['share_class_dir_beats_row']:.2f}; controls/class {a_['ratio_controls_over_class']:.2f} / {b_['ratio_controls_over_class']:.2f} | {time.time() - t0:.0f}s")
log(summ); record("e591b_removal_controls", res, summ)
