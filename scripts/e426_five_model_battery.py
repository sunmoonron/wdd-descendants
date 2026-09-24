"""e426: the native-vocabulary findings that rest on Pythia alone, run on all five original models (final checkpoints,
middle depth, 6 sequences), one pass per model:
 (a) vocabulary, not accent: loss recovered at k 4 and 16 by the own words, their rotation, Gaussian words with the own
     vocabulary's second moment (covA) and signed sums of 8 own words of one family (mix8); the share of the
     own-minus-rotation gap carried by covA and mix8 (e399, e405: near zero at the end of Pythia's training);
 (b) the weights know what is function-light: the states' top-8 principal directions' share of the variance against
     their share of the trace of the downstream readers' Gram (weights only; e408: 0.86 against 0.06 in Pythia);
 (c) errors in the readers' words: per-position loss gradients and centred states as unit directions, fraction
     unexplained by 16 OMP words from the writers (MLP rows of blocks 0..L) and the downstream readers (MLP input rows
     of later blocks), each against its rotation (e412: at the end, states favour writers, errors favour readers);
 (d) Zipf-like usage: rank-frequency slope of native-word usage in 16-word descriptions against a rotation (e414).
Pre-registered, in all five: (a) covA and mix8 shares below one half at k 16; (b) readers' trace share on the top-8
below a third of the variance share; (c) writers ahead on states, readers ahead on errors; (d) own slope steeper."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2      # GPT-2: 6, as elsewhere
E = eval_ids(name); ev = E[:6].to(DEV); fit = E[6:12].to(DEV)
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV)
lv = Level(model, arch, ev, L); fl = Level(model, arch, fit, L); SX = (fl.Xc.T @ fl.Xc) / fl.Xc.shape[0]
evX, UX = torch.linalg.eigh(SX.double()); UM = UX[:, -8:].float(); res = dict(model=name, level=L, gap=lv.gap)
# (a)
groups = [torch.nonzero(typ == t)[:, 0] for t in (T_TOK, T_POS, T_MLP, T_ATT, T_BIAS) if (typ == t).any()]
KS = [4, 16]; cells = {nm: describe(lv, V, KS) for nm, V in [("own", A), ("rot", rotate(A, seed=7)), ("covA", gauss_like(A.shape[0], (A.T @ A) / A.shape[0], seed=1)), ("mix8", mixtures(A, groups, m=8, seed=4))]}
share = lambda ctl, k: (cells[ctl][str(k)]["rec"] - cells["rot"][str(k)]["rec"]) / max(cells["own"][str(k)]["rec"] - cells["rot"][str(k)]["rec"], 1e-9)
res["a"] = dict(cells=cells, shares={f"{c}_k{k}": share(c, k) for c in ("covA", "mix8") for k in KS})
# (b)
GR, GU = reader_gram(model, arch, list(range(L + 1, arch.NB)))
res["b"] = dict(var_top8=(evX[-8:].sum() / evX.sum()).item(), readers_top8=share_on(GR, UM), unembed_top8=share_on(GU, UM))
# (c) gradients (true labels) and states at positions 1..T-2
for p in model.parameters(): p.requires_grad_(False)
def grads(ids):
    out = []; torch.set_grad_enabled(True)
    try:
        for s0 in range(0, ids.shape[0], 2):
            leaf = {}
            def hk(m, i, o):
                xo = o[0] if isinstance(o, tuple) else o; y = xo.detach().requires_grad_(True); leaf["x"] = y
                return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
            h = arch.layers[L].register_forward_hook(hk)
            try: lg = model(ids[s0:s0 + 2]).logits.float()
            finally: h.remove()
            token_loss(lg, ids[s0:s0 + 2]).sum().backward(); out.append(leaf["x"].grad[:, 1:-1].reshape(-1, arch.D).detach())
    finally: torch.set_grad_enabled(False)
    return torch.cat(out)
Gd = grads(ev); keep = Gd.norm(dim=-1) > 0; Gd = unitr(Gd[keep]); Sd = unitr(lv.Xc.view(ev.shape[0], -1, arch.D)[:, :-1].reshape(-1, arch.D)[keep])
Wr = A[typ == T_MLP].clone(); Rr = reader_rows(arch, list(range(L + 1, arch.NB)))
def fvu16(X, V):
    V = unitr(V); sel, _, _ = omp(X, V, 16, batch=256, record_err=False); _, err = refit(X, V, sel); return (err / X.pow(2).sum(-1)).mean().item()
c = {}
for tg, X in [("grad", Gd), ("state", Sd)]:
    for nm, V in [("W", Wr), ("W_rot", rotate(Wr, seed=7)), ("R", Rr), ("R_rot", rotate(Rr, seed=7))]: c[f"{tg}:{nm}"] = fvu16(X, V)
res["c"] = dict(fvu=c, adv={f"{tg}:{nm}": c[f"{tg}:{nm}_rot"] - c[f"{tg}:{nm}"] for tg in ("grad", "state") for nm in ("W", "R")})
# (d)
def slope(V):
    sel, _, _ = omp(lv.Xc, unitr(V), 16, batch=256, record_err=False); f = torch.bincount(sel.flatten(), minlength=V.shape[0]).float().sort(descending=True).values; f = f[f > 0]
    r = torch.arange(1, f.numel() + 1, device=DEV).float(); hi = min(1000, f.numel()); x, y = r[9:hi].log(), f[9:hi].log()
    return (((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()).item()
res["d"] = dict(slope_own=slope(A), slope_rot=slope(rotate(A, seed=7)))
a, b, cc, d = res["a"], res["b"], res["c"]["adv"], res["d"]
summ = (f"{name} L{L}: (a) k16 own {cells['own']['16']['rec']:.2f} rot {cells['rot']['16']['rec']:.2f} covA {cells['covA']['16']['rec']:.2f} mix8 {cells['mix8']['16']['rec']:.2f} -> shares covA {a['shares']['covA_k16']:.2f} mix8 {a['shares']['mix8_k16']:.2f} | "
        f"(b) top-8 variance {b['var_top8']:.2f}, readers' trace {b['readers_top8']:.3f}, unembedding {b['unembed_top8']:.3f} | "
        f"(c) advantage over rotation, states W {cc['state:W']:.3f} R {cc['state:R']:.3f}; errors W {cc['grad:W']:.3f} R {cc['grad:R']:.3f} | (d) Zipf slope own {d['slope_own']:.2f} rot {d['slope_rot']:.2f}")
log(summ); record(f"e426_battery_{name}", res, summ)
