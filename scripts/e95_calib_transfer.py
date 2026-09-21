"""e95: does the coefficient calibration transfer across corpora? Fit the 2-parameter log-linear calibration of
e12 on WikiText (all sequences) and apply it to Pile states of the same model; also fit on Pile and apply to
WikiText. Median relative error before/after on the held-out corpus."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; res = dict(model=tag)
def ident(c):
    L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L); sel, cof, err = get_omp(c, L, A=A, X=X)
    tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); ct = tc[:, 0].to(DEV); hit = sel == row[:, None]; idn = hit.any(1) & typ
    return (cof * hit).sum(1)[idn], ct[idn]
cw, tw = ident(Cache(tag)); cp, tp = ident(Cache(tag + "_pile"))
def fit(chat, ct):
    x, y = chat.abs().log(), ct.abs().log(); b = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum(); return (y.mean() - b * x.mean()).item(), b.item()
def apply(chat, a, b): return torch.sign(chat) * torch.exp(a + b * chat.abs().clamp_min(1e-6).log())
relerr = lambda est, ct: ((est - ct).abs() / ct.abs()).median().item()
aw, bw = fit(cw, tw); ap, bp = fit(cp, tp)
res.update(wikitext=dict(base=relerr(cw, tw), self_fit=relerr(apply(cw, aw, bw), tw), pile_fit_applied=relerr(apply(cw, ap, bp), tw)),
           pile=dict(base=relerr(cp, tp), self_fit=relerr(apply(cp, ap, bp), tp), wikitext_fit_applied=relerr(apply(cp, aw, bw), tp)), params=dict(wikitext=(aw, bw), pile=(ap, bp)))
record(f"e95_calib_{tag}", res, f"wikitext: base {res['wikitext']['base']:.3f} self {res['wikitext']['self_fit']:.3f} pile-fit {res['wikitext']['pile_fit_applied']:.3f} | pile: base {res['pile']['base']:.3f} self {res['pile']['self_fit']:.3f} wikitext-fit {res['pile']['wikitext_fit_applied']:.3f} | params wt (a {aw:.2f}, b {bw:.2f}) pile (a {ap:.2f}, b {bp:.2f})")
