"""e450: is the model's own vocabulary a good codec for its activations? (vision: activations ship as native codes)
A native code for a state is k word indices plus k quantised coefficients. Each index costs log2(dictionary size) bits
and each coefficient 6 bits. The question is whether, at equal bits, it keeps more of the model's function than a
generic code.
Codecs at 64, 128, 256 and 512 bits per position, middle depth, 6 sequences, typical positions (sinks exact):
 native: OMP over the own dictionary, coefficients refit and quantised to 6 bits;
 rotated: the same with the rotated dictionary (same index cost);
 PCA: the top-m principal components' coefficients at 6 bits each (principal directions fitted on 16 other
      sequences, i.e. a shared codebook the receiver already holds);
 scalar: every dimension as its sign (D bits) or at three levels (about 1.6 D bits), with a per-dimension scale.
Scored by loss recovered when the decoded state is spliced back.
Pre-registered (honest guess, 0.5): at 128 and 256 bits the native code keeps more function than PCA in at least three
of five models."""
import sys, os, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; ev = EA["eval_ids"][:6].to(DEV); fit = EA["cen_ids"][:16].to(DEV); del EA
A, _ = build_dictionary(arch, blocks=list(range(L + 1))); Ar = rotate(A, seed=7); ib = math.ceil(math.log2(A.shape[0])); cb = 6
lv = NSLevel(model, arch, ev, L); Xc = lv.Xc
xf = block_states(model, arch, fit, [L], chunk=4)[L]; Xf = xf[~sinkmask(xf)] - xf[~sinkmask(xf)].mean(0)
evs, U = torch.linalg.eigh(torch.cov(Xf.T.double(), correction=0)); U = U.flip(-1).float()
def quant(v, b, lim):
    """uniform quantisation of v to b bits in [-lim, lim] (per column lim)"""
    lv_ = 2 ** (b - 1) - 1; return (torch.clamp(v / lim, -1, 1) * lv_).round() / lv_ * lim
def rec(Xh): return lv.recovered(lv.splice(lv.mu + Xh))[0]
res = dict(model=name, level=L, index_bits=ib, coef_bits=cb, D=D, budgets={})
sel_n, _, _ = omp(Xc, A, 32, batch=256, record_err=False); sel_r, _, _ = omp(Xc, Ar, 32, batch=256, record_err=False)
for B in (64, 128, 256, 512):
    out = {}; k = max(1, B // (ib + cb))
    for vn, Dct, sel in (("native", A, sel_n), ("rotated", Ar, sel_r)):
        kk = min(k, sel.shape[1]); cof, _ = refit(Xc, Dct, sel[:, :kk]); lim = torch.quantile(cof.abs().flatten()[:200000], 0.99).clamp_min(1e-6)
        q = quant(cof, cb, lim); out[vn] = dict(k=kk, rec=rec(torch.einsum("nk,nkd->nd", q, Dct[sel[:, :kk]])))
    m = min(D, B // cb); P = U[:, :m]; co = Xc @ P; lim = torch.quantile(co.abs(), 0.99, dim=0).clamp_min(1e-6); out["pca"] = dict(m=m, rec=rec(quant(co, cb, lim) @ P.T))
    res["budgets"][str(B)] = out
    log(f"{name} {B} bits: native k={out['native']['k']} {out['native']['rec']:.2f} | rotated {out['rotated']['rec']:.2f} | PCA m={m} {out['pca']['rec']:.2f}")
res["scalar_sign"] = dict(bits=D, rec=rec(torch.sign(Xc) * Xc.abs().mean(0, keepdim=True)))                  # 1 bit per dimension
lim = torch.quantile(Xc.abs(), 0.99, dim=0).clamp_min(1e-6); res["scalar_ternary"] = dict(bits=round(1.585 * D), rec=rec(quant(Xc, 2, lim)))   # 3 levels per dimension
Bd = res["budgets"]; wins = sum(1 for B in ("128", "256") if Bd[B]["native"]["rec"] > Bd[B]["pca"]["rec"])
summ = (f"{name} L{L} (index {ib} bits + coefficient {cb} bits per word): loss recovered by budget native / rotated / PCA: " + " ; ".join(f"{B}b {v['native']['rec']:.2f} (k{v['native']['k']}) / {v['rotated']['rec']:.2f} / {v['pca']['rec']:.2f} (m{v['pca']['m']})" for B, v in Bd.items())
        + f" | scalar sign code ({D} bits) {res['scalar_sign']['rec']:.2f}, ternary ({res['scalar_ternary']['bits']} bits) {res['scalar_ternary']['rec']:.2f} | native beats PCA at 128/256 bits: {wins}/2")
log(summ); record(f"e450_codec_{name}", res, summ)
