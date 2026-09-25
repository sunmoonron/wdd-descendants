"""e490: why refitting compensates. e484 found that after erasing words from a native description, refitting the
remaining words recovers part of the loss (+0.02 to +0.08), while for rotated words it costs. A remaining word can
stand in for an erased one only if the two overlap. Mutual coherence (Tropp 2004) is the sparse-coding name for that
overlap, and it is what limits OMP's recovery guarantees; here it is measured within the chosen supports, native
against rotated (principal components have none by construction), and set against the per-position refit gain.
Setup: middle depth, 8 x 256 evaluation tokens, typical positions, 16 words. Per position: the mean absolute cosine
between the chosen words (within-support coherence), the log condition number of the support's Gram matrix; the
refit gain after erasing a random half, in reconstruction (FVU) and in the spliced per-position loss.
Reported: median coherence and condition number, native against rotated; Spearman of coherence with the refit gain,
and the mean refit gain in the top against the bottom coherence quartile.
Models (argument): the five.
Pre-registered (honest guesses):
- native supports are more coherent than rotated ones (median at least 1.5 times) in all five (0.7);
- the refit gain in FVU rises with coherence (Spearman above 0.3) (0.6);
- the same holds for the loss gain, more weakly (above 0.1) (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16; D = arch.D
ids = eval_ids(name)[:8, :256].to(DEV); sp = Splicer(model, arch, ids, L, chunk=2)
X = block_states(model, arch, ids, [L], chunk=4)[L]; flat = X.reshape(-1, D); keep = ~sinkmask(flat); mu = flat[keep].mean(0); Xc = flat[keep] - mu; N = Xc.shape[0]
lm = sp.lossmask(keep.view(X.shape[0], -1))
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
g = torch.Generator(device=DEV).manual_seed(0); order = torch.rand(N, K, generator=g, device=DEV).argsort(1); kept = order[:, K // 2:]
def loss_of(Xh):
    new = flat.clone(); new[keep] = mu + Xh; return sp.run(new.view_as(X))["loss"][lm]
res = dict(model=name, level=L, k=K, n=N, kinds={})
for kind, Dct in (("native", Au), ("rotated", Ar)):
    sel, _, _ = omp(Xc, Dct, K, batch=1024, record_err=False); cof, _ = refit(Xc, Dct, sel); W = Dct[sel]
    G = W @ W.transpose(1, 2); off = G - torch.eye(K, device=DEV)[None]; coh = off.abs().sum((1, 2)) / (K * (K - 1))
    ev = torch.linalg.eigvalsh(G); logcond = torch.log10(ev[:, -1] / ev[:, 0].clamp_min(1e-9))
    Wk = W.gather(1, kept[:, :, None].expand(-1, -1, D)); Ck = cof.gather(1, kept)
    plain = torch.einsum("nk,nkd->nd", Ck, Wk); cr = torch.linalg.lstsq(Wk.transpose(1, 2), Xc[:, :, None]).solution[:, :, 0]; refit_rec = torch.einsum("nk,nkd->nd", cr, Wk)
    xn2 = Xc.pow(2).sum(1).clamp_min(1e-9); gain_fvu = ((Xc - plain).pow(2).sum(1) - (Xc - refit_rec).pow(2).sum(1)) / xn2
    l_plain = loss_of(plain); l_refit = loss_of(refit_rec); gain_loss = l_plain - l_refit
    n_ = min(gain_loss.numel(), coh.numel()); c_ = coh[:n_] if coh.numel() >= n_ else coh
    def spear(x, y):
        rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-9))
    q = coh.quantile(torch.tensor([0.25, 0.75], device=DEV)); lo, hi = coh <= q[0], coh >= q[1]
    res["kinds"][kind] = dict(coherence_median=float(coh.median()), coherence_p90=float(coh.quantile(0.9)), log10_condition_median=float(logcond.median()),
                              refit_gain_fvu_mean=float(gain_fvu.mean()), refit_gain_loss_mean=float(gain_loss.mean()), spearman_coherence_fvu_gain=spear(coh, gain_fvu),
                              spearman_coherence_loss_gain=spear(coh[:gain_loss.numel()], gain_loss) if gain_loss.numel() == coh.numel() else None,
                              fvu_gain_top_quartile=float(gain_fvu[hi].mean()), fvu_gain_bottom_quartile=float(gain_fvu[lo].mean()), loss_gain_top_quartile=float(gain_loss[hi[:gain_loss.numel()]].mean()) if gain_loss.numel() == coh.numel() else None, loss_gain_bottom_quartile=float(gain_loss[lo[:gain_loss.numel()]].mean()) if gain_loss.numel() == coh.numel() else None)
    r = res["kinds"][kind]; log(f"{name} {kind}: coherence median {r['coherence_median']:.3f} (p90 {r['coherence_p90']:.3f}), log10 condition {r['log10_condition_median']:.2f}; refit gain FVU {r['refit_gain_fvu_mean']:.4f}, loss {r['refit_gain_loss_mean']:+.4f}; Spearman with coherence: FVU {r['spearman_coherence_fvu_gain']:+.2f}, loss {r['spearman_coherence_loss_gain']}; FVU gain top/bottom coherence quartile {r['fvu_gain_top_quartile']:.4f}/{r['fvu_gain_bottom_quartile']:.4f}")
Rn, Rr = res["kinds"]["native"], res["kinds"]["rotated"]
res["checks"] = dict(native_coherence_1_5x=Rn["coherence_median"] >= 1.5 * Rr["coherence_median"], fvu_gain_tracks_coherence=Rn["spearman_coherence_fvu_gain"] > 0.3, loss_gain_tracks_coherence=(Rn["spearman_coherence_loss_gain"] or 0) > 0.1)
summ = (f"{name} L{L}: within-support coherence (mean |cos| of the 16 words) native {Rn['coherence_median']:.3f} against rotated {Rr['coherence_median']:.3f} (p90 {Rn['coherence_p90']:.3f} / {Rr['coherence_p90']:.3f}); log10 condition number {Rn['log10_condition_median']:.2f} / {Rr['log10_condition_median']:.2f}; "
        f"refit gain after erasing half: FVU native {Rn['refit_gain_fvu_mean']:.4f}, rotated {Rr['refit_gain_fvu_mean']:.4f}; loss native {Rn['refit_gain_loss_mean']:+.4f}, rotated {Rr['refit_gain_loss_mean']:+.4f}; Spearman of native coherence with the FVU gain {Rn['spearman_coherence_fvu_gain']:+.2f} and the loss gain {Rn['spearman_coherence_loss_gain']}; FVU gain in the top / bottom coherence quartile {Rn['fvu_gain_top_quartile']:.4f} / {Rn['fvu_gain_bottom_quartile']:.4f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e490_coherence_{name}", res, summ)
