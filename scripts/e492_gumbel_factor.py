"""e492: the provenance factor across depth and training. e483 found that the competitor maximum of the native
dictionary scales with sqrt(2 ln m) at 2.9-6.2 times the slope of the rotated dictionary, whose slope equals the
second-moment prediction exactly. That factor is a one-number measure of how far the model's own atoms are aligned
with its states beyond their Gram matrix. Here it is measured at five depths in each model, and across Pythia-410m's
training, where it should start near 1 (own = rotated at initialisation, e401) and rise as the vocabulary forms
(e395, e443).
Setup: 8 x 256 evaluation tokens, typical positions; blocks 1, NB/4, NB/2, 3NB/4 and NB-2; the dictionary up to the
block; sub-dictionaries of 2^10, 2^12, 2^14 and all atoms; the factor is the ratio of the native slope to the rotated
slope of the mean maximum against sqrt(2 ln m'); also the ratio of the mean maxima at the full dictionary.
Models (arguments): name [revision].
Pre-registered (honest guesses):
- the factor is above 2 at every depth in every final model (0.6);
- it is largest in the early blocks and falls with depth (0.5);
- in Pythia it is within 0.2 of 1 at step 1000 and above 2 by step 16000 (0.5)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB
blocks = sorted({1, NB // 4, NB // 2, (3 * NB) // 4, NB - 2}); ids = eval_ids(name)[:8, :256].to(DEV)
S_ = block_states(model, arch, ids, blocks, chunk=4)
def maxcorr(Ux, Dct):
    out = torch.empty(Ux.shape[0], device=DEV)
    for s in range(0, Ux.shape[0], 256): out[s:s + 256] = (Ux[s:s + 256] @ Dct.T).abs().max(1).values
    return out
def slope(U, Dct, ms, g):
    means = []
    for m in ms:
        idx = torch.randperm(Dct.shape[0], generator=g, device=DEV)[:m] if m < Dct.shape[0] else torch.arange(Dct.shape[0], device=DEV); means.append(float(maxcorr(U, Dct[idx]).mean()))
    xs = torch.tensor([math.sqrt(2 * math.log(m)) for m in ms]); ys = torch.tensor(means); return float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum()), means
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, arch.D); keep = ~sinkmask(X); U = unitr(X[keep] - X[keep].mean(0))
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; m_all = Au.shape[0]; ms = [2 ** 10, 2 ** 12, 2 ** 14, m_all]
    g = torch.Generator(device=DEV).manual_seed(0); sn, mn = slope(U, Au, ms, g); g = torch.Generator(device=DEV).manual_seed(0); sr, mr = slope(U, Ar, ms, g)
    res["by_block"][b] = dict(m=m_all, native_slope=sn, rotated_slope=sr, factor=sn / sr, native_mean_max=mn[-1], rotated_mean_max=mr[-1], mean_max_ratio=mn[-1] / mr[-1])
    log(f"{name}{' ' + rev if rev else ''} block {b} (m {m_all}): native slope {sn:.4f}, rotated {sr:.4f}, factor {sn / sr:.2f}; mean maximum {mn[-1]:.3f} / {mr[-1]:.3f} (ratio {mn[-1] / mr[-1]:.2f})")
    del Au, Ar; torch.cuda.empty_cache()
fs = [res["by_block"][b]["factor"] for b in blocks]
res["checks"] = dict(factor_over_2_everywhere=all(f > 2 for f in fs), falls_with_depth=fs[0] == max(fs) and fs[-1] < fs[0])
summ = f"{name}{' ' + rev if rev else ''}: provenance factor (native over rotated slope of the competitor maximum) by block " + ", ".join(f"{b}: {res['by_block'][b]['factor']:.2f} (mean-max ratio {res['by_block'][b]['mean_max_ratio']:.2f})" for b in blocks) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e492_gumbelfactor_{name}{'_' + rev if rev else ''}", res, summ)
