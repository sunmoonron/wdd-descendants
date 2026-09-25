"""e484: the description as a code over an erasure channel (Shannon 1948; Elias 1955). A 16-word description is a code
for the state; erasing words is the erasure channel. Two things a code can have: redundancy (the function survives
losing some words) and error correction (the remaining words, refitted, make up for the lost ones). e465 and e466 asked
what a description omits; here the question is how it degrades.
Setup: middle depth, 8 x 256 evaluation tokens, typical positions; 16 native words, 16 rotated words, and the top 16
principal components fitted on other sequences. Erase 0, 1, 2, 4, 8 or 12 of the 16 parts: at random, the largest
coefficients first, or the smallest first; splice and measure loss recovered (1 = the clean state, 0 = the mean state).
With refit, the remaining words' coefficients are refitted by least squares before splicing (the correction).
Reported: the erasure curves; (v2) the ratio of a random half kept with refit to the intact 8-word description chosen by OMP (1 = the words are interchangeable); the number of words that can be erased at random, with refit, before loss recovered
falls below 0.9 of the intact description's; and the share of the intact value kept after erasing half the words.
Models (argument): the five.
Pre-registered (honest guesses):
- native descriptions degrade more gracefully than rotated ones: after erasing half at random, they keep a larger
  share of their intact value in all five (0.6);
- refitting recovers part of what is erased for native words (at f = 8, refit adds at least 0.05 of loss recovered)
  (0.5);
- erasing the largest coefficient first costs more than erasing the smallest first, for every dictionary (0.8)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16; D = arch.D
E = eval_ids(name); ids = E[:8, :256].to(DEV); fit = E[8:16, :256].to(DEV)
sp = Splicer(model, arch, ids, L, chunk=2)
X = block_states(model, arch, ids, [L], chunk=4)[L]; flat = X.reshape(-1, D); keep = ~sinkmask(flat); mu = flat[keep].mean(0); Xc = flat[keep] - mu; N = Xc.shape[0]
lm = sp.lossmask(keep.view(X.shape[0], -1)); base = sp.c["loss"][lm].mean().item(); ms = flat.clone(); ms[keep] = mu; mean_loss = sp.run(ms.view_as(X))["loss"][lm].mean().item()
Ff = block_states(model, arch, fit, [L], chunk=4)[L].reshape(-1, D); Ff = Ff[~sinkmask(Ff)]; P16, _ = pcs(Ff, K)
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
def rec(Xh): new = flat.clone(); new[keep] = mu + Xh; return (mean_loss - sp.run(new.view_as(X))["loss"][lm].mean().item()) / max(mean_loss - base, 1e-9)
sel_n, _, _ = omp(Xc, Au, K, batch=1024, record_err=False); cof_n, _ = refit(Xc, Au, sel_n)
sel_r, _, _ = omp(Xc, Ar, K, batch=1024, record_err=False); cof_r, _ = refit(Xc, Ar, sel_r)
Zp = Xc @ P16
FR = [0, 1, 2, 4, 8, 12]; g = torch.Generator(device=DEV).manual_seed(0)
# v2: the intact 8-word descriptions as the reference for "half of 16 erased": the ratio says whether the 16 words are interchangeable
sel8n, _, _ = omp(Xc, Au, 8, batch=1024, record_err=False); c8n, _ = refit(Xc, Au, sel8n); sel8r, _, _ = omp(Xc, Ar, 8, batch=1024, record_err=False); c8r, _ = refit(Xc, Ar, sel8r)
ref8 = dict(native=rec(torch.einsum("nk,nkd->nd", c8n, Au[sel8n])), rotated=rec(torch.einsum("nk,nkd->nd", c8r, Ar[sel8r])), pca=rec(Zp[:, :8] @ P16[:, :8].T))
res = dict(model=name, level=L, k=K, erased=FR, curves={}, intact_k8=ref8)
for kind, W, C in (("native", Au[sel_n], cof_n), ("rotated", Ar[sel_r], cof_r), ("pca", P16.T[None].expand(N, -1, -1), Zp)):
    for mode in ("random", "largest_first", "smallest_first"):
        if mode == "random": order = torch.rand(N, K, generator=g, device=DEV).argsort(1)
        elif mode == "largest_first": order = C.abs().argsort(1, descending=True)
        else: order = C.abs().argsort(1)
        plain, refitted = [], []
        for f in FR:
            kept = order[:, f:]; Wk = W.gather(1, kept[:, :, None].expand(-1, -1, D)); Ck = C.gather(1, kept)
            plain.append(rec(torch.einsum("nk,nkd->nd", Ck, Wk)))
            if kind == "pca" or f == 0: refitted.append(plain[-1]); continue
            cr = torch.linalg.lstsq(Wk.transpose(1, 2), Xc[:, :, None]).solution[:, :, 0]; refitted.append(rec(torch.einsum("nk,nkd->nd", cr, Wk)))
        res["curves"][f"{kind}_{mode}"] = dict(plain=plain, refit=refitted)
        log(f"{name} {kind} {mode}: plain " + "/".join(f"{v:.3f}" for v in plain) + " | refit " + "/".join(f"{v:.3f}" for v in refitted))
Cv = res["curves"]
def tolerance(kind):
    c = Cv[f"{kind}_random"]["refit"]; return max([f for f, v in zip(FR, c) if v >= 0.9 * c[0]], default=0)
def half_share(kind): c = Cv[f"{kind}_random"]["refit"]; return c[FR.index(8)] / max(c[0], 1e-9)
res["summary"] = {kind: dict(intact=Cv[f"{kind}_random"]["plain"][0], erasable_at_0_9=tolerance(kind), half_erased_share=half_share(kind), half_erased_over_chosen_8=Cv[f"{kind}_random"]["refit"][FR.index(8)] / max(ref8[kind], 1e-9), refit_gain_at_8=Cv[f"{kind}_random"]["refit"][FR.index(8)] - Cv[f"{kind}_random"]["plain"][FR.index(8)],
                              largest_first_at_4=Cv[f"{kind}_largest_first"]["plain"][FR.index(4)], smallest_first_at_4=Cv[f"{kind}_smallest_first"]["plain"][FR.index(4)]) for kind in ("native", "rotated", "pca")}
S = res["summary"]
res["checks"] = dict(native_more_graceful_than_rotated=S["native"]["half_erased_share"] > S["rotated"]["half_erased_share"], refit_gains_0_05=S["native"]["refit_gain_at_8"] >= 0.05, largest_costs_more=all(S[k]["largest_first_at_4"] < S[k]["smallest_first_at_4"] for k in S))
summ = (f"{name} L{L}: loss recovered intact / after erasing 1, 2, 4, 8, 12 of 16 at random with refit: " + " | ".join(f"{k} " + "/".join(f"{v:.2f}" for v in Cv[f'{k}_random']['refit']) for k in ("native", "rotated", "pca"))
        + " || share of the intact value kept after erasing half: " + ", ".join(f"{k} {S[k]['half_erased_share']:.2f}" for k in S) + "; words erasable before 0.9 of intact: " + ", ".join(f"{k} {S[k]['erasable_at_0_9']}" for k in S)
        + "; random half kept over the chosen 8 (interchangeability): " + ", ".join(f"{k} {S[k]['half_erased_over_chosen_8']:.2f}" for k in S) + "; refit gain at half erased: " + ", ".join(f"{k} {S[k]['refit_gain_at_8']:+.3f}" for k in ("native", "rotated")) + "; erasing 4 largest / 4 smallest: " + ", ".join(f"{k} {S[k]['largest_first_at_4']:.2f}/{S[k]['smallest_first_at_4']:.2f}" for k in S) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e484_erasure_{name}", res, summ)
