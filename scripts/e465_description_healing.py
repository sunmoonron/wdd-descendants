"""e465: after a state is replaced by its 16-word native description, does the network regrow what the description left
out? This is the "description under amnesia" test, suggested by an external review.
e388 found the 16-word description keeps much of the loss. Here the question is dynamic. Downstream of the splice, does
the network move back toward the natural trajectory (healing), or does the omission persist? And is the native
description more generative than other descriptions of the same size?
Setup: the middle depth L, 8 x 256 evaluation tokens, typical positions replaced and sinks kept exact. Four
replacements of the state:
- native: 16 own words (OMP);
- rotated: 16 rotated words (same Gram matrix, no provenance);
- PCA: the top 16 principal components;
- random: the state plus a random vector of the same norm as the native omission (the same amount of error in a
  random direction).
Downstream, at blocks L+1, L+2 and L+4, the divergence from the natural run relative to the omission at L is:
median over positions of |h' - h| (at the block) / |x' - x| (at L). Below 1 means healing. The state grows with depth,
so the divergence relative to the centred state's norm is reported too. Also: loss recovered, and at block L+2 the
overlap (Jaccard) of the 16-word native descriptions of the spliced and natural states.
Models (argument): gpt2, qwen05.
Pre-registered (honest guesses):
- native omissions heal more (lower relative divergence at L+4) than rotated, PCA and equal-size random omissions (0.5);
- downstream, the spliced state is described with mostly the same native words as the natural state (Jaccard at least
  0.5) (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16; TR = [L + 1, L + 2, L + 4]
ids = eval_ids(name)[:8, :256].to(DEV)
sp = Splicer(model, arch, ids, L, track=TR, chunk=2)
X = block_states(model, arch, ids, [L], chunk=4)[L]                            # [B, T-1, D]
flat = X.reshape(-1, arch.D); keep = ~sinkmask(flat); mu = flat[keep].mean(0); Xc = flat[keep] - mu
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
def omp_rec(Dct):
    sel, _, _ = omp(Xc, Dct, K, batch=1024, record_err=False); cof, _ = refit(Xc, Dct, sel); return torch.einsum("nk,nkd->nd", cof, Dct[sel]), sel
rec_nat, sel_nat = omp_rec(Au); rec_rot, _ = omp_rec(Ar)
ev_, U_ = torch.linalg.eigh(torch.cov(Xc.T.double(), correction=0)); P = U_.flip(-1)[:, :K].float(); rec_pca = Xc @ P @ P.T
omission = (Xc - rec_nat).norm(dim=-1); g = torch.Generator(device=DEV).manual_seed(0)
rnd = torch.randn(Xc.shape, generator=g, device=DEV); rnd = rnd / rnd.norm(dim=-1, keepdim=True) * omission[:, None]
variants = {"native": mu + rec_nat, "rotated": mu + rec_rot, "pca": mu + rec_pca, "random": flat[keep] + rnd}
lm = sp.lossmask(keep.view(X.shape[0], -1))
base_loss = sp.c["loss"][lm].mean().item()
mean_state = flat.clone(); mean_state[keep] = mu; mean_loss = sp.run(mean_state.view_as(X))["loss"][lm].mean().item()
A2, _ = build_dictionary(arch, blocks=list(range(L + 3))); Au2 = unitr(A2); del A2
res = dict(model=name, level=L, k=K, blocks=TR, variants={})
for vn, rep in variants.items():
    newflat = flat.clone(); newflat[keep] = rep; r = sp.run(newflat.view_as(X), track=True)
    d0 = (newflat - flat)[keep].norm(dim=-1)
    out = dict(omission_rel=float((d0 / Xc.norm(dim=-1)).median()), loss_recovered=(mean_loss - r["loss"][lm].mean().item()) / max(mean_loss - base_loss, 1e-9), heal={}, rel_state={})
    for b_ in TR:
        h1 = r["track"][b_].reshape(-1, arch.D)[keep]; h0 = sp.c["track"][b_].reshape(-1, arch.D)[keep]
        dv = (h1 - h0).norm(dim=-1); mb = h0.mean(0)
        out["heal"][b_ - L] = float((dv / d0.clamp_min(1e-9)).median()); out["rel_state"][b_ - L] = float((dv / (h0 - mb).norm(dim=-1)).median())
        if b_ == L + 2:
            s1, _, _ = omp(h1 - mb, Au2, K, batch=1024, record_err=False); s0, _, _ = omp(h0 - mb, Au2, K, batch=1024, record_err=False)
            jac = [len(set(a.tolist()) & set(c.tolist())) / len(set(a.tolist()) | set(c.tolist())) for a, c in zip(s1, s0)]
            out["jaccard_L2"] = sum(jac) / len(jac)
    res["variants"][vn] = out
    log(f"{name} {vn}: omission {out['omission_rel']:.2f} of the state, loss recovered {out['loss_recovered']:.2f} | divergence / omission at +1/+2/+4: " + " ".join(f"{out['heal'][k]:.2f}" for k in (1, 2, 4))
        + " | relative to the state: " + " ".join(f"{out['rel_state'][k]:.2f}" for k in (1, 2, 4)) + f" | native-word overlap with the natural run at +2 {out['jaccard_L2']:.2f}")
V = res["variants"]
res["checks"] = dict(native_heals_most=all(V["native"]["heal"][4] < V[v]["heal"][4] for v in ("rotated", "pca", "random")), downstream_same_words=V["native"]["jaccard_L2"] >= 0.5)
summ = (f"{name} L{L}, splice of 16-word descriptions: " + " || ".join(f"{vn}: omission {o['omission_rel']:.2f}, recovered {o['loss_recovered']:.2f}, divergence/omission +1/+2/+4 {o['heal'][1]:.2f}/{o['heal'][2]:.2f}/{o['heal'][4]:.2f}, "
        f"relative to state {o['rel_state'][1]:.2f}/{o['rel_state'][2]:.2f}/{o['rel_state'][4]:.2f}, word overlap +2 {o['jaccard_L2']:.2f}" for vn, o in V.items()) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e465_healing_{name}", res, summ)
