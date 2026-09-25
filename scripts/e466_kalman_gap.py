"""e466: the "Kalman gap". How many extra dimensions must be added to a 16-word native description before the model's
future is preserved? Suggested by an external review, following e465: the network does not regrow what 16 words leave
out.
Setup: the middle depth L, WikiText-2 evaluation text. Directions are fitted on 4 sequences and splices scored on 4
others (typical positions replaced, sinks exact). Codes of total dimension 16 + r, for r = 0, 4, 8, 16, 32, 64:
- native + remainder: 16 own words (OMP), plus the projection of what they leave out on the top r principal directions
  of the native remainder;
- rotated + remainder: the same with 16 rotated words;
- PCA: the top 16 + r principal components of the states.
Measured: loss recovered (e388's splice), the mean next-token KL divergence from the clean run, and the divergence of
the state 4 blocks later relative to its norm (e465). The gap is the smallest r at which 95% of the loss is recovered.
Models (argument): gpt2, qwen05.
Pre-registered (honest guesses):
- native plus 8-16 remainder directions recovers 95% of the loss (0.5);
- native plus remainder beats PCA at every r up to 32 (0.6);
- the gap closes with fewer extra dimensions for native than for rotated words (0.7)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16; RS = [0, 4, 8, 16, 32, 64]
E = eval_ids(name); fit_ids, ids = E[:4, :256].to(DEV), E[4:8, :256].to(DEV)
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
def states(i):
    X = block_states(model, arch, i, [L], chunk=4)[L]; f = X.reshape(-1, arch.D); return X, f, ~sinkmask(f)
Xf, ff, kf = states(fit_ids); mu = ff[kf].mean(0); Cf = ff[kf] - mu
def desc(Xc, Dct):
    sel, _, _ = omp(Xc, Dct, K, batch=1024, record_err=False); cof, _ = refit(Xc, Dct, sel); return torch.einsum("nk,nkd->nd", cof, Dct[sel])
def top_pcs(Z, n):
    ev_, U_ = torch.linalg.eigh(torch.cov(Z.T.double(), correction=0)); return U_.flip(-1)[:, :n].float()
Rn_fit, Rr_fit = Cf - desc(Cf, Au), Cf - desc(Cf, Ar)
Pn, Pr, Ps = top_pcs(Rn_fit, max(RS)), top_pcs(Rr_fit, max(RS)), top_pcs(Cf, K + max(RS))
sp = Splicer(model, arch, ids, L, track=[L + 4], chunk=2)
X, flat, keep = states(ids); Xc = flat[keep] - mu; lm = sp.lossmask(keep.view(X.shape[0], -1))
base_loss = sp.c["loss"][lm].mean().item(); ms = flat.clone(); ms[keep] = mu; mean_loss = sp.run(ms.view_as(X))["loss"][lm].mean().item()
lp_clean = None
def score(rep):
    new = flat.clone(); new[keep] = mu + rep; r = sp.run(new.view_as(X), track=True)
    h1 = r["track"][L + 4].reshape(-1, arch.D)[keep]; h0 = sp.c["track"][L + 4].reshape(-1, arch.D)[keep]
    rec = (mean_loss - r["loss"][lm].mean().item()) / max(mean_loss - base_loss, 1e-9)
    kl_proxy = (r["loss"][lm] - sp.c["loss"][lm]).mean().item()                      # mean change of the loss (nats)
    div = ((h1 - h0).norm(dim=-1) / (h0 - h0.mean(0)).norm(dim=-1)).median().item()
    return dict(rec=rec, d_loss=kl_proxy, logit_cos=r["lcos"][lm].mean().item(), div4=div)
Dn, Dr = desc(Xc, Au), desc(Xc, Ar); Rn, Rr = Xc - Dn, Xc - Dr
res = dict(model=name, level=L, k=K, r=RS, variants={"native_plus_rem": {}, "rotated_plus_rem": {}, "pca": {}})
for r_ in RS:
    res["variants"]["native_plus_rem"][r_] = score(Dn + (Rn @ Pn[:, :r_] @ Pn[:, :r_].T if r_ else 0))
    res["variants"]["rotated_plus_rem"][r_] = score(Dr + (Rr @ Pr[:, :r_] @ Pr[:, :r_].T if r_ else 0))
    res["variants"]["pca"][r_] = score(Xc @ Ps[:, :K + r_] @ Ps[:, :K + r_].T)
    log(f"{name} r={r_}: loss recovered native+rem {res['variants']['native_plus_rem'][r_]['rec']:.3f}, rotated+rem {res['variants']['rotated_plus_rem'][r_]['rec']:.3f}, pca {res['variants']['pca'][r_]['rec']:.3f} | "
        f"divergence at +4 {res['variants']['native_plus_rem'][r_]['div4']:.2f}/{res['variants']['rotated_plus_rem'][r_]['div4']:.2f}/{res['variants']['pca'][r_]['div4']:.2f}")
gap = lambda v: next((r_ for r_ in RS if res["variants"][v][r_]["rec"] >= 0.95), None)
res["gap_to_95"] = {v: gap(v) for v in res["variants"]}
V = res["variants"]
res["checks"] = dict(native_95_by_16=(res["gap_to_95"]["native_plus_rem"] is not None and res["gap_to_95"]["native_plus_rem"] <= 16),
                     native_beats_pca_to_32=all(V["native_plus_rem"][r_]["rec"] > V["pca"][r_]["rec"] for r_ in RS if r_ <= 32),
                     native_closes_faster_than_rotated=(res["gap_to_95"]["native_plus_rem"] or 999) < (res["gap_to_95"]["rotated_plus_rem"] or 999))
row = lambda v: " ".join(f"{V[v][r_]['rec']:.2f}" for r_ in RS)
summ = (f"{name} L{L}, loss recovered at r = {RS} extra dimensions: native+remainder {row('native_plus_rem')} | rotated+remainder {row('rotated_plus_rem')} | pca(16+r) {row('pca')} | "
        f"95% reached at r = {res['gap_to_95']} | divergence at +4 (native+rem): " + " ".join(f"{V['native_plus_rem'][r_]['div4']:.2f}" for r_ in RS) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e466_kalmangap_{name}", res, summ)
