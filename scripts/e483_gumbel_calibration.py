"""e483: the instrument's resolution as an extreme-value law (Fisher and Tippett 1928, Gumbel 1935). WDD identifies a
write when its projection on the state beats the largest projection of the competing atoms (the prominence law, e56,
e127). The largest of m nearly independent projections has a Gumbel distribution whose location grows as
sigma * sqrt(2 ln m) and whose scale shrinks as sigma / sqrt(2 ln m), with sigma^2 = u^T C_A u the second moment of the
atoms along the unit state u. This run fits that law to the five models and derives the instrument's detection curve
from it with no free parameter beyond the fit.
Setup: middle depth, 8 x 256 evaluation tokens, typical positions, the dictionary up to the middle block.
- Distribution: the competitor maximum M(x) = max_j |<u, a_j>| over positions; Gumbel fit by moments and its KS
  distance, against a normal fit.
- Scaling: the mean of M over random sub-dictionaries of m = 2^10, 2^12, 2^14 atoms and all, against sqrt(2 ln m);
  the fitted slope against the predicted sigma.
- Detection curve: a random MLP row injected into a random state at relative size s (0.05 to 0.5 of the state's
  norm); measured: it is OMP's first pick; predicted: the Gumbel CDF at the realised prominence q = <x', a>/|x'|.
  Also the size at which half the injections are detected (the instrument's resolution).
v2 adds the rotated dictionary's maximum and scaling: with the same Gram matrix and no alignment with the states, its slope should match the second-moment prediction if the native excess is provenance.
Models (argument): the five.
Pre-registered (honest guesses):
- the competitor maximum is closer to Gumbel than to normal (smaller KS) in all five (0.6);
- the mean of M scales linearly in sqrt(2 ln m) with R^2 above 0.95 (0.7);
- the Gumbel-predicted detection rate is within 0.1 of the measured one at every size (0.5)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2
ids = eval_ids(name)[:8, :256].to(DEV)
X = block_states(model, arch, ids, [L], chunk=4)[L].reshape(-1, arch.D); keep = ~sinkmask(X); Xc = X[keep] - X[keep].mean(0); N = Xc.shape[0]
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; m_all = Au.shape[0]; typ = lab["type"].to(DEV)
U = unitr(Xc)
def maxcorr(Ux, Dct):
    out = torch.empty(Ux.shape[0], device=DEV)
    for s in range(0, Ux.shape[0], 256): out[s:s + 256] = (Ux[s:s + 256] @ Dct.T).abs().max(1).values
    return out
EULER = 0.5772156649
def gumbel_fit(v): beta = v.std() * math.sqrt(6) / math.pi; return float(v.mean() - EULER * beta), float(beta)
def ks(v, F):
    z = torch.sort(v).values; emp = torch.arange(1, z.numel() + 1, device=DEV) / z.numel(); return float((F(z) - emp).abs().max())
M = maxcorr(U, Au); mu, beta = gumbel_fit(M)
ks_g = ks(M, lambda z: torch.exp(-torch.exp(-(z - mu) / beta))); mean, sd = M.mean(), M.std(); ks_n = ks(M, lambda z: 0.5 * (1 + torch.erf((z - mean) / (sd * math.sqrt(2)))))
g = torch.Generator(device=DEV).manual_seed(0); ms = [2 ** 10, 2 ** 12, 2 ** 14, m_all]; means = []
for m in ms:
    idx = torch.randperm(m_all, generator=g, device=DEV)[:m] if m < m_all else torch.arange(m_all, device=DEV); means.append(float(maxcorr(U, Au[idx]).mean()))
CA = (Au.T @ Au) / m_all; sig = ((U @ CA) * U).sum(1).sqrt(); sigma = float(sig.mean())
# v2: the same for the rotated dictionary (same Gram matrix, no alignment with the states): is the excess slope provenance?
Mr = maxcorr(U, Ar); mur, betar = gumbel_fit(Mr); means_r = []
for m in ms:
    idx = torch.randperm(m_all, generator=g, device=DEV)[:m] if m < m_all else torch.arange(m_all, device=DEV); means_r.append(float(maxcorr(U, Ar[idx]).mean()))
yr = torch.tensor(means_r); slope_r = float(((xs - xs.mean()) * (yr - yr.mean())).sum() / ((xs - xs.mean()) ** 2).sum()) if False else None
xs = torch.tensor([math.sqrt(2 * math.log(m)) for m in ms]); ys = torch.tensor(means)
slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum()); icpt = float(ys.mean() - slope * xs.mean())
r2 = float(1 - ((ys - (slope * xs + icpt)) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())
yr = torch.tensor(means_r); slope_r = float(((xs - xs.mean()) * (yr - yr.mean())).sum() / ((xs - xs.mean()) ** 2).sum()); CAr = (Ar.T @ Ar) / m_all; sigma_r = float(((U @ CAr) * U).sum(1).sqrt().mean())
mlp_idx = torch.nonzero(typ == T_MLP)[:, 0]; gp = torch.Generator(device=DEV).manual_seed(1)
pos = torch.randperm(N, generator=gp, device=DEV)[:200]; atoms = mlp_idx[torch.randint(0, mlp_idx.numel(), (200,), generator=gp, device=DEV)]
SIZES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]; curve = []
for s in SIZES:
    x = Xc[pos]; a = Au[atoms]; xp = x + s * x.norm(dim=-1, keepdim=True) * a; q = (xp * a).sum(1) / xp.norm(dim=-1)
    c = (unitr(xp) @ Au.T).abs(); det = (c.argmax(1) == atoms).float(); pred = torch.exp(-torch.exp(-(q - mu) / beta))
    curve.append(dict(size=s, prominence=float(q.mean()), detected=float(det.mean()), predicted=float(pred.mean())))
p50 = next((c["size"] for c in curve if c["detected"] >= 0.5), None)
res = dict(model=name, level=L, n_positions=N, m=m_all, gumbel=dict(mu=mu, beta=beta, ks=ks_g), normal_ks=ks_n, scaling=dict(m=ms, mean_max=means, slope=slope, intercept=icpt, r2=r2, sigma_predicted=sigma, slope_over_sigma=slope / sigma), detection=curve, half_detection_size=p50, rotated=dict(mu=mur, beta=betar, mean_max=means_r, slope=slope_r, sigma_predicted=sigma_r, slope_over_sigma=slope_r / sigma_r))
res["checks"] = dict(gumbel_closer_than_normal=ks_g < ks_n, scaling_r2_over_0_95=r2 > 0.95, prediction_within_0_1=all(abs(c["detected"] - c["predicted"]) <= 0.1 for c in curve))
summ = (f"{name} L{L}, m={m_all}, {N} positions: competitor maximum Gumbel(mu {mu:.3f}, beta {beta:.4f}), KS {ks_g:.3f} against normal {ks_n:.3f} | scaling of the mean maximum with sqrt(2 ln m): slope {slope:.4f} (predicted sigma {sigma:.4f}, ratio {slope / sigma:.2f}), R^2 {r2:.3f}, means " + "/".join(f"{v:.3f}" for v in means)
        + f" | rotated dictionary: Gumbel(mu {mur:.3f}, beta {betar:.4f}), slope {slope_r:.4f} (sigma {sigma_r:.4f}, ratio {slope_r / sigma_r:.2f}), means " + "/".join(f"{v:.3f}" for v in means_r) + " | detection of an injected MLP row, size: measured / predicted (prominence): " + ", ".join(f"{c['size']}: {c['detected']:.2f}/{c['predicted']:.2f} ({c['prominence']:.2f})" for c in curve) + f"; half detection at size {p50} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e483_gumbel_{name}", res, summ)
