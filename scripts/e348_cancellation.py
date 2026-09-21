"""e348: cancel a natural perturbation with its estimated negative, and with a functionally equivalent unrelated one.
For each candidate neuron at its own tokens: the natural footprint's core coordinate z; injected at the block-(L+1)
input: (i) the negative of the descendant centroid direction scaled to the footprint norm (physical cancellation),
(ii) the negative of the core part only (coordinate cancellation, the complement untouched), (iii) a random vector
of the same norm; measured: the residual logit deviation relative to the ablation alone (restoration), and the
residual core norm; plus norm-preserving steering: a coordinate direction added while the token's residual norm is
held fixed, the paired logit direction response against the unconstrained one."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Ud, Sd, Vd = torch.linalg.svd(nat["dl"], full_matrices=False); Zs = Ud[:, :256] * Sd[:256][None]; B = Vd[:256].T; Zs = Zs - Zs.mean(0, keepdim=True); Up, Sp, Wt = torch.linalg.svd(Fc.T @ Zs, full_matrices=False); P = Up[:, :d]; u = Up[:, :8].T; l = unit((B @ Wt[:8].T).T); res = {"physical": [], "coordinate": [], "random": []}
for k in range(min(S.K, 10)):
    rows = S.idx[S.lab_i == k]; base = S.run(positions=rows); r_ab = S.run(S.tn, positions=rows, ablate_positions=rows); A = r_ab["lg"] - base["lg"]; foot = (base[L] - r_ab[L])[rows]; c = foot.mean(0)
    for nm, v in (("physical", c), ("coordinate", (c @ P) @ P.T), ("random", unit(torch.randn(S.D, device=DEV)) * c.norm())):
        inj = torch.zeros(S.NT, S.D, device=DEV); inj[rows] = v[None]; r = S.run(S.tn, positions=rows, ablate_positions=rows, inject=inj, inject_block=L + 1); R = r["lg"] - base["lg"]; res[nm].append(dict(restoration=(1 - R.norm(dim=1) / A.norm(dim=1).clamp_min(1e-9)).median().item(), core_residual=(((base[L] - r[L])[rows] @ P).norm(dim=1) / (foot @ P).norm(dim=1).clamp_min(1e-9)).median().item()))
summ = {nm: dict(restoration=float(torch.tensor([x["restoration"] for x in v]).median()), core_residual=float(torch.tensor([x["core_residual"] for x in v]).median())) for nm, v in res.items()}
sub = S.foreign[:1024]; base = S.run(positions=sub); lg0 = base["lg"]; x0 = S.S0[L][sub]; fnorm = F.norm(dim=1).median(); steer = []
for k in range(8):
    inj = torch.zeros(S.NT, S.D, device=DEV); inj[sub] = fnorm * u[k][None]; r = S.run(positions=sub, inject=inj, inject_block=L + 1); dl = r["lg"] - lg0; dl = dl - dl.mean(1, keepdim=True); free = (dl @ l[k]).median().item(); inj2 = torch.zeros(S.NT, S.D, device=DEV); inj2[sub] = unit(x0 + fnorm * u[k][None]) * x0.norm(dim=1, keepdim=True) - x0; r2 = S.run(positions=sub, inject=inj2, inject_block=L + 1); dl2 = r2["lg"] - lg0; dl2 = dl2 - dl2.mean(1, keepdim=True); steer.append((free, (dl2 @ l[k]).median().item()))
st = torch.tensor(steer); ratio = (st[:, 1] / st[:, 0].abs().clamp_min(1e-9)).median().item()
log(f"{tag} (K {S.K}): cancelling a natural footprint at the block-(L+1) input: physical negative restores {summ['physical']['restoration']:+.2f} of the logit deviation (core residual {summ['physical']['core_residual']:.2f}); coordinate-only negative {summ['coordinate']['restoration']:+.2f} ({summ['coordinate']['core_residual']:.2f}); random vector {summ['random']['restoration']:+.2f} ({summ['random']['core_residual']:.2f}) | norm-preserving steering keeps {ratio:.2f} of the unconstrained paired response (all 8 same sign: {(torch.sign(st[:, 0]) == torch.sign(st[:, 1])).float().mean().item():.2f})")
record(f"e348_cancel_{tag}", dict(model=tag, L=L, K=S.K, cancellation=summ, norm_preserving_ratio=ratio, steer=steer), f"physical {summ['physical']['restoration']:+.2f} coord {summ['coordinate']['restoration']:+.2f} random {summ['random']['restoration']:+.2f} | norm-preserving {ratio:.2f}")
