"""e293: the dynamics of the causal core. z = the projection of the per-token descendant onto a 16-dimensional
supervised subspace (function PLS, and identity) at level L; z' the same at L+2 and L+4, in the basis recomputed
there and in the fixed basis of L. The linear map A with z' ~ A z is fitted on half the tokens and characterised on
the other half: R2 (how much of the future core is linear in the present core), gain (geometric mean of singular
values) and spread (largest/smallest), distance from a scaled rotation (Procrustes residual), mixing (off-diagonal
energy in the fixed basis), rotation (fraction of complex eigenvalues), and new dimensions (1 - R2)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; d = 16; futures = [lf for lf in (L + 2, L + 4) if lf < NB - 1]
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L] + futures, NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]
G = dl[tr] @ dl[tr].T; evg, Vg = torch.linalg.eigh(G); Z = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Z = Z - Z.mean(0, keepdim=True)
def bases(X):
    Xc = X - X[tr].mean(0, keepdim=True); cents = torch.stack([Xc[tr][lab_i[tr] == k].mean(0) for k in range(K)]); w = torch.bincount(lab_i[tr], minlength=K).float(); Sb = (cents * w[:, None]).T @ cents / w.sum(); return Xc, {"function": torch.linalg.svd(Xc[tr].T @ Z, full_matrices=False)[0][:, :d], "identity": torch.linalg.eigh(Sb)[1].flip(1)[:, :d]}
FL, SL = bases((S0i[L] - S1i[L])[idx]); out = {}
for lf in futures:
    Ff, Sf = bases((S0i[lf] - S1i[lf])[idx])
    for nm in ("function", "identity"):
        zL = FL @ SL[nm]
        for basis, S_use in (("recomputed", Sf[nm]), ("fixed", SL[nm])):
            zf = Ff @ S_use; A = torch.linalg.lstsq(zL[tr], zf[tr]).solution; pred = zL[te] @ A; r2 = 1 - ((zf[te] - pred) ** 2).sum().item() / ((zf[te] - zf[tr].mean(0, keepdim=True)) ** 2).sum().item(); sv = torch.linalg.svdvals(A); U_, _, Vh_ = torch.linalg.svd(A); Om = U_ @ Vh_; s = (A * Om).sum() / (Om * Om).sum(); proc = ((A - s * Om).norm() / A.norm()).item(); ev = torch.linalg.eigvals(A); rec = dict(r2=r2, gain=sv.log().mean().exp().item(), spread=(sv.max() / sv.min().clamp_min(1e-9)).item(), procrustes_residual=proc, offdiag_energy=((A ** 2).sum() - (A.diagonal() ** 2).sum()).item() / (A ** 2).sum().item(), complex_fraction=(ev.imag.abs() > 1e-6 * ev.abs().max()).float().mean().item(), max_rotation=(ev.imag.abs() / ev.abs().clamp_min(1e-9)).max().item()); out[f"{nm}_{lf}_{basis}"] = rec
log(f"{tag} (K {K}): " + " | ".join(f"{key}: R2 {v['r2']:.2f}, gain {v['gain']:.2f}, spread {v['spread']:.1f}, Procrustes residual {v['procrustes_residual']:.2f}, off-diagonal {v['offdiag_energy']:.2f}, complex eigenvalues {v['complex_fraction']:.2f} (max rotation {v['max_rotation']:.2f})" for key, v in out.items()))
record(f"e293_dynamics_{tag}", dict(model=tag, b=b, L=L, K=K, d=d, maps=out), " | ".join(f"{key}: R2 {v['r2']:.2f} gain {v['gain']:.2f} spread {v['spread']:.1f} procrustes {v['procrustes_residual']:.2f} offdiag {v['offdiag_energy']:.2f} complex {v['complex_fraction']:.2f}" for key, v in out.items()))
