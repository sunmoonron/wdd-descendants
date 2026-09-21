"""e302: dose-response and interaction in causal coordinates. Eight PLS coordinates u_k with paired logit directions
l_k (as e289). Dose-response: inject delta x the natural footprint norm along u_k for delta in {-2, -1, -0.5, 0.5,
1, 2}, read the paired scalar F_k(delta) = median over tokens of the centred logit change projected on l_k; report
per coordinate the saturation ratio F(2)/(2 F(1)), the sign asymmetry (F(1) + F(-1)) / |F(1)|, and the linear R2 of
the six doses. Interaction: inject u_i + u_j for all pairs; I_ij = E_ij - E_i - E_j relative to the mean single
response norm; the matrix's mean, maximum, diagonal (self-interaction = amplitude nonlinearity) and first-mode share."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); F = (S0i[L] - S1i[L])[idx]; Fc = F - F.mean(0, keepdim=True); fnorm = F.norm(dim=1).median()
Ud, Sd, Vd = torch.linalg.svd(dl, full_matrices=False); Zs = Ud[:, :256] * Sd[:256][None]; B = Vd[:256].T; Zs = Zs - Zs.mean(0, keepdim=True); Up, Sp, Wt = torch.linalg.svd(Fc.T @ Zs, full_matrices=False); nk = 8; u = Up[:, :nk].T; l = unit((B @ Wt[:nk].T).T)
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0][:1024]; base = run(positions=foreign); lg0 = base["lg"]
def resp(v):
    inj = torch.zeros(NT, D, device=DEV); inj[foreign] = fnorm * v[None]; r = run(positions=foreign, inject=inj, inject_block=L + 1); dlr = r["lg"] - lg0; return (dlr - dlr.mean(1, keepdim=True))
deltas = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]; dose = {}
for k in range(nk):
    Fk = {dd: (resp(dd * u[k]) @ l[k]).median().item() for dd in deltas}; X = torch.tensor(deltas, device=DEV); Y = torch.tensor([Fk[dd] for dd in deltas], device=DEV); slope = (X * Y).sum() / (X * X).sum(); r2 = 1 - ((Y - slope * X) ** 2).sum().item() / (Y ** 2).sum().item(); dose[k] = dict(saturation=Fk[2.0] / (2 * Fk[1.0]) if abs(Fk[1.0]) > 1e-9 else float("nan"), asymmetry=(Fk[1.0] + Fk[-1.0]) / abs(Fk[1.0]) if abs(Fk[1.0]) > 1e-9 else float("nan"), linear_r2=r2, F1=Fk[1.0], F2=Fk[2.0], Fm1=Fk[-1.0])
E = {k: resp(u[k]) for k in range(nk)}; En = {k: E[k].norm(dim=1).median().item() for k in range(nk)}; I = torch.zeros(nk, nk, device=DEV)
for i in range(nk):
    for j in range(i, nk):
        Eij = resp(u[i] + u[j]); I[i, j] = I[j, i] = ((Eij - E[i] - E[j]).norm(dim=1) / (0.5 * (E[i].norm(dim=1) + E[j].norm(dim=1))).clamp_min(1e-9)).median().item()
off = I[~torch.eye(nk, dtype=torch.bool, device=DEV)]; sv = torch.linalg.svdvals(I); inter = dict(mean_offdiag=off.mean().item(), max_offdiag=off.max().item(), mean_diag=I.diagonal().mean().item(), first_mode_share=(sv[0] ** 2 / (sv ** 2).sum()).item())
sat = [dose[k]["saturation"] for k in range(nk)]; asym = [dose[k]["asymmetry"] for k in range(nk)]; r2s = [dose[k]["linear_r2"] for k in range(nk)]
log(f"{tag} (K {K}): dose-response over 8 coordinates: saturation F(2)/2F(1) " + "/".join(f"{v:.2f}" for v in sat) + "; asymmetry (F(1)+F(-1))/|F(1)| " + "/".join(f"{v:+.2f}" for v in asym) + "; linear R2 " + "/".join(f"{v:.2f}" for v in r2s) + f" | interaction matrix: off-diagonal mean {inter['mean_offdiag']:.2f} max {inter['max_offdiag']:.2f}, diagonal (self) {inter['mean_diag']:.2f}, first-mode share {inter['first_mode_share']:.2f}")
record(f"e302_dose_{tag}", dict(model=tag, b=b, L=L, K=K, dose={str(k): v for k, v in dose.items()}, interaction=inter, matrix=I.tolist()), "saturation " + "/".join(f"{v:.2f}" for v in sat) + " asymmetry " + "/".join(f"{v:+.2f}" for v in asym) + " linear R2 " + "/".join(f"{v:.2f}" for v in r2s) + f" | interaction offdiag {inter['mean_offdiag']:.2f} max {inter['max_offdiag']:.2f} diag {inter['mean_diag']:.2f} mode1 {inter['first_mode_share']:.2f}")
