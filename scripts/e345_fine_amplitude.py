"""e345: fine amplitude sweep and second derivatives per coordinate. Along each of eight core directions, injections at
0.03, 0.1, 0.3, 1, 3, 10 and 30 times the natural footprint norm, both signs: the paired scalar response, fitted per
coordinate as a + b x + c x^2 + e x^3 over the log-spaced doses; the amplitude at which the response leaves linearity
(relative deviation > 20%), the sign-asymmetry onset, and whether KL, the logits and the top-1 probability leave
linearity at the same amplitude."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Ud, Sd, Vd = torch.linalg.svd(nat["dl"], full_matrices=False); Zs = Ud[:, :256] * Sd[:256][None]; B = Vd[:256].T; Zs = Zs - Zs.mean(0, keepdim=True); Up, Sp, Wt = torch.linalg.svd(Fc.T @ Zs, full_matrices=False); nk = 8; u = Up[:, :nk].T; l = unit((B @ Wt[:nk].T).T); fnorm = F.norm(dim=1).median(); sub = S.foreign[:1024]; base = S.run(positions=sub); lg0 = base["lg"]; lp0 = torch.log_softmax(lg0, -1); doses = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]; out = {}
def probe(v, al):
    inj = torch.zeros(S.NT, S.D, device=DEV); inj[sub] = al * fnorm * v[None]; r = S.run(positions=sub, inject=inj, inject_block=L + 1); dl = r["lg"] - lg0; dl = dl - dl.mean(1, keepdim=True); lp1 = torch.log_softmax(r["lg"], -1); return (dl @ l.T).median(0).values, (lp0.exp() * (lp0 - lp1)).sum(1).median().item(), (lp1.exp().max(1).values - lp0.exp().max(1).values).median().item()
for k in range(nk):
    Fp = {}; Fm = {}; KLp = {}; T1p = {}
    for al in doses: Fp[al], KLp[al], T1p[al] = probe(u[k], al); Fm[al], _, _ = probe(-u[k], al)
    fk = torch.tensor([Fp[al][k].item() for al in doses]); fmk = torch.tensor([Fm[al][k].item() for al in doses]); x = torch.tensor(doses); slope = (fk[:2] / x[:2]).mean(); lin_dev = (fk - slope * x).abs() / (slope * x).abs().clamp_min(1e-9); asym = (fk + fmk).abs() / fk.abs().clamp_min(1e-9); kl = torch.tensor([KLp[al] for al in doses]); kl_quad = kl[:2].mean() / (x[:2] ** 2).mean(); kl_dev = (kl - kl_quad * x ** 2).abs() / (kl_quad * x ** 2).clamp_min(1e-9)
    out[k] = dict(onset_linear=next((al for al, dv in zip(doses, lin_dev.tolist()) if dv > 0.2), float("inf")), onset_asymmetry=next((al for al, av in zip(doses, asym.tolist()) if av > 0.2), float("inf")), onset_kl_quadratic=next((al for al, dv in zip(doses, kl_dev.tolist()) if dv > 0.3), float("inf")), response=fk.tolist(), response_minus=fmk.tolist(), kl=kl.tolist(), top1=[T1p[al] for al in doses], second_derivative=((fk[3] + fmk[3]) / 2).item() / (slope.abs().item() + 1e-9))
med = lambda key: float(torch.tensor([out[k][key] for k in out]).median())
log(f"{tag} (8 coordinates, doses {doses}x the footprint norm): median onset of nonlinearity (relative deviation > 20%) {med('onset_linear')}x, of sign asymmetry {med('onset_asymmetry')}x, of KL leaving quadratic {med('onset_kl_quadratic')}x; second derivative relative to slope at 1x {med('second_derivative'):+.2f} | per-coordinate linear onsets: " + " ".join(f"{out[k]['onset_linear']}" for k in out) + " | responses along coordinate 0: " + "/".join(f"{v:+.2f}" for v in out[0]['response']))
record(f"e345_fineamp_{tag}", dict(model=tag, L=L, per_coordinate={str(k): v for k, v in out.items()}), f"onset linear {med('onset_linear')}x asym {med('onset_asymmetry')}x kl {med('onset_kl_quadratic')}x second-deriv {med('second_derivative'):+.2f}")
