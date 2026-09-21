"""e356: one-to-many: the same physical perturbation in different contexts. Thirty random directions injected at the
block-3 input, each at ~100 tokens; per direction the dispersion of its per-token coordinate (mean cosine of the
per-token coordinates to their centroid), of its per-token logit effect, and of its per-token image; the fraction of
a direction's coordinate variance explained by the token's residual state (ridge from the state's top-64 PCs,
held-out), and whether context dispersion is larger for the model's own writes than for random directions."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; b = S.b; S = Setup(tag, NS=16, levels=[b, L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, Bv = logit_scores(nat["dl"], tr); P = pls(Fc[tr], Z[tr], d); Kf = 30; torch.manual_seed(0); fams = {"random": unit(torch.randn(Kf, S.D, device=DEV)), "wdd": S.W2[torch.randperm(S.DFF, device=DEV)[:Kf]]}; out = {}
for nm, V in fams.items():
    r = S.inject_family(V, b + 1, seed=1); a = r["a"]; z = r["F"][L] @ P; Fl = r["dl"] @ Bv; X = S.S0[b][r["pos"]]; Xc = X - X.mean(0, keepdim=True); U = torch.linalg.svd(Xc, full_matrices=False)[2][:64]; Xp = Xc @ U.T; coh_z, coh_f, coh_i, r2s = [], [], [], []
    for k in range(Kf):
        m = a == k; zk = z[m]; coh_z.append(((unit(zk) * unit(zk.mean(0))[None]).sum(1)).mean().item()); coh_f.append(((unit(Fl[m]) * unit(Fl[m].mean(0))[None]).sum(1)).mean().item()); coh_i.append(((unit(r["F"][L][m]) * unit(r["F"][L][m].mean(0))[None]).sum(1)).mean().item())
        n = int(m.sum()); rows = torch.nonzero(m)[:, 0]; ta, tb = rows[::2], rows[1::2]; W = ridge(Xp[ta], zk[::2] - zk[::2].mean(0, keepdim=True), 1e-1); pred = Xp[tb] @ W + zk[::2].mean(0, keepdim=True); r2s.append(1 - ((z[tb] - pred) ** 2).sum().item() / ((z[tb] - zk[::2].mean(0, keepdim=True)) ** 2).sum().item())
    out[nm] = dict(coordinate_coherence=float(torch.tensor(coh_z).median()), effect_coherence=float(torch.tensor(coh_f).median()), image_coherence=float(torch.tensor(coh_i).median()), coordinate_r2_from_state=float(torch.tensor(r2s).median()))
log(f"{tag}: per-direction coherence across contexts (coordinate / effect / image) and coordinate R2 from the token's state: " + " | ".join(f"{nm}: {v['coordinate_coherence']:.2f} / {v['effect_coherence']:.2f} / {v['image_coherence']:.2f}; R2 {v['coordinate_r2_from_state']:+.2f}" for nm, v in out.items()))
record(f"e356_onetomany_{tag}", dict(model=tag, L=L, per_family=out), " | ".join(f"{nm} coh {v['coordinate_coherence']:.2f}/{v['effect_coherence']:.2f}/{v['image_coherence']:.2f} r2 {v['coordinate_r2_from_state']:+.2f}" for nm, v in out.items()))
