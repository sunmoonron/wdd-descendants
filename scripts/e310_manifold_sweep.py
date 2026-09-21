"""e310: does the low-dimensional functional coordinate require the perturbation to lie near the model's natural
residual manifold? The natural manifold at the injection level = the top-64 principal directions of the block-2
output over typical tokens. Families of 30 vectors injected at the block-3 input at the natural median amplitude:
the sweep delta_alpha = sqrt(1 - alpha^2) delta_perp + alpha delta_nat for alpha in 0, 0.1, 0.25, 0.5, 0.75, 1
(delta_perp random with its manifold component removed, delta_nat covariance-matched), and synthetic controls:
heavy-tailed (Student t, 1.5 degrees of freedom), sparse (5 coordinates), low-rank (a random 4-dimensional subspace),
and shuffled residual states. For each: the e286 ladder (identity / function / future dimensions and full scores), the
physical rank of the images, the gain, and the median logit response."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lf = L + 2 if L + 2 < NB - 1 else NB - 2; Kf = 30
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [b, L, lf], NT, b); S0 = run(); typ = typical_mask(S0[L]); pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median()
torch.manual_seed(0); Sc = S0[b][typ] - S0[b][typ].mean(0, keepdim=True); Unat = torch.linalg.svd(Sc, full_matrices=False)[2][:64]; perp = lambda V: unit(V - (V @ Unat.T) @ Unat); nat = unit((torch.randn(Kf, len(Sc), device=DEV) / len(Sc) ** 0.5) @ Sc); base_perp = perp(torch.randn(Kf, D, device=DEV))
fams = {}
for al in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0): fams[f"alpha_{al}"] = unit((1 - al ** 2) ** 0.5 * base_perp + al * nat)
fams["heavy_tailed"] = unit(torch.distributions.StudentT(1.5).sample((Kf, D)).to(DEV)); sp = torch.zeros(Kf, D, device=DEV); sp[torch.arange(Kf, device=DEV)[:, None], torch.randint(0, D, (Kf, 5), device=DEV)] = torch.randn(Kf, 5, device=DEV); fams["sparse5"] = unit(sp); Q4 = torch.linalg.qr(torch.randn(D, 4, device=DEV))[0]; fams["lowrank4"] = unit((torch.randn(Kf, 4, device=DEV) @ Q4.T)); fams["shuffled_states"] = unit(Sc[torch.randperm(len(Sc), device=DEV)[:Kf]][:, torch.randperm(D, device=DEV)])
base = run(positions=pool); lg0 = base["lg"]; dims = [dd for dd in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, D) if dd <= D]; out = {}
def knn(Ptr, Pte, target, tr, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return target[tr][nn].mean(1)
def prank(X):
    ev = torch.linalg.eigvalsh(X.T @ X / len(X)).clamp_min(0); return (ev.sum() ** 2 / (ev ** 2).sum()).item()
for nm, V in fams.items():
    a = torch.randint(0, Kf, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * V[a]; r = run(positions=pool, inject=inj, inject_block=b + 1); F = (r[L] - S0[L])[pool]; Ff = unit((r[lf] - S0[lf])[pool]); dlr = r["lg"] - lg0; dlc = dlr - dlr.mean(1, keepdim=True); dln = unit(dlc)
    split = torch.rand(len(pool), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]; Fc = F - F[tr].mean(0, keepdim=True); G = dln[tr] @ dln[tr].T; evg, Vg = torch.linalg.eigh(G); Zt = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Zt = Zt - Zt.mean(0, keepdim=True)
    cents = torch.stack([Fc[tr][a[tr] == k].mean(0) for k in range(Kf)]); w = torch.bincount(a[tr], minlength=Kf).float(); Sb = (cents * w[:, None]).T @ cents / w.sum(); U_id = torch.linalg.eigh(Sb)[1].flip(1).T; U_fn = torch.linalg.svd(Fc[tr].T @ Zt, full_matrices=False)[0].T; U_fu = torch.linalg.svd(Fc[tr].T @ (Ff[tr] - Ff[tr].mean(0, keepdim=True)), full_matrices=False)[0].T
    def score(U, rr, obs):
        P = U[:min(rr, U.shape[0])].T; Ptr, Pte = Fc[tr] @ P, Fc[te] @ P
        if obs == "identity": return accuracy(Pte, centroids(Ptr, a[tr], Kf), a[te])
        if obs == "function": return ((unit(knn(Ptr, Pte, dln, tr)) * dln[te]).sum(1)).median().item()
        return ((unit(knn(Ptr, Pte, Ff, tr)) * Ff[te]).sum(1)).median().item()
    rec = {}
    for obs, U in (("identity", U_id), ("function", U_fn), ("future", U_fu)):
        curve = {rr: score(U, rr, obs) for rr in dims}; full = curve[dims[-1]]; ch = 1 / Kf if obs == "identity" else 0.0; rec[obs] = dict(dim=next((rr for rr in dims if curve[rr] - ch >= 0.9 * (full - ch)), dims[-1]), full=full)
    rec["physical_prank"] = prank(Fc); rec["gain"] = (F.norm(dim=1) / s_inj).median().item(); rec["logit_response"] = dlc.norm(dim=1).median().item(); rec["manifold_fraction"] = ((V @ Unat.T) ** 2).sum(1).mean().item(); out[nm] = rec
log(f"{tag}: family: id-dim/fn-dim/fut-dim (full scores) | physical rank | gain | logit response | manifold fraction of the injected vectors :: " + " ; ".join(f"{nm}: {v['identity']['dim']}/{v['function']['dim']}/{v['future']['dim']} ({v['identity']['full']:.2f}/{v['function']['full']:.2f}/{v['future']['full']:.2f}) | {v['physical_prank']:.0f} | {v['gain']:.2f} | {v['logit_response']:.1f} | {v['manifold_fraction']:.2f}" for nm, v in out.items()))
record(f"e310_manifold_{tag}", dict(model=tag, b=b, L=L, future=lf, per_family=out), " ; ".join(f"{nm}: {v['identity']['dim']}/{v['function']['dim']}/{v['future']['dim']} ({v['identity']['full']:.2f}/{v['function']['full']:.2f}/{v['future']['full']:.2f}) rank {v['physical_prank']:.0f} gain {v['gain']:.2f} logit {v['logit_response']:.1f} mf {v['manifold_fraction']:.2f}" for nm, v in out.items()))
