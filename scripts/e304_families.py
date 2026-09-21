"""e304: is the dimensionality hierarchy special to WDD writes? Five perturbation families injected at the block-3
input at the natural median amplitude, 30 vectors each spread over the typical tokens: WDD writes of block 2, random
unit directions, covariance-matched random directions (from the block-2 output covariance), attention-head write
directions of blocks 0-2 (OV bases from the dictionary) and MLP writes of a later block (block 5). For each family the
per-token images at L and L+2 and the per-token logit change give the e286 ladder: supervised dimensions for 90% of
the full score on vector identity, function (kNN cosine of the logit change) and future (kNN cosine of the L+2
image), plus the PCA reconstruction dimension and the full scores."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; lf = L + 2 if L + 2 < NB - 1 else NB - 2; Kf = 30
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [b, L, lf], NT, b); S0 = run(); typ = typical_mask(S0[L]); pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20)
torch.manual_seed(0); Rm = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); Ratt = A[(lab["type"] == T_ATT) & (lab["block"] <= b)].float().to(DEV); R5 = A[(lab["type"] == T_MLP) & (lab["block"] == min(5, NB - 2))].float().to(DEV); Sc = S0[b][typ] - S0[b][typ].mean(0, keepdim=True)
fams = {"wdd_writes": unit(Rm[keep[:Kf]] if len(keep) >= Kf else Rm[torch.randperm(len(Rm), device=DEV)[:Kf]]), "random": unit(torch.randn(Kf, D, device=DEV)), "cov_matched_random": unit((torch.randn(Kf, len(Sc), device=DEV) / len(Sc) ** 0.5) @ Sc), "attention_writes": unit(Ratt[torch.randperm(len(Ratt), device=DEV)[:Kf]]), "later_mlp_writes": unit(R5[torch.randperm(len(R5), device=DEV)[:Kf]])}
base = run(positions=pool); lg0 = base["lg"]; dims = [dd for dd in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, D) if dd <= D]; out = {}
def knn(Ptr, Pte, target, tr, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return target[tr][nn].mean(1)
for nm, V in fams.items():
    a = torch.randint(0, Kf, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * V[a]; r = run(positions=pool, inject=inj, inject_block=b + 1); F = (r[L] - S0[L])[pool]; Ff = unit((r[lf] - S0[lf])[pool]); dlr = r["lg"] - lg0; dln = unit(dlr - dlr.mean(1, keepdim=True)); lab_ = a
    split = torch.rand(len(pool), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]; Fc = F - F[tr].mean(0, keepdim=True); G = dln[tr] @ dln[tr].T; evg, Vg = torch.linalg.eigh(G); Zt = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Zt = Zt - Zt.mean(0, keepdim=True)
    cents = torch.stack([Fc[tr][lab_[tr] == k].mean(0) for k in range(Kf)]); w = torch.bincount(lab_[tr], minlength=Kf).float(); Sb = (cents * w[:, None]).T @ cents / w.sum(); U_id = torch.linalg.eigh(Sb)[1].flip(1).T; U_fn = torch.linalg.svd(Fc[tr].T @ Zt, full_matrices=False)[0].T; U_fu = torch.linalg.svd(Fc[tr].T @ (Ff[tr] - Ff[tr].mean(0, keepdim=True)), full_matrices=False)[0].T; U_pca = torch.linalg.svd(Fc[tr], full_matrices=False)[2]
    def score(U, rr, obs):
        P = U[:min(rr, U.shape[0])].T; Ptr, Pte = Fc[tr] @ P, Fc[te] @ P
        if obs == "identity": return accuracy(Pte, centroids(Ptr, lab_[tr], Kf), lab_[te])
        if obs == "function": return ((unit(knn(Ptr, Pte, dln, tr)) * dln[te]).sum(1)).median().item()
        return ((unit(knn(Ptr, Pte, Ff, tr)) * Ff[te]).sum(1)).median().item()
    rec = {}
    for obs, U in (("identity", U_id), ("function", U_fn), ("future", U_fu)):
        curve = {rr: score(U, rr, obs) for rr in dims}; full = curve[dims[-1]]; ch = 1 / Kf if obs == "identity" else 0.0; rec[obs] = dict(dim=next((rr for rr in dims if curve[rr] - ch >= 0.9 * (full - ch)), dims[-1]), full=full)
    recon = {rr: 1 - ((Fc[te] - (Fc[te] @ U_pca[:rr].T) @ U_pca[:rr]) ** 2).sum().item() / (Fc[te] ** 2).sum().item() for rr in dims}; rec["reconstruction"] = dict(dim=next((rr for rr in dims if recon[rr] >= 0.9 * recon[dims[-1]]), dims[-1]), full=recon[dims[-1]]); rec["gain"] = (F.norm(dim=1) / s_inj).median().item(); out[nm] = rec
log(f"{tag} (30 vectors per family, injected at block {b + 1}): family: identity-dim / function-dim / future-dim / reconstruction-dim (full scores; gain) :: " + " ; ".join(f"{nm}: {v['identity']['dim']}/{v['function']['dim']}/{v['future']['dim']}/{v['reconstruction']['dim']} ({v['identity']['full']:.2f}/{v['function']['full']:.2f}/{v['future']['full']:.2f}; gain {v['gain']:.2f})" for nm, v in out.items()))
record(f"e304_families_{tag}", dict(model=tag, b=b, L=L, future=lf, per_family=out), " ; ".join(f"{nm}: id {v['identity']['dim']} fn {v['function']['dim']} fut {v['future']['dim']} rec {v['reconstruction']['dim']} (full {v['identity']['full']:.2f}/{v['function']['full']:.2f}/{v['future']['full']:.2f}, gain {v['gain']:.2f})" for nm, v in out.items()))
