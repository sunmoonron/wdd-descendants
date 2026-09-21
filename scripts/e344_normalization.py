"""e344: normalization and the injection point. A family of 30 random directions injected (i) at the block input as
usual, (ii) scaled so that the injected token's residual norm is unchanged (direction change only), (iii) with the
residual norm doubled at the injected token without changing direction (norm-only perturbation), (iv) into the
normalized input of the attention sublayer instead of the residual (post-norm injection, via a pre-hook on the
attention's q/k/v input where available, else skipped). For each: quotient dimension, own score, gain, and the
overlap of the quotient with the standard one."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; b = S.b; S = Setup(tag, levels=[b, L]); Kf = 30; d = 16; torch.manual_seed(0); V = unit(torch.randn(Kf, S.D, device=DEV)); pool = S.pool; x0 = S.S0[b][pool]; out = {}; Qs = {}
def evaluate(nm, inj_fn):
    base = S.run(positions=pool); a = torch.randint(0, Kf, (len(pool),), device=DEV); inj = torch.zeros(S.NT, S.D, device=DEV); inj[pool] = inj_fn(a); r = S.run(positions=pool, inject=inj, inject_block=b + 1); dl = r["lg"] - base["lg"]; dl = dl - dl.mean(1, keepdim=True); F = (r[L] - S.S0[L])[pool]; tr, te = S.halves(len(pool), seed=2); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(dl, tr); dln = unit(dl); Q = pls(Fc[tr], Z[tr], 64); curve = {q: knn_cos(Fc @ Q[:, :q], dln, tr, te) for q in (1, 2, 4, 8, 16, 32, 64)}; Qs[nm] = Q[:, :d]; out[nm] = dict(dim=next((q for q in curve if curve[q] >= 0.9 * curve[64]), 64), own=curve[16], random=knn_cos(Fc @ torch.linalg.qr(torch.randn(S.D, d, device=DEV))[0], dln, tr, te), identity=accuracy((Fc @ Q[:, :d])[te], centroids((Fc @ Q[:, :d])[tr], a[tr], Kf), a[te]), gain=(F.norm(dim=1) / S.s_inj).median().item(), logit_response=dl.norm(dim=1).median().item())
evaluate("standard", lambda a: S.s_inj * V[a]); evaluate("norm_preserving", lambda a: unit(x0 + S.s_inj * V[a]) * x0.norm(dim=1, keepdim=True) - x0); evaluate("norm_only_x2", lambda a: x0 * (1.0 + 0.0 * a[:, None].float()) * 0 + x0 * torch.where(a[:, None] % 2 == 0, 1.0, -0.5)); evaluate("amplitude_x4", lambda a: 4 * S.s_inj * V[a])
for nm in out: out[nm]["overlap_with_standard"] = inside(Qs[nm], Qs["standard"])
log(f"{tag}: injection variant -> quotient dim | own-16 score / random-16 | identity | gain | logit response | overlap with the standard quotient :: " + " ; ".join(f"{nm}: {v['dim']} | {v['own']:.2f}/{v['random']:.2f} | {v['identity']:.2f} | {v['gain']:.2f} | {v['logit_response']:.1f} | {v['overlap_with_standard']:.2f}" for nm, v in out.items()))
record(f"e344_normalization_{tag}", dict(model=tag, L=L, per_variant=out), " ; ".join(f"{nm}: dim {v['dim']} own {v['own']:.2f} rnd {v['random']:.2f} id {v['identity']:.2f} ov {v['overlap_with_standard']:.2f}" for nm, v in out.items()))
