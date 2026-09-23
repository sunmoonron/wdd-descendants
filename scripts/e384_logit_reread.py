"""e384: e357's joint non-additivity and e366's conditional natural-text test, re-read on linear readouts. Mean-ablation
as in the originals. (1) Induction data: the top-8 heads by single effect on the induction loss; joint over sum of
ablating all 8, and the mean signed pairwise interaction of the top-12, on the loss, the next token's logit and its
centred logit. (2) Natural text (8 x 512 tokens), the induction top-12 and 12 random heads, scored at
induction-applicable positions, at size-matched random positions and at all positions, on the same three readouts:
interaction magnitude top vs random (relative norm of the per-token joint effect minus the sum of singles), mean
signed interaction of the top pairs, and joint over sum of the top-8."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam); NB, NH = arch.NB, arch.NH; rng = random.Random(0); torch.manual_seed(0); KT = 12
heads = [(l, h) for l in range(NB) for h in range(NH)]
def readouts(lg, ids, pos=None):
    lg = lg[:, :-1].float() if pos is None else lg[:, pos].float(); y = ids[:, 1:] if pos is None else ids[:, pos + 1]
    lp = torch.log_softmax(lg, -1); cl = lg.gather(-1, y[..., None])[..., 0]
    return dict(loss=-lp.gather(-1, y[..., None])[..., 0].flatten(), logit=-cl.flatten(), centred=-(cl - lg.mean(-1)).flatten())
ind, off, half = induction_batch(tok, c, n=8, half=128, seed=0); pos = torch.arange(off + half, off + 2 * half - 1, device=DEV)
MAi, MMi, lg = capture_means(model, arch, ind); Bi = readouts(lg, ind, pos); del lg
Ei = {x: {k: v - Bi[k] for k, v in readouts(ablate(model, arch, ind, heads=[x], MA=MAi, MM=MMi), ind, pos).items()} for x in heads}
order = sorted(heads, key=lambda x: -Ei[x]["loss"].mean().item()); top = order[:KT]; U = top + rng.sample(order[KT:], KT)
res = dict(model=tag, top=[list(x) for x in top], induction={}, natural={})
J = {k: v - Bi[k] for k, v in readouts(ablate(model, arch, ind, heads=top[:8], MA=MAi, MM=MMi), ind, pos).items()}
S = {k: [] for k in Bi}
for x in range(KT):
    for y in range(x + 1, KT):
        P = {k: v - Bi[k] for k, v in readouts(ablate(model, arch, ind, heads=[top[x], top[y]], MA=MAi, MM=MMi), ind, pos).items()}
        for k in S: a, b = Ei[top[x]][k].mean().item(), Ei[top[y]][k].mean().item(); S[k].append((P[k].mean().item() - a - b) / max(0.5 * (abs(a) + abs(b)), 1e-9))
for k in Bi:
    res["induction"][k] = dict(joint_over_sum=J[k].mean().item() / max(sum(Ei[x][k].mean().item() for x in top[:8]), 1e-9), mean_S=sum(S[k]) / len(S[k]), share_super=sum(s > 0.2 for s in S[k]) / len(S[k]), share_sub=sum(s < -0.2 for s in S[k]) / len(S[k]))
nat = c.s["eval_ids"][:8].to(DEV); B, T = nat.shape; M = torch.zeros(B, T - 1, dtype=torch.bool)
for b, row in enumerate(nat.tolist()):
    last = {}
    for t in range(T - 1):
        a = row[t]
        if a in last and row[last[a] + 1] == row[t + 1]: M[b, t] = True
        last[a] = t
M = M.flatten().to(DEV); ia = torch.nonzero(M)[:, 0]; rest = torch.nonzero(~M)[:, 0]; g = torch.Generator().manual_seed(3); ir = rest[torch.randperm(len(rest), generator=g)[:len(ia)].to(DEV)]; iall = torch.arange(len(M), device=DEV)
MAn, MMn, lg = capture_means(model, arch, nat); Bn = readouts(lg, nat); del lg
eff = lambda hs: {k: v - Bn[k] for k, v in readouts(ablate(model, arch, nat, heads=hs, MA=MAn, MM=MMn), nat).items()}
E1 = {x: eff([x]) for x in U}; n = len(U); PE = {(i, j): eff([U[i], U[j]]) for i in range(n) for j in range(i + 1, n)}; ES = eff(top[:8])
for mn, ix in dict(applicable=ia, matched_random=ir, all=iall).items():
    res["natural"][mn] = {}
    for k in Bn:
        I = torch.zeros(n, n); Sg = torch.zeros(n, n)
        for (i, j), Eij in PE.items():
            a, b = E1[U[i]][k][ix], E1[U[j]][k][ix]; e = Eij[k][ix]; lin = a + b
            I[i, j] = I[j, i] = ((e - lin).norm() / (0.5 * (a.norm() + b.norm())).clamp_min(1e-9)).item(); Sg[i, j] = Sg[j, i] = ((e.mean() - lin.mean()) / (0.5 * (a.mean().abs() + b.mean().abs())).clamp_min(1e-9)).item()
        st = block_stats(I, Sg, KT); js = ES[k][ix].mean().item() / max(sum(E1[x][k][ix].mean().item() for x in top[:8]), 1e-9)
        res["natural"][mn][k] = dict(I_top=st["I_top"], I_random=st["I_random"], signed_top=st["signed_top"], joint_over_sum_top8=js)
fi = lambda d: f"joint/sum {d['joint_over_sum']:.2f}, mean S {d['mean_S']:+.2f} (super {d['share_super']:.2f}, sub {d['share_sub']:.2f})"
fn = lambda d: f"I {d['I_top']:.2f}/{d['I_random']:.2f} signed {d['signed_top']:+.2f} j/s {d['joint_over_sum_top8']:.2f}"
log(f"{tag}: INDUCTION " + " | ".join(f"{k}: {fi(res['induction'][k])}" for k in Bi) + " || NATURAL " + " | ".join(f"{mn}: " + "; ".join(f"{k} {fn(res['natural'][mn][k])}" for k in Bn) for mn in res["natural"]))
record(f"e384_logitreread_{tag}", res, " | ".join(f"induction {k} j/s {res['induction'][k]['joint_over_sum']:.2f} S {res['induction'][k]['mean_S']:+.2f}" for k in Bi) + " | " + " | ".join(f"applicable {k} I {res['natural']['applicable'][k]['I_top']:.2f}/{res['natural']['applicable'][k]['I_random']:.2f}" for k in Bn))
