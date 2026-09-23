"""e366: is the natural-text blind spot an averaging artefact? The induction heads (top-12 by single-head ablation
effect on repeated random tokens) and 12 random heads, mean-ablated singly and in pairs on natural text, with the loss
scored separately at induction-applicable positions (the current token occurred earlier and the next token repeats
the earlier continuation), at a size-matched random sample of the other positions, and at all positions. For each
scoring: pairwise interaction top vs random heads, signed interaction, joint non-additivity of the top-8, and the
mean single effect. If the circuit reappears when scoring is restricted to the positions that use it, the old
natural-text negatives were a property of population averaging, not of the model."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam); NB, NH = arch.NB, arch.NH; rng = random.Random(0); torch.manual_seed(0); KT = 12
heads = [(l, h) for l in range(NB) for h in range(NH)]
ind, off, half = induction_batch(tok, c, n=8, half=128, seed=0); MAi, MMi, lg = capture_means(model, arch, ind); bi = induction_loss(lg, ind, off, half); del lg
Ei = torch.tensor([(induction_loss(ablate(model, arch, ind, heads=[x], MA=MAi, MM=MMi), ind, off, half) - bi).mean().item() for x in heads]); order = Ei.argsort(descending=True).tolist(); top = [heads[i] for i in order[:KT]]; U = top + rng.sample([heads[i] for i in order[KT:]], KT)
nat = c.s["eval_ids"][:int(os.environ.get("WDD_NNAT", "8"))].to(DEV); B, T = nat.shape; M = torch.zeros(B, T - 1, dtype=torch.bool)
for b, row in enumerate(nat.tolist()):
    last = {}
    for t in range(T - 1):
        a = row[t]
        if a in last and row[last[a] + 1] == row[t + 1]: M[b, t] = True
        last[a] = t
M = M.flatten().to(DEV); idx_ind = torch.nonzero(M)[:, 0]; rest = torch.nonzero(~M)[:, 0]; g = torch.Generator().manual_seed(3); idx_rnd = rest[torch.randperm(len(rest), generator=g)[:len(idx_ind)].to(DEV)]; idx_all = torch.arange(len(M), device=DEV)
MAn, MMn, lgn = capture_means(model, arch, nat); base = token_loss(lgn, nat); del lgn
masks = dict(applicable=idx_ind, matched_random=idx_rnd, all=idx_all)
def eff(hs): return token_loss(ablate(model, arch, nat, heads=hs, MA=MAn, MM=MMn), nat) - base
E1 = {x: eff([x]) for x in U}; n = len(U); PE = {}
for i in range(n):
    for j in range(i + 1, n): PE[(i, j)] = eff([U[i], U[j]])
ES = eff(top[:8])
res = dict(model=tag, n_applicable=len(idx_ind), share_applicable=len(idx_ind) / len(M), top_heads=[list(x) for x in top], top_effects_induction=[Ei[heads.index(x)].item() for x in top])
for mn, ix in masks.items():
    I = torch.zeros(n, n); Sg = torch.zeros(n, n)
    for (i, j), Eij in PE.items():
        a, b = E1[U[i]][ix], E1[U[j]][ix]; e = Eij[ix]; lin = a + b
        I[i, j] = I[j, i] = ((e - lin).norm() / (0.5 * (a.norm() + b.norm())).clamp_min(1e-9)).item(); Sg[i, j] = Sg[j, i] = ((e.mean() - lin.mean()) / (0.5 * (a.mean().abs() + b.mean().abs())).clamp_min(1e-9)).item()
    st = block_stats(I, Sg, KT); lin8 = sum(E1[x][ix] for x in top[:8]); na = ((ES[ix] - lin8).norm() / sum(E1[x][ix].norm() for x in top[:8]).clamp_min(1e-9)).item()
    res[mn] = dict(stats=st, na_top8=na, joint_top8=ES[ix].mean().item(), sum_singles_top8=lin8.mean().item(), mean_single_top=torch.stack([E1[x][ix].mean() for x in top]).mean().item(), mean_single_random=torch.stack([E1[x][ix].mean() for x in U[KT:]]).mean().item())
log(f"{tag}: induction-applicable positions {len(idx_ind)} ({100 * res['share_applicable']:.1f}% of natural tokens) | " + " | ".join(f"{mn.upper()}: interaction top {res[mn]['stats']['I_top']:.2f} vs random {res[mn]['stats']['I_random']:.2f}, signed top {res[mn]['stats']['signed_top']:+.2f}, non-additivity top-8 {res[mn]['na_top8']:.2f} (joint {res[mn]['joint_top8']:+.3f} vs sum of singles {res[mn]['sum_singles_top8']:+.3f}), mean single effect top {res[mn]['mean_single_top']:+.4f} vs random {res[mn]['mean_single_random']:+.4f}" for mn in masks))
record(f"e366_condnat_{tag}", res, " | ".join(f"{mn}: I {res[mn]['stats']['I_top']:.2f}/{res[mn]['stats']['I_random']:.2f} NA {res[mn]['na_top8']:.2f} single {res[mn]['mean_single_top']:+.3f}/{res[mn]['mean_single_random']:+.3f}" for mn in masks))
