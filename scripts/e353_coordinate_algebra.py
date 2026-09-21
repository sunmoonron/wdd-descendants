"""e353: higher-order coordinate algebra. Along the eight core directions at the natural footprint norm: triples
(third-order residuals of F(a+b+c) against the pairwise-corrected prediction), dense superposition of all eight with
random signs against the sum of singles, sparse superpositions of 1, 2, 4, 8 active coordinates (relative error of
additivity by sparsity), and a permutation test: coordinates permuted before injection produce the permuted effects
(the transport does not care which index is which)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Ud, Sd, Vd = torch.linalg.svd(nat["dl"], full_matrices=False); Zs = Ud[:, :256] * Sd[:256][None]; B = Vd[:256].T; Zs = Zs - Zs.mean(0, keepdim=True); Up, Sp, Wt = torch.linalg.svd(Fc.T @ Zs, full_matrices=False); nk = 8; u = Up[:, :nk].T; fnorm = F.norm(dim=1).median(); sub = S.foreign[:1024]; base = S.run(positions=sub); lg0 = base["lg"]
def E(v):
    inj = torch.zeros(S.NT, S.D, device=DEV); inj[sub] = fnorm * v[None]; r = S.run(positions=sub, inject=inj, inject_block=L + 1); dl = r["lg"] - lg0; return dl - dl.mean(1, keepdim=True)
single = [E(u[k]) for k in range(nk)]; pair = {}; torch.manual_seed(0); err = lambda act, pred: ((act - pred).norm(dim=1) / act.norm(dim=1).clamp_min(1e-9)).median().item()
trip = []
for t in range(6):
    i, j, k = torch.randperm(nk)[:3].tolist()
    for (a_, b_) in ((i, j), (i, k), (j, k)):
        if (a_, b_) not in pair: pair[(a_, b_)] = E(u[a_] + u[b_])
    act = E(u[i] + u[j] + u[k]); lin = single[i] + single[j] + single[k]; corr = pair[(i, j)] + pair[(i, k)] + pair[(j, k)] - lin; trip.append((err(act, lin), err(act, corr)))
sparse = {}
for m in (1, 2, 4, 8):
    errs = []
    for t in range(4):
        sel = torch.randperm(nk)[:m].tolist(); sg = torch.sign(torch.randn(m)); v = sum(sg[q].item() * u[sel[q]] for q in range(m)); errs.append(err(E(v), sum(sg[q].item() * single[sel[q]] for q in range(m))))
    sparse[m] = float(torch.tensor(errs).median())
perm = torch.randperm(nk); pe = [E(u[perm[k]]) for k in range(nk)]; perm_cos = float(torch.tensor([((unit(pe[k]) * unit(single[perm[k]])).sum(1)).median().item() for k in range(nk)]).median())
res = dict(triple_err_linear=float(torch.tensor([x[0] for x in trip]).median()), triple_err_pair_corrected=float(torch.tensor([x[1] for x in trip]).median()), sparse_err={str(k): v for k, v in sparse.items()}, permutation_cos=perm_cos)
log(f"{tag}: triples: additivity error {res['triple_err_linear']:.2f}, after pairwise correction {res['triple_err_pair_corrected']:.2f} | additivity error by number of active coordinates 1/2/4/8: " + "/".join(f"{sparse[m]:.2f}" for m in (1, 2, 4, 8)) + f" | permuted coordinates reproduce the permuted effects at cos {perm_cos:.2f}")
record(f"e353_coordalgebra_{tag}", dict(model=tag, L=L, **res), f"triple {res['triple_err_linear']:.2f}->{res['triple_err_pair_corrected']:.2f} sparse " + "/".join(f"{sparse[m]:.2f}" for m in (1, 2, 4, 8)) + f" perm {perm_cos:.2f}")
