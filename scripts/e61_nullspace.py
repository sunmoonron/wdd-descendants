"""e61: the identifiability limit, quantified. Any reading of the state x = A^T c recovers c only modulo the null
space of A (dimension >= m - d). For the true MLP ledger c (blocks 0..L): the fraction of its energy in the null
space of the MLP dictionary (invisible in principle from the state), the same per block for the increment
(block dictionary), and for the token's true top-3 writes specifically. Also the min-norm reading
c_min = A (A^T A)^-1 x_mlp (the canonical dual-frame coefficients, Duffin-Schaeffer): its correlation with the
true c, overall and for the top-3 writes, and the sign agreement."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L, types=(T_MLP,))          # MLP atoms of blocks 0..L, in (block, neuron) order
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV)                    # [N, M] true coefficients (unit-atom convention)
Xm = C @ A                                                                                          # [N, D] the MLP part of the state (exact)
G = A.T @ A + 1e-6 * torch.eye(c.D, device=DEV); Cmin = torch.linalg.solve(G, Xm.T).T @ A.T            # [N, M] min-norm coefficients: A (A^T A)^-1 x
null = C - Cmin                                                                                     # component of the true ledger invisible from the state
res = dict(model=tag, L=L, M=A.shape[1] if False else A.shape[0], D=c.D)
res["ledger_energy_in_nullspace"] = ((null[typ] ** 2).sum() / (C[typ] ** 2).sum()).item()
top3 = C.abs().topk(3, dim=1).indices; e_top = (C.gather(1, top3) ** 2).sum(1); e_top_null = (null.gather(1, top3) ** 2).sum(1)
res["top3_energy_in_nullspace_median"] = (e_top_null / e_top)[typ].median().item()
res["minnorm_vs_true"] = dict(pearson_all=torch.corrcoef(torch.stack([Cmin[typ].flatten()[::97], C[typ].flatten()[::97]]))[0, 1].item(),
                              sign_agree_top3=((Cmin.gather(1, top3) * C.gather(1, top3)) > 0)[typ].float().mean().item(),
                              ratio_top1_median=(Cmin.gather(1, top3[:, :1]) / C.gather(1, top3[:, :1]))[typ].median().item())
# per-block increments: the block's MLP ledger vs the block's MLP dictionary
per_block = []
for b in range(L + 1):
    Ab = A[b * c.DFF:(b + 1) * c.DFF]; Cb = led[b][ids].to(DEV); Xb = Cb @ Ab; Gb = Ab.T @ Ab + 1e-6 * torch.eye(c.D, device=DEV)
    Cbmin = torch.linalg.solve(Gb, Xb.T).T @ Ab.T; per_block.append(((Cb - Cbmin)[typ] ** 2).sum().item() / (Cb[typ] ** 2).sum().item())
res["increment_ledger_energy_in_nullspace_by_block"] = per_block
record(f"e61_null_{tag}", res, f"M {A.shape[0]} D {c.D} | ledger energy invisible from the state (null space of the MLP dictionary): {res['ledger_energy_in_nullspace']:.3f}; of the top-3 true writes: {res['top3_energy_in_nullspace_median']:.3f} | per-block increments: " + " ".join(f"b{b}:{v:.2f}" for b, v in enumerate(per_block)) + f" | min-norm reading: r {res['minnorm_vs_true']['pearson_all']:.2f} sign(top3) {res['minnorm_vs_true']['sign_agree_top3']:.2f} ratio(top1) {res['minnorm_vs_true']['ratio_top1_median']:.2f}")
