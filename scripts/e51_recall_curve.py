"""e51: recall as a function of support size from the cached OMP order (k=1..64), the type/block of the FIRST
atom picked, and whether the first atom is a true write (ledger >= 5% of max). Where in the greedy order does the
dominant write get picked?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 3); rows3 = torch.stack([c.atom_index(L, tb[:, j], tn[:, j]) for j in range(3)], 1).to(DEV)
hit = (sel == rows3[:, :1]); ks = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
curve = {k: hit[:, :k].any(1)[typ].float().mean().item() for k in ks}
pos = torch.where(hit.any(1), hit.float().argmax(1), torch.full((c.NT,), -1, device=DEV)); pm = pos[typ & hit.any(1)]
typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
led = c.ledger(L); C = torch.cat([led[b] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * C.abs().max(1, keepdim=True).values
first = sel[:, 0]; f_mlp = typA[first] == T_MLP; key = torch.where(f_mlp, blkA[first] * c.DFF + idxA[first], torch.zeros_like(first))
f_real = f_mlp & (C.gather(1, key[:, None])[:, 0].abs() >= thr[:, 0])
res = dict(model=tag, L=L, recall_curve=curve, pick_position_median=pm.float().median().item(), pick_position_q90=pm.float().quantile(0.9).item(),
           first_atom_types={int(t): (typA[first] == t)[typ].float().mean().item() for t in range(5)}, first_atom_is_real_write=f_real[typ].float().mean().item(),
           first_atom_is_dominant=hit[:, 0][typ].float().mean().item(), first_atom_in_top3=(sel[:, :1] == rows3).any(1)[typ].float().mean().item(),
           first_atom_block_hist={int(b): ((blkA[first] == b) & f_mlp)[typ].float().mean().item() for b in range(L + 1)},
           first_atom_energy_share=((err[:, 0] / (X ** 2).sum(1)))[typ].median().item())
record(f"e51_curve_{tag}", res, "recall@k " + " ".join(f"{k}:{v:.2f}" for k, v in curve.items()) + f" | pick pos med {res['pick_position_median']:.0f} q90 {res['pick_position_q90']:.0f} | first atom types {res['first_atom_types']} real {res['first_atom_is_real_write']:.2f} dominant {res['first_atom_is_dominant']:.2f} top3 {res['first_atom_in_top3']:.2f} | residual after 1 atom {res['first_atom_energy_share']:.2f}")
