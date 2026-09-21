"""e16: the dictionary decomposed over itself (meta). Each mid-layer MLP atom over the rest of the dictionary
(k=8): how sparse is an atom in the other atoms, by the block distance of the atoms that explain it; the same for
embedding rows (which neurons re-write token directions); and rotated control. Zero data."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); A, lab = c.dictionary(c.NB - 1); typ, blk = lab["type"].to(DEV), lab["block"].to(DEV)
K = 8; res = dict(model=tag, K=K, blocks={})
for b in (0, L, c.NB - 1):
    m = torch.nonzero((typ == T_MLP) & (blk == b))[:, 0]; T = A[m[sub(len(m), 1024)]]
    others = torch.ones(A.shape[0], dtype=torch.bool, device=DEV); others[m] = False       # exclude the block's own atoms (incl. self)
    Ao = A[others]; lo = {k: v[others.cpu()] for k, v in lab.items()}
    sel, cof, err = omp(T, Ao, K); sb = lo["block"].to(DEV)[sel]; st = lo["type"].to(DEV)[sel]
    selr, cofr, errr = omp(T, rotate(Ao), K)
    same_bl = lambda q: (sb == q).float().mean().item()
    res["blocks"][b] = dict(fvu1=fvu(err[:, 0], T), fvu4=fvu(err[:, 3], T), fvu8=fvu(err[:, 7], T), rot_fvu8=fvu(errr[:, 7], T),
                            support_types={int(t): (st == t).float().mean().item() for t in range(5)},
                            support_block_hist={int(q): same_bl(q) for q in range(-1, c.NB)},
                            frac_earlier=(sb[(sb >= 0)] < b).float().mean().item(), frac_later=(sb[(sb >= 0)] > b).float().mean().item())
    log(f"{tag} block {b} atoms over others: fvu1 {res['blocks'][b]['fvu1']:.3f} fvu8 {res['blocks'][b]['fvu8']:.3f} rot {res['blocks'][b]['rot_fvu8']:.3f} earlier {res['blocks'][b]['frac_earlier']:.2f} later {res['blocks'][b]['frac_later']:.2f} types {res['blocks'][b]['support_types']}")
# embedding rows over the block atoms only
E = A[typ == T_TOK][sub(int((typ == T_TOK).sum()), 2048)]; Ab = A[typ >= T_MLP]; lb = lab["block"][(lab["type"] >= T_MLP)].to(DEV)
sel, cof, err = omp(E, Ab, K); selr, _, errr = omp(E, rotate(Ab), K)
res["emb_over_blocks"] = dict(fvu8=fvu(err[:, 7], E), rot_fvu8=fvu(errr[:, 7], E), block_hist={int(q): (lb[sel] == q).float().mean().item() for q in range(c.NB)})
record(f"e16_atoms_{tag}", res, " | ".join(f"b{b}: fvu8 {v['fvu8']:.2f} (rot {v['rot_fvu8']:.2f}) earlier {v['frac_earlier']:.2f}" for b, v in res["blocks"].items()) + f" | emb over blocks fvu8 {res['emb_over_blocks']['fvu8']:.2f} rot {res['emb_over_blocks']['rot_fvu8']:.2f} block0 share {res['emb_over_blocks']['block_hist'][0]:.2f}")
