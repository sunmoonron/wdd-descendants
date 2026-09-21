"""e22: identification as a function of (level read, block born). At each level L, OMP recall of the dominant
write split by the block that wrote it, plus the share of tokens whose dominant write comes from each block.
Explains whether a recall collapse with depth is late-born writes being unreadable or old writes being erased."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
M = {}; share = {}; fvu_by_pos = {}
for L in range(c.NB):
    X = c.X(L)[ids]; Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
    tb, tn, tc = c.top_writes(L, 1); tb, tn = tb[ids, 0], tn[ids, 0]; row = c.atom_index(L, tb, tn).to(DEV)
    sel, cof, err = omp(X, A, 64); hit = (sel == row[:, None]).any(1); tbd = tb.to(DEV)
    M[L] = {b: (hit[typ & (tbd == b)].float().mean().item() if (typ & (tbd == b)).sum() > 30 else None) for b in range(L + 1)}
    share[L] = {b: (typ & (tbd == b)).float().sum().item() / typ.float().sum().item() for b in range(L + 1)}
    # the dominant write's magnitude relative to the state, by birth block
    rel = tc[ids, 0].abs().to(DEV) / Xraw.norm(dim=1)
    fvu_by_pos[L] = {b: (rel[typ & (tbd == b)].median().item() if (typ & (tbd == b)).sum() > 30 else None) for b in range(L + 1)}
    log(f"{tag} L{L}: recall by birth " + " ".join(f"b{b}:{v:.2f}({share[L][b]:.2f})" for b, v in M[L].items() if v is not None))
record(f"e22_birth_{tag}", dict(model=tag, N=N, recall=M, share=share, rel_mag=fvu_by_pos), " | ".join(f"L{L}: " + " ".join(f"b{b}:{v:.2f}/{share[L][b]:.2f}" for b, v in M[L].items() if v is not None) for L in M))
