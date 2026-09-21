"""e104 (recursion in depth): each block's increment decomposed over the atoms of EARLIER blocks only (does a block
re-write directions that earlier blocks own?), over its own atoms, and over later blocks' atoms (control), at k=16,
on typical states: FVU by dictionary; plus the fraction of the increment's energy along the previous increment's
direction (cos between consecutive increments)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = 4096; ids = sub(c.NT, N); rows = []
for b in range(1, c.NB - 1):
    D_ = c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV); D_ = D_ - D_.mean(0); typ = typical_mask(c.X(b, center=False)[ids])
    Aown, _ = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); Apast, _ = c.dictionary(b, blocks=list(range(b)), types=(T_MLP, T_ATT, T_BIAS)); Afut, _ = c.dictionary(c.NB - 1, blocks=list(range(b + 1, c.NB)), types=(T_MLP, T_ATT, T_BIAS))
    n = min(Aown.shape[0], Apast.shape[0], Afut.shape[0]); g = torch.Generator().manual_seed(b)
    sub_ = lambda M: M[torch.randperm(M.shape[0], generator=g)[:n].to(DEV)]
    out = {}
    for nm, Ad in (("own", sub_(Aown)), ("past", sub_(Apast)), ("future", sub_(Afut)), ("rot_own", rotate(sub_(Aown)))):
        _, _, e = omp(D_, Ad, 16); out[nm] = fvu(e[:, 15], D_, typ)
    prev = c.s["H"][b][ids].float().to(DEV) - c.s["H"][b - 1][ids].float().to(DEV); cosprev = ((D_ * prev).sum(1) / (D_.norm(dim=1) * prev.norm(dim=1)).clamp_min(1e-6))[typ]
    rows.append(dict(b=b, n_atoms_each=n, fvu16=out, cos_prev_increment_med=cosprev.median().item()))
    log(f"{tag} b{b}: own {out['own']:.2f} past {out['past']:.2f} future {out['future']:.2f} rot {out['rot_own']:.2f} | cos(prev increment) {rows[-1]['cos_prev_increment_med']:+.2f}")
record(f"e104_incpast_{tag}", dict(model=tag, rows=rows), " | ".join(f"b{r['b']}: own {r['fvu16']['own']:.2f} past {r['fvu16']['past']:.2f} fut {r['fvu16']['future']:.2f} rot {r['fvu16']['rot_own']:.2f}" for r in rows))
