"""e15: zero-data, weight-space WDD (meta). Each block's READ directions (MLP input rows; gate rows for gated
models) are decomposed by OMP over the WRITE atoms of earlier blocks + embeddings, over later blocks' atoms, and
over a rotated copy. If the model composes (Elhage 2021), earlier writes should explain read directions better
than later ones; if the stream is just a shared coordinate system, both should tie. Also max |cos| read-vs-write
by block distance, and the fraction of read energy inside the top-r principal subspace of the earlier writes."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); lab = c.d["lab"]; A_all = c.d["A"].to(DEV); typ, blk = lab["type"].to(DEV), lab["block"].to(DEV)
K = 16; rows = []
for b in range(1, c.NB - 1):
    R = c.d["RD"][b].to(DEV); R = R / R.norm(dim=1, keepdim=True)                       # read directions (unit) [DFF, D]
    Rs = R[sub(R.shape[0], 1024)]
    earlier = (blk < b) & (blk >= 0) | (blk < 0) ; later = blk > b
    Ae, Al = A_all[earlier], A_all[later]
    n = min(Ae.shape[0], Al.shape[0]); g = torch.Generator().manual_seed(b)
    Ae_ = Ae[torch.randperm(Ae.shape[0], generator=g)[:n].to(DEV)]; Al_ = Al[torch.randperm(Al.shape[0], generator=g)[:n].to(DEV)]   # size-matched
    out = {}
    for nm, Ad in (("earlier", Ae_), ("later", Al_), ("earlier_rot", rotate(Ae_)), ("later_rot", rotate(Al_))):
        _, _, err = omp(Rs, Ad, K); out[nm] = dict(fvu8=fvu(err[:, 7], Rs), fvu16=fvu(err[:, 15], Rs), maxcos_med=(Rs @ Ad.T).abs().max(1).values.median().item())
    if c.d["RG"] is not None:
        G_ = c.d["RG"][b].to(DEV); G_ = (G_ / G_.norm(dim=1, keepdim=True))[sub(G_.shape[0], 1024)]
        _, _, err = omp(G_, Ae_, K); out["gate_earlier"] = dict(fvu16=fvu(err[:, 15], G_)); _, _, err = omp(G_, Al_, K); out["gate_later"] = dict(fvu16=fvu(err[:, 15], G_))
    # by block distance: median max|cos| of read dirs to the MLP write atoms of block b' 
    dist = {}
    for bp in range(c.NB):
        m = (typ == T_MLP) & (blk == bp); dist[bp] = (Rs @ A_all[m].T).abs().max(1).values.median().item()
    rows.append(dict(b=b, n_atoms_each=n, omp=out, maxcos_by_write_block=dist))
    log(f"{tag} b{b}: read dirs FVU16 earlier {out['earlier']['fvu16']:.3f} later {out['later']['fvu16']:.3f} (rot {out['earlier_rot']['fvu16']:.3f}/{out['later_rot']['fvu16']:.3f}) maxcos e/l {out['earlier']['maxcos_med']:.3f}/{out['later']['maxcos_med']:.3f}")
record(f"e15_compose_{tag}", dict(model=tag, K=K, rows=rows), " | ".join(f"b{r['b']}: e {r['omp']['earlier']['fvu16']:.2f} l {r['omp']['later']['fvu16']:.2f} rot {r['omp']['earlier_rot']['fvu16']:.2f}" for r in rows))
