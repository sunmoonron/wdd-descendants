"""e01: the depth map. At every level L: typical-state FVU (weight vs rotated) at k=32/64, OMP top-1 recall,
one-shot recall, recall of the token's own embedding atom, support composition by atom type, sink variance share.
Question: is the middle layer special, and does the token identity (a KNOWN write) stay readable with depth?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; N = int(os.environ.get("WDD_N", 4096)); K = 64
c = Cache(tag); ids = sub(c.NT, N); tok_ids = c.s["eval_ids"].reshape(-1)[ids]
rows = []
for L in range(c.NB):
    X = c.X(L)[ids]; Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw)
    A, lab = c.dictionary(L)
    sel, cof, err = omp(X, A, K); selr, cofr, errr = omp(X, rotate(A), K)
    tb, tn, tc = c.top_writes(L, 1); true_row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
    rec = recall_curve(sel, true_row); sel1, _, _ = oneshot(X, A, K); rec1 = recall_curve(sel1, true_row, ks=(K,))
    # the token's own embedding row is atom row == token id (tok atoms come first)
    emb_row = tok_ids.to(DEV); emb_rec = recall_curve(sel, emb_row)
    emb_cof = (cof * (sel == emb_row[:, None])).sum(1)     # recovered coefficient of the token atom
    enorm = c.d["emb"][0].norm(dim=-1)[tok_ids].to(DEV)    # true coefficient at birth (unit-atom convention)
    surv = (Xraw @ c.d["A"][:c.n_tok_emb].to(DEV).T).gather(1, emb_row[:, None])[:, 0] / enorm  # survival of the token write
    types = lab["type"].to(DEV)[sel]
    comp = {int(t): (types == t).float().mean().item() for t in range(5)}
    e = (X ** 2).sum(-1); sink_share = (e[~typ].sum() / e.sum()).item()
    r = dict(L=L, atoms=A.shape[0], n_typ=int(typ.sum()), sink_share=sink_share,
             fvu32=fvu(err[:, 31], X, typ), fvu64=fvu(err[:, 63], X, typ), rot32=fvu(errr[:, 31], X, typ), rot64=fvu(errr[:, 63], X, typ),
             fvu32_sink=fvu(err[:, 31], X, ~typ), rot32_sink=fvu(errr[:, 31], X, ~typ),
             recall=rec, oneshot64=rec1[K], emb_recall=emb_rec, emb_cof_ratio_med=(emb_cof / enorm)[emb_cof != 0].median().item() if (emb_cof != 0).any() else None,
             emb_survival_med=surv[typ].median().item(), support_types=comp)
    rows.append(r); log(f"{tag} L{L}: fvu32 {r['fvu32']:.3f} rot {r['rot32']:.3f} recall64 {rec[K]:.3f} oneshot {rec1[K]:.3f} emb_recall64 {emb_rec[K]:.3f} emb_surv {r['emb_survival_med']:.2f}")
record(f"e01_layers_{tag}", dict(model=tag, N=N, rows=rows), " | ".join(f"L{r['L']}:{r['fvu32']:.2f}/{r['rot32']:.2f} rec{r['recall'][K]:.2f} emb{r['emb_recall'][K]:.2f}" for r in rows))
