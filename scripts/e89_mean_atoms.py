"""e89: the centering mean decomposed. OMP of the level-L mean state mu (the DC component every token shares) over
the dictionary at k=16: which atoms (type, block, neuron) make up the mean, how much of it is one atom, and are
those the same neurons that dominate the sink states / the most frequent dominant writers."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); A, lab = c.dictionary(L); mu = c.s["mu"][L + 1].to(DEV)[None]
sel, cof, err = omp(mu, A, 16); typA, blkA, idxA = lab["type"], lab["block"], lab["index"]
atoms = [dict(type=int(typA[i]), block=int(blkA[i]), index=int(idxA[i]), coef=float(cof[0, j]), cum_fvu=float(err[0, j] / (mu ** 2).sum())) for j, i in enumerate(sel[0].tolist())]
tb, tn, tc = c.top_writes(L, 1); key = (tb[:, 0] * c.DFF + tn[:, 0]); uk, cnt = key.unique(return_counts=True); top_dom = [(int(k) // c.DFF, int(k) % c.DFF) for k in uk[cnt.topk(10).indices].tolist()]
mean_atoms_mlp = [(a["block"], a["index"]) for a in atoms if a["type"] == T_MLP]
res = dict(model=tag, L=L, mean_norm=mu.norm().item(), typical_norm=c.X(L, center=False).norm(dim=1).median().item(), atoms=atoms, fvu1=atoms[0]["cum_fvu"], fvu4=atoms[3]["cum_fvu"], fvu16=atoms[-1]["cum_fvu"],
           overlap_with_top10_dominant_writers=len(set(mean_atoms_mlp) & set(top_dom)), top10_dominant_writers=top_dom)
record(f"e89_mean_{tag}", res, f"|mu| {res['mean_norm']:.1f} vs typical |x| {res['typical_norm']:.1f} | fvu after 1/4/16 atoms {res['fvu1']:.3f}/{res['fvu4']:.3f}/{res['fvu16']:.3f} | atoms: " + " ".join(f"t{a['type']}b{a['block']}#{a['index']}({a['coef']:+.1f})" for a in atoms[:8]) + f" | overlap with top-10 dominant writers {res['overlap_with_top10_dominant_writers']}")
