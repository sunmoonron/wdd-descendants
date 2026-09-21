"""e126 (audit follow-up): does the sink neuron's atom sit in typical supports (a constant offset the cached mean
carries)? Fraction of typical k=64 supports containing the most frequent sink atom (from e20), its median
coefficient there relative to the token's largest recovered coefficient, and its selection rank."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); Xraw = c.X(L, center=False); typ = typical_mask(Xraw); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); S = torch.nonzero(~typ)[:, 0]; ss, _, _ = omp(X[S], A, 4); u, cnt = ss[:, 0].unique(return_counts=True); sink_atom = u[cnt.argmax()]
hit = sel == sink_atom; present = hit.any(1); rank = torch.where(present, hit.float().argmax(1), torch.full_like(present, -1, dtype=torch.long))
coef = (cof * hit).sum(1); rel = coef.abs() / cof.abs().max(1).values
res = dict(model=tag, L=L, sink_atom=dict(type=int(lab["type"][sink_atom]), block=int(lab["block"][sink_atom]), index=int(lab["index"][sink_atom])), frac_typical_supports=present[typ].float().mean().item(),
           rank_median=rank[typ & present].float().median().item() if (typ & present).any() else None, rel_coef_median=rel[typ & present].median().item() if (typ & present).any() else None, mean_proj_on_sink_atom_over_typ_norm=((X[typ] @ A[sink_atom]).abs().median() / X[typ].norm(dim=1).median()).item())
record(f"e126_sinkatom_{tag}", res, f"sink atom b{res['sink_atom']['block']}#{res['sink_atom']['index']} in {res['frac_typical_supports']:.2f} of typical supports (rank med {res['rank_median']}, coef/max {res['rel_coef_median']}) | median |x_c . sink atom| / |x_c| {res['mean_proj_on_sink_atom_over_typ_norm']:.3f}")
