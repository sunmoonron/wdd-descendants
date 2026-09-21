"""e20: the sink states. Which atoms reconstruct them, how many, is it the same neuron across sink tokens and
across levels, and how much of a sink state is one atom."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); res = dict(model=tag, levels={})
for L in sorted(set([1, 2, mid(c), c.NB - 1])):
    Xraw = c.X(L, center=False); X = c.X(L); typ = typical_mask(Xraw); A, lab = c.dictionary(L)
    S = torch.nonzero(~typ)[:, 0]
    if len(S) == 0: continue
    sel, cof, err = omp(X[S], A, 8); st, sb, si = lab["type"].to(DEV)[sel], lab["block"].to(DEV)[sel], lab["index"].to(DEV)[sel]
    e = (X[S] ** 2).sum(1); first = sel[:, 0]; u, cnt = first.unique(return_counts=True); top = u[cnt.argmax()]
    fr = (err[:, 0] / e)
    tb, tn, tc = c.top_writes(L, 1)
    res["levels"][L] = dict(n_sink=int(len(S)), positions=(S % CTX).unique().tolist()[:10], norm_ratio_med=(Xraw[S].norm(dim=1).median() / Xraw[typ].norm(dim=1).median()).item(),
                            fvu1=fvu(err[:, 0], X[S]), fvu2=fvu(err[:, 1], X[S]), fvu8=fvu(err[:, 7], X[S]),
                            first_atom=dict(type=int(lab["type"][top]), block=int(lab["block"][top]), index=int(lab["index"][top]), share_of_sinks=(cnt.max() / len(S)).item()),
                            first_atom_is_true_dominant=(first == c.atom_index(L, tb[S.cpu(), 0], tn[S.cpu(), 0]).to(DEV)).float().mean().item(),
                            top_channel=int(Xraw[S].abs().mean(0).argmax()), top_channel_share=((Xraw[S] ** 2).mean(0).max() / (Xraw[S] ** 2).mean(0).sum()).item())
    log(f"{tag} L{L}: {len(S)} sinks, norm x{res['levels'][L]['norm_ratio_med']:.0f}, fvu1 {res['levels'][L]['fvu1']:.4f} fvu2 {res['levels'][L]['fvu2']:.4f}, first atom {res['levels'][L]['first_atom']} true-dominant {res['levels'][L]['first_atom_is_true_dominant']:.2f}, channel {res['levels'][L]['top_channel']} share {res['levels'][L]['top_channel_share']:.2f}")
record(f"e20_sinks_{tag}", res, " | ".join(f"L{L}: n {v['n_sink']} fvu1 {v['fvu1']:.3f} atom t{v['first_atom']['type']} b{v['first_atom']['block']} #{v['first_atom']['index']} ({v['first_atom']['share_of_sinks']:.2f}) ch{v['top_channel']} {v['top_channel_share']:.2f}" for L, v in res["levels"].items()))
