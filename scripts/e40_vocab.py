"""e40: does observability line up with vocabulary-space interpretability? For the always- and never-identified
neurons (dominant-write recall > 0.9 / < 0.1), the unembedding projection of the atom (W_U d): kurtosis, top-10
token concentration, and the fraction whose top token is a function word / punctuation. Uses only the checkpoint."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); hit = (sel == row[:, None]).any(1)
key = row[typ]; h = hit[typ].float(); uk, inv, cnt = key.unique(return_inverse=True, return_counts=True)
rec = torch.zeros(len(uk), device=DEV).index_add_(0, inv, h) / cnt; m = cnt >= 20
model, tok, fam = load_model(c.name); WU = model.get_output_embeddings().weight.detach().float().to(DEV); del model
def stats(group):
    if len(group) == 0: return None
    P = A[group] @ WU.T                                                                     # [n, V]
    z = (P - P.mean(1, keepdim=True)) / P.std(1, keepdim=True); kurt = (z ** 4).mean(1) - 3
    top = P.topk(10, dim=1); conc = (top.values.abs().sum(1) / P.abs().sum(1))
    toks = [tok.decode([int(i)]) for i in top.indices[:, 0].tolist()]
    fw = sum(1 for t in toks if t.strip().lower() in {"the", "of", "and", "a", "in", "to", "is", ",", ".", "that", "for", "on", "with", "as", "at", "by", "an", "or", "it", "from", "was", "be", "this", "are", "were", "which", "s", "'s", "-", "(", ")", "\n"} or not t.strip().isalnum()) / len(toks)
    return dict(n=len(group), kurtosis_med=kurt.median().item(), top10_conc_med=conc.median().item(), frac_function_or_punct=fw, examples=toks[:8], recall_mean=rec[torch.isin(uk, group)].mean().item())
res = dict(model=tag, L=L, always=stats(uk[m & (rec > 0.9)]), never=stats(uk[m & (rec < 0.1)]), middle=stats(uk[m & (rec > 0.4) & (rec < 0.6)]), all=stats(uk[m]))
record(f"e40_vocab_{tag}", res, " | ".join(f"{k}: n {v['n']} kurt {v['kurtosis_med']:.1f} conc {v['top10_conc_med']:.3f} func {v['frac_function_or_punct']:.2f} eg {v['examples'][:4]}" for k, v in res.items() if isinstance(v, dict) and v))
