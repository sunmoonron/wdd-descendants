"""e24: does a neuron's identifiability transfer across corpora? Per-neuron recall on WikiText vs on the Pile
(same model, same dictionary, different states). If identification is a neuron property, the two should agree."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c1 = Cache(tag); c2 = Cache(tag + "_pile"); L = mid(c1)
def per_neuron(c):
    X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L); sel, cof, err = get_omp(c, L, A=A, X=X)
    tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); hit = (sel == row[:, None]).any(1)
    key = (tb[:, 0] * c.DFF + tn[:, 0]).to(DEV)[typ]; h = hit[typ].float()
    uk, inv, cnt = key.unique(return_inverse=True, return_counts=True)
    rec = torch.zeros(len(uk), device=DEV).index_add_(0, inv, h) / cnt
    return dict(zip(uk.tolist(), zip(rec.tolist(), cnt.tolist()))), hit[typ].float().mean().item(), fvu(err[:, 63], X, typ)
p1, r1, f1 = per_neuron(c1); p2, r2, f2 = per_neuron(c2)
common = [k for k in p1 if k in p2 and p1[k][1] >= 20 and p2[k][1] >= 20]
a = torch.tensor([p1[k][0] for k in common]); b = torch.tensor([p2[k][0] for k in common])
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
res = dict(model=tag, L=L, recall_wikitext=r1, recall_pile=r2, fvu64_wikitext=f1, fvu64_pile=f2, n_common_neurons=len(common), spearman=spearman(a, b), pearson=torch.corrcoef(torch.stack([a, b]))[0, 1].item(),
           mean_abs_diff=(a - b).abs().mean().item(), agree_always=((a > 0.9) & (b > 0.9)).sum().item() / max(1, (a > 0.9).sum().item()), agree_never=((a < 0.1) & (b < 0.1)).sum().item() / max(1, (a < 0.1).sum().item()))
record(f"e24_transfer_{tag}", res, f"recall wt {r1:.3f} pile {r2:.3f} | per-neuron recall across corpora: n {len(common)} Spearman {res['spearman']:.2f} Pearson {res['pearson']:.2f} |diff| {res['mean_abs_diff']:.2f} always-agree {res['agree_always']:.2f} never-agree {res['agree_never']:.2f}")
