"""e142: the neuron phenotype over training (Pythia-410m checkpoints 16k, 32k, 64k, 143k, mid layer). Per neuron
dominant on >= 20 typical tokens at the final checkpoint: its identification rate at each checkpoint (same neuron
index), the correlation of per-neuron rates across checkpoints, the fraction of final never-neurons that were
readable earlier and of final always-neurons that were unreadable earlier."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tags = ["pythia410_step16000", "pythia410_step32000", "pythia410_step64000", "pythia410"]; per = {}
for tg in tags:
    c = Cache(tg); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L); sel, cof, err = get_omp(c, L, A=A, X=X)
    tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); hit = (sel == row[:, None]).any(1)
    key = (tb[:, 0] * c.DFF + tn[:, 0]).to(DEV)[typ]; uk, inv, cnt = key.unique(return_inverse=True, return_counts=True); rec = torch.zeros(len(uk), device=DEV).index_add_(0, inv, hit[typ].float()) / cnt
    per[tg] = {int(k): (r, int(n)) for k, r, n in zip(uk.tolist(), rec.tolist(), cnt.tolist())}
final = per["pythia410"]; keys = [k for k, (r, n) in final.items() if n >= 20]
res = dict(n_final_neurons=len(keys), pairs={})
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
for tg in tags[:-1]:
    common = [k for k in keys if k in per[tg] and per[tg][k][1] >= 10]; a = torch.tensor([per[tg][k][0] for k in common]); b = torch.tensor([final[k][0] for k in common])
    res["pairs"][tg] = dict(n_common=len(common), spearman=spearman(a, b), final_never_readable_earlier=((a > 0.5) & (b < 0.1)).sum().item() / max(1, (b < 0.1).sum().item()), final_always_unreadable_earlier=((a < 0.5) & (b > 0.9)).sum().item() / max(1, (b > 0.9).sum().item()),
                             frac_still_dominant=len(common) / len(keys), mean_recall_then=a.mean().item(), mean_recall_final_on_common=b.mean().item())
record("e142_phenotype_training_pythia410", res, " | ".join(f"{tg.split('_')[-1]}: n {v['n_common']} (still dominant {v['frac_still_dominant']:.2f}) Spearman with final {v['spearman']:.2f}; final never readable then {v['final_never_readable_earlier']:.2f}; final always unreadable then {v['final_always_unreadable_earlier']:.2f}" for tg, v in res["pairs"].items()))
