"""e59b: readability vs importance at the neuron level. 200 neurons that are dominant on >= 20 typical tokens,
each ablated (activation zeroed) on 16 held-out sequences: Spearman of the ablation cost with per-neuron
identification rate, with mean prominence, with mean raw survival, and with the neuron's mean |coefficient|."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); Xraw = c.X(L, center=False); typ = typical_mask(Xraw); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); hit = (sel == row[:, None]).any(1)
d = A[row]; prom = (X * d).sum(1).abs() / X.norm(dim=1); surv = (Xraw * d).sum(1) / tc[:, 0].to(DEV); mag = tc[:, 0].abs().to(DEV)
key = (tb[:, 0] * c.DFF + tn[:, 0]).to(DEV)[typ]; uk, inv, cnt = key.unique(return_inverse=True, return_counts=True)
agg = lambda v: torch.zeros(len(uk), device=DEV).index_add_(0, inv, v[typ].float()) / cnt
rec, pr, sv, mg = agg(hit), agg(prom), agg(surv.clamp(-2, 3)), agg(mag); m = cnt >= 20
g = torch.Generator().manual_seed(1); cand = torch.nonzero(m)[:, 0]; pick = cand[torch.randperm(len(cand), generator=g)[:200].to(DEV)]
model, tok, fam = load_model(c.name); arch = Arch(model, fam); ids = c.s["eval_ids"][16:32].to(DEV)
def ce(b=None, j=None):
    hs = []
    if b is not None:
        def f(mod, inp):
            x = inp[0].clone(); x[..., j] = 0; return (x,)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(f))
    out = model(ids, labels=ids); [h.remove() for h in hs]; return out.loss.item()
base = ce(); dce = torch.tensor([ce(int(uk[i]) // c.DFF, int(uk[i]) % c.DFF) - base for i in pick.tolist()], device=DEV)
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
res = dict(model=tag, L=L, n=len(pick), base_ce=base, dce_median=dce.median().item(), dce_q90=dce.quantile(0.9).item(),
           spearman_dce_vs=dict(recall=spearman(dce, rec[pick]), prominence=spearman(dce, pr[pick]), raw_survival=spearman(dce, sv[pick]), mean_abs_coef=spearman(dce, mg[pick]), count=spearman(dce, cnt[pick].float())),
           dce_by_recall_tercile=[dce[rec[pick] <= rec[pick].quantile(1 / 3)].mean().item(), dce[(rec[pick] > rec[pick].quantile(1 / 3)) & (rec[pick] <= rec[pick].quantile(2 / 3))].mean().item(), dce[rec[pick] > rec[pick].quantile(2 / 3)].mean().item()])
record(f"e59b_rank_{tag}", res, f"n {len(pick)} dCE med {res['dce_median']:.4f} q90 {res['dce_q90']:.4f} | Spearman(dCE, recall) {res['spearman_dce_vs']['recall']:+.2f} prominence {res['spearman_dce_vs']['prominence']:+.2f} raw-survival {res['spearman_dce_vs']['raw_survival']:+.2f} |coef| {res['spearman_dce_vs']['mean_abs_coef']:+.2f} count {res['spearman_dce_vs']['count']:+.2f} | mean dCE by recall tercile low/mid/high " + " ".join(f"{v:.4f}" for v in res['dce_by_recall_tercile']))
