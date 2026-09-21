"""e59: can WDD be fooled? Is identifiability related to causal importance? For neurons that are always / never
identified when dominant (from the cached mid-layer OMP), ablate each neuron (zero its activation) on 16 held-out
sequences and measure the cross-entropy increase; compare the distributions, plus a random-neuron control matched
by block. If never-neurons matter as much or more, WDD's readable set is not the important set."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); hit = (sel == row[:, None]).any(1)
key = (tb[:, 0] * c.DFF + tn[:, 0]).to(DEV)[typ]; h = hit[typ].float(); uk, inv, cnt = key.unique(return_inverse=True, return_counts=True)
rec = torch.zeros(len(uk), device=DEV).index_add_(0, inv, h) / cnt; m = cnt >= 20
g = torch.Generator().manual_seed(0)
def pick(mask, n=40):
    u = uk[mask]; return u[torch.randperm(len(u), generator=g)[:n].to(DEV)]
groups = dict(always=pick(m & (rec > 0.9)), never=pick(m & (rec < 0.1)), middle=pick(m & (rec > 0.4) & (rec < 0.6)))
model, tok, fam = load_model(c.name); arch = Arch(model, fam); ids = c.s["eval_ids"][16:32].to(DEV)
def ce(hook_b=None, hook_j=None):
    hs = []
    if hook_b is not None:
        def f(mod, inp):
            x = inp[0].clone(); x[..., hook_j] = 0; return (x,)
        hs.append(arch.mlp_lin(hook_b).register_forward_pre_hook(f))
    out = model(ids, labels=ids); [hh.remove() for hh in hs]; return out.loss.item()
base = ce(); res = dict(model=tag, L=L, base_ce=base, groups={})
for name, u in groups.items():
    d = []
    for kk in u.tolist():
        b, j = kk // c.DFF, kk % c.DFF; d.append(dict(block=b, neuron=j, dce=ce(b, j) - base))
    # block-matched random control
    ctrl = []
    for kk in u.tolist():
        b = kk // c.DFF; j = int(torch.randint(0, c.DFF, (1,), generator=g)); ctrl.append(ce(b, j) - base)
    dd = torch.tensor([x["dce"] for x in d]); cc = torch.tensor(ctrl)
    res["groups"][name] = dict(n=len(d), dce_median=dd.median().item(), dce_mean=dd.mean().item(), dce_q90=dd.quantile(0.9).item(), frac_above_0p01=(dd > 0.01).float().mean().item(), control_median=cc.median().item(), control_mean=cc.mean().item(), top=sorted(d, key=lambda x: -x["dce"])[:3])
    log(f"{tag} {name}: n {len(d)} dCE median {dd.median():.4f} mean {dd.mean():.4f} q90 {dd.quantile(0.9):.4f} | control median {cc.median():.4f} mean {cc.mean():.4f}")
record(f"e59_importance_{tag}", res, " | ".join(f"{k}: n {v['n']} dCE med {v['dce_median']:.4f} mean {v['dce_mean']:.4f} q90 {v['dce_q90']:.4f} (ctrl {v['control_mean']:.4f})" for k, v in res["groups"].items()))
