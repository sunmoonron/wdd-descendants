"""e160: per-write functional attribution vs first-order attribution. For 8 sequences at the mid layer: for each of
the first 12 OMP atoms that are real writes, remove that atom's contribution from the state alone and splice
(dCE per atom position), plus the same for the true top-3 writes exactly; compare the per-token distribution of
ablation effects with the first-order (gradient) attribution of the same writes: Spearman, and the share of the
total ablation effect carried by the single most prominent identified write."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; ids = torch.arange(NT)
Xraw = c.X(L, center=False)[ids]; mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; A, lab = c.dictionary(L); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * C.abs().max(1, keepdim=True).values
sel, cof, err = omp(X, A, 64); ismlp = typA[sel] == T_MLP; key = torch.where(ismlp, blkA[sel] * c.DFF + idxA[sel], torch.zeros_like(sel)); real = ismlp & (C.gather(1, key).abs() >= thr)
store = {}
def hook_g(mod, inp, out):
    o = out[0] if isinstance(out, tuple) else out; o.retain_grad(); store["h"] = o; return out
h = arch.layers[L].register_forward_hook(hook_g)
with torch.enable_grad(): loss = model(ids_seq, labels=ids_seq).loss; loss.backward()
h.remove(); G = store["h"].grad.detach().float().reshape(NT, c.D) * NT; model.zero_grad()
def ce_tokens(rep):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out; o = o.clone(); o[:] = rep.view(NS, CTX, c.D).to(o.dtype); return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
    hh = arch.layers[L].register_forward_hook(hook); logits = model(ids_seq).logits.float(); hh.remove()
    lp = torch.log_softmax(logits[:, :-1], -1); nll = -lp.gather(2, ids_seq[:, 1:, None])[:, :, 0]; return torch.cat([nll, torch.zeros(NS, 1, device=DEV)], 1).reshape(NT)
base = ce_tokens(Xraw); abl = torch.zeros(NT, 12, device=DEV); grad_attr = torch.zeros(NT, 12, device=DEV); valid = torch.zeros(NT, 12, dtype=torch.bool, device=DEV)
for j in range(12):
    v = cof[:, j][:, None] * A[sel[:, j]]; m = real[:, j]; rep = Xraw.clone(); rep[m] = rep[m] - v[m]; d = ce_tokens(rep) - base
    abl[:, j] = d; valid[:, j] = m; grad_attr[:, j] = (G * v).sum(1)                                                   # first-order: g . (c d)
typ = typical_mask(Xraw); ok = typ & (valid.sum(1) >= 3)
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
vv = valid & ok[:, None]; res = dict(model=tag, L=L, n_tokens=int(ok.sum()), spearman_ablation_vs_gradient_pooled=spearman(abl[vv], -grad_attr[vv]), spearman_ablation_vs_abs_coef=spearman(abl[vv].abs(), cof[:, :12][vv].abs()),
                                  ablation_dce_per_real_atom_median=abl[vv].median().item(), ablation_dce_mean=abl[vv].mean().item(), frac_negative=(abl[vv] < 0).float().mean().item(),
                                  share_of_total_effect_by_first_real_atom=((abl * valid.float())[ok].abs()[:, 0].sum() / (abl * valid.float())[ok].abs().sum()).item())
# rank-1 concentration: fraction of tokens where one atom carries > 50% of the total |dCE|
tot = (abl * valid.float()).abs(); res["frac_tokens_one_atom_dominates"] = ((tot.max(1).values / tot.sum(1).clamp_min(1e-9)) > 0.5)[ok].float().mean().item()
record(f"e160_perwrite_{tag}", res, f"tokens {res['n_tokens']} | per-real-atom ablation dCE median {res['ablation_dce_per_real_atom_median']:+.4f} mean {res['ablation_dce_mean']:+.4f} (negative in {res['frac_negative']:.2f}) | Spearman(ablation, first-order attribution) {res['spearman_ablation_vs_gradient_pooled']:+.2f}; (|ablation|, |coef|) {res['spearman_ablation_vs_abs_coef']:+.2f} | first real atom carries {res['share_of_total_effect_by_first_real_atom']:.2f} of the total effect; one atom dominates in {res['frac_tokens_one_atom_dominates']:.2f} of tokens")
