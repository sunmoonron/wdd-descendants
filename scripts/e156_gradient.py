"""e156: what the loss cares about vs what WDD reads. For 16 sequences, the gradient g of the next-token loss with
respect to the level-L residual state (one backward pass). First-order importance of true write i to the loss is
c_i (g . d_i) (the write's direction dotted with the gradient, times its coefficient). Per token: the top-3 writes
by |c (g.d)| vs the ledger top-3 by |c| vs WDD's support (OMP and dual): overlap; the fraction of gradient-important
writes that WDD identifies; Spearman between the WDD-recovered coefficient and the gradient importance among
identified writes; and does prominence predict |g.d| (are readable directions the loss-sensitive ones)?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; ids = torch.arange(NT)
store = {}
def hook(mod, inp, out):
    o = out[0] if isinstance(out, tuple) else out; o.retain_grad(); store["h"] = o; return out
h = arch.layers[L].register_forward_hook(hook)
with torch.enable_grad():
    loss = model(ids_seq, labels=ids_seq).loss; loss.backward()
h.remove(); G = store["h"].grad.detach().float().reshape(NT, c.D) * NT                                                     # per-token gradient (loss is a mean)
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); mlp_rows = torch.cat([c.atom_index(L, torch.full((c.DFF,), b), torch.arange(c.DFF)) for b in range(L + 1)]).to(DEV); Am = A[mlp_rows]
imp = C * (G @ Am.T)                                                                                                       # [NT, M] first-order loss attribution of each write
S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
so, co, _ = omp(X, A, 64); sd, cd, _ = oneshot(X, A, 64, whiten=Winv)
top_imp = imp.abs().topk(3, dim=1).indices; top_c = C.abs().topk(3, dim=1).indices; rows_imp = mlp_rows[top_imp]; rows_c = mlp_rows[top_c]
def found(sel, rows): return (sel[:, :, None] == rows[:, None, :]).any(1)                                                   # [NT, 3]
res = dict(model=tag, L=L, NT=NT, overlap_imp_vs_ledger_top3=(top_imp[:, :, None] == top_c[:, None, :]).any(2).float().mean().item(),
           top1_imp_is_ledger_top1=(top_imp[:, 0] == top_c[:, 0])[typ].float().mean().item(),
           wdd_finds=dict(imp_top1_omp=found(so, rows_imp)[:, 0][typ].float().mean().item(), imp_top1_dual=found(sd, rows_imp)[:, 0][typ].float().mean().item(), imp_top3_omp=found(so, rows_imp)[typ].float().mean().item(), imp_top3_dual=found(sd, rows_imp)[typ].float().mean().item(),
                          ledger_top1_omp=found(so, rows_c)[:, 0][typ].float().mean().item(), ledger_top1_dual=found(sd, rows_c)[:, 0][typ].float().mean().item()))
# among identified true writes: Spearman(recovered coefficient magnitude, |gradient importance|); and prominence vs |g.d|
ismlp = lab["type"].to(DEV)[so] == T_MLP; key = torch.where(ismlp, lab["block"].to(DEV)[so] * c.DFF + lab["index"].to(DEV)[so], torch.zeros_like(so))
thr = 0.05 * C.abs().max(1, keepdim=True).values; real = ismlp & (C.gather(1, key).abs() >= thr)
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
rr = real & typ[:, None]; res["spearman_recovered_coef_vs_grad_importance"] = spearman(co[rr].abs(), imp.gather(1, key)[rr].abs()); res["spearman_true_coef_vs_grad_importance"] = spearman(C.gather(1, key)[rr].abs(), imp.gather(1, key)[rr].abs())
gd = (G @ Am.T).abs(); prom_all = (X @ Am.T).abs() / X.norm(dim=1, keepdim=True); big = C.abs() >= thr
res["spearman_prominence_vs_grad_sensitivity_real_writes"] = spearman(prom_all[big & typ[:, None]][::7], gd[big & typ[:, None]][::7])
res["share_of_loss_attribution_in_wdd_support"] = ((imp.abs() * torch.zeros_like(imp).scatter_(1, key.clamp(min=0), real.float()).clamp(max=1))[typ].sum() / imp.abs()[typ].sum()).item()
record(f"e156_grad_{tag}", res, f"gradient-important top-1 = ledger top-1 in {res['top1_imp_is_ledger_top1']:.2f} of tokens (top-3 overlap {res['overlap_imp_vs_ledger_top3']:.2f}) | WDD finds the gradient-important top-1: omp {res['wdd_finds']['imp_top1_omp']:.2f} dual {res['wdd_finds']['imp_top1_dual']:.2f} (ledger top-1: {res['wdd_finds']['ledger_top1_omp']:.2f}/{res['wdd_finds']['ledger_top1_dual']:.2f}); top-3 {res['wdd_finds']['imp_top3_omp']:.2f}/{res['wdd_finds']['imp_top3_dual']:.2f} | among identified real writes: Spearman(recovered |c|, |loss attribution|) {res['spearman_recovered_coef_vs_grad_importance']:+.2f} (true |c|: {res['spearman_true_coef_vs_grad_importance']:+.2f}) | prominence vs gradient sensitivity {res['spearman_prominence_vs_grad_sensitivity_real_writes']:+.2f} | share of |loss attribution| carried by WDD's real atoms {res['share_of_loss_attribution_in_wdd_support']:.2f}")
