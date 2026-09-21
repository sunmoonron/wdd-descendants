"""e171: the simplest system with accumulation. Toy: sparse features x (n, sparsity s, importance decay) are
written into a d-dim stream by W1 (block 0 = feature writes); block 1 is a residual MLP (m hidden units, ReLU)
that reads the stream and writes back; a linear readout reconstructs x from the final stream, trained with
importance-weighted MSE. Dictionary = columns of W1 (block-0 atoms) + output rows of block 1 (block-1 atoms), all
known. Measures on the FINAL stream: identification of the largest active feature (OMP / dual), its prominence,
the law's prediction, and IMPORTANCE = loss increase when that feature's block-0 write is removed from the final
stream (holding block 1 fixed) vs when it is removed at the input (block 1 re-runs). Correlation of readability
with each importance, for block-1 strengths (hidden width m) in (0, 64, 256, 1024)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
torch.manual_seed(0); rows = []; n, d, s = 400, 64, 0.05; imp = 0.9 ** torch.arange(n, device=DEV)
for m in (0, 64, 256, 1024):
    W1 = torch.nn.Parameter(torch.randn(d, n, device=DEV) * 0.1); Wr = torch.nn.Parameter(torch.randn(n, d, device=DEV) * 0.1); br = torch.nn.Parameter(torch.zeros(n, device=DEV)); params = [W1, Wr, br]
    if m:
        Win = torch.nn.Parameter(torch.randn(m, d, device=DEV) * 0.1); Wout = torch.nn.Parameter(torch.randn(m, d, device=DEV) * 0.1); bin_ = torch.nn.Parameter(torch.zeros(m, device=DEV)); params += [Win, Wout, bin_]
    opt = torch.optim.Adam(params, lr=2e-3)
    def fwd(x):
        h0 = x @ W1.T; h1 = h0 + (torch.relu(h0 @ Win.T + bin_) @ Wout if m else 0); return h0, h1, torch.relu(h1 @ Wr.T + br)
    with torch.enable_grad():
        for step in range(4000):
            x = torch.rand(1024, n, device=DEV) * (torch.rand(1024, n, device=DEV) < s); _, _, y = fwd(x); loss = (imp * (y - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    x = torch.rand(4096, n, device=DEV) * (torch.rand(4096, n, device=DEV) < s); h0, h1, y = fwd(x); base = (imp * (y - x) ** 2).mean(1)
    norms1 = W1.detach().norm(dim=0); A0 = (W1.detach() / norms1.clamp_min(1e-6)).T; C0 = x * norms1[None]
    atoms = [A0]
    if m: no = Wout.detach().norm(dim=1); atoms.append(Wout.detach() / no.clamp_min(1e-6)[:, None])
    A = torch.cat(atoms); active = (C0.abs() > 0).any(1); top = C0.abs().argmax(1); hc = h1.detach() - h1.detach().mean(0)
    prom = (hc * A0[top]).sum(1).abs() / hc.norm(dim=1).clamp_min(1e-9); k = 16
    so, _, _ = omp(hc, A, k); S_ = A.T @ A; ev, V = torch.linalg.eigh(S_); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sd, _, _ = oneshot(hc, A, k, whiten=Winv)
    hit_o, hit_d = (so == top[:, None]).any(1), (sd == top[:, None]).any(1)
    Sig = (hc.T @ hc) / hc.shape[0]; ev2, V2 = torch.linalg.eigh(Sig); half = V2 @ torch.diag(ev2.clamp_min(0).sqrt()) @ V2.T; z = torch.randn(2048, d, device=DEV) @ half; z = z / z.norm(dim=1, keepdim=True); lvl = (z @ A.T).abs().topk(k, dim=1).values[:, -1].mean().item()
    # importance 1: remove the feature's block-0 write from the FINAL stream (block 1 fixed): loss increase
    v = (C0.gather(1, top[:, None]) * A0[top]); y1 = torch.relu((h1.detach() - v) @ Wr.detach().T + br.detach()); imp_final = (imp * (y1 - x) ** 2).mean(1) - base.detach()
    # importance 2: remove the feature at the input (block 1 re-runs)
    x2 = x.clone(); x2[torch.arange(len(x)), top] = 0; _, _, y2 = fwd(x2); imp_input = (imp * (y2 - x) ** 2).mean(1) - base.detach()
    # survival of the feature write in the final stream
    surv = ((h1.detach() * A0[top]).sum(1) / C0.gather(1, top[:, None])[:, 0].clamp_min(1e-9))
    def spearman(a, b):
        ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    a_ = active
    rows.append(dict(m=m, recall_omp=hit_o[a_].float().mean().item(), recall_dual=hit_d[a_].float().mean().item(), law_pred=(prom[a_] > lvl).float().mean().item(), prom_med=prom[a_].median().item(), survival_med=surv[a_].median().item(),
                     rho_hit_vs_imp_final=spearman(hit_o[a_].float(), imp_final[a_]), rho_hit_vs_imp_input=spearman(hit_o[a_].float(), imp_input[a_]), rho_prom_vs_imp_input=spearman(prom[a_], imp_input[a_]), rho_surv_vs_imp_input=spearman(surv[a_], imp_input[a_]),
                     rho_hit_vs_feature_importance=spearman(hit_o[a_].float(), imp[top[a_]]), block1_share_of_stream=((h1.detach() - h0.detach())[a_].norm(dim=1) / h1.detach()[a_].norm(dim=1)).median().item() if m else 0.0))
    log(f"m={m}: recall omp {rows[-1]['recall_omp']:.2f} dual {rows[-1]['recall_dual']:.2f} law {rows[-1]['law_pred']:.2f} | survival med {rows[-1]['survival_med']:.2f} block-1 share {rows[-1]['block1_share_of_stream']:.2f} | rho(hit, importance at input) {rows[-1]['rho_hit_vs_imp_input']:+.2f} (final-stream removal {rows[-1]['rho_hit_vs_imp_final']:+.2f}); rho(prominence, importance) {rows[-1]['rho_prom_vs_imp_input']:+.2f}; rho(survival, importance) {rows[-1]['rho_surv_vs_imp_input']:+.2f}; rho(hit, feature importance weight) {rows[-1]['rho_hit_vs_feature_importance']:+.2f}")
record("e171_toy_twolayer", dict(rows=rows), " | ".join(f"m{r['m']}: recall {r['recall_omp']:.2f}/{r['recall_dual']:.2f} law {r['law_pred']:.2f} surv {r['survival_med']:.2f} rho(hit,imp) {r['rho_hit_vs_imp_input']:+.2f} rho(prom,imp) {r['rho_prom_vs_imp_input']:+.2f}" for r in rows))
