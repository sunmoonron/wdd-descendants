"""e175: the many-block toy. Sparse features written by block 0 (W1), then B residual blocks (each a ReLU MLP of
width m reading the stream and writing back with its own output rows), a linear readout trained with
importance-weighted reconstruction. Dictionary = block-0 atoms + every block's output rows (all known). At the
final stream: identification of the largest active feature's block-0 atom (OMP@16 / dual@16), its raw survival,
the law's prediction, the cancel index of block writes, importance by input removal, and rho(readable, importance)
for B in (1, 4, 8) at m = 256. Does depth alone produce the real models' decoupling and cancellation?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
torch.manual_seed(0); rows = []; n, d, s, m = 400, 64, 0.05, 256; imp = 0.9 ** torch.arange(n, device=DEV)
for B in (1, 4, 8):
    W1 = torch.nn.Parameter(torch.randn(d, n, device=DEV) * 0.1); Wr = torch.nn.Parameter(torch.randn(n, d, device=DEV) * 0.1); br = torch.nn.Parameter(torch.zeros(n, device=DEV)); params = [W1, Wr, br]; blocks = []
    for b in range(B):
        Win = torch.nn.Parameter(torch.randn(m, d, device=DEV) * 0.1); bi = torch.nn.Parameter(torch.zeros(m, device=DEV)); Wout = torch.nn.Parameter(torch.randn(m, d, device=DEV) * (0.1 / math.sqrt(B))); blocks.append((Win, bi, Wout)); params += [Win, bi, Wout]
    opt = torch.optim.Adam(params, lr=2e-3)
    def fwd(x):
        h = x @ W1.T; hs = [h]; ws = []
        for Win, bi, Wout in blocks:
            w = torch.relu(h @ Win.T + bi) @ Wout; h = h + w; hs.append(h); ws.append(w)
        return hs, ws, torch.relu(h @ Wr.T + br)
    with torch.enable_grad():
        for step in range(4000):
            x = torch.rand(1024, n, device=DEV) * (torch.rand(1024, n, device=DEV) < s); _, _, y = fwd(x); loss = (imp * (y - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    x = torch.rand(4096, n, device=DEV) * (torch.rand(4096, n, device=DEV) < s); hs, ws, y = fwd(x); base = (imp * (y - x) ** 2).mean(1).detach(); hL = hs[-1].detach()
    norms1 = W1.detach().norm(dim=0); A0 = (W1.detach() / norms1.clamp_min(1e-6)).T; atoms = [A0]
    for Win, bi, Wout in blocks: atoms.append(Wout.detach() / Wout.detach().norm(dim=1, keepdim=True).clamp_min(1e-6))
    A = torch.cat(atoms); C0 = x * norms1[None]; active = (C0.abs() > 0).any(1); top = C0.abs().argmax(1); hc = hL - hL.mean(0); dtop = A0[top]
    prom = (hc * dtop).sum(1).abs() / hc.norm(dim=1).clamp_min(1e-9); k = 16
    so, _, _ = omp(hc, A, k); S_ = A.T @ A; ev, V = torch.linalg.eigh(S_); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sd, _, _ = oneshot(hc, A, k, whiten=Winv)
    hit_o, hit_d = (so == top[:, None]).any(1), (sd == top[:, None]).any(1)
    Sig = (hc.T @ hc) / hc.shape[0]; ev2, V2 = torch.linalg.eigh(Sig); half = V2 @ torch.diag(ev2.clamp_min(0).sqrt()) @ V2.T; z = torch.randn(2048, d, device=DEV) @ half; z = z / z.norm(dim=1, keepdim=True); lvl = (z @ A.T).abs().topk(k, dim=1).values[:, -1].mean().item()
    x2 = x.clone(); x2[torch.arange(len(x)), top] = 0; _, _, y2 = fwd(x2); imp_input = ((imp * (y2 - x) ** 2).mean(1) - base).detach()
    surv = (hL * dtop).sum(1) / C0.gather(1, top[:, None])[:, 0].clamp_min(1e-9); comps = [hs[0].detach()] + [w.detach() for w in ws]; cancel = ((hL ** 2).sum(1) / sum((v ** 2).sum(1) for v in comps)).median().item()
    contrib = [((w.detach() * dtop).sum(1) / C0.gather(1, top[:, None])[:, 0].clamp_min(1e-9))[active].median().item() for w in ws]
    def spearman(a, b):
        ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    a_ = active; rows.append(dict(B=B, recall_omp=hit_o[a_].float().mean().item(), recall_dual=hit_d[a_].float().mean().item(), law_pred=(prom[a_] > lvl).float().mean().item(), prom_med=prom[a_].median().item(), survival_med=surv[a_].median().item(), cancel_index=cancel, block_contrib_to_survival=contrib,
                                  rho_hit_vs_importance=spearman(hit_o[a_].float(), imp_input[a_]), rho_prom_vs_importance=spearman(prom[a_], imp_input[a_]), rho_surv_vs_importance=spearman(surv[a_], imp_input[a_]), frac_erased=(surv[a_] < 0.5).float().mean().item()))
    log(f"B={B}: recall omp {rows[-1]['recall_omp']:.2f} dual {rows[-1]['recall_dual']:.2f} law {rows[-1]['law_pred']:.2f} | survival med {rows[-1]['survival_med']:.2f} erased {rows[-1]['frac_erased']:.2f} cancel index {cancel:.2f} block contributions {[round(v, 2) for v in contrib]} | rho(hit, importance) {rows[-1]['rho_hit_vs_importance']:+.2f} rho(prom, imp) {rows[-1]['rho_prom_vs_importance']:+.2f} rho(surv, imp) {rows[-1]['rho_surv_vs_importance']:+.2f}")
record("e175_toy_multiblock", dict(rows=rows), " | ".join(f"B{r['B']}: recall {r['recall_omp']:.2f}/{r['recall_dual']:.2f} law {r['law_pred']:.2f} surv {r['survival_med']:.2f} erased {r['frac_erased']:.2f} cancel {r['cancel_index']:.2f} rho(hit,imp) {r['rho_hit_vs_importance']:+.2f} rho(prom,imp) {r['rho_prom_vs_importance']:+.2f}" for r in rows))
