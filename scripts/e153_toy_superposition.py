"""e153: WDD in the toy model of superposition (Elhage et al. 2022). Train the ReLU toy (n features, d dims,
sparsity s, importance decay) for 2000 steps; the learned W columns are the feature write directions (the
dictionary), the features are the ledger. Decompose hidden states with OMP / dual projection over W: identification
of the largest active feature vs its prominence, the covariance-matched competitor level, and importance. Does
the prominence law reproduce, and is readability tied to feature importance in the toy?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
torch.manual_seed(0); rows = []
for n, d, s in ((400, 40, 0.05), (400, 40, 0.15), (1600, 40, 0.03), (1600, 80, 0.03)):
    imp = (0.9 ** torch.arange(n, device=DEV)); W = torch.nn.Parameter(torch.randn(d, n, device=DEV) * 0.1); bb = torch.nn.Parameter(torch.zeros(n, device=DEV)); opt = torch.optim.Adam([W, bb], lr=1e-3)
    with torch.enable_grad():
        for step in range(3000):
            x = torch.rand(1024, n, device=DEV) * (torch.rand(1024, n, device=DEV) < s); h = x @ W.T; y = torch.relu(h @ W + bb); loss = (imp * (y - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    W_ = W.detach(); x = torch.rand(4096, n, device=DEV) * (torch.rand(4096, n, device=DEV) < s); h = x @ W_.T                                   # hidden states
    norms = W_.norm(dim=0); A = (W_ / norms.clamp_min(1e-6)).T; C = x * norms[None]                                                         # unit atoms, ledger coefficients (unit-atom convention)
    active = (C.abs() > 0).any(1); h, C, x = h[active], C[active], x[active]; hc = h - h.mean(0); top = C.abs().argmax(1); d_top = A[top]
    prom = (hc * d_top).sum(1).abs() / hc.norm(dim=1).clamp_min(1e-9); k = min(16, n // 4)
    so, _, _ = omp(hc, A, k); S_ = A.T @ A; ev, V = torch.linalg.eigh(S_); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sd, _, _ = oneshot(hc, A, k, whiten=Winv); s1, _, _ = oneshot(hc, A, k)
    hit_o, hit_d, hit_1 = (so == top[:, None]).any(1), (sd == top[:, None]).any(1), (s1 == top[:, None]).any(1)
    Sig = (hc.T @ hc) / hc.shape[0]; ev2, V2 = torch.linalg.eigh(Sig); half = V2 @ torch.diag(ev2.clamp_min(0).sqrt()) @ V2.T; z = torch.randn(2048, d, device=DEV) @ half; z = z / z.norm(dim=1, keepdim=True)
    lvl = (z @ A.T).abs().topk(k, dim=1).values[:, -1].mean().item(); pred = (prom > lvl).float().mean().item()
    n_active = (C.abs() > 0).sum(1).float().mean().item(); represented = (norms > 0.5).float().mean().item()
    def spearman(a, b):
        ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    rows.append(dict(n=n, d=d, sparsity=s, features_represented=represented, mean_active=n_active, recall_omp=hit_o.float().mean().item(), recall_oneshot=hit_1.float().mean().item(), recall_dual=hit_d.float().mean().item(), pred_from_prominence=pred, competitor_level=lvl,
                     spearman_hit_vs_importance=spearman(hit_o.float(), imp[top]), spearman_prom_vs_importance=spearman(prom, imp[top]), spearman_hit_vs_norm=spearman(hit_o.float(), norms[top])))
    log(f"n{n} d{d} s{s}: represented {represented:.2f}, active/state {n_active:.1f} | recall omp {rows[-1]['recall_omp']:.2f} oneshot {rows[-1]['recall_oneshot']:.2f} dual {rows[-1]['recall_dual']:.2f} | law pred {pred:.2f} (level {lvl:.2f}) | hit vs importance rho {rows[-1]['spearman_hit_vs_importance']:+.2f}, prominence vs importance {rows[-1]['spearman_prom_vs_importance']:+.2f}")
record("e153_toy_superposition", dict(rows=rows), " | ".join(f"n{r['n']}d{r['d']}s{r['sparsity']}: rep {r['features_represented']:.2f} omp {r['recall_omp']:.2f} os {r['recall_oneshot']:.2f} dual {r['recall_dual']:.2f} law {r['pred_from_prominence']:.2f} imp-rho {r['spearman_hit_vs_importance']:+.2f}" for r in rows))
