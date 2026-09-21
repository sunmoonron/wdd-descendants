"""e174: a toy with REDUNDANCY (the reinforcing-near-twin regime of S1f, in a trained system). Block 0 writes sparse
features x along W1; block 1 reads the stream through a ReLU MLP and writes along fixed TWIN directions of W1's
columns (cosine ~0.9 to each feature's direction), with a learned per-feature gain (so block 1 can re-write each
feature); the readout reconstructs x from the sum. Dictionary = block-0 atoms + block-1 twin atoms. Measures:
identification of the largest active feature's BLOCK-0 atom (OMP / dual) and of either its block-0 or twin atom
('write credited to the feature regardless of block'), its survival, importance by input removal, and the
readability-importance correlations, as the twin gain is encouraged (0, 0.5, 1, 2 x)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
torch.manual_seed(0); rows = []; n, d, s, m = 400, 64, 0.05, 512; imp = 0.9 ** torch.arange(n, device=DEV)
for gain_target in (0.0, 0.5, 1.0, 2.0):
    W1 = torch.nn.Parameter(torch.randn(d, n, device=DEV) * 0.1); Wr = torch.nn.Parameter(torch.randn(n, d, device=DEV) * 0.1); br = torch.nn.Parameter(torch.zeros(n, device=DEV))
    Win = torch.nn.Parameter(torch.randn(m, d, device=DEV) * 0.1); bin_ = torch.nn.Parameter(torch.zeros(m, device=DEV)); Wmid = torch.nn.Parameter(torch.randn(n, m, device=DEV) * 0.05)
    params = [W1, Wr, br, Win, bin_, Wmid]; opt = torch.optim.Adam(params, lr=2e-3)
    def twins():
        u = W1 / W1.norm(dim=0, keepdim=True).clamp_min(1e-6); noise = torch.randn_like(u); noise = noise - (noise * u).sum(0, keepdim=True) * u; noise = noise / noise.norm(dim=0, keepdim=True).clamp_min(1e-6)
        return 0.9 * u + math.sqrt(1 - 0.81) * noise                                                                     # [d, n] unit twin directions (recomputed with a fixed noise seed below)
    torch.manual_seed(1); T = twins().detach()
    def fwd(x):
        h0 = x @ W1.T; z = torch.relu(h0 @ Win.T + bin_) @ Wmid.T                                                          # per-feature block-1 coefficient
        h1 = h0 + z @ T.T; return h0, z, h1, torch.relu(h1 @ Wr.T + br)
    with torch.enable_grad():
        for step in range(4000):
            x = torch.rand(1024, n, device=DEV) * (torch.rand(1024, n, device=DEV) < s); h0, z, h1, y = fwd(x)
            loss = (imp * (y - x) ** 2).mean() + 0.1 * ((z - gain_target * (x * W1.norm(dim=0)[None])) ** 2).mean()          # encourage block 1 to re-write each feature with the target gain
            opt.zero_grad(); loss.backward(); opt.step()
    x = torch.rand(4096, n, device=DEV) * (torch.rand(4096, n, device=DEV) < s); h0, z, h1, y = fwd(x); base = (imp * (y - x) ** 2).mean(1).detach()
    norms1 = W1.detach().norm(dim=0); A0 = (W1.detach() / norms1.clamp_min(1e-6)).T; A1 = T.T; A = torch.cat([A0, A1]); C0 = x * norms1[None]; active = (C0.abs() > 0).any(1); top = C0.abs().argmax(1)
    hc = h1.detach() - h1.detach().mean(0); prom = (hc * A0[top]).sum(1).abs() / hc.norm(dim=1).clamp_min(1e-9); k = 16
    so, _, _ = omp(hc, A, k); S_ = A.T @ A; ev, V = torch.linalg.eigh(S_); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sd, _, _ = oneshot(hc, A, k, whiten=Winv)
    hit_o, hit_d = (so == top[:, None]).any(1), (sd == top[:, None]).any(1); hit_twin_o = (so == (top + n)[:, None]).any(1); hit_either = hit_o | hit_twin_o
    x2 = x.clone(); x2[torch.arange(len(x)), top] = 0; _, _, _, y2 = fwd(x2); imp_input = ((imp * (y2 - x) ** 2).mean(1) - base).detach(); surv = ((h1.detach() * A0[top]).sum(1) / C0.gather(1, top[:, None])[:, 0].clamp_min(1e-9))
    def spearman(a, b):
        ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    a_ = active; zg = (z.detach().gather(1, top[:, None])[:, 0] / C0.gather(1, top[:, None])[:, 0].clamp_min(1e-9))
    rows.append(dict(gain_target=gain_target, realized_gain_med=zg[a_].median().item(), recall_block0_omp=hit_o[a_].float().mean().item(), recall_block0_dual=hit_d[a_].float().mean().item(), recall_twin_omp=hit_twin_o[a_].float().mean().item(), recall_either_omp=hit_either[a_].float().mean().item(), survival_med=surv[a_].median().item(), prom_med=prom[a_].median().item(),
                     rho_hit0_vs_importance=spearman(hit_o[a_].float(), imp_input[a_]), rho_either_vs_importance=spearman(hit_either[a_].float(), imp_input[a_]), rho_prom_vs_importance=spearman(prom[a_], imp_input[a_]), importance_med=imp_input[a_].median().item()))
    log(f"gain {gain_target} (realized {rows[-1]['realized_gain_med']:.2f}): block-0 atom recall omp {rows[-1]['recall_block0_omp']:.2f} dual {rows[-1]['recall_block0_dual']:.2f}; twin {rows[-1]['recall_twin_omp']:.2f}; either {rows[-1]['recall_either_omp']:.2f} | survival {rows[-1]['survival_med']:.2f} prominence {rows[-1]['prom_med']:.2f} | rho(block-0 hit, importance) {rows[-1]['rho_hit0_vs_importance']:+.2f}, (either, importance) {rows[-1]['rho_either_vs_importance']:+.2f}, (prominence, importance) {rows[-1]['rho_prom_vs_importance']:+.2f}")
record("e174_toy_redundancy", dict(rows=rows), " | ".join(f"gain {r['gain_target']} (real {r['realized_gain_med']:.2f}): b0 {r['recall_block0_omp']:.2f}/{r['recall_block0_dual']:.2f} twin {r['recall_twin_omp']:.2f} either {r['recall_either_omp']:.2f} surv {r['survival_med']:.2f} rho(b0,imp) {r['rho_hit0_vs_importance']:+.2f} rho(prom,imp) {r['rho_prom_vs_importance']:+.2f}" for r in rows))
