"""S1b (GPU): diffuse vs concentrated cancellation. The largest write loses a fraction f of its projection either
to ONE anti-aligned partner atom (concentrated) or to a sum of 32 small writes on random atoms projected to remove
the same amount along d (diffuse). Also the real-model regime check: recall vs prominence bins for both, so the
synthetic law can be overlaid on e56/e57."""
import os, json, math, time, itertools, torch
torch.set_default_device("cuda"); torch.manual_seed(0); RES = "/workspace/wdd/results"; d = 256; N = 512; k = 64
def unit(v): return v / v.norm(dim=-1, keepdim=True)
def omp(X, A, k):
    n = X.shape[0]; r = X.clone(); S = torch.zeros(n, 0, dtype=torch.long); taken = torch.zeros(n, A.shape[0], dtype=torch.bool)
    for step in range(k):
        pick = (r @ A.T).abs().masked_fill(taken, -1).argmax(1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
        As = A[S]; G = As @ As.transpose(1, 2) + 1e-5 * torch.eye(step + 1); c = torch.cholesky_solve(As @ X[:, :, None], torch.linalg.cholesky(G)); r = X - (c.transpose(1, 2) @ As)[:, 0]
    return S
def oneshot(X, A, k, W=None): return ((X @ W.T if W is not None else X) @ A.T).abs().topk(k, dim=1).indices
rows = []
for m_over_d, K, f, mode in itertools.product((32, 128), (16, 64), (0.5, 0.8, 0.95, 1.1), ("concentrated", "diffuse")):
    m = m_over_d * d; A = unit(torch.randn(m, d)); S = A.T @ A; ev, V = torch.linalg.eigh(S); W = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
    idx = torch.stack([torch.randperm(m)[:K] for _ in range(N)]); c = torch.exp(torch.randn(N, K)) * torch.where(torch.rand(N, K) < 0.5, -1.0, 1.0)
    c[:, 0] = c.abs().max(1).values * 1.5; top = idx[:, 0]; X = torch.einsum("nk,nkd->nd", c, A[idx]); dtop = A[top]
    if mode == "concentrated":
        part = torch.randint(0, m, (N,)); A[part] = unit(-0.8 * dtop + 0.6 * unit(torch.randn(N, d) - (torch.randn(N, d) * dtop).sum(1, keepdim=True) * dtop))
        X = X + (f * c[:, 0] / 0.8)[:, None] * A[part]                                   # removes f*c along dtop
    else:
        others = torch.stack([torch.randperm(m)[:32] for _ in range(N)]); B = A[others]     # [N,32,d] random atoms, small writes
        proj = (B * dtop[:, None]).sum(2)                                                  # [N,32] cosines with dtop
        w = torch.randn(N, 32); w = w * torch.sign(proj) * -1                               # each small write opposes dtop
        scale = (f * c[:, 0]) / (w * proj).sum(1).abs().clamp_min(1e-6); w = -w * scale[:, None] * torch.sign((w * proj).sum(1))[:, None] * -1
        w = w * (-(f * c[:, 0]) / (w * proj).sum(1))[:, None]                                # exact: total projection = -f*c
        X = X + torch.einsum("nk,nkd->nd", w, B)
    Xc = X - X.mean(0); prom = (Xc * dtop).sum(1).abs() / Xc.norm(dim=1)
    so = omp(Xc, A, k); s1 = oneshot(Xc, A, k); sd = oneshot(Xc, A, k, W); h = (so == top[:, None]).any(1); h1 = (s1 == top[:, None]).any(1)
    r = dict(m_over_d=m_over_d, K=K, f=f, mode=mode, recall_omp=h.float().mean().item(), recall_oneshot=h1.float().mean().item(), recall_dual=(sd == top[:, None]).any(1).float().mean().item(),
             prom_med=prom.median().item(), gauss_floor=math.sqrt(2 * math.log(m) / d), raw_survival=1 - f,
             bins={f"{lo:.2f}": (h1[(prom >= lo) & (prom < lo + 0.05)].float().mean().item() if ((prom >= lo) & (prom < lo + 0.05)).sum() > 10 else None) for lo in [i * 0.05 for i in range(12)]})
    rows.append(r); print(f"m/d {m_over_d} K {K} f {f} {mode}: omp {r['recall_omp']:.2f} oneshot {r['recall_oneshot']:.2f} dual {r['recall_dual']:.2f} prom {r['prom_med']:.2f} floor {r['gauss_floor']:.2f}", flush=True)
json.dump(dict(rows=rows), open(f"{RES}/s1b_diffuse.json", "w"), indent=1)
with open(f"{RES}/FINDINGS.log", "a") as fh: fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + " s1b_diffuse: " + " | ".join(f"m{r['m_over_d']} K{r['K']} f{r['f']} {r['mode'][:4]}: omp {r['recall_omp']:.2f} os {r['recall_oneshot']:.2f} prom {r['prom_med']:.2f}" for r in rows) + "\n")
