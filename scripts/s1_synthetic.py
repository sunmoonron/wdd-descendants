"""S1 (GPU, foundational track): is WDD identification just generic sparse recovery? Synthetic states x = sum of K
'writes' c_i a_i over a random unit dictionary (m atoms in d dims), heavy-tailed |c| (lognormal sigma), random or
positive signs, a controllable fraction of cancelling pairs (the largest write gets an anti-aligned partner with
cosine -rho and coefficient -alpha c), a DC offset, and clustered atoms to raise coherence. Measures top-1 recall
of the largest write under OMP(k=64), one-shot, dual one-shot, and records prominence, so the recall-vs-prominence
law can be compared with the real models (e56). Sweeps m/d, K, cancellation, coherence."""
import sys, os, json, math, time, itertools
import torch
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(0)
RES = "/workspace/wdd/results"; d = 256; N = 512; k = 64
def unit(v): return v / v.norm(dim=-1, keepdim=True)
def make_dict(m, kappa):
    A = torch.randn(m, d)
    if kappa > 0:                       # clustered atoms: 64 centers, each atom = center + noise
        C = torch.randn(64, d); A = A + kappa * C[torch.randint(0, 64, (m,))]
    return unit(A)
def omp(X, A, k):
    n = X.shape[0]; r = X.clone(); S = torch.zeros(n, 0, dtype=torch.long); taken = torch.zeros(n, A.shape[0], dtype=torch.bool)
    for step in range(k):
        pick = (r @ A.T).abs().masked_fill(taken, -1).argmax(1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
        As = A[S]; G = As @ As.transpose(1, 2) + 1e-5 * torch.eye(step + 1); c = torch.cholesky_solve(As @ X[:, :, None], torch.linalg.cholesky(G)); r = X - (c.transpose(1, 2) @ As)[:, 0]
    return S
def oneshot(X, A, k, W=None):
    sc = (X @ W.T if W is not None else X) @ A.T; return sc.abs().topk(k, dim=1).indices
rows = []
for m_over_d, K, sigma, cancel, kappa, positive in itertools.product((8, 32, 128), (4, 16, 64, 256), (1.0,), (0.0, 0.5, 1.0), (0.0, 1.0), (False, True)):
    m = m_over_d * d; A = make_dict(m, kappa); S = A.T @ A; ev, V = torch.linalg.eigh(S); W = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
    idx = torch.stack([torch.randperm(m)[:K] for _ in range(N)]); c = torch.exp(sigma * torch.randn(N, K)); 
    if not positive: c = c * torch.where(torch.rand(N, K) < 0.5, -1.0, 1.0)
    c[:, 0] = c[:, 0].abs().clamp_min(1) * c.abs().max(1).values / c[:, 0].abs().clamp_min(1e-6)          # make slot 0 the largest write
    X = torch.einsum("nk,nkd->nd", c, A[idx]); top = idx[:, 0]
    if cancel > 0:                                                                                          # anti-aligned partner atom exists in the dictionary and writes -alpha*c
        part = torch.randint(0, m, (N,)); rho = 0.8
        A[part] = unit(-rho * A[top] + math.sqrt(1 - rho ** 2) * unit(torch.randn(N, d) - (torch.randn(N, d) * A[top]).sum(1, keepdim=True) * A[top]))
        X = X + (cancel * c[:, 0])[:, None] * A[part] * (-1) * (-1)  # partner write: coefficient +cancel*c along a direction with cos -0.8 to the top atom => cancels 0.8*cancel of it
    mu = torch.randn(d); mu = mu / mu.norm() * X.norm(dim=1).median() * 0.5; Xc = (X + mu) - (X + mu).mean(0)
    prom = (Xc * A[top]).sum(1).abs() / Xc.norm(dim=1); t0 = time.time()
    so = omp(Xc, A, k); s1 = oneshot(Xc, A, k); sd = oneshot(Xc, A, k, W)
    r = dict(m_over_d=m_over_d, K=K, cancel=cancel, kappa=kappa, positive=positive, recall_omp=(so == top[:, None]).any(1).float().mean().item(), recall_oneshot=(s1 == top[:, None]).any(1).float().mean().item(),
             recall_dual=(sd == top[:, None]).any(1).float().mean().item(), prom_med=prom.median().item(), coherence_med=None, time=time.time() - t0,
             prom_bins={f"{lo:.2f}": ((so == top[:, None]).any(1)[(prom >= lo) & (prom < lo + 0.1)].float().mean().item() if ((prom >= lo) & (prom < lo + 0.1)).sum() > 10 else None) for lo in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]})
    rows.append(r); print(f"m/d {m_over_d} K {K} cancel {cancel} kappa {kappa} pos {positive}: omp {r['recall_omp']:.2f} oneshot {r['recall_oneshot']:.2f} dual {r['recall_dual']:.2f} prom {r['prom_med']:.2f} ({r['time']:.0f}s)", flush=True)
    json.dump(dict(rows=rows, d=d, N=N, k=k), open(f"{RES}/s1_synthetic.json", "w"), indent=1)
with open(f"{RES}/FINDINGS.log", "a") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " s1_synthetic: done, " + str(len(rows)) + " cells\n")
