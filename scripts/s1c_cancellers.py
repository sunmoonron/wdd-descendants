"""S1c (GPU): the anatomy of cancellation that defeats OMP. The largest write loses a fraction f=0.8 of its
projection to P partner writes (P in 1, 8, 32, 128) whose directions have cosine -rho (0.8 for P=1, else 0.5 or 0.2)
with the write; partners are either dictionary atoms (in_dict, like other MLP writes) or NOT atoms (like attention
writes). Which configurations reproduce the real regime (one-shot >= OMP, recall set by prominence)?"""
import os, json, math, time, itertools, torch
torch.set_default_device("cuda"); torch.manual_seed(0); RES = "/workspace/wdd/results"; d = 256; N = 512; k = 64; f = 0.8
def unit(v): return v / v.norm(dim=-1, keepdim=True)
def omp(X, A, k):
    n = X.shape[0]; r = X.clone(); S = torch.zeros(n, 0, dtype=torch.long); taken = torch.zeros(n, A.shape[0], dtype=torch.bool)
    for step in range(k):
        pick = (r @ A.T).abs().masked_fill(taken, -1).argmax(1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
        As = A[S]; G = As @ As.transpose(1, 2) + 1e-5 * torch.eye(step + 1); c = torch.cholesky_solve(As @ X[:, :, None], torch.linalg.cholesky(G)); r = X - (c.transpose(1, 2) @ As)[:, 0]
    return S
def oneshot(X, A, k, W=None): return ((X @ W.T if W is not None else X) @ A.T).abs().topk(k, dim=1).indices
rows = []; m = 64 * d
for P, rho, in_dict, K in itertools.product((1, 8, 32, 128), (0.8, 0.5, 0.2), (True, False), (16, 64)):
    if P == 1 and rho != 0.8: continue
    if P > 1 and rho == 0.8: continue
    A = unit(torch.randn(m, d)); idx = torch.stack([torch.randperm(m)[:K] for _ in range(N)]); c = torch.exp(torch.randn(N, K)) * torch.where(torch.rand(N, K) < 0.5, -1.0, 1.0)
    c[:, 0] = c.abs().max(1).values * 1.5; top = idx[:, 0]; X = torch.einsum("nk,nkd->nd", c, A[idx]); dtop = A[top]
    noise = torch.randn(N, P, d); noise = unit(noise - (noise * dtop[:, None]).sum(2, keepdim=True) * dtop[:, None])
    part_dirs = unit(-rho * dtop[:, None] + math.sqrt(1 - rho ** 2) * noise)                    # [N,P,d], cos -rho with dtop
    if in_dict:
        pidx = torch.stack([torch.randperm(m)[K:K + P] for _ in range(N)]); A[pidx] = part_dirs
    coef = (f * c[:, 0] / (P * rho))[:, None].expand(N, P)                                        # each partner removes f*c/P along dtop
    X = X + torch.einsum("np,npd->nd", coef, part_dirs)
    S = A.T @ A; ev, V = torch.linalg.eigh(S); W = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
    Xc = X - X.mean(0); prom = (Xc * dtop).sum(1).abs() / Xc.norm(dim=1)
    so = omp(Xc, A, k); s1 = oneshot(Xc, A, k); sd = oneshot(Xc, A, k, W)
    r = dict(P=P, rho=rho, in_dict=in_dict, K=K, recall_omp=(so == top[:, None]).any(1).float().mean().item(), recall_oneshot=(s1 == top[:, None]).any(1).float().mean().item(), recall_dual=(sd == top[:, None]).any(1).float().mean().item(),
             prom_med=prom.median().item(), partner_energy_over_write=((coef ** 2).sum(1) / c[:, 0] ** 2).median().item(), floor=math.sqrt(2 * math.log(m) / d))
    rows.append(r); print(f"P {P} rho {rho} in_dict {in_dict} K {K}: omp {r['recall_omp']:.2f} oneshot {r['recall_oneshot']:.2f} dual {r['recall_dual']:.2f} prom {r['prom_med']:.2f} partner-energy {r['partner_energy_over_write']:.2f}", flush=True)
json.dump(dict(rows=rows, f=f), open(f"{RES}/s1c_cancellers.json", "w"), indent=1)
with open(f"{RES}/FINDINGS.log", "a") as fh: fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + " s1c_cancellers: " + " | ".join(f"P{r['P']} rho{r['rho']} {'dict' if r['in_dict'] else 'nodict'} K{r['K']}: omp {r['recall_omp']:.2f} os {r['recall_oneshot']:.2f} prom {r['prom_med']:.2f}" for r in rows) + "\n")
