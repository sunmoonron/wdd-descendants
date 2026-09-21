"""S1f (GPU): heavy reinforcement. Later writers add f=2-4x the original write along its direction (real GPT-2: raw survival 2-5) at cosine 0.5 or 0.9; the original becomes a minority owner of its direction. Does identification of the ORIGINAL writer fail (aliasing by reinforcement)?: the largest write is
accompanied by P partner writes with cosine rho to it (rho < 0 cancelling, > 0 reinforcing), total projection
f*c along the write; partners are APPENDED to the dictionary as new atoms (in_dict) or kept out (like attention).
Which structures reproduce the real regime: one-shot >= OMP, recall governed by prominence?"""
import os, json, math, time, itertools, torch
torch.set_default_device("cuda"); torch.manual_seed(0); RES = "/workspace/wdd/results"; d = 256; N = 512; k = 64; K = 64
def unit(v): return v / v.norm(dim=-1, keepdim=True)
def omp(X, A, k):
    n = X.shape[0]; r = X.clone(); S = torch.zeros(n, 0, dtype=torch.long); taken = torch.zeros(n, A.shape[0], dtype=torch.bool)
    for step in range(k):
        pick = (r @ A.T).abs().masked_fill(taken, -1).argmax(1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
        As = A[S]; G = As @ As.transpose(1, 2) + 1e-5 * torch.eye(step + 1); c = torch.cholesky_solve(As @ X[:, :, None], torch.linalg.cholesky(G)); r = X - (c.transpose(1, 2) @ As)[:, 0]
    return S
def oneshot(X, A, k, W=None): return ((X @ W.T if W is not None else X) @ A.T).abs().topk(k, dim=1).indices
rows = []; m = 64 * d
for P, rho, f, in_dict in itertools.product((1, 8, 32), (0.5, 0.9), (2.0, 4.0), (True, False)):
    A0 = unit(torch.randn(m, d)); idx = torch.stack([torch.randperm(m)[:K] for _ in range(N)]); c = torch.exp(torch.randn(N, K)) * torch.where(torch.rand(N, K) < 0.5, -1.0, 1.0)
    c[:, 0] = c.abs().max(1).values * 1.5; top = idx[:, 0]; X = torch.einsum("nk,nkd->nd", c, A0[idx]); dtop = A0[top]
    noise = torch.randn(N, P, d); noise = unit(noise - (noise * dtop[:, None]).sum(2, keepdim=True) * dtop[:, None])
    part = unit(rho * dtop[:, None] + math.sqrt(1 - rho ** 2) * noise)                          # [N,P,d]
    coef = (f * c[:, 0] / (P * rho))[:, None].expand(N, P) * (1 if rho > 0 else 1)               # projection along dtop = f*c (rho>0 reinforce) or -f*c... sign via rho
    coef = (f * c[:, 0] / (P * abs(rho)))[:, None].expand(N, P)                                  # partner coefficient positive; direction carries the sign of rho
    X = X + torch.einsum("np,npd->nd", coef, part)
    A = torch.cat([A0, part.reshape(N * P, d)]) if in_dict else A0
    S = A.T @ A; ev, V = torch.linalg.eigh(S); W = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
    Xc = X - X.mean(0); prom = (Xc * dtop).sum(1).abs() / Xc.norm(dim=1)
    so = omp(Xc, A, k); s1 = oneshot(Xc, A, k); sd = oneshot(Xc, A, k, W)
    own_partner_rows = (m + torch.arange(N)[:, None] * P + torch.arange(P)[None]) if in_dict else None
    r = dict(P=P, rho=rho, f=f, in_dict=in_dict, recall_omp=(so == top[:, None]).any(1).float().mean().item(), recall_oneshot=(s1 == top[:, None]).any(1).float().mean().item(), recall_dual=(sd == top[:, None]).any(1).float().mean().item(),
             prom_med=prom.median().item(), raw_survival=1 + f * (1 if rho > 0 else -1), m_total=A.shape[0], floor=math.sqrt(2 * math.log(A.shape[0]) / d),
             own_partners_in_omp_support=((so[:, :, None] == own_partner_rows[:, None, :]).any(1).float().mean().item() if in_dict else None))
    rows.append(r); print(f"P {P} rho {rho} in_dict {in_dict}: omp {r['recall_omp']:.2f} oneshot {r['recall_oneshot']:.2f} dual {r['recall_dual']:.2f} prom {r['prom_med']:.2f} surv {r['raw_survival']:.1f} partners-selected {r['own_partners_in_omp_support']}", flush=True)
json.dump(dict(rows=rows), open(f"{RES}/s1f_majority.json", "w"), indent=1)
with open(f"{RES}/FINDINGS.log", "a") as fh: fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + " s1f_majority: " + " | ".join(f"P{r['P']} rho{r['rho']} {'dict' if r['in_dict'] else 'nodict'}: omp {r['recall_omp']:.2f} os {r['recall_oneshot']:.2f} prom {r['prom_med']:.2f}" for r in rows) + "\n")
