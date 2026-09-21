"""S2 v2 (GPU; coupled writes get APPENDED atoms owned by their block): two constructions.
(A) Layered synthetic stream: L blocks, each with its own atoms and K writes per block; with probability p a block's
write is placed along a NEW atom that is near-collinear (cos 0.9) with an earlier block's write direction, with
sign + (reinforce) or - (cancel). Identification of each block's dominant write from the FINAL state (all atoms)
vs from the block's own INCREMENT (block atoms). When does state-level reading fail while increment reading holds?
(B) Non-uniqueness counterexample: 1000 dense writers whose sum lies exactly in the span of 10 other atoms. OMP
reconstructs perfectly with the 10 non-writers: FVU ~ 0, real share 0. Provenance cannot be read off a state."""
import os, json, math, time, itertools, torch
torch.set_default_device("cuda"); torch.manual_seed(1); RES = "/workspace/wdd/results"
d = 256; N = 512
def unit(v): return v / v.norm(dim=-1, keepdim=True)
def omp(X, A, k):
    n = X.shape[0]; r = X.clone(); S = torch.zeros(n, 0, dtype=torch.long); taken = torch.zeros(n, A.shape[0], dtype=torch.bool)
    for step in range(k):
        pick = (r @ A.T).abs().masked_fill(taken, -1).argmax(1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
        As = A[S]; G = As @ As.transpose(1, 2) + 1e-5 * torch.eye(step + 1); c = torch.cholesky_solve(As @ X[:, :, None], torch.linalg.cholesky(G)); r = X - (c.transpose(1, 2) @ As)[:, 0]
    return S, c[:, :, 0], (r ** 2).sum(1)
out = dict(A=[], B={})
L, mb, K = 8, 4 * d, 16
for p, sign_mode in itertools.product((0.0, 0.25, 0.5, 0.75, 1.0), ("reinforce", "cancel", "mixed")):
    A = unit(torch.randn(L * mb, d)); blk = torch.arange(L * mb) // mb
    X = torch.zeros(N, d); inc = torch.zeros(L, N, d); dom = torch.zeros(L, N, dtype=torch.long); prev_dirs = []   # per block: (atom idx, coef) of writes
    written = [[] for _ in range(N)]
    for b in range(L):
        idx = torch.randint(b * mb, (b + 1) * mb, (N, K)); c = torch.exp(torch.randn(N, K)) * torch.where(torch.rand(N, K) < 0.5, -1.0, 1.0)
        c[:, 0] = c.abs().max(1).values * 1.5                                                   # slot 0 = the block's dominant write
        if b > 0 and p > 0:
            m = torch.rand(N) < p                                                              # dominant write becomes a copy-direction write of an earlier dominant write
            for i in torch.nonzero(m)[:, 0].tolist():
                eb = torch.randint(0, b, (1,)).item(); ea = dom[eb, i]                          # earlier dominant atom
                a_new = unit(0.9 * A[ea] + math.sqrt(1 - 0.81) * unit(torch.randn(d) - (torch.randn(d) @ A[ea]) * A[ea]))
                A = torch.cat([A, a_new[None]]); blk = torch.cat([blk, torch.tensor([b])]); idx[i, 0] = A.shape[0] - 1   # appended atom, owned by block b
                s = 1.0 if sign_mode == "reinforce" else (-1.0 if sign_mode == "cancel" else (1.0 if torch.rand(1) < 0.5 else -1.0))
                c[i, 0] = s * c[i, 0].abs()
        w = torch.einsum("nk,nkd->nd", c, A[idx]); inc[b] = w; X = X + w; dom[b] = idx[:, 0]
    Xc = X - X.mean(0); S_state, _, _ = omp(Xc, A, 64)
    rows = []
    for b in range(L):
        rows_b = torch.nonzero(blk == b)[:, 0]; Ab = A[rows_b]; S_inc, _, _ = omp(inc[b], Ab, 8)
        rec_state = (S_state == dom[b][:, None]).any(1).float().mean().item(); rec_inc = (rows_b[S_inc] == dom[b][:, None]).any(1).float().mean().item()
        rows.append((b, rec_state, rec_inc))
    r = dict(p=p, sign_mode=sign_mode, recall_state_by_block=[x[1] for x in rows], recall_increment_by_block=[x[2] for x in rows], recall_state_mean=sum(x[1] for x in rows) / L, recall_increment_mean=sum(x[2] for x in rows) / L)
    out["A"].append(r); print(f"p {p} {sign_mode}: state {r['recall_state_mean']:.2f} increment {r['recall_increment_mean']:.2f} | by block state " + " ".join(f"{x[1]:.2f}" for x in rows), flush=True)
# (B) non-uniqueness: 1000 dense writers whose sum lies in the span of 10 other atoms
A = unit(torch.randn(32 * d, d)); T = torch.arange(10); Dw = torch.arange(10, 1010); fvu, real, hitT = [], [], []
for i in range(64):
    z = torch.randn(10); target = z @ A[T]; sol = torch.linalg.lstsq(A[Dw].T, target[:, None]).solution[:, 0]   # 1000 coefficients, exact (underdetermined)
    x = sol @ A[Dw]; assert (x - target).norm() / target.norm() < 1e-4
    S, c, e = omp((x - 0)[None], A, 64); fvu.append((e / (x ** 2).sum()).item()); real.append(torch.isin(S[0], Dw).float().mean().item()); hitT.append(torch.isin(T, S[0]).float().mean().item())
out["B"] = dict(fvu64=sum(fvu) / len(fvu), support_real_share=sum(real) / len(real), frac_of_the_10_nonwriters_selected=sum(hitT) / len(hitT), dense_coef_l1_over_sparse=None)
print("B:", out["B"], flush=True)
json.dump(out, open(f"{RES}/s2_layered.json", "w"), indent=1)
with open(f"{RES}/FINDINGS.log", "a") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" s2_layered: A: " + " | ".join(f"p{r['p']} {r['sign_mode']}: state {r['recall_state_mean']:.2f} inc {r['recall_increment_mean']:.2f}" for r in out["A"]) + f" || B: fvu {out['B']['fvu64']:.3f} real share {out['B']['support_real_share']:.2f} nonwriters selected {out['B']['frac_of_the_10_nonwriters_selected']:.2f}\n")
