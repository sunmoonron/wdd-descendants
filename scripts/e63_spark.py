"""e63: uniqueness of the true support (Kruskal 1977 / spark). For each token's true top-k MLP atoms (k=3,8,16),
the condition number and smallest singular value of A_S, and the ERC of the top-8 support; the fraction of tokens
whose top-8 true support is an ERC-recoverable set. Is the ceiling ever a non-uniqueness of the true support?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 2048; ids = sub(c.NT, N); A, lab = c.dictionary(L); typ = typical_mask(c.X(L, center=False)[ids])
tb, tn, tc = c.top_writes(L, 16); rows = torch.stack([c.atom_index(L, tb[ids, j], tn[ids, j]) for j in range(16)], 1).to(DEV); res = dict(model=tag, L=L)
for k in (3, 8, 16):
    As = A[rows[:, :k]]; sv = torch.linalg.svdvals(As); res[f"k{k}"] = dict(smin_median=sv[:, -1][typ].median().item(), cond_median=(sv[:, 0] / sv[:, -1])[typ].median().item(), frac_cond_above_10=((sv[:, 0] / sv[:, -1]) > 10)[typ].float().mean().item())
S8 = rows[:, :8]; erc = torch.zeros(N, device=DEV)
for s in range(0, N, 256):
    S_ = S8[s:s + 256]; As = A[S_]; G = As @ As.transpose(1, 2) + 1e-6 * torch.eye(8, device=DEV); P = torch.linalg.solve(G, As); Cc = (P @ A.T).abs().sum(1); Cc.scatter_(1, S_, 0.0); erc[s:s + 256] = Cc.max(1).values
res["erc8"] = dict(median=erc[typ].median().item(), frac_below_1=(erc[typ] < 1).float().mean().item())
record(f"e63_spark_{tag}", res, " | ".join(f"k{k}: smin {res[f'k{k}']['smin_median']:.2f} cond {res[f'k{k}']['cond_median']:.1f} cond>10 {res[f'k{k}']['frac_cond_above_10']:.2f}" for k in (3, 8, 16)) + f" | ERC(top-8) median {res['erc8']['median']:.2f} <1: {res['erc8']['frac_below_1']:.2f}")
