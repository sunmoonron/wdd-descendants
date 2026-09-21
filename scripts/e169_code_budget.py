"""e169: how small a code preserves the state's predictive geometry? kNN-10 next-token accuracy from the Jaccard
overlap of OMP supports truncated to the first k picks (k = 4, 8, 16, 32, 64) and from the union of per-block
increment supports (8 per block), vs state cosine. (gpt2 and smollm2.)"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 8192; ids = sub(c.NT, N); X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
tokall = c.s["eval_ids"].reshape(-1); nxt = torch.cat([tokall[1:], tokall[:1]])[ids].to(DEV); pos = (torch.arange(c.NT) % CTX)[ids].to(DEV); ok = typ & (pos < CTX - 1)
sel, cof, err = get_omp(c, L, A=A, X=c.X(L)); sel = sel[ids.to(DEV)]
def knn_acc(sim, k=10):
    sim = sim.clone(); sim.fill_diagonal_(-1e9); nn = sim.topk(k, dim=1).indices; pred = torch.mode(nxt[nn], dim=1).values; return (pred == nxt)[ok].float().mean().item()
def jacc(sel_):
    M = torch.zeros(N, A.shape[0], device=DEV, dtype=torch.float16); M.scatter_(1, sel_, 1.0); inter = (M @ M.T).float(); sz = M.float().sum(1); return inter / (sz[:, None] + sz[None, :] - inter).clamp_min(1)
Xn = X / X.norm(dim=1, keepdim=True); res = dict(model=tag, L=L, state_cosine=knn_acc(Xn @ Xn.T), by_k={})
for k in (4, 8, 16, 32, 64): res["by_k"][k] = knn_acc(jacc(sel[:, :k]))
parts = []; typA, blkA = lab["type"].to(DEV), lab["block"].to(DEV)
for b in range(L + 1):
    D_ = c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV); Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); rows_b = torch.nonzero((blkA == b) & (typA >= T_MLP))[:, 0]; sb, _, _ = omp(D_, Ab, 8); parts.append(rows_b[sb])
res["increment_union_8_per_block"] = knn_acc(jacc(torch.cat(parts, 1)))
record(f"e169_codebudget_{tag}", res, f"state cosine {res['state_cosine']:.3f} | support Jaccard by k: " + " ".join(f"k{k}:{v:.3f}" for k, v in res["by_k"].items()) + f" | increment-union code {res['increment_union_8_per_block']:.3f}")
