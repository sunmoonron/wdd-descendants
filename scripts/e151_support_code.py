"""e151: the support as a code. Does the set of 64 selected atoms preserve the state's predictive geometry?
kNN next-token accuracy (k=10, majority vote) on 8192 tokens using (a) cosine similarity of the centered states,
(b) Jaccard overlap of the OMP supports, (c) Jaccard of the dual-projection supports, (d) cosine of the sparse
coefficient vectors, (e) random supports of 64 atoms (null); plus the 1-NN agreement between the state and
support geometries."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 8192; ids = sub(c.NT, N); X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
tokall = c.s["eval_ids"].reshape(-1); nxt = torch.cat([tokall[1:], tokall[:1]])[ids].to(DEV); pos = (torch.arange(c.NT) % CTX)[ids].to(DEV); ok = typ & (pos < CTX - 1)
sel, cof, err = get_omp(c, L, A=A, X=c.X(L)); sel, cof = sel[ids.to(DEV)], cof[ids.to(DEV)]
S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sd, cd, _ = oneshot(X, A, 64, whiten=Winv)
def knn_acc(sim, k=10):
    sim = sim.clone(); sim.fill_diagonal_(-1e9); nn = sim.topk(k, dim=1).indices; votes = nxt[nn]
    pred = torch.mode(votes, dim=1).values; return (pred == nxt)[ok].float().mean().item(), nn[:, 0]
def jaccard(sel_):
    M = torch.zeros(N, A.shape[0], device=DEV, dtype=torch.float16); M.scatter_(1, sel_, 1.0); inter = (M @ M.T).float(); return inter / (128 - inter)
Xn = X / X.norm(dim=1, keepdim=True); sim_state = Xn @ Xn.T
acc_state, nn_state = knn_acc(sim_state); acc_omp, nn_omp = knn_acc(jaccard(sel)); acc_dual, nn_dual = knn_acc(jaccard(sd))
Cv = torch.zeros(N, A.shape[0], device=DEV, dtype=torch.float16); Cv.scatter_(1, sel, cof.half()); Cn = Cv / Cv.float().norm(dim=1, keepdim=True).half(); acc_coef, nn_coef = knn_acc((Cn @ Cn.T).float())
g = torch.Generator().manual_seed(0); rnd = torch.randint(0, A.shape[0], (N, 64), generator=g).to(DEV); acc_rand, _ = knn_acc(jaccard(rnd))
res = dict(model=tag, L=L, N=N, knn10_next_token_acc=dict(state_cosine=acc_state, omp_support_jaccard=acc_omp, dual_support_jaccard=acc_dual, omp_coef_cosine=acc_coef, random_support=acc_rand, majority_baseline=(nxt[ok] == torch.mode(nxt[ok]).values).float().mean().item()),
           nn1_agreement_with_state=dict(omp_support=(nn_omp == nn_state)[ok].float().mean().item(), dual_support=(nn_dual == nn_state)[ok].float().mean().item(), omp_coef=(nn_coef == nn_state)[ok].float().mean().item()))
record(f"e151_code_{tag}", res, f"kNN-10 next-token acc: state cosine {acc_state:.3f} | OMP support Jaccard {acc_omp:.3f} | dual support {acc_dual:.3f} | OMP coef cosine {acc_coef:.3f} | random support {acc_rand:.3f} | majority {res['knn10_next_token_acc']['majority_baseline']:.3f} || 1-NN agreement with state geometry: omp {res['nn1_agreement_with_state']['omp_support']:.2f} dual {res['nn1_agreement_with_state']['dual_support']:.2f} coef {res['nn1_agreement_with_state']['omp_coef']:.2f}")
