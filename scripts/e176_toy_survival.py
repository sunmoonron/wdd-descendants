"""e176: the many-block toy as a testbed for the survival curve and the aliasing mechanism. B=6 residual blocks
(width 256). At the final stream: (a) identification of the largest active feature's block-0 atom at the WRITE
level (trivial), the block-0 INCREMENT level (h0 over block-0 atoms), and the STATE level (dual@16); (b) when the
state-level reading misses, which block's atom takes the direction (the alias's block and its cosine with the
feature direction); (c) importance by input removal vs readability at the increment level (where readability is
high) to test whether readability and importance decouple at the level where reading works."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
torch.manual_seed(0); n, d, s, m, B = 400, 64, 0.05, 256, 6; imp = 0.9 ** torch.arange(n, device=DEV)
W1 = torch.nn.Parameter(torch.randn(d, n, device=DEV) * 0.1); Wr = torch.nn.Parameter(torch.randn(n, d, device=DEV) * 0.1); br = torch.nn.Parameter(torch.zeros(n, device=DEV)); params = [W1, Wr, br]; blocks = []
for b in range(B):
    Win = torch.nn.Parameter(torch.randn(m, d, device=DEV) * 0.1); bi = torch.nn.Parameter(torch.zeros(m, device=DEV)); Wout = torch.nn.Parameter(torch.randn(m, d, device=DEV) * (0.1 / math.sqrt(B))); blocks.append((Win, bi, Wout)); params += [Win, bi, Wout]
opt = torch.optim.Adam(params, lr=2e-3)
def fwd(x):
    h = x @ W1.T; hs = [h]; ws = []
    for Win, bi, Wout in blocks:
        w = torch.relu(h @ Win.T + bi) @ Wout; h = h + w; hs.append(h); ws.append(w)
    return hs, ws, torch.relu(h @ Wr.T + br)
with torch.enable_grad():
    for step in range(4000):
        x = torch.rand(1024, n, device=DEV) * (torch.rand(1024, n, device=DEV) < s); _, _, y = fwd(x); loss = (imp * (y - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
x = torch.rand(4096, n, device=DEV) * (torch.rand(4096, n, device=DEV) < s); hs, ws, y = fwd(x); base = (imp * (y - x) ** 2).mean(1).detach(); hL = hs[-1].detach(); h0 = hs[0].detach()
norms1 = W1.detach().norm(dim=0); A0 = (W1.detach() / norms1.clamp_min(1e-6)).T; atoms = [A0]; blk_of = [torch.zeros(n, dtype=torch.long, device=DEV)]
for b, (Win, bi, Wout) in enumerate(blocks): atoms.append(Wout.detach() / Wout.detach().norm(dim=1, keepdim=True).clamp_min(1e-6)); blk_of.append(torch.full((m,), b + 1, dtype=torch.long, device=DEV))
A = torch.cat(atoms); blkA = torch.cat(blk_of); C0 = x * norms1[None]; active = (C0.abs() > 0).any(1); top = C0.abs().argmax(1); dtop = A0[top]
def dual(Xc, Ad, k):
    S_ = Ad.T @ Ad; ev, V = torch.linalg.eigh(S_); Wi = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; return oneshot(Xc, Ad, k, whiten=Wi)[0]
k = 16; hc = hL - hL.mean(0); h0c = h0 - h0.mean(0)
s_state = dual(hc, A, k); s_inc = dual(h0c, A0, k); s_state_omp, _, _ = omp(hc, A, k)
hit_state, hit_inc, hit_state_omp = (s_state == top[:, None]).any(1), (s_inc == top[:, None]).any(1), (s_state_omp == top[:, None]).any(1)
# aliasing: for misses, the atom with the largest |cos| to the feature direction among the state support
sup_dirs = A[s_state]; cosv = torch.einsum("nkd,nd->nk", sup_dirs, dtop).abs(); best = cosv.argmax(1); alias_blk = blkA[s_state.gather(1, best[:, None])[:, 0]]; alias_cos = cosv.max(1).values
miss = active & ~hit_state; x2 = x.clone(); x2[torch.arange(len(x)), top] = 0; _, _, y2 = fwd(x2); imp_input = ((imp * (y2 - x) ** 2).mean(1) - base).detach()
surv = (hL * dtop).sum(1) / C0.gather(1, top[:, None])[:, 0].clamp_min(1e-9)
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
a_ = active; res = dict(B=B, survival_curve=dict(write=1.0, increment_dual=hit_inc[a_].float().mean().item(), state_dual=hit_state[a_].float().mean().item(), state_omp=hit_state_omp[a_].float().mean().item()), survival_med=surv[a_].median().item(),
                        alias=dict(cos_med=alias_cos[miss].median().item(), block_hist={int(b): (alias_blk[miss] == b).float().mean().item() for b in range(B + 1)}, frac_alias_cos_above_0p5=(alias_cos[miss] > 0.5).float().mean().item()),
                        rho_increment_hit_vs_importance=spearman(hit_inc[a_].float(), imp_input[a_]), rho_surv_vs_importance=spearman(surv[a_], imp_input[a_]), rho_feature_weight_vs_importance=spearman(imp[top[a_]], imp_input[a_]))
record("e176_toy_survival", res, f"B={B}: survival curve write 1.00 -> increment {res['survival_curve']['increment_dual']:.2f} -> state dual {res['survival_curve']['state_dual']:.2f} (omp {res['survival_curve']['state_omp']:.2f}); write survival med {res['survival_med']:.2f} | misses: the support atom closest to the feature direction has cos {res['alias']['cos_med']:.2f} (>0.5 in {res['alias']['frac_alias_cos_above_0p5']:.2f}) and comes from block " + " ".join(f"b{b}:{v:.2f}" for b, v in res['alias']['block_hist'].items()) + f" | rho(increment readability, importance) {res['rho_increment_hit_vs_importance']:+.2f}; rho(survival, importance) {res['rho_surv_vs_importance']:+.2f}; rho(feature weight, importance) {res['rho_feature_weight_vs_importance']:+.2f}")
