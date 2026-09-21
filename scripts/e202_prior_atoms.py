"""e202: what are the geometric-prior atoms? The 100 atoms most often selected on covariance-matched noise (dual and
OMP): their type mix, their block mix, their alignment with the top principal directions of the state covariance,
and whether the corresponding neurons are real workhorses (activation rate and dominance rate in real data) or
rarely used. Also: fraction of the real-state support that falls on these prior atoms, and their precision."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); mu = c.s["mu"][L + 1].to(DEV); X = (Xraw - mu); A, lab = c.dictionary(L); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV); NA = A.shape[0]
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * C.abs().max(1, keepdim=True).values; dom_key = C.abs().argmax(1)
torch.manual_seed(0); Xtr = c.X(L)[sub(c.NT, 8192, seed=5)]; Xtr = Xtr[typical_mask(Xtr + mu)]; Sig = Xtr.T @ Xtr / len(Xtr); ev, V = torch.linalg.eigh(Sig); Ch = V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T; PC = V[:, -8:].T
G = torch.randn(4096, c.D, device=DEV) @ Ch; G = G * (X[typ].norm(dim=1).median() / G.norm(dim=1, keepdim=True))
S = A.T @ A; ev2, V2 = torch.linalg.eigh(S); Winv = V2 @ torch.diag(1 / (ev2 + 1e-2 * ev2[-1])) @ V2.T
# neuron usage in real data: activation rate (|ledger| >= 5% of token max) and dominance rate per dictionary atom
use = torch.zeros(NA, device=DEV); domc = torch.zeros(NA, device=DEV); ismlp_all = typA == T_MLP; key_all = torch.where(ismlp_all, blkA * c.DFF + idxA, torch.zeros_like(blkA))
act_rate = (C.abs() >= thr).float().mean(0); dom_rate = torch.bincount(dom_key, minlength=C.shape[1]).float() / len(C); use[ismlp_all] = act_rate[key_all[ismlp_all]]; domc[ismlp_all] = dom_rate[key_all[ismlp_all]]
out = {}
for dec in ("omp", "dual"):
    solve = (lambda Z: omp(Z, A, 64)[0]) if dec == "omp" else (lambda Z: oneshot(Z, A, 64, whiten=Winv)[0])
    fn = torch.bincount(solve(G).reshape(-1), minlength=NA).float() / len(G); top = fn.topk(100).indices; sel = solve(X); onprior = torch.isin(sel, top)
    ismlp = typA[sel] == T_MLP; key = torch.where(ismlp, blkA[sel] * c.DFF + idxA[sel], torch.zeros_like(sel)); real = ismlp & (C.gather(1, key).abs() >= thr)
    pc_cos = (A[top] @ PC.T).abs().max(1).values; rnd = torch.randperm(NA, device=DEV)[:100]; pc_cos_rand = (A[rnd] @ PC.T).abs().max(1).values
    rec = dict(type_mix={int(t): round((typA[top] == t).float().mean().item(), 2) for t in typA[top].unique()}, block_mix={int(b): round((blkA[top] == b).float().mean().item(), 2) for b in blkA[top].unique()}, pc_cos_med=pc_cos.median().item(), pc_cos_rand_med=pc_cos_rand.median().item(), frac_pc_cos_above_0p3=(pc_cos > 0.3).float().mean().item(), act_rate_prior=use[top][typA[top] == T_MLP].median().item() if (typA[top] == T_MLP).any() else float("nan"), act_rate_all=use[ismlp_all].median().item(), dom_rate_prior=domc[top][typA[top] == T_MLP].mean().item() if (typA[top] == T_MLP).any() else float("nan"), dom_rate_all=domc[ismlp_all].mean().item(), share_of_real_support=onprior[typ].float().mean().item(), precision_on_prior=(real & onprior)[typ].float().sum().item() / onprior[typ].float().sum().clamp_min(1).item(), precision_off_prior=(real & ~onprior)[typ].float().sum().item() / (~onprior)[typ].float().sum().clamp_min(1).item(), null_mass_top100=fn.topk(100).values.sum().item() / fn.sum().item())
    out[dec] = rec
    log(f"{tag} {dec}: top-100 null atoms hold {rec['null_mass_top100']:.2f} of null selections; types {rec['type_mix']} blocks {rec['block_mix']} | max |cos| with top-8 PCs: {rec['pc_cos_med']:.2f} (random atoms {rec['pc_cos_rand_med']:.2f}), >0.3 in {rec['frac_pc_cos_above_0p3']:.2f} | neuron activation rate: prior {rec['act_rate_prior']:.3f} vs all {rec['act_rate_all']:.3f}; dominance rate prior {rec['dom_rate_prior']:.4f} vs all {rec['dom_rate_all']:.4f} | they take {rec['share_of_real_support']:.2f} of the real support with precision {rec['precision_on_prior']:.2f} vs {rec['precision_off_prior']:.2f} off-prior")
record(f"e202_prioratoms_{tag}", dict(model=tag, L=L, results=out), " | ".join(f"{dec}: prior atoms max|cos| with top PCs {r['pc_cos_med']:.2f} (random {r['pc_cos_rand_med']:.2f}); activation rate {r['act_rate_prior']:.3f} vs all {r['act_rate_all']:.3f}; dominance {r['dom_rate_prior']:.4f} vs {r['dom_rate_all']:.4f}; share of real support {r['share_of_real_support']:.2f}, precision on-prior {r['precision_on_prior']:.2f} vs off {r['precision_off_prior']:.2f}; types {r['type_mix']}" for dec, r in out.items()))
