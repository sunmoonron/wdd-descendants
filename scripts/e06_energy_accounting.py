"""e06: exact accounting. The state is exactly emb + sum of attention writes + sum of MLP writes (+ biases).
Per level: energy shares, the self-cancellation index ||sum||^2 / sum ||.||^2, block-pair write cosines,
token-embedding survival, and the ORACLE sparse approximation: how well do the k largest TRUE writes (MLP
neurons and attention heads, by norm) explain the centered state, vs what OMP explains with k atoms it chooses?
The gap between OMP's FVU and the oracle's is what OMP 'invents'."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
tok_ids = c.s["eval_ids"].reshape(-1)[ids]; E = c.d["emb"][0].to(DEV); etok = E[tok_ids.to(DEV)]
rows = []; Lm = mid(c)
# per-block component writes on the subsample (attention per head, MLP per block)
def head_writes(b):
    HI = c.s["HI"][b][ids].float().to(DEV); WO = c.d["WO"][b].to(DEV)
    return torch.stack([HI[:, h * c.HD:(h + 1) * c.HD] @ WO[h * c.HD:(h + 1) * c.HD] for h in range(c.NH)], 1)   # [N, NH, D]
def mlp_write(b):
    return c.acts[b][ids].float().to(DEV) @ c.wdir_cpu(b).to(DEV)
for L in range(c.NB):
    Xraw = c.X(L, center=False)[ids]; X = c.X(L)[ids]; typ = typical_mask(Xraw); mu = c.s["mu"][L + 1].to(DEV)
    att = [c.s["ATT"][b][ids].float().to(DEV) for b in range(L + 1)]; mlp = [mlp_write(b) for b in range(L + 1)]
    mb = [c.d["mlp_bias"][b] for b in range(L + 1)]
    for b in range(L + 1):
        if mb[b] is not None: mlp[b] = mlp[b] + mb[b].to(DEV)
    e0 = c.s["H"][0][ids].float().to(DEV)
    tot = e0 + sum(att) + sum(mlp); chk = ((tot - Xraw).norm() / Xraw.norm()).item()
    comps = [e0] + att + mlp; ss = sum((v ** 2).sum(1) for v in comps)
    x2 = (Xraw ** 2).sum(1)
    cancel = (x2[typ].sum() / ss[typ].sum()).item()
    e_att = sum((v ** 2).sum(1) for v in att); e_mlp = sum((v ** 2).sum(1) for v in mlp); e_emb = (e0 ** 2).sum(1)
    cos_x = lambda v: ((v * Xraw).sum(1) / (v.norm(dim=1) * Xraw.norm(dim=1)).clamp_min(1e-6))[typ].mean().item()
    blockw = [att[b] + mlp[b] for b in range(L + 1)]
    M = torch.zeros(L + 1, L + 1)
    for i in range(L + 1):
        for j in range(L + 1):
            M[i, j] = ((blockw[i] * blockw[j]).sum(1) / (blockw[i].norm(dim=1) * blockw[j].norm(dim=1)).clamp_min(1e-6))[typ].mean().item()
    emb_surv = ((Xraw * etok).sum(1) / (etok ** 2).sum(1))[typ]
    r = dict(L=L, ledger_check=chk, cancel_index=cancel, share_emb=(e_emb[typ].sum() / ss[typ].sum()).item(), share_att=(e_att[typ].sum() / ss[typ].sum()).item(),
             share_mlp=(e_mlp[typ].sum() / ss[typ].sum()).item(), cos_emb_x=cos_x(e0), cos_att_x=[cos_x(v) for v in att], cos_mlp_x=[cos_x(v) for v in mlp],
             block_cos=M.tolist(), emb_survival_med=emb_surv.median().item(), emb_survival_q10=emb_surv.quantile(0.1).item(), emb_survival_q90=emb_surv.quantile(0.9).item(),
             state_norm_med=Xraw.norm(dim=1)[typ].median().item(), sum_of_norms_med=sum(v.norm(dim=1) for v in comps)[typ].median().item())
    if L == Lm:
        # oracle sparse approximation with the true writes (unit atoms = write directions, true coefficients = norms)
        hw = [head_writes(b) for b in range(L + 1)]                                   # per block [N, NH, D]
        led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV)  # [N, (L+1)*DFF]
        hn = torch.cat([hw[b].norm(dim=2) for b in range(L + 1)], 1)                      # [N, (L+1)*NH]
        allmag = torch.cat([C.abs(), hn], 1)
        oracle = {}
        for k in (8, 16, 32, 64):
            top = allmag.topk(k, dim=1).indices
            # reconstruct: sum of the selected true writes
            rec = torch.zeros_like(Xraw); vecs = torch.zeros(N, k, c.D, device=DEV)
            for j in range(k):
                t = top[:, j]; ismlp = t < C.shape[1]
                bi = torch.where(ismlp, t // c.DFF, (t - C.shape[1]) // c.NH); ni = torch.where(ismlp, t % c.DFF, (t - C.shape[1]) % c.NH)
                v = torch.zeros(N, c.D, device=DEV)
                for b in range(L + 1):
                    m1 = ismlp & (bi == b)
                    if m1.any(): v[m1] = (C[m1][:, b * c.DFF:(b + 1) * c.DFF].gather(1, ni[m1][:, None]) * (c.d["A"][c.atom_index(L, torch.full((int(m1.sum()),), b), ni[m1].cpu())].to(DEV)))
                    m2 = (~ismlp) & (bi == b)
                    if m2.any(): v[m2] = hw[b][m2, ni[m2]]
                vecs[:, j] = v; rec += v
            fv_sum = fvu(((X - (rec - mu)) ** 2).sum(1), X, typ)
            # least-squares refit of the k true directions to the centered state
            U = vecs / vecs.norm(dim=2, keepdim=True).clamp_min(1e-6)
            G = U @ U.transpose(1, 2) + 1e-5 * torch.eye(k, device=DEV); cc = torch.cholesky_solve(U @ X[:, :, None], torch.linalg.cholesky(G))[:, :, 0]
            fv_fit = fvu(((X - torch.einsum("nk,nkd->nd", cc, U)) ** 2).sum(1), X, typ)
            oracle[k] = dict(fvu_sum=fv_sum, fvu_refit=fv_fit, mlp_frac=(top < C.shape[1]).float().mean().item())
        Ad, _ = c.dictionary(L); sel, cof, err = get_omp(c, L, A=Ad, X=c.X(L)); e_ids = err[ids]
        r["oracle"] = oracle; r["omp_fvu"] = {k: fvu(e_ids[:, k - 1], X, typ) for k in (8, 16, 32, 64)}
        log(f"{tag} L{L} ORACLE: " + " ".join(f"k{k}: true-sum {oracle[k]['fvu_sum']:.3f} true-refit {oracle[k]['fvu_refit']:.3f} omp {r['omp_fvu'][k]:.3f}" for k in oracle))
    rows.append(r); log(f"{tag} L{L}: cancel {cancel:.3f} shares emb/att/mlp {r['share_emb']:.2f}/{r['share_att']:.2f}/{r['share_mlp']:.2f} emb_surv {r['emb_survival_med']:.2f} chk {chk:.4f}")
record(f"e06_energy_{tag}", dict(model=tag, N=N, rows=rows), " | ".join(f"L{r['L']}: cancel {r['cancel_index']:.2f} embsurv {r['emb_survival_med']:.2f}" for r in rows) + f" || oracle@64 sum {rows[Lm]['oracle'][64]['fvu_sum']:.3f} refit {rows[Lm]['oracle'][64]['fvu_refit']:.3f} vs omp {rows[Lm]['omp_fvu'][64]:.3f}")
