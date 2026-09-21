"""e74: what does WDD leave behind? The k=64 OMP residual of each state, projected onto the TRUE writes: which
blocks' and which kinds of writes make up the residual (energy shares), the survival of the dominant write inside
the residual, and how much residual energy is attention vs MLP vs embedding (exact, via the ledger)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
X = c.X(L)[ids]; Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=c.X(L)); sel, cof = sel[ids.to(DEV)], cof[ids.to(DEV)]
R = X - torch.einsum("nk,nkd->nd", cof, A[sel])                                                  # residual (centered frame)
mu = c.s["mu"][L + 1].to(DEV); comps = {"emb": c.s["H"][0][ids].float().to(DEV)}
for b in range(L + 1):
    comps[f"att{b}"] = c.s["ATT"][b][ids].float().to(DEV); w = c.acts[b][ids].float().to(DEV) @ c.wdir_cpu(b).to(DEV)
    if c.d["mlp_bias"][b] is not None: w = w + c.d["mlp_bias"][b].to(DEV)
    comps[f"mlp{b}"] = w
# least-squares split of the residual over the component vectors of the same token (each token: R ~ sum_i beta_i comp_i)
names = list(comps); V = torch.stack([comps[n] for n in names], 1)                                 # [N, C, D]
G = V @ V.transpose(1, 2) + 1e-3 * torch.eye(len(names), device=DEV); beta = torch.linalg.solve(G, (V @ R[:, :, None]))[:, :, 0]
contrib = beta[:, :, None] * V; e_r = (R ** 2).sum(1)
share = {n: ((contrib[:, i] * R).sum(1) / e_r.clamp_min(1e-6))[typ].mean().item() for i, n in enumerate(names)}   # signed share of residual energy
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV); d = A[row]; ct = tc[ids, 0].to(DEV)
res = dict(model=tag, L=L, residual_fvu=(e_r[typ].sum() / (X[typ] ** 2).sum()).item(), residual_share_by_component=share,
           att_total=sum(v for k, v in share.items() if k.startswith("att")), mlp_total=sum(v for k, v in share.items() if k.startswith("mlp")),
           dominant_write_in_residual=((R * d).sum(1) / ct)[typ].median().item(), dominant_write_in_state=((X * d).sum(1) / ct)[typ].median().item(),
           cos_residual_attention=((R * sum(comps[k] for k in names if k.startswith("att"))).sum(1) / (R.norm(dim=1) * sum(comps[k] for k in names if k.startswith("att")).norm(dim=1)))[typ].median().item())
record(f"e74_residual_{tag}", res, f"residual fvu {res['residual_fvu']:.3f} | residual energy shares: emb {share['emb']:.2f} attention {res['att_total']:.2f} mlp {res['mlp_total']:.2f} | by block att " + " ".join(f"{share[f'att{b}']:.2f}" for b in range(L + 1)) + " mlp " + " ".join(f"{share[f'mlp{b}']:.2f}" for b in range(L + 1)) + f" | dominant write left in residual {res['dominant_write_in_residual']:.2f} (in state {res['dominant_write_in_state']:.2f}) | cos(residual, attention sum) {res['cos_residual_attention']:.2f}")
