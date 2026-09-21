"""e333: the quotient of the whole MLP write from the ledger. At typical tokens the entire block-2 MLP output is
ablated (the write of all atoms at once); the footprint's coordinate at L is compared with the ledger-weighted sum
of the atoms' exact coordinates over the top-k atoms of the ledger (k = 8, 32, 128, all), and with the same sum
using operator coordinates; against the coordinate predicted from the write vector itself pushed through the
operator (no atoms), and against random coefficients."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); P = pls(Fc[tr], Z[tr], d)
torch.manual_seed(0); img = torch.zeros(S.DFF, S.D, device=DEV); cnt = torch.zeros(S.DFF, device=DEV); per = min(1024, S.DFF)
for p in range(max(1, S.DFF // per) * 6):
    start = (p % max(1, S.DFF // per)) * per; ids = (start + torch.arange(per, device=DEV)) % S.DFF; a = ids[torch.randint(0, per, (len(S.pool),), device=DEV)]; inj = torch.zeros(S.NT, S.D, device=DEV); inj[S.pool] = S.s_inj * S.W2[a]; r = S.run(inject=inj, inject_block=S.b + 1); img.index_add_(0, a, (r[L] - S.S0[L])[S.pool]); cnt.index_add_(0, a, torch.ones(len(S.pool), device=DEV))
img = img / cnt[:, None].clamp_min(1) / S.s_inj; Zat = img @ P; T = S.fit_T(S.b + 1, L, seed=3); Zop = (S.W2 @ T) @ P
def pre(m, inp): x = inp[0].clone(); return (torch.zeros_like(x),)
h = S.arch.mlp_lin(S.b).register_forward_pre_hook(pre); r = S.run(positions=S.pool); h.remove(); dS = (S.S0[L] - r[L])[S.pool]; act = dS @ P; coef = S.led[S.pool]; wnorm = S.arch.wdir(S.b).to(DEV).norm(dim=1); coef_dir = coef
out = {}
for k in (8, 32, 128, S.DFF):
    top = coef.abs().topk(k, dim=1).indices; mask = torch.zeros_like(coef).scatter_(1, top, 1.0); pred = (coef * mask) @ Zat; out[f"exact_top{k}"] = ((unit(act) * unit(pred)).sum(1)).median().item(); out[f"operator_top{k}"] = ((unit(act) * unit((coef * mask) @ Zop)).sum(1)).median().item()
wv = coef @ S.W2; out["write_through_operator"] = ((unit(act) * unit((wv @ T) @ P)).sum(1)).median().item(); out["random_coefficients"] = ((unit(act) * unit((torch.randn_like(coef) * coef.abs().mean()) @ Zat)).sum(1)).median().item(); out["ledger_l1_top8_share"] = (coef.abs().topk(8, dim=1).values.sum(1) / coef.abs().sum(1)).median().item()
log(f"{tag} (K {S.K}): coordinate of the whole block-2 MLP write predicted from the ledger x exact atom coordinates, top-8/32/128/all atoms: " + "/".join(f"{out[f'exact_top{k}']:.2f}" for k in (8, 32, 128, S.DFF)) + "; with operator coordinates " + "/".join(f"{out[f'operator_top{k}']:.2f}" for k in (8, 32, 128, S.DFF)) + f"; from the write vector through the operator {out['write_through_operator']:.2f}; random coefficients {out['random_coefficients']:.2f}; top-8 atoms hold {out['ledger_l1_top8_share']:.2f} of the ledger's L1 mass")
record(f"e333_mlpquot_{tag}", dict(model=tag, L=L, K=S.K, **out), "exact top8/32/128/all " + "/".join(f"{out[f'exact_top{k}']:.2f}" for k in (8, 32, 128, S.DFF)) + " op " + "/".join(f"{out[f'operator_top{k}']:.2f}" for k in (8, 32, 128, S.DFF)) + f" write-op {out['write_through_operator']:.2f} random {out['random_coefficients']:.2f}")
