"""e07: gauge invariance. SwiGLU has the symmetry (R*)^m x S_m per block (scale up row j by s, down column j by 1/s;
permute neurons; flip the sign of both). The function is unchanged, weight norms are scrambled, but WDD's unit
atoms and coefficients are invariant. Checks: hidden states identical, dictionary identical up to sign+perm,
OMP FVU/recall identical, weight-norm rankings destroyed (Spearman)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
name = sys.argv[1] if len(sys.argv) > 1 else "smollm2"
model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2
ids = corpus_ids(tok, "wikitext", "test")[:8 * CTX].view(8, CTX).to(DEV)
def states_and_acts(model):
    acts = {}
    hs = [arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, i: acts.__setitem__(b_, i[0].detach().float().cpu()))(b)) for b in range(L + 1)]
    H = []
    for b in range(L + 1):
        H.append(arch.layers[b].register_forward_hook((lambda b_: lambda m, i, o: acts.__setitem__(("H", b_), (o[0] if isinstance(o, tuple) else o).detach().float().cpu()))(b)))
    model(ids); [h.remove() for h in hs + H]
    return acts
a0 = states_and_acts(model); X0 = a0[("H", L)].view(-1, arch.D).to(DEV); A0, lab0 = build_dictionary(arch, blocks=list(range(L + 1)))
led0 = torch.cat([a0[b].view(-1, arch.DFF) * arch.wdir(b).norm(dim=1).cpu() for b in range(L + 1)], 1)
norm0 = torch.cat([arch.wdir(b).norm(dim=1).cpu() for b in range(L + 1)])
# apply the gauge
g = torch.Generator().manual_seed(0); perms = {}
for b in range(L + 1):
    l = arch.layers[b]; m = arch.DFF
    s = torch.exp(torch.randn(m, generator=g) * 1.0).to(DEV) * torch.where(torch.rand(m, generator=g) < 0.5, -1.0, 1.0).to(DEV)
    perm = torch.randperm(m, generator=g).to(DEV); perms[b] = perm
    l.mlp.up_proj.weight.data = (l.mlp.up_proj.weight.data * s[:, None])[perm]
    l.mlp.gate_proj.weight.data = l.mlp.gate_proj.weight.data[perm]
    l.mlp.down_proj.weight.data = (l.mlp.down_proj.weight.data / s[None, :])[:, perm]
a1 = states_and_acts(model); X1 = a1[("H", L)].view(-1, arch.D).to(DEV); A1, lab1 = build_dictionary(arch, blocks=list(range(L + 1)))
led1 = torch.cat([a1[b].view(-1, arch.DFF) * arch.wdir(b).norm(dim=1).cpu() for b in range(L + 1)], 1)
norm1 = torch.cat([arch.wdir(b).norm(dim=1).cpu() for b in range(L + 1)])
state_diff = ((X1 - X0).norm() / X0.norm()).item()
# atom matching: each transformed MLP atom should match exactly one original atom with |cos| = 1
m0 = lab0["type"] == T_MLP; m1 = lab1["type"] == T_MLP
G = (A1[m1.to(DEV)] @ A0[m0.to(DEV)].T).abs(); best = G.max(1)
atom_match = dict(min_best_cos=best.values.min().item(), frac_exact=(best.values > 0.9999).float().mean().item())
# ledger invariance: coefficients up to permutation and sign (|c| equal)
inv_perm = torch.cat([perms[b].cpu() + b * arch.DFF for b in range(L + 1)])
led_diff = ((led1.abs() - led0[:, inv_perm].abs()).norm() / led0.norm()).item()
# weight-norm ranking destroyed
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
rho_norm = spearman(norm1, norm0[inv_perm])
# OMP on the same states with both dictionaries
X = X0 - X0.mean(0); sel0, cof0, err0 = omp(X, A0, 64); sel1, cof1, err1 = omp(X, A1, 64)
typ = typical_mask(X0)
# top-1 write identification in both frames
t0 = led0.abs().argmax(1).to(DEV); t1 = led1.abs().argmax(1).to(DEV)
rows0 = torch.nonzero(m0)[:, 0].to(DEV)[t0]; rows1 = torch.nonzero(m1)[:, 0].to(DEV)[t1]
r0 = (sel0 == rows0[:, None]).any(1)[typ].float().mean().item(); r1 = (sel1 == rows1[:, None]).any(1)[typ].float().mean().item()
res = dict(model=name, L=L, state_rel_diff=state_diff, atom_match=atom_match, ledger_abs_rel_diff=led_diff, spearman_weight_norm=rho_norm,
           fvu32=(fvu(err0[:, 31], X, typ), fvu(err1[:, 31], X, typ)), recall64=(r0, r1), same_support_frac=None)
# support identity through the permutation (MLP atoms only)
map1to0 = best.indices  # transformed MLP atom -> original MLP atom (indices within MLP subsets)
mlp_rows0 = torch.nonzero(m0)[:, 0].to(DEV); mlp_rows1 = torch.nonzero(m1)[:, 0].to(DEV)
lut = torch.full((A1.shape[0],), -1, dtype=torch.long, device=DEV); lut[mlp_rows1] = mlp_rows0[map1to0]
nonmlp = torch.nonzero(~m1)[:, 0].to(DEV); lut[nonmlp] = nonmlp   # embeddings/attention rows are unchanged and in the same positions
s1m = lut[sel1]; res["same_support_frac"] = (torch.sort(s1m, 1).values == torch.sort(sel0, 1).values).float().mean().item()
record(f"e07_gauge_{name}", res, f"state diff {state_diff:.1e} | atoms match {atom_match['frac_exact']:.3f} (min cos {atom_match['min_best_cos']:.4f}) | ledger diff {led_diff:.1e} | weight-norm Spearman {rho_norm:.3f} | fvu32 {res['fvu32'][0]:.3f}/{res['fvu32'][1]:.3f} recall {r0:.3f}/{r1:.3f} support identical {res['same_support_frac']:.3f}")
