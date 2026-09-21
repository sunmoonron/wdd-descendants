"""e240: why does the transported dictionary decode worse? At the mid layer, build the transported atoms of one
source block (b = 2) as in e236 and compare with the native atoms: (i) cosine of each transported atom with its own
native atom (how much of the original direction the image keeps); (ii) Babel-1 coherence (median over atoms of the
largest |cos| with another atom) and the share of the dictionary's energy in its top principal direction, for the
native block-2 atoms, the transported block-2 atoms, and transported atoms of random directions; (iii) the cosine of
transported atoms with the 'random-direction footprint' centroid (the shared disturbance direction of e224)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; lab = c.d["lab"]; A = c.d["A"]; DFF = c.DFF; b = 2; run = make_runner(model, arch, c, ids_seq, [L], NT)
R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); S0 = run(); H = S0[L]; s_b = (c.acts[b][:NT].float().to(DEV) * c.d["WN"][b].to(DEV)[None]).abs().max(1).values.median().clamp_min(1e-3)
def transport(dirs):
    acc = torch.zeros(dirs.shape[0], c.D, device=DEV); cnt = torch.zeros(dirs.shape[0], device=DEV); torch.manual_seed(0)
    for p in range(3):
        idx = torch.randint(0, dirs.shape[0], (NT,), device=DEV); S1 = run(inject=s_b * dirs[idx], inject_block=b + 1); img = (S1[L] - H) / s_b; acc.index_add_(0, idx, img); cnt.index_add_(0, idx, torch.ones(NT, device=DEV))
    T = acc / cnt.clamp_min(1)[:, None]; return T[cnt > 0], cnt > 0
T, ok = transport(R); Tn = unit(T); Rn = R[ok]; rnd = unit(torch.randn(DFF, c.D, device=DEV)); Tr, okr = transport(rnd); Trn = unit(Tr)
def babel(M):
    G = M @ M.T; G.fill_diagonal_(0); return G.abs().max(1).values.median().item()
def top_share(M):
    Mc = M - M.mean(0, keepdim=True); s = torch.linalg.svdvals(Mc); e = s ** 2; return (e[0] / e.sum()).item(), ((e.sum() ** 2) / (e ** 2).sum()).item()
own = (Tn * Rn).sum(1); rc = unit(Tr.mean(0, keepdim=True)); res = dict(own_direction_cos_med=own.median().item(), own_direction_cos_q10=own.quantile(0.1).item(), babel_native=babel(Rn), babel_transported=babel(Tn), babel_transported_random=babel(Trn), top_share_native=top_share(Rn)[0], top_share_transported=top_share(Tn)[0], dim_native=top_share(Rn)[1], dim_transported=top_share(Tn)[1], dim_transported_random=top_share(Trn)[1], cos_with_random_centroid_transported=(Tn @ rc.T)[:, 0].abs().median().item(), cos_with_random_centroid_native=(Rn @ rc.T)[:, 0].abs().median().item(), gain_med=T.norm(dim=1).median().item())
log(f"{tag} block {b} -> level {L}: transported atom keeps cos {res['own_direction_cos_med']:.2f} with its native atom (10th pct {res['own_direction_cos_q10']:.2f}); gain {res['gain_med']:.2f} | Babel-1 coherence native {res['babel_native']:.2f} vs transported {res['babel_transported']:.2f} (transported random directions {res['babel_transported_random']:.2f}) | top-PC energy share native {res['top_share_native']:.3f} vs transported {res['top_share_transported']:.3f}; effective dimension native {res['dim_native']:.0f}, transported {res['dim_transported']:.0f}, transported random {res['dim_transported_random']:.0f} | |cos| with the random-footprint centroid: transported {res['cos_with_random_centroid_transported']:.2f} vs native {res['cos_with_random_centroid_native']:.2f}")
record(f"e240_tcoherence_{tag}", dict(model=tag, b=b, L=L, **res), f"own-direction cos {res['own_direction_cos_med']:.2f}; Babel-1 native {res['babel_native']:.2f} -> transported {res['babel_transported']:.2f} (random dirs {res['babel_transported_random']:.2f}); top-PC share native {res['top_share_native']:.3f} -> transported {res['top_share_transported']:.3f}; effective dim {res['dim_native']:.0f} -> {res['dim_transported']:.0f}; |cos| with random-footprint centroid {res['cos_with_random_centroid_native']:.2f} -> {res['cos_with_random_centroid_transported']:.2f}")
