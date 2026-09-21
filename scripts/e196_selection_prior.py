"""e196: the selection prior of the dictionary. OMP@64 and dual@64 on (i) isotropic Gaussian vectors, (ii)
covariance-matched Gaussian vectors (same second moments as the centered typical states), (iii) the real centered
states. Reports the block/type shares of the support in each case and the per-atom selection-frequency
correlation (Spearman) between noise and real states: how much of WDD's support structure is a property of
(dictionary, covariance) alone, and what the provenance-driven excess looks like."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); mu = c.s["mu"][L + 1].to(DEV); X = (Xraw - mu)[typ]; A, lab = c.dictionary(L); typA, blkA = lab["type"].to(DEV), lab["block"].to(DEV); NA = A.shape[0]
torch.manual_seed(0); Xtr = c.X(L)[sub(c.NT, N, seed=5)]; Xtr = Xtr[typical_mask(Xtr + mu)]; Sig = Xtr.T @ Xtr / len(Xtr); ev, V = torch.linalg.eigh(Sig); Ch = V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T
G_iso = torch.randn(len(X), c.D, device=DEV) * X.norm(dim=1, keepdim=True) / math.sqrt(c.D); G_cov = torch.randn(len(X), c.D, device=DEV) @ Ch; G_cov = G_cov * (X.norm(dim=1, keepdim=True) / G_cov.norm(dim=1, keepdim=True))
S = A.T @ A; ev2, V2 = torch.linalg.eigh(S); Winv = V2 @ torch.diag(1 / (ev2 + 1e-2 * ev2[-1])) @ V2.T
def freq(sel): return torch.bincount(sel.reshape(-1), minlength=NA).float() / len(sel)
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
out = {}
for dec in ("omp", "dual"):
    solve = (lambda Z: omp(Z, A, 64)[0]) if dec == "omp" else (lambda Z: oneshot(Z, A, 64, whiten=Winv)[0])
    F = {nm: freq(solve(Z)) for nm, Z in (("real", X), ("iso", G_iso), ("cov", G_cov))}
    shares = {nm: dict(emb=(typA[solve(Z)] != T_MLP).float().mean().item()) for nm, Z in (("real", X),)} if False else {}
    rec = {}
    for nm in F:
        f = F[nm]; rec[nm] = dict(top100_mass=f.sort(descending=True).values[:100].sum().item() / f.sum().item(), n_used=(f > 0).sum().item(), share_by_block={int(b): round(f[blkA == b].sum().item() / f.sum().item(), 3) for b in range(-1, L + 1)}, share_nonmlp=f[typA != T_MLP].sum().item() / f.sum().item())
    rec["spearman_real_iso"] = spearman(F["real"], F["iso"]); rec["spearman_real_cov"] = spearman(F["real"], F["cov"]); rec["spearman_iso_cov"] = spearman(F["iso"], F["cov"])
    top_real = F["real"].topk(200).indices; rec["top200_overlap_cov"] = len(set(top_real.tolist()) & set(F["cov"].topk(200).indices.tolist())) / 200; rec["top200_overlap_iso"] = len(set(top_real.tolist()) & set(F["iso"].topk(200).indices.tolist())) / 200
    out[dec] = rec
    log(f"{tag} {dec}: per-atom selection frequency Spearman real~cov-noise {rec['spearman_real_cov']:.2f}, real~iso-noise {rec['spearman_real_iso']:.2f} | top-200 atom overlap real/cov {rec['top200_overlap_cov']:.2f} real/iso {rec['top200_overlap_iso']:.2f} | top-100 mass real {rec['real']['top100_mass']:.2f} cov {rec['cov']['top100_mass']:.2f} iso {rec['iso']['top100_mass']:.2f} | non-MLP share real {rec['real']['share_nonmlp']:.2f} cov {rec['cov']['share_nonmlp']:.2f} iso {rec['iso']['share_nonmlp']:.2f} | block shares real " + " ".join(f"b{b}:{v:.2f}" for b, v in rec['real']['share_by_block'].items()) + " | cov " + " ".join(f"b{b}:{v:.2f}" for b, v in rec['cov']['share_by_block'].items()))
record(f"e196_prior_{tag}", dict(model=tag, L=L, results=out), " | ".join(f"{dec}: Spearman(real, cov-noise) {r['spearman_real_cov']:.2f}, (real, iso-noise) {r['spearman_real_iso']:.2f}; top-200 overlap real/cov {r['top200_overlap_cov']:.2f}; top-100 mass real {r['real']['top100_mass']:.2f} vs cov {r['cov']['top100_mass']:.2f}; non-MLP share real {r['real']['share_nonmlp']:.2f} vs cov {r['cov']['share_nonmlp']:.2f}" for dec, r in out.items()))
