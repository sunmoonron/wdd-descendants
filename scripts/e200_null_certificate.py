"""e200: a per-atom null model as a certificate (the geometric prior subtracted). From covariance-matched Gaussian
vectors, estimate for every atom (i) its null selection frequency under OMP@64 and dual@64 and (ii) the mean and
std of its whitened (dual) score. On real states: (a) atom-level filter: drop support atoms whose null selection
frequency is in the top q of the dictionary; (b) token-level z-score: keep dual atoms with |z| = |score - mu_a| /
sigma_a above a threshold. Precision (real = ledger |coef| >= 5% of the token max) and dominant-write recall,
before and after. Complements e195 (stability selection works for OMP, not for the dual)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; A, lab = c.dictionary(L); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV); NA = A.shape[0]
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * C.abs().max(1, keepdim=True).values
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
torch.manual_seed(0); Xtr = c.X(L)[sub(c.NT, 8192, seed=5)]; Xtr = Xtr[typical_mask(Xtr + mu)]; Sig = Xtr.T @ Xtr / len(Xtr); ev, V = torch.linalg.eigh(Sig); Ch = V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T
G = torch.randn(8192, c.D, device=DEV) @ Ch; G = G * (X.norm(dim=1).median() / G.norm(dim=1, keepdim=True))
S = A.T @ A; ev2, V2 = torch.linalg.eigh(S); Winv = V2 @ torch.diag(1 / (ev2 + 1e-2 * ev2[-1])) @ V2.T; B = (Winv @ A.T)                                   # whitened analysis operator [D, NA]
def is_real(sel):
    ismlp = typA[sel] == T_MLP; key = torch.where(ismlp, blkA[sel] * c.DFF + idxA[sel], torch.zeros_like(sel)); return ismlp & (C.gather(1, key).abs() >= thr)
# null statistics per atom
mu_a = torch.zeros(NA, device=DEV); m2 = torch.zeros(NA, device=DEV)
for i in range(0, len(G), 1024): Z = G[i:i + 1024] @ B; mu_a += Z.sum(0); m2 += (Z ** 2).sum(0)
mu_a /= len(G); sd_a = (m2 / len(G) - mu_a ** 2).clamp_min(1e-12).sqrt()
f_null = {"omp": torch.bincount(omp(G[:4096], A, 64)[0].reshape(-1), minlength=NA).float() / 4096, "dual": torch.bincount(oneshot(G[:4096], A, 64, whiten=Winv)[0].reshape(-1), minlength=NA).float() / 4096}
out = {}
for dec in ("omp", "dual"):
    sel = omp(X, A, 64)[0] if dec == "omp" else oneshot(X, A, 64, whiten=Winv)[0]; real = is_real(sel); dom = sel == row[:, None]; base = real[typ].float().mean().item(); dom_base = dom.any(1)[typ].float().mean().item(); rec = dict(base_precision=base, base_dominant_recall=dom_base)
    for q in (0.9, 0.99):
        cut = f_null[dec].quantile(q); keep = f_null[dec][sel] <= cut; rec[f"atomfilter_q{q}"] = dict(kept=keep[typ].float().mean().item(), precision=real[typ & keep.all(1)[:, None].expand_as(real) | typ[:, None].expand_as(real)][keep[typ]].float().mean().item() if False else (real & keep)[typ].float().sum().item() / keep[typ].float().sum().clamp_min(1).item(), dominant_recall=(dom & keep).any(1)[typ].float().mean().item())
    if dec == "dual":
        score = torch.gather(X @ B, 1, sel); z = (score - mu_a[sel]).abs() / sd_a[sel]
        for zt in (2.0, 3.0, 4.0):
            keep = z > zt; rec[f"z>{zt}"] = dict(kept=keep[typ].float().mean().item(), precision=(real & keep)[typ].float().sum().item() / keep[typ].float().sum().clamp_min(1).item(), dominant_recall=(dom & keep).any(1)[typ].float().mean().item())
        # law check: is the dominant atom's z-score the reading statistic? recall by z bin of the dominant atom
        zdom = ((X @ B)[torch.arange(len(X)), row] - mu_a[row]).abs() / sd_a[row]; bins = [(0, 2), (2, 3), (3, 4), (4, 6), (6, 1e9)]
        rec["dominant_recall_by_z"] = {f"{a}-{b}": (dom.any(1)[typ & (zdom >= a) & (zdom < b)].float().mean().item() if (typ & (zdom >= a) & (zdom < b)).sum() > 30 else None) for a, b in bins}
    out[dec] = rec
    log(f"{tag} {dec}: base precision {base:.2f}, dominant recall {dom_base:.2f} | drop atoms in the top-10% null-frequency: kept {rec['atomfilter_q0.9']['kept']:.2f} precision {rec['atomfilter_q0.9']['precision']:.2f} dominant recall {rec['atomfilter_q0.9']['dominant_recall']:.2f}; top-1%: kept {rec['atomfilter_q0.99']['kept']:.2f} precision {rec['atomfilter_q0.99']['precision']:.2f} recall {rec['atomfilter_q0.99']['dominant_recall']:.2f}" + (" | z-score certificate: " + " ".join(f"|z|>{zt}: kept {rec[f'z>{zt}']['kept']:.2f} precision {rec[f'z>{zt}']['precision']:.2f} recall {rec[f'z>{zt}']['dominant_recall']:.2f}" for zt in (2.0, 3.0, 4.0)) + " | dominant recall by its z: " + " ".join(f"{k}:{(v if v is not None else float('nan')):.2f}" for k, v in rec['dominant_recall_by_z'].items()) if dec == "dual" else ""))
record(f"e200_nullcert_{tag}", dict(model=tag, L=L, results=out), f"omp: base precision {out['omp']['base_precision']:.2f} -> null-frequency filter (top 10% dropped) {out['omp']['atomfilter_q0.9']['precision']:.2f} (dominant recall {out['omp']['base_dominant_recall']:.2f} -> {out['omp']['atomfilter_q0.9']['dominant_recall']:.2f}) | dual: base {out['dual']['base_precision']:.2f} -> filter {out['dual']['atomfilter_q0.9']['precision']:.2f} (recall {out['dual']['base_dominant_recall']:.2f} -> {out['dual']['atomfilter_q0.9']['dominant_recall']:.2f}); z-score |z|>3: precision {out['dual']['z>3.0']['precision']:.2f} kept {out['dual']['z>3.0']['kept']:.2f} recall {out['dual']['z>3.0']['dominant_recall']:.2f} | dominant recall by z bin: " + " ".join(f"{k}:{(v if v is not None else float('nan')):.2f}" for k, v in out['dual']['dominant_recall_by_z'].items()))
