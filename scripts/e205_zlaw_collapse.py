"""e205: scaling collapse of the z-law. dual@k for k in (8, 16, 64, 256): the extreme-value account predicts the
reading threshold moves as the k-th largest null |z| (z_k) and that the dominant-recall curves in z - z_k for the
different k collapse onto one curve. Reports z_k per k, the recall curves in z - z_k, and the spread of the
crossover across k (a collapse means a spread near zero)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; A, lab = c.dictionary(L); NA = A.shape[0]
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
torch.manual_seed(0); Xtr = c.X(L)[sub(c.NT, 8192, seed=5)]; Xtr = Xtr[typical_mask(Xtr + mu)]; Sig = Xtr.T @ Xtr / len(Xtr); ev, V = torch.linalg.eigh(Sig); Ch = V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T
G = torch.randn(4096, c.D, device=DEV) @ Ch; G = G * (X[typ].norm(dim=1).median() / G.norm(dim=1, keepdim=True))
S = A.T @ A; ev2, V2 = torch.linalg.eigh(S); Winv = V2 @ torch.diag(1 / (ev2 + 1e-2 * ev2[-1])) @ V2.T; B = Winv @ A.T
mu_a = torch.zeros(NA, device=DEV); m2 = torch.zeros(NA, device=DEV)
for i in range(0, len(G), 512): Z = G[i:i + 512] @ B; mu_a += Z.sum(0); m2 += (Z ** 2).sum(0)
mu_a /= len(G); sd_a = (m2 / len(G) - mu_a ** 2).clamp_min(1e-12).sqrt(); zdom = ((X @ B)[torch.arange(len(X)), row] - mu_a[row]).abs() / sd_a[row]
edges = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 6, 100]; out = {}
for k in (8, 16, 64, 256):
    zk = []
    for i in range(0, 2048, 512): Z = ((G[i:i + 512] @ B) - mu_a) / sd_a; zk.append(Z.abs().topk(k, dim=1).values[:, -1])
    zk = torch.cat(zk).median().item(); sel = oneshot(X, A, k, whiten=Winv)[0]; hit = (sel == row[:, None]).any(1); curve = {}
    for a, b in zip(edges[:-1], edges[1:]):
        m = typ & (zdom - zk >= a) & (zdom - zk < b); curve[f"{a}..{b}"] = hit[m].float().mean().item() if m.sum() > 30 else None
    pts = [((a + b) / 2, v) for (a, b), v in zip(zip(edges[:-1], edges[1:]), curve.values()) if v is not None and b < 100]; z50 = None
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if y0 < 0.5 <= y1: z50 = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0); break
    out[k] = dict(zk=zk, iid=math.sqrt(2 * math.log(NA / k)), overall_recall=hit[typ].float().mean().item(), curve=curve, crossover_rel=z50)
    log(f"{tag} k={k}: z_k {zk:.2f} (iid {out[k]['iid']:.2f}), overall dominant recall {out[k]['overall_recall']:.2f} | recall vs z - z_k: " + " ".join(f"[{kk}]:{(v if v is not None else float('nan')):.2f}" for kk, v in curve.items()) + f" | crossover {(z50 if z50 is not None else float('nan')):+.2f}")
xs = [out[k]["crossover_rel"] for k in out if out[k]["crossover_rel"] is not None]
record(f"e205_collapse_{tag}", dict(model=tag, L=L, results={str(k): v for k, v in out.items()}), "z_k by k: " + " ".join(f"k{k}:{out[k]['zk']:.2f}(iid {out[k]['iid']:.2f})" for k in out) + " | overall recall: " + " ".join(f"k{k}:{out[k]['overall_recall']:.2f}" for k in out) + " | crossover z - z_k: " + " ".join(f"k{k}:{(out[k]['crossover_rel'] if out[k]['crossover_rel'] is not None else float('nan')):+.2f}" for k in out) + (f" | spread {max(xs) - min(xs):.2f}" if len(xs) > 1 else ""))
