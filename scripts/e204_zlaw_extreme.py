"""e204: the z-score law and its parameter-free threshold. For the dual (whitened) score and the raw score, the per-
atom null mean/std come from covariance-matched noise. The predicted reading threshold for a k-sparse (k=64)
decoder is the k-th largest null |z| over the NA atoms (extreme-value level; iid approximation sqrt(2 ln(NA/k))).
Reports: the empirical k-th largest null |z| (median over noise samples), the iid prediction, and the dominant
write's recall as a function of z - z_k for dual@64 and OMP@64 (raw z), plus the crossover z50."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; A, lab = c.dictionary(L); NA = A.shape[0]
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
torch.manual_seed(0); Xtr = c.X(L)[sub(c.NT, 8192, seed=5)]; Xtr = Xtr[typical_mask(Xtr + mu)]; Sig = Xtr.T @ Xtr / len(Xtr); ev, V = torch.linalg.eigh(Sig); Ch = V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T
G = torch.randn(4096, c.D, device=DEV) @ Ch; G = G * (X[typ].norm(dim=1).median() / G.norm(dim=1, keepdim=True))
S = A.T @ A; ev2, V2 = torch.linalg.eigh(S); Winv = V2 @ torch.diag(1 / (ev2 + 1e-2 * ev2[-1])) @ V2.T
sel = {"dual": oneshot(X, A, 64, whiten=Winv)[0], "omp": omp(X, A, 64)[0]}; out = {}
for nm, B in (("dual", Winv @ A.T), ("omp", A.T)):
    mu_a = torch.zeros(NA, device=DEV); m2 = torch.zeros(NA, device=DEV); zk = []
    for i in range(0, len(G), 512): Z = G[i:i + 512] @ B; mu_a += Z.sum(0); m2 += (Z ** 2).sum(0)
    mu_a /= len(G); sd_a = (m2 / len(G) - mu_a ** 2).clamp_min(1e-12).sqrt()
    for i in range(0, 2048, 512): Z = ((G[i:i + 512] @ B) - mu_a) / sd_a; zk.append(Z.abs().topk(64, dim=1).values[:, -1])
    zk = torch.cat(zk).median().item(); iid = math.sqrt(2 * math.log(NA / 64)); zdom = ((X @ B)[torch.arange(len(X)), row] - mu_a[row]).abs() / sd_a[row]; hit = (sel[nm] == row[:, None]).any(1)
    edges = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 6, 100]; curve = {}
    for a, b in zip(edges[:-1], edges[1:]):
        m = typ & (zdom - zk >= a) & (zdom - zk < b); curve[f"{a}..{b}"] = (hit[m].float().mean().item(), int(m.sum())) if m.sum() > 30 else (None, int(m.sum()))
    # crossover: smallest bin center with recall >= 0.5 (linear interp on the bin sequence)
    pts = [((a + b) / 2, v[0]) for (a, b), v in zip(zip(edges[:-1], edges[1:]), curve.values()) if v[0] is not None and b < 100]; z50 = None
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if y0 < 0.5 <= y1: z50 = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0); break
    out[nm] = dict(NA=NA, zk_empirical=zk, zk_iid=iid, curve=curve, crossover_rel=z50, crossover_abs=(None if z50 is None else z50 + zk))
    log(f"{tag} {nm}: NA {NA}, k-th largest null |z| empirical {zk:.2f} (iid prediction {iid:.2f}) | dominant recall vs (z - z_k): " + " ".join(f"[{k}]:{(v[0] if v[0] is not None else float('nan')):.2f}" for k, v in curve.items()) + f" | crossover at z - z_k = {(z50 if z50 is not None else float('nan')):+.2f} (absolute z50 {(z50 + zk if z50 is not None else float('nan')):.2f})")
record(f"e204_zlaw_{tag}", dict(model=tag, L=L, results=out), " | ".join(f"{nm}: z_k empirical {r['zk_empirical']:.2f} vs iid {r['zk_iid']:.2f}; recall crossover at z - z_k = {(r['crossover_rel'] if r['crossover_rel'] is not None else float('nan')):+.2f} (z50 {(r['crossover_abs'] if r['crossover_abs'] is not None else float('nan')):.2f}); recall at z-z_k in [-1,-0.5) {(r['curve']['-1..-0.5'][0] if r['curve']['-1..-0.5'][0] is not None else float('nan')):.2f}, [0,0.5) {(r['curve']['0..0.5'][0] if r['curve']['0..0.5'][0] is not None else float('nan')):.2f}, [1,2) {(r['curve']['1..2'][0] if r['curve']['1..2'][0] is not None else float('nan')):.2f}" for nm, r in out.items()))
