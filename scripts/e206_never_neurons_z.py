"""e206: never-neurons explained by geometry? Per MLP neuron (blocks <= L) with >= 20 dominant typical tokens at
level L: its dual@64 readability (fraction of its dominant tokens where its atom is read), and three geometric
factors: the median |coef| of its dominant writes, its self-gain a^T Winv a (how much of its own write the
whitened decoder sees), and its null std sigma_a (how strongly covariance-matched noise projects on it). The
predicted own-write z is coef * self-gain / sigma_a. Reports Spearman of readability with each factor and with the
predicted z, and the factor profile of never-read (readability < 0.05) vs always-read (> 0.95) neurons."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 16384; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; A, lab = c.dictionary(L); NA = A.shape[0]
tb, tn, tc = c.top_writes(L, 1); tb, tn, tc = tb[ids, 0], tn[ids, 0], tc[ids, 0]; row = c.atom_index(L, tb, tn).to(DEV)
torch.manual_seed(0); Xtr = c.X(L)[sub(c.NT, 8192, seed=5)]; Xtr = Xtr[typical_mask(Xtr + mu)]; Sig = Xtr.T @ Xtr / len(Xtr); ev, V = torch.linalg.eigh(Sig); Ch = V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T
G = torch.randn(4096, c.D, device=DEV) @ Ch; G = G * (X[typ].norm(dim=1).median() / G.norm(dim=1, keepdim=True))
S = A.T @ A; ev2, V2 = torch.linalg.eigh(S); Winv = V2 @ torch.diag(1 / (ev2 + 1e-2 * ev2[-1])) @ V2.T; B = Winv @ A.T
mu_a = torch.zeros(NA, device=DEV); m2 = torch.zeros(NA, device=DEV)
for i in range(0, len(G), 512): Z = G[i:i + 512] @ B; mu_a += Z.sum(0); m2 += (Z ** 2).sum(0)
mu_a /= len(G); sd_a = (m2 / len(G) - mu_a ** 2).clamp_min(1e-12).sqrt(); selfgain = (A * (A @ Winv)).sum(1)
sel = oneshot(X, A, 64, whiten=Winv)[0]; hit = (sel == row[:, None]).any(1)
rows_ = row[typ]; hits = hit[typ]; coefs = tc.to(DEV).abs()[typ]; uniq, inv, cnt = torch.unique(rows_, return_inverse=True, return_counts=True); keep = cnt >= 20
read = torch.zeros(len(uniq), device=DEV).index_add_(0, inv, hits.float()) / cnt; medc = torch.zeros(len(uniq), device=DEV)
for i in torch.nonzero(keep)[:, 0].tolist(): medc[i] = coefs[inv == i].median()
u = uniq[keep]; read = read[keep]; medc = medc[keep]; sg = selfgain[u]; sd = sd_a[u]; zpred = medc * sg / sd
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
rho = dict(coef=spearman(read, medc), selfgain=spearman(read, sg), sigma=spearman(read, -sd), zpred=spearman(read, zpred), coef_over_sigma=spearman(read, medc / sd))
never, always = read < 0.05, read > 0.95
prof = {nm: dict(n=int(m.sum()), coef=medc[m].median().item(), selfgain=sg[m].median().item(), sigma=sd[m].median().item(), zpred=zpred[m].median().item()) for nm, m in (("never", never), ("always", always), ("all", torch.ones_like(never)))}
log(f"{tag}: {len(u)} neurons with >= 20 dominant tokens; Spearman(readability, .): |coef| {rho['coef']:+.2f}, self-gain {rho['selfgain']:+.2f}, -sigma_a {rho['sigma']:+.2f}, coef/sigma {rho['coef_over_sigma']:+.2f}, predicted z {rho['zpred']:+.2f} | never-read (n {prof['never']['n']}): coef {prof['never']['coef']:.1f} self-gain {prof['never']['selfgain']:.2f} sigma {prof['never']['sigma']:.2f} z {prof['never']['zpred']:.1f} | always-read (n {prof['always']['n']}): coef {prof['always']['coef']:.1f} self-gain {prof['always']['selfgain']:.2f} sigma {prof['always']['sigma']:.2f} z {prof['always']['zpred']:.1f} | all: coef {prof['all']['coef']:.1f} self-gain {prof['all']['selfgain']:.2f} sigma {prof['all']['sigma']:.2f} z {prof['all']['zpred']:.1f}")
record(f"e206_neverz_{tag}", dict(model=tag, L=L, n_neurons=len(u), spearman=rho, profiles=prof), f"Spearman(readability, predicted own-write z) {rho['zpred']:+.2f} vs |coef| {rho['coef']:+.2f}, self-gain {rho['selfgain']:+.2f}, -sigma {rho['sigma']:+.2f} | never-read vs always-read neurons: coef {prof['never']['coef']:.1f} vs {prof['always']['coef']:.1f}, self-gain {prof['never']['selfgain']:.2f} vs {prof['always']['selfgain']:.2f}, sigma {prof['never']['sigma']:.2f} vs {prof['always']['sigma']:.2f}, predicted z {prof['never']['zpred']:.1f} vs {prof['always']['zpred']:.1f} (n {prof['never']['n']} / {prof['always']['n']} of {len(u)})")
