"""e118 (audit follow-up): is the 'structured competition' just anisotropy? The maximum |cos| of a random direction
with the dictionary when the direction is drawn (a) isotropically, (b) from a Gaussian with the typical-state
covariance, (c) as a real centered state; plus the 64th-largest. If (b) matches (c), the real competitor level is
explained by the state covariance alone, with no reference to write structure."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L); Xt = X[typ]
Sig = (Xt.T @ Xt) / Xt.shape[0]; ev, V = torch.linalg.eigh(Sig); half = V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T
g = torch.Generator(device=DEV).manual_seed(0); z = torch.randn(2048, c.D, generator=g, device=DEV); zi = z / z.norm(dim=1, keepdim=True); zc = z @ half; zc = zc / zc.norm(dim=1, keepdim=True)
xr = Xt[sub(Xt.shape[0], 2048).to(DEV)]; xr = xr / xr.norm(dim=1, keepdim=True)
def stats(Z):
    mx, k64 = [], []
    for s in range(0, Z.shape[0], 512):
        G = (Z[s:s + 512] @ A.T).abs(); mx.append(G.max(1).values); k64.append(G.topk(64, dim=1).values[:, -1])
    mx, k64 = torch.cat(mx), torch.cat(k64); return dict(max_mean=mx.mean().item(), max_median=mx.median().item(), k64_mean=k64.mean().item())
p = ev / ev.sum(); deff = math.exp(-(p * (p + 1e-12).log()).sum().item())
res = dict(model=tag, L=L, m=A.shape[0], d=c.D, d_eff=deff, gauss_floor_d=math.sqrt(2 * math.log(A.shape[0]) / c.D), gauss_floor_deff=math.sqrt(2 * math.log(A.shape[0]) / deff), isotropic=stats(zi), covariance_matched=stats(zc), real_states=stats(xr))
record(f"e118_aniso_{tag}", res, f"d_eff {deff:.0f} | gauss floor d {res['gauss_floor_d']:.3f} d_eff {res['gauss_floor_deff']:.3f} | max|cos| mean: isotropic {res['isotropic']['max_mean']:.3f} covariance-matched {res['covariance_matched']['max_mean']:.3f} real states {res['real_states']['max_mean']:.3f} | 64th: {res['isotropic']['k64_mean']:.3f} / {res['covariance_matched']['k64_mean']:.3f} / {res['real_states']['k64_mean']:.3f}")
