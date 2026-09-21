"""e355: is the low dimension a kernel phenomenon? The functional kernel K(a, b) = <F(a), F(b)> over 150 perturbations
of three families (exact logit effects): its participation rank and the number of eigenvalues for 90% of the trace,
against the linear causal rank (participation rank of the images' coordinates and of the images themselves), and a
kernel built from the descendants' coordinates (K_z = <z_a, z_b>) against the functional kernel: the alignment of the
two Gram matrices (kernel-target alignment) and the rank at which K_z explains K."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, Bv = logit_scores(nat["dl"], tr); P = pls(Fc[tr], Z[tr], d); lab = S.c.d["lab"]; A = S.c.d["A"]; torch.manual_seed(0); Ratt = A[(lab["type"] == T_ATT) & (lab["block"] <= S.b)].float().to(DEV)
V = torch.cat([S.W2[torch.randperm(S.DFF, device=DEV)[:50]], unit(torch.randn(50, S.D, device=DEV)), unit(Ratt[torch.randperm(len(Ratt), device=DEV)[:50]])]); N = len(V); img = torch.zeros(N, S.D, device=DEV); Fl = torch.zeros(N, Bv.shape[1], device=DEV); cnt = torch.zeros(N, device=DEV)
for p in range(3):
    r = S.inject_family(V, S.b + 1, seed=p); img.index_add_(0, r["a"], r["F"][L]); Fl.index_add_(0, r["a"], r["dl"] @ Bv); cnt.index_add_(0, r["a"], torch.ones(len(r["a"]), device=DEV))
img = img / cnt[:, None].clamp_min(1); Fl = Fl / cnt[:, None].clamp_min(1); Kf = Fl @ Fl.T; z = img @ P; Kz = z @ z.T; Kimg = img @ img.T
def pr(K): ev = torch.linalg.eigvalsh(K).clamp_min(0).flip(0); return (ev.sum() ** 2 / (ev ** 2).sum()).item(), int((ev.cumsum(0) / ev.sum() >= 0.9).nonzero()[0].item()) + 1
kta = lambda K1, K2: ((K1 * K2).sum() / (K1.norm() * K2.norm())).item(); pf, nf = pr(Kf); pz, nz = pr(Kz); pi, ni = pr(Kimg)
ev, U = torch.linalg.eigh(Kz); U = U.flip(1); expl = {}
for r_ in (1, 2, 4, 8, 16):
    Ur = U[:, :r_]; Pr = Ur @ Ur.T; expl[r_] = kta(Pr @ Kf @ Pr, Kf)
res = dict(N=N, functional_kernel_prank=pf, functional_kernel_n90=nf, coordinate_kernel_prank=pz, coordinate_kernel_n90=nz, image_kernel_prank=pi, image_kernel_n90=ni, alignment_coordinate_vs_functional=kta(Kz, Kf), alignment_image_vs_functional=kta(Kimg, Kf), alignment_random16=kta((img @ torch.linalg.qr(torch.randn(S.D, d, device=DEV))[0]) @ (img @ torch.linalg.qr(torch.randn(S.D, d, device=DEV))[0]).T, Kf), functional_explained_by_coordinate_rank={str(k): v for k, v in expl.items()})
log(f"{tag} (N {N}): functional kernel participation rank {pf:.1f} (90% of trace in {nf} eigenvalues); coordinate kernel {pz:.1f} ({nz}); image kernel {pi:.1f} ({ni}) | alignment with the functional kernel: coordinates {res['alignment_coordinate_vs_functional']:.2f}, full images {res['alignment_image_vs_functional']:.2f}, random-16 {res['alignment_random16']:.2f} | functional kernel explained by the coordinate kernel's top 1/2/4/8/16 eigenvectors: " + "/".join(f"{expl[k]:.2f}" for k in (1, 2, 4, 8, 16)))
record(f"e355_kernel_{tag}", dict(model=tag, L=L, **res), f"K_f prank {pf:.1f} n90 {nf}; K_z prank {pz:.1f} n90 {nz}; K_img {pi:.1f}/{ni}; align z {res['alignment_coordinate_vs_functional']:.2f} img {res['alignment_image_vs_functional']:.2f} rand {res['alignment_random16']:.2f}")
