"""e332: WDD as the foundation of the quotient. The functional dictionary Z = P^T J w over all block-2 atoms (exact
coordinates for 1024 atoms by injection, operator coordinates for all): effective rank; how many atoms an orthogonal
matching pursuit needs to represent a random point of the quotient (the sparsity of the quotient in the atom basis)
against random unit vectors as a dictionary; and whether the atoms' coordinates are more concentrated in the
quotient than random directions (fraction of an atom's transported energy inside the quotient)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); P = pls(Fc[tr], Z[tr], d)
torch.manual_seed(0); atoms = torch.randperm(S.DFF, device=DEV)[:1024]; V = S.W2[atoms]; img = torch.zeros(1024, S.D, device=DEV); cnt = torch.zeros(1024, device=DEV)
for p in range(6):
    r = S.inject_family(V, S.b + 1, seed=p); img.index_add_(0, r["a"], r["F"][L]); cnt.index_add_(0, r["a"], torch.ones(len(r["a"]), device=DEV))
img = img / cnt[:, None].clamp_min(1); Zat = img @ P; T = S.fit_T(S.b + 1, L, seed=3); Zop = (S.W2 @ T) @ P; Rn = unit(torch.randn(1024, S.D, device=DEV)); Zrn = (Rn @ T) @ P; Rimg = torch.zeros(1024, S.D, device=DEV); cnt2 = torch.zeros(1024, device=DEV)
for p in range(3):
    r = S.inject_family(Rn, S.b + 1, seed=20 + p); Rimg.index_add_(0, r["a"], r["F"][L]); cnt2.index_add_(0, r["a"], torch.ones(len(r["a"]), device=DEV))
Rimg = Rimg / cnt2[:, None].clamp_min(1)
def omp_needed(Dict, targets, tol=0.1, kmax=64):
    Dn = unit(Dict); need = []
    for t in targets:
        res = t.clone(); sel = []
        for k in range(kmax):
            j = int((Dn @ res).abs().argmax()); sel.append(j); A = Dn[sel]; coef = torch.linalg.lstsq(A.T, t[:, None]).solution[:, 0]; res = t - A.T @ coef
            if res.norm() / t.norm() < tol: break
        need.append(len(sel))
    return float(torch.tensor(need).float().median())
torch.manual_seed(5); targets = [unit(torch.randn(d, device=DEV)) for _ in range(24)]; conc_atoms = ((img @ P) ** 2).sum(1) / (img ** 2).sum(1).clamp_min(1e-9); conc_rand = ((Rimg @ P) ** 2).sum(1) / (Rimg ** 2).sum(1).clamp_min(1e-9)
res = dict(K=S.K, prank_exact_1024=prank(Zat), prank_operator_all=prank(Zop), prank_random_operator=prank(Zrn), omp_atoms_needed=omp_needed(Zat, targets), omp_random_needed=omp_needed(unit(torch.randn(1024, d, device=DEV)), targets), omp_random_transported_needed=omp_needed(Zrn, targets), quotient_energy_fraction_atoms=conc_atoms.median().item(), quotient_energy_fraction_random=conc_rand.median().item(), operator_vs_exact_cos=((unit(Zop[atoms]) * unit(Zat)).sum(1)).median().item())
log(f"{tag} (K {S.K}): functional dictionary over block-2 atoms: effective rank {res['prank_exact_1024']:.1f} (1024 exact), {res['prank_operator_all']:.1f} (all, operator; random directions {res['prank_random_operator']:.1f}); atoms needed by matching pursuit to represent a random quotient point to 10% {res['omp_atoms_needed']:.0f} vs {res['omp_random_needed']:.0f} for random 16-vectors and {res['omp_random_transported_needed']:.0f} for transported random directions; fraction of transported energy inside the quotient: atoms {res['quotient_energy_fraction_atoms']:.3f} vs random directions {res['quotient_energy_fraction_random']:.3f}; operator vs exact coordinates cos {res['operator_vs_exact_cos']:.2f}")
record(f"e332_funcdict_{tag}", dict(model=tag, L=L, **res), f"prank exact {res['prank_exact_1024']:.1f} op {res['prank_operator_all']:.1f} rand {res['prank_random_operator']:.1f} | OMP atoms {res['omp_atoms_needed']:.0f} rand16 {res['omp_random_needed']:.0f} transported-rand {res['omp_random_transported_needed']:.0f} | quotient energy atoms {res['quotient_energy_fraction_atoms']:.3f} rand {res['quotient_energy_fraction_random']:.3f}")
