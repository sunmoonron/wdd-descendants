"""e399: what makes a network's own words good at describing it? The rotated vocabulary (e395) keeps the vocabulary's size,
norms and whole Gram matrix and discards only its orientation to the states, so it answers "is it orientation", not "is
it more than second-order alignment". Controls, Pythia-410m at the middle depth (argument: revision), 8 sequences:
 own, and three rotations (seeds 7, 11, 13; the null's spread);
 covA: Gaussian words with the own vocabulary's second moment (same anisotropy and effective rank, no individual words);
 covX, covX_sqrt: Gaussian words with the states' covariance (fitted on 8 other sequences) or its square root, i.e. words
   placed where the states have variance;
 mix8: each word replaced by a random signed sum of 8 words of its own family (same span, no individual words);
 MLP rows only: the writers of blocks 0..L (w_own) against their rotation, the downstream readers (MLP input rows of
   blocks L+1.., scaled by the norm's gain and centred: r_down) against their rotation, and the readers of the writing
   blocks themselves (r_up). This asks whether the advantage belongs to the writers or to any operator that touches
   the state (a privileged coordinate system shared by readers and writers).
Pre-registered: covA at the rotation level (the own vocabulary's second moment is close to isotropic, e397); covX lower
in FVU than own at small k but below own in loss recovered at k 16 (variance is not function); mix8 at the rotation
level (individual words matter); w_own well above w_rot at every k. No prediction for the readers."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
rev = sys.argv[1] if len(sys.argv) > 1 else "step143000"
c = Cache("pythia410"); ev = c.s["eval_ids"][:8].to(DEV); fit = c.s["eval_ids"][8:16].to(DEV); KS = [4, 8, 16, 32, 64]
model, tok, fam = load_model("pythia410", revision=rev); arch = Arch(model, fam); L = arch.NB // 2
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV)
lv = Level(model, arch, ev, L); fl = Level(model, arch, fit, L); SX = (fl.Xc.T @ fl.Xc) / fl.Xc.shape[0]; del fl
res = dict(rev=rev, level=L, k=KS, gap=lv.gap, n_seq=8, cells={})
def run(name, V):
    res["cells"][name] = describe(lv, V, KS); log(f"{rev} {name}: rec k4..64 {fmt(res['cells'][name], KS)}")
run("own", A)
for sd in [7, 11, 13]: run(f"rot{sd}", rotate(A, seed=sd))
run("covA", gauss_like(A.shape[0], (A.T @ A) / A.shape[0], seed=1))
run("covX", gauss_like(A.shape[0], SX, seed=2)); run("covX_sqrt", gauss_like(A.shape[0], SX, seed=3, power=0.25))
run("mix8", mixtures(A, [torch.nonzero(typ == t)[:, 0] for t in (T_TOK, T_MLP, T_ATT, T_BIAS) if (typ == t).any()], m=8, seed=4))
W = A[typ == T_MLP]; run("w_own", W); run("w_rot", rotate(W, seed=7))
R = reader_rows(arch, list(range(L + 1, arch.NB))); run("r_down", R); run("r_down_rot", rotate(R, seed=7))
run("r_up", reader_rows(arch, list(range(L + 1))))
ev_, _ = torch.linalg.eigh(SX.double()); res["state_cov_pr"] = (ev_.sum() ** 2 / (ev_ ** 2).sum()).item()
g = lambda n: res["cells"][n]["16"]["rec"]; rots = [g(f"rot{s}") for s in (7, 11, 13)]
summ = (f"{rev} k16: own {g('own'):.2f} rot {min(rots):.2f}-{max(rots):.2f} covA {g('covA'):.2f} covX {g('covX'):.2f} covX_sqrt {g('covX_sqrt'):.2f} "
        f"mix8 {g('mix8'):.2f} | MLP: writers {g('w_own'):.2f} (rot {g('w_rot'):.2f}) downstream readers {g('r_down'):.2f} (rot {g('r_down_rot'):.2f}) upstream readers {g('r_up'):.2f}")
log(summ); record(f"e399_ownwords_pythia410_{rev}", res, summ)
