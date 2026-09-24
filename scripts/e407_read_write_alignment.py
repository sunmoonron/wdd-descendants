"""e407: read alignment against write alignment. e399 found that the input directions of the downstream MLPs (the readers
of the state) also describe the state better than their rotation, by less than the writers do. Is the readers'
advantage an accent (second-order) or a vocabulary (individual rows), and does it add to the writers' or repeat it?
Pythia-410m (argument: revision), middle depth, 8 sequences, MLP rows only: the writers of blocks 0..L (W) and the
downstream readers of blocks L+1.. (R, gain-scaled and centred), each as is, rotated, as Gaussian words with the
family's own second moment (covA) and as signed sums of 8 rows (mix8), and their union W+R and its rotation.
Pre-registered: at the end of training the readers' advantage is word-level (covA and mix8 at the rotation level), as
the writers' is; no prediction on whether the union adds to the writers."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
rev = sys.argv[1] if len(sys.argv) > 1 else "step143000"
c = Cache("pythia410"); ev = c.s["eval_ids"][:8].to(DEV); KS = [4, 8, 16, 32, 64]
model, tok, fam = load_model("pythia410", revision=rev); arch = Arch(model, fam); L = arch.NB // 2
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); W = A[lab["type"].to(DEV) == T_MLP].clone(); del A
R = reader_rows(arch, list(range(L + 1, arch.NB))); lv = Level(model, arch, ev, L)
res = dict(rev=rev, level=L, k=KS, gap=lv.gap, n_w=W.shape[0], n_r=R.shape[0], cells={})
def run(nm, V): res["cells"][nm] = describe(lv, V, KS); log(f"{rev} {nm}: rec k4..64 {fmt(res['cells'][nm], KS)}")
allidx = lambda V: [torch.arange(V.shape[0], device=DEV)]
for tag, V in [("W", W), ("R", R)]:
    run(tag, V); run(f"{tag}_rot", rotate(V, seed=7)); run(f"{tag}_covA", gauss_like(V.shape[0], (V.T @ V) / V.shape[0], seed=1)); run(f"{tag}_mix8", mixtures(V, allidx(V), m=8, seed=4))
U = torch.cat([W, R]); run("WR", U); run("WR_rot", rotate(U, seed=7))
g = lambda n, k=16: res["cells"][n][str(k)]["rec"]
sh = lambda t, ctl, k: (g(f"{t}_{ctl}", k) - g(f"{t}_rot", k)) / max(g(t, k) - g(f"{t}_rot", k), 1e-9)
res["shares"] = {f"{t}_{ctl}_k{k}": sh(t, ctl, k) for t in ("W", "R") for ctl in ("covA", "mix8") for k in (4, 16)}
summ = (f"{rev} k16: writers {g('W'):.2f} (rot {g('W_rot'):.2f}, covA {g('W_covA'):.2f}, mix8 {g('W_mix8'):.2f}) readers {g('R'):.2f} (rot {g('R_rot'):.2f}, covA {g('R_covA'):.2f}, mix8 {g('R_mix8'):.2f}) "
        f"union {g('WR'):.2f} (rot {g('WR_rot'):.2f}) | k4: W {g('W', 4):.2f} R {g('R', 4):.2f} WR {g('WR', 4):.2f} | second-order shares k16: W covA {res['shares']['W_covA_k16']:.2f} mix8 {res['shares']['W_mix8_k16']:.2f}, "
        f"R covA {res['shares']['R_covA_k16']:.2f} mix8 {res['shares']['R_mix8_k16']:.2f}")
log(summ); record(f"e407_readwrite_pythia410_{rev}", res, summ)
