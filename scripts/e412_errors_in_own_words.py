"""e412: do the network's errors speak its language? Everything so far described the forward state. Here the backward
signal: the gradient of the next-token loss (true labels, summed over positions) with respect to the middle-depth
state at each position. It is the direction in which learning would push the state and, weighted by each neuron's
activation, the direction in which one step moves that neuron's write row, so the write rows are, over training, sums
of such gradients. Pythia-410m (argument: revision), 8 sequences. Each per-position gradient and each centred state is
normalised to unit length (every position counts equally) and described by OMP (k 4-64, fraction of the direction's
energy unexplained), with the writers of blocks 0..L (MLP rows), the downstream readers (MLP input rows of the later
blocks, gain-scaled and centred), the full own vocabulary, rotations of each, and Gaussian words with the target's own
second moment (from 8 other sequences; a data-derived control). Pre-registered: (1) trained, the writers describe the
gradients better than their rotation; (2) the readers describe the gradients better than the writers do, the reverse of
the forward states (the gradient reaches the state through the readers); (3) at step 0 own and rotated agree on both."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
rev = sys.argv[1] if len(sys.argv) > 1 else "step143000"
c = Cache("pythia410"); ev = c.s["eval_ids"][:8].to(DEV); fit = c.s["eval_ids"][8:16].to(DEV); KS = [4, 8, 16, 32, 64]
model, tok, fam = load_model("pythia410", revision=rev); arch = Arch(model, fam); L = arch.NB // 2
for p in model.parameters(): p.requires_grad_(False)
def grads_and_states(ids):
    gs, xs = [], []
    torch.set_grad_enabled(True)
    try:
        for s0 in range(0, ids.shape[0], 2):
            x_ids = ids[s0:s0 + 2]; leaf = {}
            def hk(m, i, o):
                xo = o[0] if isinstance(o, tuple) else o; y = xo.detach().requires_grad_(True); leaf["x"] = y
                return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
            h = arch.layers[L].register_forward_hook(hk)
            try: lg = model(x_ids).logits.float()
            finally: h.remove()
            token_loss(lg, x_ids).sum().backward()
            gs.append(leaf["x"].grad[:, 1:-1].reshape(-1, arch.D).detach()); xs.append(leaf["x"][:, 1:-1].reshape(-1, arch.D).detach()); del lg   # the last position is not scored: zero gradient
    finally: torch.set_grad_enabled(False)
    return torch.cat(gs), torch.cat(xs)
Ge, Xe = grads_and_states(ev); Gf, Xf = grads_and_states(fit)
ke, kf = Ge.norm(dim=-1) > 0, Gf.norm(dim=-1) > 0; Ge, Xe, Gf, Xf = Ge[ke], Xe[ke], Gf[kf], Xf[kf]
Xe = Xe - Xf.mean(0, keepdim=True); Xf = Xf - Xf.mean(0, keepdim=True)
T = dict(grad=unitr(Ge), state=unitr(Xe)); C2 = dict(grad=(unitr(Gf).T @ unitr(Gf)) / Gf.shape[0], state=(unitr(Xf).T @ unitr(Xf)) / Xf.shape[0])
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); W = A[lab["type"].to(DEV) == T_MLP].clone(); R = reader_rows(arch, list(range(L + 1, arch.NB)))
def fvu_curve(X, V):
    V = unitr(V); sel, _, _ = omp(X, V, max(KS), batch=256, record_err=False); out = {}
    for k in KS:
        cof, err = refit(X, V, sel[:, :k]); out[str(k)] = (err / X.pow(2).sum(-1)).mean().item()
    del sel; return out
res = dict(rev=rev, level=L, k=KS, fvu={})
for tg, X in T.items():
    for nm, V in [("W", W), ("W_rot", rotate(W, seed=7)), ("R", R), ("R_rot", rotate(R, seed=7)), ("own", A), ("own_rot", rotate(A, seed=7)), ("cov", gauss_like(A.shape[0], C2[tg], seed=2))]:
        res["fvu"][f"{tg}:{nm}"] = fvu_curve(X, V); log(f"{rev} {tg} {nm}: FVU k4..64 " + " ".join(f"{res['fvu'][f'{tg}:{nm}'][str(k)]:.3f}" for k in KS))
f = lambda n, k=16: res["fvu"][n][str(k)]
summ = (f"{rev} FVU k16 (lower is better) grad: W {f('grad:W'):.3f} (rot {f('grad:W_rot'):.3f}) R {f('grad:R'):.3f} (rot {f('grad:R_rot'):.3f}) own {f('grad:own'):.3f} (rot {f('grad:own_rot'):.3f}) cov {f('grad:cov'):.3f} | "
        f"state: W {f('state:W'):.3f} (rot {f('state:W_rot'):.3f}) R {f('state:R'):.3f} (rot {f('state:R_rot'):.3f}) own {f('state:own'):.3f} (rot {f('state:own_rot'):.3f}) cov {f('state:cov'):.3f}")
log(summ); record(f"e412_errors_pythia410_{rev}", res, summ)
