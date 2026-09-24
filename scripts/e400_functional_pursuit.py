"""e400: functional description length. So far words were chosen by Euclidean fit to the state (OMP) and only judged by
function (the splice), so the self-description length measured the function kept by a variance-chasing description.
Here the words are chosen under the network's own functional metric: the Fisher information of its next-token
distribution with respect to the middle-depth state, G = E[g g^T] with g the gradient of the log-likelihood of a token
sampled from the model's own prediction (8 sequences disjoint from the 8 described; two samples per position),
normalised to mean eigenvalue 1 with a 1% ridge; OMP then runs on x G^(1/2) over atoms a G^(1/2) (renormalised), and
the reconstruction is taken back to the residual coordinates. Pythia-410m at the end of training (argument: revision).
Vocabularies: own, two rotations, the step-4000 words (false friends, e397) and their rotation, the step-1000 words,
and Gaussian words with the states' covariance (e399 covX). Both metrics on the same states.
Pre-registered: (1) the functional pursuit shortens the own self-description (90% of the loss at fewer than the 64
words Euclidean OMP needs); (2) the own-minus-rotation gap at k 16 persists (at least half its Euclidean size), so the
own words' advantage is not an artifact of chasing variance; (3) if false friends come from chasing high-variance,
function-light directions, the step-4000 words are no longer below the mean state at k 4-8 under the Fisher metric."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
rev = sys.argv[1] if len(sys.argv) > 1 else "step143000"
c = Cache("pythia410"); ev = c.s["eval_ids"][:8].to(DEV); fit = c.s["eval_ids"][8:16].to(DEV); KS = [4, 8, 16, 32, 64]
V = {}
for r in ["step1000", "step4000"]:
    m, tok, fam = load_model("pythia410", revision=r); a = Arch(m, fam); L = a.NB // 2; V[r], _ = build_dictionary(a, blocks=list(range(L + 1))); del m, a
model, tok, fam = load_model("pythia410", revision=rev); arch = Arch(model, fam); L = arch.NB // 2; own, _ = build_dictionary(arch, blocks=list(range(L + 1)))
for p in model.parameters(): p.requires_grad_(False)
# Fisher metric at the output of block L, from the model's own predictive distribution on the fit sequences
torch.set_grad_enabled(True); G = torch.zeros(arch.D, arch.D, device=DEV, dtype=torch.float64); n = 0; gen = torch.Generator(device=DEV).manual_seed(0)
for s0 in range(0, fit.shape[0], 2):
    ids = fit[s0:s0 + 2]; leaf = {}
    def hk(m, i, o):
        xo = o[0] if isinstance(o, tuple) else o; y = xo.detach().requires_grad_(True); leaf["x"] = y
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    for smp in range(2):
        h = arch.layers[L].register_forward_hook(hk)
        try: lg = model(ids).logits.float()
        finally: h.remove()
        lp = torch.log_softmax(lg[:, :-1], -1)
        with torch.no_grad(): y = torch.multinomial(lp.exp().reshape(-1, lp.shape[-1]), 1, generator=gen).view(lp.shape[0], -1)
        (-lp.gather(2, y[..., None]).sum()).backward()
        gx = leaf["x"].grad[:, 1:].reshape(-1, arch.D).double(); G += gx.T @ gx; n += gx.shape[0]; del lg, lp, gx, leaf["x"]
torch.set_grad_enabled(False); G = (G / n).float()
evG, UG = torch.linalg.eigh(G.double()); pr = (evG.sum() ** 2 / (evG ** 2).sum()).item()
M = G / (G.trace() / arch.D) + 0.01 * torch.eye(arch.D, device=DEV); evM, UM = torch.linalg.eigh(M.double()); S = ((UM * evM.clamp_min(0).sqrt()) @ UM.T).float()
lv = Level(model, arch, ev, L); fl = Level(model, arch, fit, L); SX = (fl.Xc.T @ fl.Xc) / fl.Xc.shape[0]; del fl
evX = torch.linalg.eigvalsh(SX.double()); topX = torch.linalg.eigh(SX.double())[1][:, -8:].float()
res = dict(rev=rev, level=L, k=KS, gap=lv.gap, fisher_pr=pr, fisher_top_share=(evG[-8:].sum() / evG.sum()).item(),
           fisher_weight_on_top8_state_pcs=((topX.T @ G @ topX).trace() / G.trace()).item(), state_top8_var_share=(evX[-8:].sum() / evX.sum()).item(), cells={})
log(f"{rev} Fisher PR {pr:.1f}, top-8 eigen share {res['fisher_top_share']:.2f}; the states' top-8 PCs hold {res['state_top8_var_share']:.2f} of the variance and {res['fisher_weight_on_top8_state_pcs']:.3f} of the Fisher trace")
vocab = dict(own=own, rot7=rotate(own, seed=7), rot11=rotate(own, seed=11), f4000=V["step4000"], f4000_rot=rotate(V["step4000"], seed=7), v1000=V["step1000"],
             covX=gauss_like(own.shape[0], SX, seed=2))
for nm, A in vocab.items():
    for metric, Sm in [("euc", None), ("fisher", S)]:
        res["cells"][f"{nm}:{metric}"] = describe(lv, A, KS, S=Sm); log(f"{rev} {nm} {metric}: rec k4..64 {fmt(res['cells'][f'{nm}:{metric}'], KS)}")
def k90(cell):
    for k in KS:
        if cell[str(k)]["rec"] >= 0.9: return k
    return None
res["k90"] = {k_: k90(v) for k_, v in res["cells"].items()}
g = lambda n: res["cells"][n]["16"]["rec"]
summ = (f"{rev} k16 euc/fisher: own {g('own:euc'):.2f}/{g('own:fisher'):.2f} rot7 {g('rot7:euc'):.2f}/{g('rot7:fisher'):.2f} rot11 {g('rot11:euc'):.2f}/{g('rot11:fisher'):.2f} "
        f"f4000 {g('f4000:euc'):.2f}/{g('f4000:fisher'):.2f} f4000_rot {g('f4000_rot:euc'):.2f}/{g('f4000_rot:fisher'):.2f} covX {g('covX:euc'):.2f}/{g('covX:fisher'):.2f} | "
        f"k90 own {res['k90']['own:euc']}/{res['k90']['own:fisher']} rot7 {res['k90']['rot7:euc']}/{res['k90']['rot7:fisher']} | f4000 k4,k8 fisher "
        f"{res['cells']['f4000:fisher']['4']['rec']:.2f},{res['cells']['f4000:fisher']['8']['rec']:.2f}")
log(summ); record(f"e400_funcpursuit_pythia410_{rev}", res, summ)
