"""e398b: the control e398 lacked. A ridge-linear map fitted on paired residual states sends every translated atom into the
span of the target's states (its rows are combinations of the target's state directions), so a vocabulary pushed
through it may describe the target's states well whatever its words mean: a PCA effect, not a translation. Here the
same map is applied to a random rotation of the source vocabulary (control_lin) and to the target's own vocabulary
rotated at random (own_rot_lin: the target's own geometry, no provenance, through the target-to-target map fitted the
same way, i.e. the projection alone). Pre-registered: if control_lin reaches state_lin (within 0.05 at k 16), the
linear translation's advantage over the target's own words is the projection, and only the orthogonal translation
(e398 state_orth against control) measures shared words."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
MODELS["pythia410s1"] = ("EleutherAI/pythia-410m-seed1", "neox"); MODELS["pythia410s2"] = ("EleutherAI/pythia-410m-seed2", "neox")
NETS = ["pythia410", "pythia410s1", "pythia410s2"]; c = Cache("pythia410"); ev = c.s["eval_ids"][:3].to(DEV); fit = c.s["eval_ids"][3:11].to(DEV)
B, T = ev.shape; KS = [16, 64]; FITL = [3, 6, 9, 12]; V, S = {}, {}
def states(model, arch, ids, levels):
    out = {}; hs = [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: out.__setitem__(l_, (o[0] if isinstance(o, tuple) else o).detach().float()))(l)) for l in levels]
    try:
        with torch.no_grad(): lg = model(ids).logits.float()
    finally: [h.remove() for h in hs]
    return {l: out[l][:, 1:].reshape(-1, out[l].shape[-1]) for l in levels}, lg
for n in NETS:
    m, tok, fam = load_model(n); a = Arch(m, fam); L = a.NB // 2; A, lab = build_dictionary(a, blocks=list(range(L + 1))); V[n] = A
    st, _ = states(m, a, fit, FITL); S[n] = torch.cat([st[l] - st[l].mean(0, keepdim=True) for l in FITL]); del m, a
def ridge_map(Xa, Xb, lam=1e-3):
    G = (Xa.T @ Xa).double(); lam = lam * G.diagonal().mean(); return torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=DEV, dtype=torch.float64), (Xa.T @ Xb).double()).float()
res = dict(nets=NETS, level=L, k=KS, cells={})
for j in NETS:
    model, tok, fam = load_model(j); arch = Arch(model, fam); stj, lg = states(model, arch, ev, [L]); Lc = token_loss(lg, ev).mean().item()
    x = stj[L]; mu = x.mean(0, keepdim=True); Xc = x - mu
    def splice(Xh):
        def hk(m, i, o):
            xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, 1:] = Xh.to(xo.dtype).view(B, T - 1, -1)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hh = arch.layers[L].register_forward_hook(hk)
        try:
            with torch.no_grad(): return token_loss(model(ev).logits.float(), ev).mean().item()
        finally: hh.remove()
    Lm = splice(mu.expand(x.shape[0], -1)); gap = Lm - Lc
    def describe(A, name):
        A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-8); sel, _, _ = omp(Xc, A, max(KS), batch=256, record_err=False); cell = {}
        for k in KS:
            cof, err = refit(Xc, A, sel[:, :k]); Ls = splice(mu + torch.einsum("nk,nkd->nd", cof, A[sel[:, :k]])); cell[str(k)] = (Lm - Ls) / max(gap, 1e-9)
        res["cells"][name] = cell; log(f"{name}: k16 {cell['16']:.2f} k64 {cell['64']:.2f}"); del sel
    Wjj = ridge_map(S[j], S[j]); describe(V[j] @ Wjj, f"own_lin->{j}"); describe(rotate(V[j], seed=7) @ Wjj, f"own_rot_lin->{j}")
    for i in NETS:
        if i == j: continue
        W = ridge_map(S[i], S[j]); describe(V[i] @ W, f"state_lin:{i}->{j}"); describe(rotate(V[i], seed=7) @ W, f"control_lin:{i}->{j}"); describe(rotate(V[i], seed=11) @ W, f"control_lin2:{i}->{j}")
    del model, arch
def avg(pfx, k):
    v = [res["cells"][f"{pfx}:{i}->{j}"][str(k)] for j in NETS for i in NETS if i != j]; return sum(v) / len(v)
dg = lambda p, k: sum(res["cells"][f"{p}->{j}"][str(k)] for j in NETS) / 3
summ = " | ".join(f"k{k}: own_lin {dg('own_lin', k):.2f} own_rot_lin {dg('own_rot_lin', k):.2f} state_lin {avg('state_lin', k):.2f} control_lin {avg('control_lin', k):.2f} control_lin2 {avg('control_lin2', k):.2f}" for k in KS)
log("summary " + summ); record("e398b_linctrl_pythia410", res, summ)
