"""e396: private languages. Each checkpoint's vocabulary (its token embeddings, MLP write rows and head output bases up to
the middle depth) is used to describe every checkpoint's middle-depth state (k-sparse OMP, splice, loss recovered in the
describing target's own forward pass, as e388), for Pythia-410m at steps 1000, 4000, 16000, 33000, 63000 and 143000.
The diagonal is self-description; off the diagonal, how intelligible one stage's internal language is in another's
words. Also the cosine between the two vocabularies' MLP rows (the implementation drift of e360)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
revs = ["step1000", "step4000", "step16000", "step33000", "step63000", "step143000"]; c = Cache("pythia410"); ev = c.s["eval_ids"][:3].to(DEV); B, T = ev.shape; KS = [16, 64]
V, Wm = {}, {}
for r in revs:
    m, tok, fam = load_model("pythia410", revision=r); a = Arch(m, fam); L = a.NB // 2; A, lab = build_dictionary(a, blocks=list(range(L + 1))); V[r] = A; Wm[r] = torch.cat([a.wdir(b) for b in range(L + 1)]).to(DEV); del m, a
res = dict(revs=revs, level=L, k=KS, matrix={str(k): {} for k in KS}, row_cos={})
for j in revs:
    model, tok, fam = load_model("pythia410", revision=j); arch = Arch(model, fam); out = {}
    h = arch.layers[L].register_forward_hook(lambda m, i, o: out.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    with torch.no_grad(): Lc = token_loss(model(ev).logits.float(), ev).mean().item()
    h.remove(); x = out["x"][:, 1:].reshape(-1, out["x"].shape[-1]); mu = x.mean(0, keepdim=True); Xc = x - mu
    def splice(Xh):
        def hk(m, i, o):
            xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, 1:] = Xh.to(xo.dtype).view(B, T - 1, -1)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hh = arch.layers[L].register_forward_hook(hk)
        try:
            with torch.no_grad(): return token_loss(model(ev).logits.float(), ev).mean().item()
        finally: hh.remove()
    Lm = splice(mu.expand(x.shape[0], -1))
    for i in revs:
        sel, _, _ = omp(Xc, V[i], max(KS), batch=256, record_err=False)
        for k in KS:
            cof, err = refit(Xc, V[i], sel[:, :k]); Ls = splice(mu + torch.einsum("nk,nkd->nd", cof, V[i][sel[:, :k]])); res["matrix"][str(k)][f"{i}->{j}"] = (Lm - Ls) / max(Lm - Lc, 1e-9)
        ui, uj = Wm[i] / Wm[i].norm(dim=-1, keepdim=True).clamp_min(1e-9), Wm[j] / Wm[j].norm(dim=-1, keepdim=True).clamp_min(1e-9); res["row_cos"][f"{i}->{j}"] = (ui * uj).sum(-1).median().item()
        del sel
    del model, arch
short = lambda r: r.replace("step", "")
for k in KS:
    log(f"pythia410 L{L} mutual intelligibility, k {k} (rows: vocabulary of; columns: states of {' '.join(short(r) for r in revs)}): " + " | ".join(f"{short(i)}: " + " ".join(f"{res['matrix'][str(k)][f'{i}->{j}']:.2f}" for j in revs) for i in revs))
record("e396_intelligibility_pythia410", res, " | ".join(f"k{k} diag " + " ".join(f"{res['matrix'][str(k)][f'{r}->{r}']:.2f}" for r in revs) for k in KS))
