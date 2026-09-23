"""e389: why does WDD beat a rotated copy of its own dictionary at small k (e388)? The state contains the current token's
embedding almost verbatim, so the advantage might be only the embedding atoms. At the middle depth, the same splice test
(loss recovered and FVU for k = 8, 16, 32, 64) with sub-dictionaries of the model's own atoms: all atoms; embeddings
only (token and position rows); writes only (MLP rows, head write bases and biases, no embeddings); MLP rows only;
head write bases only; and the full dictionary rotated (the e388 control)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NB = arch.NB; torch.manual_seed(0)
ev = c.s["eval_ids"][:3].to(DEV); KS = [8, 16, 32, 64]; L = NB // 2
def states(ids):
    out = {}
    h = arch.layers[L].register_forward_hook(lambda m, i, o: out.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    try:
        with torch.no_grad(): lg = model(ids).logits.float()
    finally: h.remove()
    return out["x"], lg
def loss_with(Xnew):
    def hk(m, i, o):
        x = o[0] if isinstance(o, tuple) else o; y = x.clone(); y[:, 1:] = Xnew.to(x.dtype).view(x.shape[0], x.shape[1] - 1, -1)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): return token_loss(model(ev).logits.float(), ev).mean().item()
    finally: h.remove()
Xe, lg = states(ev); Lc = token_loss(lg, ev).mean().item(); D = Xe.shape[-1]; Xp = Xe[:, 1:].reshape(-1, D); mu = c.s["mu"][L + 1].to(DEV).float(); Xc = Xp - mu[None]; Lm = loss_with(mu[None].expand(Xp.shape[0], -1))
A, lab = c.dictionary(L); typ = lab["type"].to(DEV)
subs = dict(all=torch.ones_like(typ, dtype=torch.bool), embeddings=(typ == T_TOK) | (typ == T_POS), writes=(typ == T_MLP) | (typ == T_ATT) | (typ == T_BIAS), mlp=(typ == T_MLP), heads=(typ == T_ATT))
res = dict(model=tag, level=L, clean=Lc, mean_ablate=Lm, sub={})
for nm, m_ in list(subs.items()) + [("rotated_all", None)]:
    Dct = rotate(A) if m_ is None else A[m_]
    if Dct.shape[0] < max(KS): continue
    sel, _, _ = omp(Xc, Dct, max(KS), batch=256, record_err=False); row = dict(n_atoms=int(Dct.shape[0]))
    for k in KS:
        cof, err = refit(Xc, Dct, sel[:, :k]); Xh = mu[None] + torch.einsum("nk,nkd->nd", cof, Dct[sel[:, :k]]); Ls = loss_with(Xh)
        row[str(k)] = dict(fvu=(err.sum() / Xc.pow(2).sum()).item(), recovered=(Lm - Ls) / max(Lm - Lc, 1e-9))
    res["sub"][nm] = row; del sel
log(f"{tag} L{L}: loss recovered (FVU) by sub-dictionary and k: " + " | ".join(f"{nm} ({r['n_atoms']} atoms): " + " ".join(f"k{k} {r[str(k)]['recovered']:.2f} ({r[str(k)]['fvu']:.2f})" for k in KS) for nm, r in res["sub"].items()))
record(f"e389_whichatoms_{tag}", res, " | ".join(f"{nm}: k8 {r['8']['recovered']:.2f} k32 {r['32']['recovered']:.2f}" for nm, r in res["sub"].items()))
