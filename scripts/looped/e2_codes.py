"""e2: WDD codes of the state before layer 12 (the middle of the stack) in every loop. Dictionary: the atoms that can
have written the state (loop 1: embeddings and layers 0-11; later loops: embeddings and all 24 layers, ignoring the
between-loop tilt). Per loop: OMP to 64 atoms over the native dictionary and over a rotated copy (same Gram matrix),
one-shot ranking (top-k by |cosine|, refit), FVU at k = 8, 16, 32, 64. Also how far the between-loop norm's gain
tilts each atom (cos(g^a * atom, atom) for a = 1, 2, 3)."""
import sys; sys.path.insert(0, "/data/loopwdd/code")
from lw import *
kind = sys.argv[1]; SMOKE = "--smoke" in sys.argv
m = LM(kind); C = torch.load(f"{ROOT}/cache/{kind}_e1.pt"); loops = C["loops"]; KS = (8, 16, 32, 64); d = m.d
Q = rotation(d)
res = dict(loops={}); codes = {}
Aall, laball = m.atoms(range(m.L)); A12, lab12 = m.atoms(range(12))
if kind == "ouro":
    g = m.nf.to(DEVD); tl = {}
    for a in (1, 2, 3):
        cs = (unit(Aall * g[None] ** a) * Aall).sum(-1)
        tl[a] = {nm: dict(median=cs[laball["type"].to(DEVD) == ty].median().item(), p05=cs[laball["type"].to(DEVD) == ty].quantile(0.05).item())
                 for nm, ty in (("emb", T_EMB), ("mlp", T_MLP), ("head", T_ATT))}
    res["tilt_cos"] = tl; res["gf_stats"] = dict(mean=g.mean().item(), std=g.std().item(), min=g.min().item(), max=g.max().item())
    log("tilt", tl, res["gf_stats"])
for t in (range(2) if SMOKE else range(loops)):
    A, lab = (A12, lab12) if t == 0 else (Aall, laball)
    mu = C["fit_states"][(t, 12)][:, 1:].reshape(-1, d).mean(0)
    Xr = C["states"][(t, 12)][:, 1:].reshape(-1, d)
    if SMOKE: Xr = Xr[:512]
    X = (Xr - mu).to(DEVD); nrm = Xr.norm(dim=-1); typ = (nrm <= 10 * nrm.median()).to(DEVD); E = X.pow(2).sum(-1)
    fv = lambda err, msk=None: (err[msk].sum() / E[msk].sum()).item() if msk is not None else (err.sum() / E.sum()).item()
    row = dict(n=X.shape[0], n_atoms=A.shape[0], massive=int((~typ).sum()))
    with torch.no_grad():
        sel, cof, err = omp(X, A, 64)
        selr, cofr, errr = omp(X @ Q.T, A, 64)
        so = (X @ A.T).abs().topk(64, dim=1).indices if X.shape[0] <= 4096 else torch.cat([(X[s:s + 1024] @ A.T).abs().topk(64, dim=1).indices for s in range(0, X.shape[0], 1024)])
        for k in KS:
            _, eo = refit(X, A, so[:, :k])
            row[f"k{k}"] = dict(native=fv(err[:, k - 1]), rotated=fv(errr[:, k - 1]), oneshot=fv(eo),
                                native_typ=fv(err[:, k - 1], typ), rotated_typ=fv(errr[:, k - 1], typ), oneshot_typ=fv(eo, typ))
    ty = lab["type"].to(DEVD)[sel[:, :16]]; ly = lab["layer"].to(DEVD)[sel[:, :16]]
    row["k16_share"] = dict(emb=(ty == T_EMB).float().mean().item(), mlp=(ty == T_MLP).float().mean().item(), head=(ty == T_ATT).float().mean().item(),
                            later_layers=((ly >= 12)).float().mean().item())
    res["loops"][t] = row
    codes[t] = dict(sel=sel.int().cpu(), cof=cof.half().cpu(), err=err.cpu(), selr=selr.int().cpu(), cofr=cofr.half().cpu(), errr=errr.cpu(),
                    oneshot=so.int().cpu(), mu=mu, typ=typ.cpu())
    log(kind, f"loop {t+1}: atoms {A.shape[0]}, FVU k16 native {row['k16']['native']:.3f} rot {row['k16']['rotated']:.3f} one-shot {row['k16']['oneshot']:.3f}; "
        f"k64 native {row['k64']['native']:.3f} rot {row['k64']['rotated']:.3f} one-shot {row['k64']['oneshot']:.3f}; shares {row['k16_share']}")
    del X, sel, cof, err, selr, cofr, errr, so; torch.cuda.empty_cache()
if SMOKE: sys.exit(0)
torch.save(dict(codes=codes, lab_all=laball, lab12=lab12), f"{ROOT}/cache/{kind}_e2.pt")
jdump(res, f"{ROOT}/results/e2_{kind}.json")
