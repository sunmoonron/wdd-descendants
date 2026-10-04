"""e4: what the network does with an error. A perturbation is added to the state before layer 12 of loop t0 (Ouro,
t0 = 1..4, followed to the loop-8 exit) or before layer 6 or 12 (SmolLM2), at every position but the BOS, and its
size is followed at every recorded point downstream, relative to the clean state, with the excess loss at every exit.
Types, all at the norm of the WDD k=64 residual of that token (the error the stand-in introduces): wdd (that residual
itself), rot (the rotated dictionary's residual), iso (a random direction), atom (a random native atom), ownword (the
token's own first WDD atom), and wdd/iso at a tenth of the size."""
import sys; sys.path.insert(0, "/data/loopwdd/code")
from lw import *
kind = sys.argv[1]
m = LM(kind); C = torch.load(f"{ROOT}/cache/{kind}_e1.pt"); K2 = torch.load(f"{ROOT}/cache/{kind}_e2.pt"); d, L = m.d, m.L; Q = rotation(d)
ouro = kind == "ouro"; TLX = 8 if ouro else 1; NS = 8
ev = C["ids"][:NS].to(DEVM)
Aall, laball = m.atoms(range(L)); layer_all = laball["layer"].to(DEVD)
dict_for = lambda t, l: Aall if (t > 0 or l >= L) else Aall[layer_all < l]
REC = [(t, l) for t in range(TLX) for l in ((0, 4, 8, 12, 16, 20, 24) if ouro else range(L + 1))]
clean = {}
with torch.no_grad():
    hs = m.run(ev, loops=TLX, cb=lambda t, l, x: clean.__setitem__((t, l), x.cpu()) if (t, l) in REC else None)
    ce_clean = torch.stack([m.ce(h, ev).cpu() for h in hs])
INJ = [(t, 12) for t in range(4)] if ouro else [(0, 6), (0, 12)]
res = dict(kind=kind, ce_clean=ce_clean.mean((1, 2)).tolist(), inj={})
g = torch.Generator(device="cpu").manual_seed(3)
for (t0, l0) in INJ:
    A = dict_for(t0, l0); X0 = clean[(t0, l0)][:, 1:].reshape(-1, d).to(DEVD)
    if l0 == 12 and (t0, 12) in [(t, 12) for t in K2["codes"]]:
        cod = K2["codes"][t0]; mu = cod["mu"].to(DEVD); X = X0 - mu; n = X.shape[0]
        s = cod["sel"][:n].long().to(DEVD); c, _ = refit(X, A, s); R = recon(A, s, c) - X
        sr = cod["selr"][:n].long().to(DEVD); cr, _ = refit(X @ Q.T, A, sr); Rr = recon(A, sr, cr) @ Q - X
    else:
        mu = C["states"][(t0, l0)][:, 1:].reshape(-1, d).mean(0).to(DEVD) if (t0, l0) in C["states"] else X0.mean(0)
        X = X0 - mu
        with torch.no_grad():
            s, c, _ = omp(X, A, 64); R = recon(A, s, c) - X
            sr, cr, _ = omp(X @ Q.T, A, 64); Rr = recon(A, sr, cr) @ Q - X
    nr = R.norm(dim=-1, keepdim=True)
    iso = unit(torch.randn(X.shape, generator=g).to(DEVD)) * nr
    ai = torch.randint(0, A.shape[0], (X.shape[0],), generator=g).to(DEVD); sg = (torch.randint(0, 2, (X.shape[0], 1), generator=g).to(DEVD) * 2 - 1)
    atom = A[ai] * sg * nr
    c1, _ = refit(X, A, s[:, :1]); own = A[s[:, 0]] * torch.sign(c1) * nr
    P = dict(wdd=R, rot=unit(Rr) * nr, iso=iso, atom=atom, ownword=own, wdd_small=0.1 * R, iso_small=0.1 * iso)
    del A; torch.cuda.empty_cache()
    out = {}
    for nm, D in P.items():
        dv = {}
        def cb(t, l, x):
            if (t, l) in REC:
                xc = clean[(t, l)].to(x.device)
                dd = (x - xc)[:, 1:].norm(dim=-1); nc = xc[:, 1:].norm(dim=-1)
                dv[f"{t},{l}"] = dict(rel=(dd / nc).median().item(), rel_mean=(dd / nc).mean().item())
        x = clean[(t0, l0)].clone().to(DEVM); x[:, 1:] += D.view(NS, -1, d).to(DEVM)
        inj_rel = (D.norm(dim=-1) / X0.norm(dim=-1)).median().item()
        with torch.no_grad():
            hs = m.run_from(x, t0, l0, loops=TLX, cb=cb)
            ce = torch.stack([m.ce(h, ev).cpu() for h in hs])
        exc = (ce.mean((1, 2)) - ce_clean[t0:].mean((1, 2))).tolist()
        out[nm] = dict(inj_rel=inj_rel, dev=dv, excess_by_exit=exc)
        log(kind, f"inject {t0+1},{l0} {nm}: rel size {inj_rel:.3f}; rel dev at loop ends " +
            " ".join(f"{k}:{v['rel']:.3f}" for k, v in dv.items() if k.endswith(f",{L}") or (not ouro and k.endswith(',24'))) + f"; excess by exit {' '.join(f'{e:+.3f}' for e in exc)}")
    res["inj"][f"{t0},{l0}"] = out
    jdump(res, f"{ROOT}/results/e4_{kind}.json")
