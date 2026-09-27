"""e524: the row space against the state cloud at every thousand steps (the records e524b reads). e522 and e523 found
that a future word's alignment rises because its row rotates toward the states and into the cloud's principal
subspace, and that the maximum often moves to another position. Those were endpoint measurements over intervals of
four to sixteen thousand steps. This run records, at each Pythia checkpoint from step 1000 to step 16000 in steps of
a thousand, everything the trajectory analysis needs: the unit rows of blocks 0-12 and their norms; per block (12 and
6) the centred unit states at every measurement position with the sink mask, the extreme-value floor at every
position, the top-32 principal directions of the kept states and their variance fractions, OMP usage (the word set),
and per row the largest projection over the floor with its position, the counts of positions over the floor and over
three quarters of it, and the mean squared projection over positions.
Setup: Pythia-410m; 8 x 256 measurement tokens (e519's); blocks 12 and 6; OMP with 16 atoms over the block's
dictionary as in e519. Arguments: name start-end (checkpoint steps, thousands, inclusive)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; lo, hi = (int(x) for x in sys.argv[2].split("-")); STEPS = list(range(lo * 1000, hi * 1000 + 1, 1000))
CDIR = f"/workspace/wdd/cache/e524_{name}"; os.makedirs(CDIR, exist_ok=True); BL = [12, 6]; K = 16; NPC = 32
idsB = eval_ids(name)[:8, :256].to(DEV)
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
for n in STEPS:
    out = f"{CDIR}/step{n}.pt"
    if os.path.exists(out): log(f"step{n} cached"); continue
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF
    for p in model.parameters(): p.requires_grad_(False)
    X0 = block_states(model, arch, idsB, BL, chunk=4); Wv = torch.cat([arch.wdir(bb) for bb in range(max(BL) + 1)]); rec = dict(step=n, rows=unitr(Wv).half().cpu(), norms=Wv.norm(dim=1).cpu(), blocks={})
    for b in BL:
        X = X0[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]; mu = Xk.mean(0); Xc_all = X - mu; U_all = unitr(Xc_all); U = U_all[keep]
        A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; m = Au.shape[0]
        typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
        for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
        rows = gidx.reshape(-1); Am = Au[rows]; R = rows.numel()
        CA = Au.T @ Au / m; CR = Ar.T @ Ar / m; gm = gabs(m)
        s2 = ((U_all @ CA) * U_all).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 128] @ Ar.T).abs().max(1).values for s in range(0, N, 128)])
        r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); L_all = s2.clamp_min(1e-12).sqrt() * gm * r_cal
        sel, _, _ = omp(Xk - mu, Au, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m)[rows].float()
        ev, V = torch.linalg.eigh(Xk.sub(mu).T @ Xk.sub(mu) / N); V = V.flip(1)[:, :NPC].contiguous(); ev = ev.flip(0); evfrac = (ev.cumsum(0) / ev.sum())[:128]
        P = U_all @ Am.T; ratio = P.abs() / L_all[:, None]; ratio[~keep] = 0; S, a = ratio.max(0); cnt = (ratio > 1).sum(0); cnt75 = (ratio > 0.75).sum(0); CU = U.T @ U / N; m2 = ((Am @ CU) * Am).sum(1) * D
        rec["blocks"][b] = dict(keep=keep.cpu(), U=U_all.half().cpu(), L=L_all.cpu(), V=V.cpu(), evfrac=evfrac.cpu(), r_cal=r_cal, usage=usage.cpu(), S=S.cpu(), a=a.cpu(), cnt=cnt.cpu(), cnt75=cnt75.cpu(), m2=m2.cpu(), n_rows=R)
        log(f"{name} step{n} block {b}: {N} kept positions, floor {float(L_all[keep].median()):.3f} (calibration {r_cal:.3f}); rows over the floor {float((S > 1).float().mean()):.3f}; words' median S {float(S[usage.argsort(descending=True)[:256]].median()):.2f}; top-32 variance {float(evfrac[31]):.2f}")
        del P, ratio, Au, Ar, Am, CA, CR; torch.cuda.empty_cache()
    torch.save(rec, out); del model, X0, Wv; torch.cuda.empty_cache()
summ = f"{name}: cached steps {STEPS[0]}-{STEPS[-1]} in {CDIR}"
log(summ); record(f"e524_cache_{name}_{lo}_{hi}", dict(model=name, steps=STEPS, cache=CDIR), summ)
