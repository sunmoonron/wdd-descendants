"""e689 (session 155): readable writes against the crowd, across width. Heimersheim's objection to the essay's squeeze
experiments: superposition needs high dimension, so a layer crushed to ten neurons cannot show it. The floor gives a
count that scales: at the middle block, per position, how many MLP write rows clear the extreme-value floor (readable by
projection competition), and how much of the resultant along the most prominent row is made by a crowd of small writes
(e521): the ledger's terms c_j (d_j . d*) for every row j of the blocks up to the middle, their effective number and the
share of the sum carried by the sixteen largest. Also the share of the centred state's norm carried by its three
largest writes. One model per run; the scaling is read across runs within a family (Pythia 70m-2.8b, GPT-2 small-xl,
SmolLM2 135M-1.7B, Qwen2.5 0.5B-7B). Arguments: --model M [--smoke]. Pre-registered in e689_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e689_{name}" + ("_smoke" if SMOKE else ""); dt = torch.bfloat16 if name in ("qwen7b",) else torch.float32; M = load(name, dtype=dt); NB, D, DFF = M["NB"], M["D"], M["DFF"]; B = NB // 2; T = 256
NWIN = 4 if SMOKE else 32; L_ = lambda s: log(TAG, s)
ids = windows_from(wikitext_ids(M, "test"), NWIN, T).to(DEV)
X, ACT = [], []
with Capture(M, list(range(B + 1)), ["act", "x"], stop_after=B) as C:
    for s0 in range(0, NWIN, 4):
        run_base(M, ids[s0:s0 + 4]); X.append(C.cap[("x", B)][:, 1:].float().reshape(-1, D)); ACT.append(torch.cat([C.cap[("act", b)][:, 1:].float().reshape(-1, DFF) for b in range(B + 1)], -1).half())
X, ACT = torch.cat(X), torch.cat(ACT); A, blk, neu, nr = rows_dictionary(M, B); m = A.shape[0]; keep = ~sinkmask(X); MU = X[keep].mean(0); Xc = X - MU; U = unitr(Xc); cal = floor_calibration(U[keep], A); Lf = floor_of(U, cal)
N = X.shape[0]; L_(f"block {B} of {NB}, d {D}, {m} rows, {N} positions ({int(keep.sum())} typical), r_cal {cal['r_cal']:.3f}")
OVER, TOPR, OWN, CROWD, NEFF, TOP3, NTERMS, CHORD = [], [], [], [], [], [], [], []
for s in range(0, N, 512):
    sl = slice(s, s + 512); pr = (U[sl] @ A.T).abs() / Lf[sl, None]; over = pr > 1; OVER.append(over.sum(1)); top = pr.argmax(1); TOPR.append(pr.max(1).values)
    c = ACT[sl].float() * nr[None]   # [n, m] ledger coefficients in write units (write = c_j d_j)
    dstar = A[top]; G = (A @ dstar.T).T   # [n, m] cosines of every row with the position's top row
    terms = c * G; tot = terms.sum(1); own = terms.gather(1, top[:, None])[:, 0]; OWN.append(own / tot.abs().clamp_min(1e-9) * torch.sign(tot))
    t16 = terms.abs().topk(16, dim=1).values.sum(1); CROWD.append(1 - t16 / terms.abs().sum(1).clamp_min(1e-9)); NEFF.append(terms.abs().sum(1) ** 2 / (terms ** 2).sum(1).clamp_min(1e-12)); NTERMS.append((c.abs() > 0).sum(1))
    top3 = c.abs().topk(3, dim=1); v3 = torch.einsum("nk,nkd->nd", c.gather(1, top3.indices), A[top3.indices]); TOP3.append(v3.norm(dim=1) / Xc[sl].norm(dim=1).clamp_min(1e-9))
    CHORD.append(((Xc[sl] * dstar).sum(1)) / Xc[sl].norm(dim=1).clamp_min(1e-9))
OVER, TOPR, OWN, CROWD, NEFF, TOP3, NTERMS, CHORD = [torch.cat(v)[keep] for v in (OVER, TOPR, OWN, CROWD, NEFF, TOP3, NTERMS, CHORD)]
S = dict(over_floor_count_median=float(OVER.float().median()), over_floor_count_mean=float(OVER.float().mean()), over_floor_per_dim=float(OVER.float().median()) / D, over_floor_per_row=float(OVER.float().median()) / m, top_ratio_median=float(TOPR.median()),
         own_share_of_resultant_median=float(OWN.median()), crowd_share_median=float(CROWD.median()), neff_terms_median=float(NEFF.median()), active_terms_median=float(NTERMS.float().median()), top3_norm_share_median=float(TOP3.median()), top_row_projection_share_median=float(CHORD.median()))
res = dict(model=M["hf"], key=name, block=B, n_blocks=NB, d=D, DFF=DFF, rows=m, positions=int(keep.sum()), r_cal=cal["r_cal"], dtype=str(dt), summary=S)
summ = f"[{name} d {D}, block {B}] rows over the floor per position, median {S['over_floor_count_median']:.0f} ({S['over_floor_per_dim']:.3f} per dimension, {S['over_floor_per_row']:.4f} per row); the top row's own share of the resultant {S['own_share_of_resultant_median']:.2f}; crowd share (beyond the 16 largest terms) {S['crowd_share_median']:.2f}; effective number of terms {S['neff_terms_median']:.0f} of {S['active_terms_median']:.0f} active; three largest writes carry {S['top3_norm_share_median']:.2f} of the centred norm"
save(TAG, res, summ)
