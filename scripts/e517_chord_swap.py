"""e517: the counterfactual chord, the per-checkpoint half. e516b found that a non-word row's largest projection over
the calibrated Gumbel floor at a checkpoint predicts whether it is a word at the next (AUC 0.76-0.97), while its own
write almost never clears the floor: the projection that crosses carries the position's chord. That is a selection
criterion, not yet a mechanism. The test is counterfactual: at every position, destroy the co-writing that makes the
chord while keeping what a threshold on the row itself would use, recompute every row's largest projection over
the floor, and (in e517b) ask whether the prediction of future entry survives. If it collapses when the chord is
destroyed and stands when only the row's own write is removed, the chord is what carries rows across the floor.
Variants of the state at each position, all with the embeddings and everything outside the 64 largest MLP writes
left as they are: real; fake chord (the same 64 coefficients on random rows of the same blocks, rescaled to the
chord's norm, the candidate's own write kept where it is a writer); permuted chord (the same rows, their
coefficients permuted among them, the candidate's own coefficient restored); no chord (the 64 writes removed, the
candidate's own kept); and, on the real state, the cross-terms projection (the candidate's own write subtracted
where it is a writer). Per row: the largest projection over the floor under each variant, the largest real
projection and the floor at the position where it occurs, usage as a native word, mean absolute write.
Setup: Pythia-410m, blocks 6 and 12; 8 x 256 evaluation tokens, typical positions; the floor as in e503, e512, e516.
Arguments: name [revision]."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
BL = [6, 12]; LB = max(BL); K = 16; NW = 64; CDIR = f"/workspace/wdd/cache/e517_{name}"; os.makedirs(CDIR, exist_ok=True)
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
acts = {b: [] for b in range(LB + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)]
try: S_ = block_states(model, arch, ids, BL, chunk=4)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(LB + 1)}; del acts; rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
g = torch.Generator(device=DEV).manual_seed(0)
out = {}; res = dict(model=name, revision=rev, blocks=BL, by_block={})
for b in BL:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); m = Au.shape[0]
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    rows = gidx.reshape(-1); Am = Au[rows]; R = rows.numel()
    Xc = Xk - Xk.mean(0); sel, _, _ = omp(Xc, Au, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m)[rows].float()
    Cled = torch.cat([A_all[bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); mag = Cled.abs().mean(0); top = Cled.abs().topk(NW, dim=1); wrow = top.indices; wcoef = Cled.gather(1, wrow); del Cled
    watom = rows[wrow]; wblk = wrow // DFF
    chord = torch.einsum("nk,nkd->nd", wcoef, Au[watom]); cn = chord.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    f_rows = wblk * DFF + torch.randint(0, DFF, (N, NW), generator=g, device=DEV); fake_raw = torch.einsum("nk,nkd->nd", wcoef, Au[rows[f_rows]]); fs = cn / fake_raw.norm(dim=-1, keepdim=True).clamp_min(1e-9); fake = fake_raw * fs; del fake_raw
    perm = torch.argsort(torch.rand(N, NW, generator=g, device=DEV), dim=1); pcoef = wcoef.gather(1, perm); permuted = torch.einsum("nk,nkd->nd", pcoef, Au[watom])
    CA = Au.T @ Au / m; CR = Ar.T @ Ar / m; gm = gabs(m)
    def floor_of(Xv):
        Xcv = Xv - Xv.mean(0); Uv = unitr(Xcv); xnv = Xcv.norm(dim=-1).clamp_min(1e-9); s2 = ((Uv @ CA) * Uv).sum(1); return Uv, xnv, s2.clamp_min(1e-12).sqrt() * gm
    U0, xn0, L0 = floor_of(Xk); s2r = ((U0 @ CR) * U0).sum(1); Mr = torch.cat([(U0[s:s + 128] @ Ar.T).abs().max(1).values for s in range(0, N, 128)]); r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean())
    variants = {"real": (Xk, None), "fake": (Xk - chord + fake, "fake"), "permuted": (Xk - chord + permuted, "perm"), "nochord": (Xk - chord, "none")}
    q = {}
    fake_cos = (Au[rows[f_rows]] * Au[watom]).sum(-1)                                                                        # cosine of each fake row with the row it replaces
    for vn, (Xv, corr) in variants.items():
        Uv, xnv, Lv = floor_of(Xv); Lv = Lv * r_cal
        mx = torch.zeros(R, device=DEV); mxP = torch.zeros(R, device=DEV); fl = torch.zeros(R, device=DEV)
        if vn == "real": mxc = torch.zeros(R, device=DEV)
        for s in range(0, N, 128):
            P = Uv[s:s + 128] @ Am.T
            if corr == "fake": add = wcoef[s:s + 128] * (1 - fs[s:s + 128] * fake_cos[s:s + 128]) / xnv[s:s + 128, None]; P.scatter_add_(1, wrow[s:s + 128], add)
            elif corr == "perm": add = (wcoef[s:s + 128] - pcoef[s:s + 128]) / xnv[s:s + 128, None]; P.scatter_add_(1, wrow[s:s + 128], add)
            elif corr == "none": add = wcoef[s:s + 128] / xnv[s:s + 128, None]; P.scatter_add_(1, wrow[s:s + 128], add)
            Pa = P.abs(); ratio = Pa / Lv[s:s + 128, None]; v, a = ratio.max(0); better = v > mx; mx = torch.where(better, v, mx); mxP = torch.where(better, Pa.gather(0, a[None])[0], mxP); fl = torch.where(better, Lv[s:s + 128][a], fl)
            if vn == "real":
                Pc = P.clone(); Pc.scatter_add_(1, wrow[s:s + 128], -wcoef[s:s + 128] / xnv[s:s + 128, None]); mxc = torch.maximum(mxc, (Pc.abs() / Lv[s:s + 128, None]).max(0).values)
        q[vn] = dict(max_over_floor=mx.cpu(), max_projection=mxP.cpu(), floor_at_max=fl.cpu())
        if vn == "real": q["cross"] = dict(max_over_floor=mxc.cpu())
    out[b] = dict(usage=usage.cpu(), magnitude=mag.cpu(), variants=q)
    med = lambda v: float(q[v]["max_over_floor"].median()); over = lambda v: float((q[v]["max_over_floor"] > 1).float().mean())
    res["by_block"][b] = dict(n_rows=R, rotated_calibration=r_cal, median_floor=float((L0 * r_cal).median()), chord_norm_over_state=float((cn[:, 0] / xn0).median()), median_max_over_floor={v: med(v) for v in ("real", "fake", "permuted", "nochord", "cross")}, share_rows_over_floor={v: over(v) for v in ("real", "fake", "permuted", "nochord", "cross")})
    o = res["by_block"][b]; log(f"{name}{' ' + rev if rev else ''} block {b}: floor {o['median_floor']:.3f}, chord {o['chord_norm_over_state']:.2f} of the state; rows whose largest projection clears the floor: real {o['share_rows_over_floor']['real']:.3f}, fake chord {o['share_rows_over_floor']['fake']:.3f}, permuted {o['share_rows_over_floor']['permuted']:.3f}, no chord {o['share_rows_over_floor']['nochord']:.3f}, own write removed {o['share_rows_over_floor']['cross']:.3f}; median largest projection over the floor " + ", ".join(f"{v} {o['median_max_over_floor'][v]:.2f}" for v in o["median_max_over_floor"]))
    del Au, Ar, Am, chord, fake, permuted, CA, CR; torch.cuda.empty_cache()
torch.save(out, f"{CDIR}/{rev or 'main'}.pt")
summ = f"{name}{' ' + rev if rev else ''}: " + " | ".join(f"block {b}: rows over the floor real/fake/permuted/no chord/own removed {o['share_rows_over_floor']['real']:.3f}/{o['share_rows_over_floor']['fake']:.3f}/{o['share_rows_over_floor']['permuted']:.3f}/{o['share_rows_over_floor']['nochord']:.3f}/{o['share_rows_over_floor']['cross']:.3f}" for b, o in res["by_block"].items())
log(summ); record(f"e517_chordswap_{name}{'_' + rev if rev else ''}", res, summ)
