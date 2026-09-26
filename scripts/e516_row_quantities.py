"""e516: what governs entry into the vocabulary, the per-checkpoint half. e515b found the word set turning over at
the churn rate of the large rows under a boundary that is never a magnitude threshold. The question is what the
boundary is a threshold on. This run records, for every MLP row of blocks up to 6 and 12 at one checkpoint, the
candidate quantities, all computable at that checkpoint: usage as a native word (16-word OMP); mean absolute write;
activity; the excess kurtosis of the write coefficient across positions; the write's prominence when the row is
active (its absolute write over the centred state norm, mean over active positions); the share of positions at
which the row's own write clears the position's calibrated Gumbel floor (the extreme-value model's writer
criterion); the share of positions at which the row's atom projection clears the floor, and the row's largest
projection over the floor (the model's wordhood criterion, own coefficient and cross terms together). e516b then
predicts, from the quantities at t, which rows enter the word set at later checkpoints.
Setup: Pythia-410m, blocks 6 and 12; 8 x 256 evaluation tokens, typical positions; the floor as in e503 and e512.
Arguments: name [revision]."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
BL = [6, 12]; LB = max(BL); K = 16; NW = 64; CDIR = f"/workspace/wdd/cache/e516_{name}"; os.makedirs(CDIR, exist_ok=True)
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; ids = eval_ids(name)[:8, :256].to(DEV)
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
out = {}; res = dict(model=name, revision=rev, blocks=BL, by_block={})
for b in BL:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); U = unitr(Xc); xn = Xc.norm(dim=-1).clamp_min(1e-9); N = U.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); m = Au.shape[0]
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    sel, _, _ = omp(Xc, Au, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m)[gidx].float()
    # the calibrated floor per position
    CA = Au.T @ Au / m; CR = Ar.T @ Ar / m; s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Lr = s2r.clamp_min(1e-12).sqrt() * gabs(m)
    Mr = torch.cat([(U[s:s + 128] @ Ar.T).abs().max(1).values for s in range(0, N, 128)]); r_cal = float(Mr.mean() / Lr.mean()); L = s2.clamp_min(1e-12).sqrt() * gabs(m) * r_cal
    # per-row quantities
    Cled = torch.cat([A_all[bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); act = torch.cat([A_all[bb][keep] for bb in range(b + 1)], 1) > 0
    mag = Cled.abs().mean(0); activity = act.float().mean(0); z = (Cled - Cled.mean(0, keepdim=True)) / Cled.std(0, keepdim=True).clamp_min(1e-9); kurt = z.pow(4).mean(0) - 3; del z
    prom = Cled.abs() / xn[:, None]; prom_active = (prom * act).sum(0) / act.float().sum(0).clamp_min(1); write_over_floor = (prom > L[:, None]).float().mean(0); del prom
    rows = gidx.reshape(-1); Am = Au[rows]; above, mx = torch.zeros(rows.numel(), device=DEV), torch.zeros(rows.numel(), device=DEV)
    for s in range(0, N, 128):
        P = (U[s:s + 128] @ Am.T).abs() / L[s:s + 128, None]; above += (P > 1).float().sum(0); mx = torch.maximum(mx, P.max(0).values)
    above = above / N
    out[b] = dict(usage=usage.reshape(-1).cpu(), magnitude=mag.cpu(), activity=activity.cpu(), kurtosis=kurt.cpu(), prominence_active=prom_active.cpu(), write_over_floor_rate=write_over_floor.cpu(), projection_over_floor_rate=above.cpu(), max_projection_over_floor=mx.cpu())
    res["by_block"][b] = dict(n_rows=int(rows.numel()), rotated_calibration=r_cal, median_floor=float(L.median()), rows_with_a_write_over_floor=float((write_over_floor > 0).float().mean()), rows_with_a_projection_over_floor=float((above > 0).float().mean()), median_max_projection_over_floor=float(mx.median()))
    o = res["by_block"][b]; log(f"{name}{' ' + rev if rev else ''} block {b}: floor {o['median_floor']:.3f} (calibration {r_cal:.3f}); rows whose own write ever clears the floor {o['rows_with_a_write_over_floor']:.2f}, whose projection ever clears it {o['rows_with_a_projection_over_floor']:.2f}; median largest projection over the floor {o['median_max_projection_over_floor']:.2f}")
    del Au, Ar, Cled, act, CA, CR; torch.cuda.empty_cache()
torch.save(out, f"{CDIR}/{rev or 'main'}.pt")
summ = f"{name}{' ' + rev if rev else ''}: " + " | ".join(f"block {b}: floor {o['median_floor']:.3f}, rows clearing it by their own write {o['rows_with_a_write_over_floor']:.2f}, by projection {o['rows_with_a_projection_over_floor']:.2f}" for b, o in res["by_block"].items())
log(summ); record(f"e516_rows_{name}{'_' + rev if rev else ''}", res, summ)
