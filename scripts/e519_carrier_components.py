"""e519: who supplies the projection, the per-checkpoint half. e517b found that a row's largest projection over the
floor predicts its entry into the vocabulary and that destroying the position's chord of largest writes leaves the
prediction intact: the projection is supplied by the rest of the state. The state at a position is exactly the sum
of its parts, so the projection onto any row's atom splits exactly into the parts' projections. Here every
position's centred state is split into the token embedding, attention's output (with the biases), the row's own
write where it is one of the 64 largest, the other 63 largest writes, and the crowd of the remaining MLP writes, each
centred over positions so that the parts sum to the centred state, all in units of the real state's norm and all scored against the real floor, so that the parts add up to
the real projection. Per row: the largest projection over the floor supplied by each part alone and by three
combinations (attention with the embedding, the MLP writes together, the chord), and, at the position of the row's
real maximum, each part's share of the real projection. e519b reads the checkpoints together and asks which part's
projection predicts entry.
Setup: Pythia-410m, blocks 6 and 12; 8 x 256 evaluation tokens, typical positions; the floor as in e516 and e517.
v2: the parts centred over positions (v1 left them uncentred with the centring as a sixth part, so that a part's
projection carried the common direction: attention alone cleared the floor for 99% of rows at block 12).
Arguments: name [revision] [v2]."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
BL = [6, 12]; LB = max(BL); K = 16; NW = 64; CDIR = f"/workspace/wdd/cache/e519_{name}"; os.makedirs(CDIR, exist_ok=True)
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
acts = {b: [] for b in range(LB + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)]
try: S_ = block_states(model, arch, ids, BL, chunk=4)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(LB + 1)}; del acts; rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}
tokens = ids[:, 1:].reshape(-1); positions = torch.arange(1, T_, device=DEV).repeat(B_); has_pos = len(arch.emb) > 1
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
PARTS = ["embedding", "attention", "own", "chord_others", "crowd"]; COMBOS = {"attention_embedding": ["attention", "embedding"], "mlp_all": ["own", "chord_others", "crowd"], "chord": ["own", "chord_others"], "real": PARTS}
out = {}; res = dict(model=name, revision=rev, blocks=BL, by_block={})
for b in BL:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]; mu = Xk.mean(0); Xc = Xk - mu; U = unitr(Xc); xn = Xc.norm(dim=-1).clamp_min(1e-9)
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); m = Au.shape[0]
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    rows = gidx.reshape(-1); Am = Au[rows]; R = rows.numel()
    sel, _, _ = omp(Xc, Au, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m)[rows].float()
    Cled = torch.cat([A_all[bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); mag = Cled.abs().mean(0); top = Cled.abs().topk(NW, dim=1); wrow = top.indices; wcoef = Cled.gather(1, wrow); del Cled
    Mvec = sum(A_all[bb][keep] @ arch.wdir(bb).float() for bb in range(b + 1)); chord = torch.einsum("nk,nkd->nd", wcoef, Au[rows[wrow]])
    E = arch.emb[0].detach().float()[tokens[keep]] + (arch.emb[1].detach().float()[positions[keep]] if has_pos else 0); Att = Xk - Mvec - E; crowd = Mvec - chord
    CA = Au.T @ Au / m; CR = Ar.T @ Ar / m; gm = gabs(m); s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 128] @ Ar.T).abs().max(1).values for s in range(0, N, 128)]); r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); L = s2.clamp_min(1e-12).sqrt() * gm * r_cal
    cen = lambda M: (M - M.mean(0)) / xn[:, None]                                                                             # v2: every part centred over positions, so that the parts sum to the centred state
    comp = {"embedding": cen(E), "attention": cen(Att), "crowd": cen(crowd), "chordc": cen(chord)}
    names = PARTS + [k for k in COMBOS if k != "real"] + ["real"]; mx = {k: torch.zeros(R, device=DEV) for k in names}; real_max = torch.zeros(R, device=DEV); arg_pos = torch.zeros(R, dtype=torch.long, device=DEV); share_at_max = {k: torch.zeros(R, device=DEV) for k in PARTS}
    for s in range(0, N, 128):
        sl = slice(s, s + 128); P = {}
        for k in ("embedding", "attention", "crowd", "chordc"): P[k] = comp[k][sl] @ Am.T
        own = torch.zeros(P["embedding"].shape, device=DEV); own.scatter_add_(1, wrow[sl], wcoef[sl] / xn[sl, None]); P["own"] = own
        P["chord_others"] = P.pop("chordc") - own
        for k, parts in COMBOS.items(): P[k] = sum(P[q] for q in parts)
        Lc = L[sl, None]
        for k in names: mx[k] = torch.maximum(mx[k], (P[k].abs() / Lc).max(0).values)
        ratio = P["real"].abs() / Lc; v, a = ratio.max(0); better = v > real_max; real_max = torch.where(better, v, real_max); arg_pos = torch.where(better, a + s, arg_pos)
        den = P["real"].gather(0, a[None])[0]; den = torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
        for k in PARTS: share_at_max[k] = torch.where(better, P[k].gather(0, a[None])[0] / den, share_at_max[k])
    out[b] = dict(usage=usage.cpu(), magnitude=mag.cpu(), max_over_floor={k: v.cpu() for k, v in mx.items()}, share_at_max={k: v.cpu() for k, v in share_at_max.items()}, is_writer_at_max=(share_at_max["own"] != 0).cpu())
    res["by_block"][b] = dict(n_rows=R, median_floor=float(L.median()), share_of_state_energy=dict(embedding=float(((Xk * E).sum(1) / Xk.pow(2).sum(1)).median()), attention=float(((Xk * Att).sum(1) / Xk.pow(2).sum(1)).median()), mlp=float(((Xk * Mvec).sum(1) / Xk.pow(2).sum(1)).median())), rows_over_floor={k: float((v > 1).float().mean()) for k, v in mx.items()}, median_max_over_floor={k: float(v.median()) for k, v in mx.items()})
    o = res["by_block"][b]; log(f"{name}{' ' + rev if rev else ''} block {b}: state energy shares embedding/attention/mlp {o['share_of_state_energy']['embedding']:.2f}/{o['share_of_state_energy']['attention']:.2f}/{o['share_of_state_energy']['mlp']:.2f}; rows whose part-only projection clears the floor: " + ", ".join(f"{k} {v:.3f}" for k, v in o["rows_over_floor"].items()))
    del Au, Ar, Am, Mvec, chord, E, Att, crowd, CA, CR; torch.cuda.empty_cache()
torch.save(out, f"{CDIR}/{rev or 'main'}.pt")
summ = f"{name}{' ' + rev if rev else ''}: " + " | ".join(f"block {b}: rows over the floor by part, real {o['rows_over_floor']['real']:.3f}, attention {o['rows_over_floor']['attention']:.3f}, embedding {o['rows_over_floor']['embedding']:.3f}, crowd {o['rows_over_floor']['crowd']:.3f}, chord {o['rows_over_floor']['chord']:.3f}, attention+embedding {o['rows_over_floor']['attention_embedding']:.3f}, all MLP {o['rows_over_floor']['mlp_all']:.3f}" for b, o in res["by_block"].items())
log(summ); record(f"e519_carrier_{name}{'_' + rev if rev else ''}", res, summ)
