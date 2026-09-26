"""e512: is the onset of wordhood a transition? The program's checkpoint runs (e493, e498, e500, e503, e505-e507,
e509, e510) were made at eight or nine checkpoints, a factor of two to four apart. The hypothesis on the table is
that training moves the residual stream from a dense regime, where the maxima are collective, to an extreme-value
regime, where a few correlated large writes carry them, and that the wordhood observables change sharply while the
energy observables change smoothly. That needs finer time resolution. This run computes the cheap observables of
those sessions at one checkpoint, with no ablation, so that a dense sweep can be read across runs.
Per block (NB/4, NB/2, 3NB/4), typical positions, the writers as in e493 (the 64 largest MLP writes plus the
embeddings): the writer-win rate (the state's largest projection is on a writer); the non-writers' maximum over
their second-order Gumbel level calibrated on the rotated dictionary (the tail factor); the provenance factor (the
native maximum over the rotated); the chord's centred amplitude over the state norm; the effective number of writes
(participation ratio of the write ledger), the top-64 energy share and the largest write's prominence; the state's
energy shares on the MLP writes, attention and the embeddings; the crowd's gain along the chord (the centred small
writes' projection on the centred chord's direction over its norm); and the linear contraction gain along 32
random directions at 0.1 of the state norm.
Arguments: name [revision]."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; NW = 64; NR = 32
blocks = sorted({NB // 4, NB // 2, (3 * NB) // 4}); LB = max(blocks); ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
def ln2(b): return arch.layers[b].ln_2 if fam == "gpt2" else arch.layers[b].post_attention_layernorm
acts = {b: [] for b in range(LB + 1)}; H = {b: [] for b in blocks}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
def mkh(b):
    def pre(m, args): H[b].append(args[0].detach().float()[:, 1:].reshape(-1, D)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)] + [ln2(b).register_forward_pre_hook(mkh(b)) for b in blocks]
try: S_ = block_states(model, arch, ids, blocks, chunk=4)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(LB + 1)}; del acts; H = {b: torch.cat(v) for b, v in H.items()}; rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}
tokens = ids[:, 1:].reshape(-1); positions = torch.arange(1, T_, device=DEV).repeat(B_); has_pos = len(arch.emb) > 1
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
g = torch.Generator(device=DEV).manual_seed(0)
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); U = unitr(Xc); xn = Xc.norm(dim=-1).clamp_min(1e-9); N = U.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); anorm = A.float().norm(dim=-1); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; m = Au.shape[0]
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV)
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    Cled = torch.cat([A_all[bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); top = Cled.abs().topk(NW, dim=1); writers = gidx.reshape(-1)[top.indices]; wcoef = Cled.gather(1, top.indices); wblk = top.indices // DFF; widx = top.indices % DFF
    c2 = Cled.pow(2); eff = float((c2.sum(1).pow(2) / c2.pow(2).sum(1).clamp_min(1e-12)).median()); top64 = float((c2.topk(NW, dim=1).values.sum(1) / c2.sum(1).clamp_min(1e-12)).median()); prom = float((Cled.abs().max(1).values / xn).median()); del Cled, c2
    mt = typ == T_TOK; tok_lut = torch.full((int(idx[mt].max()) + 1,), -1, device=DEV, dtype=torch.long); tok_lut[idx[mt]] = torch.nonzero(mt)[:, 0]; tatom = tok_lut[tokens[keep]]
    emb = anorm[tatom][:, None] * Au[tatom]; Evec = arch.emb[0].detach().float()[tokens[keep]]; wall = [writers, tatom[:, None]]
    if has_pos:
        mp = typ == T_POS; pos_lut = torch.full((int(idx[mp].max()) + 1,), -1, device=DEV, dtype=torch.long); pos_lut[idx[mp]] = torch.nonzero(mp)[:, 0]; patom = pos_lut[positions[keep]]; emb = emb + anorm[patom][:, None] * Au[patom]; Evec = Evec + arch.emb[1].detach().float()[positions[keep]]; wall.append(patom[:, None])
    writers_all = torch.cat(wall, 1); nw = writers_all.shape[1]; assert bool((writers_all >= 0).all())
    # energy shares and the crowd
    Mvec = torch.zeros(N, D, device=DEV); S = torch.zeros(N, D, device=DEV)
    for bb in range(b + 1):
        a = A_all[bb][keep]; Wd = arch.wdir(bb); Mvec += a @ Wd; mask = torch.zeros(N, DFF, dtype=torch.bool, device=DEV); here = wblk == bb; mask[torch.nonzero(here)[:, 0], widx[here]] = True; S += (a * (~mask)) @ Wd
    Avec = Xk - Mvec - Evec; xn2 = Xk.pow(2).sum(1).clamp_min(1e-9); shares = dict(mlp=float(((Xk * Mvec).sum(1) / xn2).median()), attention=float(((Xk * Avec).sum(1) / xn2).median()), embedding=float(((Xk * Evec).sum(1) / xn2).median()))
    chord = torch.einsum("nk,nkd->nd", wcoef, Au[writers]) + emb; chc = chord - chord.mean(0); cn = chc.norm(dim=-1).clamp_min(1e-9); amp = float((cn / xn).median()); Sc = S - S.mean(0); crowd_gain = float(((Sc * (chc / cn[:, None])).sum(1) / cn).median())
    # the maxima
    CA = Au.T @ Au / m; CR = Ar.T @ Ar / m; s2_all = ((U @ CA) * U).sum(1); s2_rot = ((U @ CR) * U).sum(1); Mw, Mn, Mr, ssw = [], [], [], []
    for s in range(0, N, 128):
        C = (U[s:s + 128] @ Au.T).abs(); wm = torch.zeros_like(C, dtype=torch.bool); wm.scatter_(1, writers_all[s:s + 128], True); Cw_ = torch.where(wm, C, torch.zeros_like(C))
        Mw.append(Cw_.max(1).values); Mn.append(torch.where(wm, torch.zeros_like(C), C).max(1).values); ssw.append(Cw_.pow(2).sum(1)); Mr.append((U[s:s + 128] @ Ar.T).abs().max(1).values)
    Mw, Mn, Mr, ssw = map(torch.cat, (Mw, Mn, Mr, ssw)); Mall = torch.maximum(Mw, Mn); win = (Mw > Mn).float()
    sig_n = ((m * s2_all - ssw) / (m - nw)).clamp_min(1e-12).sqrt(); Ln = sig_n * gabs(m - nw); Lr = s2_rot.clamp_min(1e-12).sqrt() * gabs(m); r_cal = float(Mr.mean() / Lr.mean())
    # the linear contraction along random directions
    Hk = H[b][keep]; hn = (Hk - Hk.mean(0)).norm(dim=-1).clamp_min(1e-6); dirs = unitr(torch.randn(NR, D, generator=g, device=DEV)); base = arch.layers[b].mlp(ln2(b)(Hk)).float(); gl = []
    with torch.no_grad():
        for i in range(0, NR, 8):
            dd = dirs[i:i + 8]; eps = (0.1 * hn)[None, :, None]; y = arch.layers[b].mlp(ln2(b)(Hk[None] + eps * dd[:, None, :])).float() - base[None]; gl.append(((y * dd[:, None, :]).sum(-1) / eps[:, :, 0]).median(1).values)
    o = dict(n_states=N, n_atoms=m, writer_win_rate=float(win.mean()), tail_factor=float(Mn.mean() / (Ln.mean() * r_cal)), provenance_factor=float(Mall.mean() / Mr.mean()), writer_max_over_prominence=float(Mw.mean() / prom) if prom > 0 else None, chord_amplitude=amp, effective_writes=eff, top64_energy_share=top64, top_write_prominence=prom, energy_shares=shares, crowd_gain_along_chord=crowd_gain, linear_contraction_random=float(torch.cat(gl).median()), rotated_calibration=r_cal)
    res["by_block"][b] = o
    log(f"{name}{' ' + rev if rev else ''} block {b}: writer wins {o['writer_win_rate']:.2f}, tail factor {o['tail_factor']:.2f}, provenance factor {o['provenance_factor']:.2f}, chord amplitude {amp:.2f}, effective writes {eff:.0f}, top-64 energy {top64:.2f}, prominence {prom:.3f}, shares mlp/attn/emb {shares['mlp']:.2f}/{shares['attention']:.2f}/{shares['embedding']:.2f}, crowd gain along the chord {crowd_gain:+.2f}, linear contraction {o['linear_contraction_random']:+.3f}, rotated calibration {r_cal:.3f}")
    del Au, Ar, chord, S, Mvec, CA, CR; torch.cuda.empty_cache()
summ = (f"{name}{' ' + rev if rev else ''}: by block " + " | ".join(f"{b}: writer wins {o['writer_win_rate']:.2f}, tail {o['tail_factor']:.2f}, factor {o['provenance_factor']:.2f}, chord {o['chord_amplitude']:.2f}, effective writes {o['effective_writes']:.0f}, top-64 energy {o['top64_energy_share']:.2f}, shares mlp/attn/emb {o['energy_shares']['mlp']:.2f}/{o['energy_shares']['attention']:.2f}/{o['energy_shares']['embedding']:.2f}, crowd gain {o['crowd_gain_along_chord']:+.2f}, contraction {o['linear_contraction_random']:+.3f}" for b, o in res["by_block"].items()))
log(summ); record(f"e512_sweep_{name}{'_' + rev if rev else ''}", res, summ)
