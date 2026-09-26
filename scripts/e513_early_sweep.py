"""e513: what precedes sparsification? e512 found the developmental order at 18 Pythia checkpoints from step 512:
the effective number of writes falls first (midpoint near step 1500-1700), the writer-win rate rises next, the
contraction and the chord follow. The origin of the sparse writes is before step 512, where the program has one
checkpoint (e401 at initialisation, e493 at step 256). This run takes e512's observables below step 512, from
initialisation, and adds the candidates for the first arrow: the concentration of the loss gradient across MLP
rows (are a few rows reinforced before the writes sparsify?), the rows' liveness (the share of positions at which
each row is active, and the share of near-dead rows), the kurtosis of the write ledger, and the anisotropy of the
state cloud (the share of variance in its top 8 directions and the effective dimension).
Per block (NB/4, NB/2, 3NB/4), typical positions: e512's writer-win rate, tail factor, provenance factor, chord
amplitude, effective writes, top-64 energy share, energy shares, crowd gain along the chord and linear contraction;
plus, from one backward pass of the language-model loss on the evaluation tokens, the participation ratio of the
per-row gradient norms of the block's down-projection and up-projection (the effective number of rows receiving
gradient) and the share of the gradient's energy in its 64 largest rows; the rows' median activity and the share of
rows active at under 1% of positions; the ledger's excess kurtosis; the state cloud's effective dimension and top-8
variance share.
Arguments: name [revision]."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; NW = 64; NR = 32
blocks = sorted({NB // 4, NB // 2, (3 * NB) // 4}); LB = max(blocks); ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
def ln2(b): return arch.layers[b].ln_2 if fam == "gpt2" else arch.layers[b].post_attention_layernorm
def up_lin(b):
    l = arch.layers[b]; return l.mlp.c_fc if fam == "gpt2" else (l.mlp.up_proj if fam == "llama" else l.mlp.dense_h_to_4h)
# one backward pass for the gradient concentration across rows
for p in model.parameters(): p.requires_grad_(False)
tracked = {}
for b in blocks:
    for nm, lin in (("down", arch.mlp_lin(b)), ("up", up_lin(b))): lin.weight.requires_grad_(True); tracked[(b, nm)] = lin.weight
with torch.enable_grad():
    loss = model(ids, labels=ids).loss; loss.backward()
grad_stats = {}
for (b, nm), w in tracked.items():
    gr = w.grad.detach().float(); ax = 1 if gr.shape[0] == DFF else 0; rn = gr.norm(dim=ax) if ax == 1 else gr.norm(dim=0)                     # per-row (neuron) gradient norm
    e = rn.pow(2); grad_stats[(b, nm)] = dict(participation_ratio=float(e.sum().pow(2) / e.pow(2).sum().clamp_min(1e-30)), top64_share=float(e.topk(NW).values.sum() / e.sum().clamp_min(1e-30))); w.grad = None; w.requires_grad_(False)
lm_loss = float(loss.detach())
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
res = dict(model=name, revision=rev, blocks=blocks, lm_loss=lm_loss, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); U = unitr(Xc); xn = Xc.norm(dim=-1).clamp_min(1e-9); N = U.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); anorm = A.float().norm(dim=-1); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; m = Au.shape[0]
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV)
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    Cled = torch.cat([A_all[bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); top = Cled.abs().topk(NW, dim=1); writers = gidx.reshape(-1)[top.indices]; wcoef = Cled.gather(1, top.indices); wblk = top.indices // DFF; widx = top.indices % DFF
    c2 = Cled.pow(2); eff = float((c2.sum(1).pow(2) / c2.pow(2).sum(1).clamp_min(1e-12)).median()); top64 = float((c2.topk(NW, dim=1).values.sum(1) / c2.sum(1).clamp_min(1e-12)).median()); prom = float((Cled.abs().max(1).values / xn).median())
    z = (Cled - Cled.mean(1, keepdim=True)) / Cled.std(1, keepdim=True).clamp_min(1e-9); kurt = float((z.pow(4).mean(1) - 3).median()); del Cled, c2, z
    a_b = A_all[b][keep]; live = (a_b > 0).float().mean(0); liveness = dict(median_activity=float(live.median()), share_under_1pct=float((live < 0.01).float().mean()))
    ev = torch.linalg.eigvalsh(torch.cov(Xc.T.double(), correction=0)).flip(0).clamp_min(0); cloud = dict(effective_dimension=float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-30)), top8_variance_share=float(ev[:8].sum() / ev.sum().clamp_min(1e-30)))
    mt = typ == T_TOK; tok_lut = torch.full((int(idx[mt].max()) + 1,), -1, device=DEV, dtype=torch.long); tok_lut[idx[mt]] = torch.nonzero(mt)[:, 0]; tatom = tok_lut[tokens[keep]]
    emb = anorm[tatom][:, None] * Au[tatom]; Evec = arch.emb[0].detach().float()[tokens[keep]]; wall = [writers, tatom[:, None]]
    if has_pos:
        mp = typ == T_POS; pos_lut = torch.full((int(idx[mp].max()) + 1,), -1, device=DEV, dtype=torch.long); pos_lut[idx[mp]] = torch.nonzero(mp)[:, 0]; patom = pos_lut[positions[keep]]; emb = emb + anorm[patom][:, None] * Au[patom]; Evec = Evec + arch.emb[1].detach().float()[positions[keep]]; wall.append(patom[:, None])
    writers_all = torch.cat(wall, 1); nw = writers_all.shape[1]; assert bool((writers_all >= 0).all())
    Mvec = torch.zeros(N, D, device=DEV); S = torch.zeros(N, D, device=DEV)
    for bb in range(b + 1):
        a = A_all[bb][keep]; Wd = arch.wdir(bb); Mvec += a @ Wd; mask = torch.zeros(N, DFF, dtype=torch.bool, device=DEV); here = wblk == bb; mask[torch.nonzero(here)[:, 0], widx[here]] = True; S += (a * (~mask)) @ Wd
    Avec = Xk - Mvec - Evec; xn2 = Xk.pow(2).sum(1).clamp_min(1e-9); shares = dict(mlp=float(((Xk * Mvec).sum(1) / xn2).median()), attention=float(((Xk * Avec).sum(1) / xn2).median()), embedding=float(((Xk * Evec).sum(1) / xn2).median()))
    chord = torch.einsum("nk,nkd->nd", wcoef, Au[writers]) + emb; chc = chord - chord.mean(0); cn = chc.norm(dim=-1).clamp_min(1e-9); amp = float((cn / xn).median()); Sc = S - S.mean(0); crowd_gain = float(((Sc * (chc / cn[:, None])).sum(1) / cn).median())
    CA = Au.T @ Au / m; CR = Ar.T @ Ar / m; s2_all = ((U @ CA) * U).sum(1); s2_rot = ((U @ CR) * U).sum(1); Mw, Mn, Mr, ssw = [], [], [], []
    for s in range(0, N, 128):
        C = (U[s:s + 128] @ Au.T).abs(); wm = torch.zeros_like(C, dtype=torch.bool); wm.scatter_(1, writers_all[s:s + 128], True); Cw_ = torch.where(wm, C, torch.zeros_like(C))
        Mw.append(Cw_.max(1).values); Mn.append(torch.where(wm, torch.zeros_like(C), C).max(1).values); ssw.append(Cw_.pow(2).sum(1)); Mr.append((U[s:s + 128] @ Ar.T).abs().max(1).values)
    Mw, Mn, Mr, ssw = map(torch.cat, (Mw, Mn, Mr, ssw)); Mall = torch.maximum(Mw, Mn); win = (Mw > Mn).float()
    sig_n = ((m * s2_all - ssw) / (m - nw)).clamp_min(1e-12).sqrt(); Ln = sig_n * gabs(m - nw); Lr = s2_rot.clamp_min(1e-12).sqrt() * gabs(m); r_cal = float(Mr.mean() / Lr.mean())
    Hk = H[b][keep]; hn = (Hk - Hk.mean(0)).norm(dim=-1).clamp_min(1e-6); dirs = unitr(torch.randn(NR, D, generator=g, device=DEV)); base = arch.layers[b].mlp(ln2(b)(Hk)).float(); gl = []
    for i in range(0, NR, 8):
        dd = dirs[i:i + 8]; eps = (0.1 * hn)[None, :, None]; y = arch.layers[b].mlp(ln2(b)(Hk[None] + eps * dd[:, None, :])).float() - base[None]; gl.append(((y * dd[:, None, :]).sum(-1) / eps[:, :, 0]).median(1).values)
    o = dict(n_states=N, n_atoms=m, writer_win_rate=float(win.mean()), tail_factor=float(Mn.mean() / (Ln.mean() * r_cal)), provenance_factor=float(Mall.mean() / Mr.mean()), chord_amplitude=amp, effective_writes=eff, top64_energy_share=top64, top_write_prominence=prom, ledger_excess_kurtosis=kurt,
             energy_shares=shares, crowd_gain_along_chord=crowd_gain, linear_contraction_random=float(torch.cat(gl).median()), rotated_calibration=r_cal, gradient=dict(down=grad_stats[(b, "down")], up=grad_stats[(b, "up")]), liveness=liveness, cloud=cloud)
    res["by_block"][b] = o; gd, gu = o["gradient"]["down"], o["gradient"]["up"]
    log(f"{name}{' ' + rev if rev else ''} block {b}: writer wins {o['writer_win_rate']:.2f}, tail {o['tail_factor']:.2f}, factor {o['provenance_factor']:.2f}, chord {amp:.2f}, effective writes {eff:.0f}, top-64 energy {top64:.2f}, kurtosis {kurt:.1f}, shares mlp/attn/emb {shares['mlp']:.2f}/{shares['attention']:.2f}/{shares['embedding']:.2f}, crowd gain {crowd_gain:+.2f}, contraction {o['linear_contraction_random']:+.3f} | gradient rows down PR {gd['participation_ratio']:.0f} (top-64 share {gd['top64_share']:.2f}), up PR {gu['participation_ratio']:.0f} (top-64 {gu['top64_share']:.2f}) | rows' median activity {liveness['median_activity']:.2f}, under 1% {liveness['share_under_1pct']:.2f} | cloud eff dim {cloud['effective_dimension']:.1f}, top-8 share {cloud['top8_variance_share']:.2f}")
    del Au, Ar, chord, S, Mvec, CA, CR; torch.cuda.empty_cache()
summ = (f"{name}{' ' + rev if rev else ''} (loss {lm_loss:.2f}): by block " + " | ".join(f"{b}: wins {o['writer_win_rate']:.2f}, tail {o['tail_factor']:.2f}, chord {o['chord_amplitude']:.2f}, effective writes {o['effective_writes']:.0f}, top-64 energy {o['top64_energy_share']:.2f}, kurtosis {o['ledger_excess_kurtosis']:.1f}, shares mlp/attn {o['energy_shares']['mlp']:.2f}/{o['energy_shares']['attention']:.2f}, crowd gain {o['crowd_gain_along_chord']:+.2f}, contraction {o['linear_contraction_random']:+.3f}, gradient PR down/up {o['gradient']['down']['participation_ratio']:.0f}/{o['gradient']['up']['participation_ratio']:.0f} (top-64 share {o['gradient']['down']['top64_share']:.2f}/{o['gradient']['up']['top64_share']:.2f}), activity {o['liveness']['median_activity']:.2f} (under 1% {o['liveness']['share_under_1pct']:.2f}), cloud eff dim {o['cloud']['effective_dimension']:.0f}, top-8 {o['cloud']['top8_variance_share']:.2f}" for b, o in res["by_block"].items()))
log(summ); record(f"e513_early_{name}{'_' + rev if rev else ''}", res, summ)
