"""e498: the anatomy of the two clocks. e493 on Pythia's checkpoints found the accent (non-writing rows collectively
aligned with the states) forms by step 512 and the words (the largest writes standing out) from about step 1000. What
else changes in the network between the two? This run measures, at the same checkpoints and blocks, the quantities
that could carry either clock, so that they can be lined up: write sparsity, the largest write's prominence, the
energy the state draws from MLP writes, attention and the embedding, the effective dimension of the state cloud, the
alignment of the states' covariance with the atoms' covariance (the accent as a second-order quantity), and the two
factor parts themselves.
Setup: Pythia-410m at a checkpoint (argument), blocks 1, 6, 12, 18, 22; 8 x 256 evaluation tokens, typical positions.
Per block: the writers' and non-writers' factors (as e493); the participation ratio of the write ledger per position
(the effective number of active MLP writes of the blocks up to the block); the energy share of the 64 largest writes;
the largest write's prominence; the state's projection shares on the summed MLP writes, the attention increments and
the token embedding; the effective dimension of the centred states and the share of variance in their top 8
directions; the covariance alignment tr(C_X C_A) / (|C_X| |C_A|) of the states with the native atoms and with the
rotated atoms, and their ratio.
Arguments: name [revision]."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF
blocks = sorted({1, NB // 4, NB // 2, (3 * NB) // 4, NB - 2}); ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
acts = {b: [] for b in range(NB - 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(NB - 1)]
try: S_ = block_states(model, arch, ids, blocks, chunk=4)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(NB - 1)}; del acts
rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(NB - 1)}
tokens = ids[:, 1:].reshape(-1); E0 = arch.emb[0].detach().float()[tokens]                                     # token embeddings per position
def maxes(U, Dct, writers):
    Ma, Mw, Mn, isw = [], [], [], []
    for s in range(0, U.shape[0], 128):
        C = (U[s:s + 128] @ Dct.T).abs(); w = writers[s:s + 128]; wm = torch.zeros_like(C, dtype=torch.bool); wm.scatter_(1, w, True)
        Ma.append(C.max(1).values); Mw.append(torch.where(wm, C, torch.zeros_like(C)).max(1).values); Mn.append(torch.where(wm, torch.zeros_like(C), C).max(1).values); isw.append(wm.gather(1, C.argmax(1, keepdim=True))[:, 0])
    return torch.cat(Ma), torch.cat(Mw), torch.cat(Mn), torch.cat(isw).float()
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); U = unitr(Xc); N = U.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV)
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1):
        m = (typ == T_MLP) & (blk == bb); gidx[bb, idx[m]] = torch.nonzero(m)[:, 0]
    Cled = torch.cat([A_all[bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); top = Cled.abs().topk(64, dim=1).indices; writers = gidx.reshape(-1)[top]
    mt = typ == T_TOK; tok_lut = torch.full((int(idx[mt].max()) + 1,), -1, device=DEV, dtype=torch.long); tok_lut[idx[mt]] = torch.nonzero(mt)[:, 0]; writers = torch.cat([writers, tok_lut[tokens[keep]][:, None]], 1)
    Ma, Mw, Mn, isw = maxes(U, Au, writers); Mr = torch.cat([(U[s:s + 128] @ Ar.T).abs().max(1).values for s in range(0, N, 128)])
    # write sparsity and prominence
    c2 = Cled.pow(2); pr = c2.sum(1).pow(2) / c2.pow(2).sum(1).clamp_min(1e-12); top64_share = c2.topk(64, dim=1).values.sum(1) / c2.sum(1).clamp_min(1e-12); prom = Cled.abs().max(1).values / Xc.norm(dim=-1)
    # the state's composition: summed MLP writes of blocks 0..b, the token embedding, the rest (attention increments)
    Mvec = sum(A_all[bb][keep] @ arch.wdir(bb).float() for bb in range(b + 1)); Evec = E0[keep]; Avec = Xk - Mvec - Evec; xn2 = Xk.pow(2).sum(1).clamp_min(1e-9)
    shares = dict(mlp=float(((Xk * Mvec).sum(1) / xn2).median()), attention=float(((Xk * Avec).sum(1) / xn2).median()), embedding=float(((Xk * Evec).sum(1) / xn2).median()))
    # the state cloud and the second-order alignment
    CX = torch.cov(Xc.T.double(), correction=0); ev = torch.linalg.eigvalsh(CX).flip(0); eff_dim = float(ev.sum() ** 2 / (ev ** 2).sum()); top8 = float(ev[:8].sum() / ev.sum())
    CA = (Au.T @ Au).double() / Au.shape[0]; CR = (Ar.T @ Ar).double() / Ar.shape[0]
    cov_align = lambda C: float((CX * C).sum() / (CX.norm() * C.norm()))
    al_n, al_r = cov_align(CA), cov_align(CR)
    res["by_block"][b] = dict(factor_all=float(Ma.mean() / Mr.mean()), factor_writers=float(Mw.mean() / Mr.mean()), factor_nonwriters=float(Mn.mean() / Mr.mean()), argmax_is_writer=float(isw.mean()),
                              effective_writes=float(pr.median()), top64_energy_share=float(top64_share.median()), top_write_prominence=float(prom.median()), state_shares=shares,
                              effective_dimension=eff_dim, top8_variance_share=top8, covariance_alignment_native=al_n, covariance_alignment_rotated=al_r, covariance_alignment_ratio=al_n / max(al_r, 1e-9), state_norm_median=float(Xk.norm(dim=-1).median()))
    r = res["by_block"][b]
    log(f"{name}{' ' + rev if rev else ''} block {b}: factors all {r['factor_all']:.2f} writers {r['factor_writers']:.2f} non-writers {r['factor_nonwriters']:.2f} (writer {r['argmax_is_writer']:.2f}) | effective writes {r['effective_writes']:.0f}, top-64 energy {r['top64_energy_share']:.2f}, top prominence {r['top_write_prominence']:.3f} | shares mlp {shares['mlp']:.2f} attn {shares['attention']:.2f} emb {shares['embedding']:.2f} | eff dim {eff_dim:.1f}, top-8 {top8:.2f} | cov alignment native {al_n:.4f} rotated {al_r:.4f} ratio {r['covariance_alignment_ratio']:.2f}")
    del Au, Ar, Cled, c2; torch.cuda.empty_cache()
Bk = res["by_block"]
summ = (f"{name}{' ' + rev if rev else ''}: by block " + " | ".join(f"{b}: writers {Bk[b]['factor_writers']:.2f}, non-writers {Bk[b]['factor_nonwriters']:.2f}, effective writes {Bk[b]['effective_writes']:.0f}, top-64 energy {Bk[b]['top64_energy_share']:.2f}, prominence {Bk[b]['top_write_prominence']:.3f}, shares mlp/attn/emb {Bk[b]['state_shares']['mlp']:.2f}/{Bk[b]['state_shares']['attention']:.2f}/{Bk[b]['state_shares']['embedding']:.2f}, eff dim {Bk[b]['effective_dimension']:.0f}, cov-alignment ratio {Bk[b]['covariance_alignment_ratio']:.2f}" for b in blocks))
log(summ); record(f"e498_clocks_{name}{'_' + rev if rev else ''}", res, summ)
