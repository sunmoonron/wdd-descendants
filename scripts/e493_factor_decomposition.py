"""e493: where the provenance factor comes from. e483 and e492 found the native dictionary's competitor maximum scales
at 2.9-6.2 times the rotated dictionary's, and that the factor falls with depth and rises through training. In
x = sum_i c_i w_i + ..., an atom's correlation with the state is its own coefficient plus cross terms; a rotated atom
has only cross terms. So the factor can have two sources: write sparsity (the largest actual writes stand out) and
cross-alignment (rows that did not write here still resemble what was written). This run separates them.
Setup: 8 x 256 evaluation tokens, typical positions, blocks 1, NB/4, NB/2, 3NB/4, NB-2, the dictionary up to the block.
Per position the writers are the 64 MLP neurons of the blocks up to the block with the largest |activation x row norm|,
plus the current token's embedding (and position embedding where one exists); every other atom is a non-writer.
Measured per block: the mean competitor maximum over all atoms (M_all), over the writers (M_w), over the non-writers
(M_n), and for the rotated dictionary (M_r); the share of positions whose maximum is attained by a writer; the
correlation of the state with its single largest write against that write's prominence (its coefficient over the
state's norm). Everything is reported as a ratio to M_r, so 1 means the rotated (geometric) level.
Models (arguments): name [revision].
Pre-registered (honest guesses):
- the maximum is attained by a writer at most positions (over 0.6) in the final models (0.6);
- non-writers sit above the rotated level too (M_n / M_r at least 1.5), so both sources contribute (0.5);
- over Pythia's training the writers' part rises more than the non-writers' (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF
blocks = sorted({1, NB // 4, NB // 2, (3 * NB) // 4, NB - 2}); ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
acts = {b: [] for b in range(NB - 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None          # must return None; accumulate over chunks
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(NB - 1)]
try: S_ = block_states(model, arch, ids, blocks, chunk=4)
finally: [h.remove() for h in hs]
rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(NB - 1)}
C_all = {b: torch.cat(acts[b]) * rown[b][None] for b in range(NB - 1)}; del acts
tokens = ids[:, 1:].reshape(-1); positions = torch.arange(1, T_, device=DEV)[None].expand(B_, T_ - 1).reshape(-1)
def maxes(U, Dct, writers):
    """per position: max |corr| over all atoms, over the writer atoms, over the rest, and whether the argmax is a writer"""
    Ma, Mw, Mn, isw = [], [], [], []
    for s in range(0, U.shape[0], 128):
        C = (U[s:s + 128] @ Dct.T).abs(); w = writers[s:s + 128]; Ma.append(C.max(1).values); isw.append((torch.gather(w, 1, C.argmax(1, keepdim=True)) if False else None))
        wm = torch.zeros_like(C, dtype=torch.bool); wm.scatter_(1, w.clamp_min(0), True); wm[:, 0] = wm[:, 0] & (w == 0).any(1)   # index 0 is a real atom only if listed
        Mw.append(torch.where(wm, C, torch.zeros_like(C)).max(1).values); Mn.append(torch.where(wm, torch.zeros_like(C), C).max(1).values); isw[-1] = wm.gather(1, C.argmax(1, keepdim=True))[:, 0]
    return torch.cat(Ma), torch.cat(Mw), torch.cat(Mn), torch.cat(isw).float()
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xc = X[keep] - X[keep].mean(0); U = unitr(Xc); N = U.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); m_all = Au.shape[0]
    # global atom index of every MLP neuron of blocks 0..b, and the ledger of their coefficients at each kept position
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1):
        m = (typ == T_MLP) & (blk == bb); gidx[bb, idx[m]] = torch.nonzero(m)[:, 0]
    Cled = torch.cat([C_all[bb][keep] for bb in range(b + 1)], 1)                                     # [N, (b+1)*DFF]
    top = Cled.abs().topk(64, dim=1).indices; writers = gidx.reshape(-1)[top]                          # [N, 64] global atom indices
    tok_atoms = torch.full((int(typ.numel()),), -1, device=DEV, dtype=torch.long); mt = typ == T_TOK; tok_lut = torch.full((int(idx[mt].max()) + 1,), -1, device=DEV, dtype=torch.long); tok_lut[idx[mt]] = torch.nonzero(mt)[:, 0]
    extra = [tok_lut[tokens[keep]]]
    mp = typ == T_POS
    if mp.any():
        pos_lut = torch.full((int(idx[mp].max()) + 1,), -1, device=DEV, dtype=torch.long); pos_lut[idx[mp]] = torch.nonzero(mp)[:, 0]; extra.append(pos_lut[positions[keep].clamp_max(int(idx[mp].max()))])
    writers = torch.cat([writers] + [e_[:, None] for e_ in extra], 1)
    Ma, Mw, Mn, isw = maxes(U, Au, writers); Mr = torch.cat([(U[s:s + 128] @ Ar.T).abs().max(1).values for s in range(0, N, 128)])
    top1 = writers[:, 0]; corr_top = (U * Au[top1]).sum(1).abs(); prom_top = Cled.abs().max(1).values / Xc.norm(dim=-1)
    row = dict(m=m_all, M_all=float(Ma.mean()), M_writers=float(Mw.mean()), M_nonwriters=float(Mn.mean()), M_rotated=float(Mr.mean()), argmax_is_writer=float(isw.mean()),
               factor_all=float(Ma.mean() / Mr.mean()), factor_writers=float(Mw.mean() / Mr.mean()), factor_nonwriters=float(Mn.mean() / Mr.mean()),
               top_write_correlation=float(corr_top.mean()), top_write_prominence=float(prom_top.mean()), top_write_is_argmax=float((writers[:, 0] == torch.cat([(U[s:s + 128] @ Au.T).abs().argmax(1) for s in range(0, N, 128)])).float().mean()))
    res["by_block"][b] = row
    log(f"{name}{' ' + rev if rev else ''} block {b}: M all {row['M_all']:.3f} writers {row['M_writers']:.3f} non-writers {row['M_nonwriters']:.3f} rotated {row['M_rotated']:.3f} | factors all {row['factor_all']:.2f}, writers {row['factor_writers']:.2f}, non-writers {row['factor_nonwriters']:.2f} | argmax is a writer {row['argmax_is_writer']:.2f}, the top write {row['top_write_is_argmax']:.2f}; top write correlation {row['top_write_correlation']:.3f} against prominence {row['top_write_prominence']:.3f}")
    del Au, Ar, Cled; torch.cuda.empty_cache()
Bk = res["by_block"]; fin = [Bk[b] for b in blocks]
res["checks"] = dict(argmax_writer_over_0_6=all(r["argmax_is_writer"] > 0.6 for r in fin), nonwriters_over_1_5=all(r["factor_nonwriters"] >= 1.5 for r in fin))
summ = (f"{name}{' ' + rev if rev else ''}: by block, factor over the rotated level for all atoms / writers / non-writers, and the share of positions whose maximum is a writer: "
        + "; ".join(f"{b}: {Bk[b]['factor_all']:.2f} / {Bk[b]['factor_writers']:.2f} / {Bk[b]['factor_nonwriters']:.2f}, writer {Bk[b]['argmax_is_writer']:.2f} (top write {Bk[b]['top_write_is_argmax']:.2f})" for b in blocks)
        + " | top write: correlation " + "/".join(f"{Bk[b]['top_write_correlation']:.2f}" for b in blocks) + " against prominence " + "/".join(f"{Bk[b]['top_write_prominence']:.2f}" for b in blocks) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e493_factordecomp_{name}{'_' + rev if rev else ''}", res, summ)
