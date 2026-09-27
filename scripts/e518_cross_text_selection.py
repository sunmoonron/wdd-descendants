"""e518: the selection criterion across texts, at the end of training, in three models. e516b tested it across
time on Pythia's checkpoints: a non-word row's largest projection over the calibrated Gumbel floor predicts entry
into the vocabulary at the next checkpoint. GPT-2 and OLMo have no checkpoints, but the criterion can be tested
across texts: on one set of sequences compute every row's quantities, on a disjoint set define the word set, and ask
which quantities computed on the first set pick the rows that are words on the second but not on the first (the
cross-text entrants), against rows that are words on neither. If the criterion is a property of the row and its
chords rather than of the particular texts, it should predict here as it did across checkpoints.
Per block (NB/4, NB/2, 3NB/4), text set A (8 x 256 tokens) and text set B (a disjoint 8 x 256): on each, the word
set (the 256 rows most used in the 16-word native descriptions of the centred typical states) and, on A, every
row's usage, mean absolute write, selectivity, write kurtosis, largest projection over the floor, and the same with
the row's own write removed. Reported: the AUC of each A-quantity for the cross-text entrants against the never-words,
and for all of B's words against B's non-words; the overlap of the two word sets.
Pre-registered (honest guesses):
- the largest projection over the floor on A picks B's new words at AUC above 0.8 at the middle block of all three
  models, ahead of magnitude (0.5);
- with the own write removed it keeps at least 0.8 of the excess over 0.5 (0.5).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; K = 16; NWORD = 256; NW = 64
blocks = sorted({NB // 4, NB // 2, (3 * NB) // 4}); LB = max(blocks); allids = eval_ids(name); assert allids.shape[0] >= 16, allids.shape
sets = {"A": allids[:8, :256].to(DEV), "B": allids[8:16, :256].to(DEV)}
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
cap = {}
for nm, ids in sets.items():
    acts = {b: [] for b in range(LB + 1)}
    def mk(b):
        def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
        return pre
    hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)]
    try: S_ = block_states(model, arch, ids, blocks, chunk=4)
    finally: [h.remove() for h in hs]
    cap[nm] = dict(S=S_, A={b: torch.cat(acts[b]) for b in range(LB + 1)})
rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}
res = dict(model=name, blocks=blocks, by_block={})
for b in blocks:
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); m = Au.shape[0]
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    rows = gidx.reshape(-1); Am = Au[rows]; R = rows.numel(); CA = Au.T @ Au / m; CR = Ar.T @ Ar / m; gm = gabs(m)
    Wsets, Q = {}, {}
    for nm in ("A", "B"):
        X = cap[nm]["S"][b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); U = unitr(Xc); xn = Xc.norm(dim=-1).clamp_min(1e-9); N = U.shape[0]
        sel, _, _ = omp(Xc, Au, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m)[rows].float(); used = torch.nonzero(usage > 0)[:, 0]; Wsets[nm] = set(used[usage[used].argsort(descending=True)[:NWORD]].tolist())
        if nm == "A":
            Cled = torch.cat([cap[nm]["A"][bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); act = Cled > 0
            mag = Cled.abs().mean(0); selv = 1 - act.float().mean(0); z = (Cled - Cled.mean(0, keepdim=True)) / Cled.std(0, keepdim=True).clamp_min(1e-9); kurt = z.pow(4).mean(0) - 3; del z
            top = Cled.abs().topk(NW, dim=1); wrow = top.indices; wcoef = Cled.gather(1, wrow); del Cled
            s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 128] @ Ar.T).abs().max(1).values for s in range(0, N, 128)]); r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); L = s2.clamp_min(1e-12).sqrt() * gm * r_cal
            mx = torch.zeros(R, device=DEV); mxc = torch.zeros(R, device=DEV)
            for s in range(0, N, 128):
                P = U[s:s + 128] @ Am.T; mx = torch.maximum(mx, (P.abs() / L[s:s + 128, None]).max(0).values); Pc = P.clone(); Pc.scatter_add_(1, wrow[s:s + 128], -wcoef[s:s + 128] / xn[s:s + 128, None]); mxc = torch.maximum(mxc, (Pc.abs() / L[s:s + 128, None]).max(0).values)
            Q = dict(usage=usage, magnitude=mag, selectivity=selv, kurtosis=kurt, max_over_floor=mx, max_over_floor_own_removed=mxc, magnitude_x_selectivity=mag * selv)
    inA = torch.zeros(R, dtype=torch.bool, device=DEV); inA[list(Wsets["A"])] = True; inB = torch.zeros(R, dtype=torch.bool, device=DEV); inB[list(Wsets["B"])] = True
    ent = torch.nonzero(~inA & inB)[:, 0]; never = torch.nonzero(~inA & ~inB)[:, 0]; wB = torch.nonzero(inB)[:, 0]; nB = torch.nonzero(~inB)[:, 0]
    out = dict(n_rows=R, overlap_of_word_sets=len(Wsets["A"] & Wsets["B"]) / NWORD, n_cross_text_entrants=int(ent.numel()), auc_entrants={k: auc(v[ent], v[never]) for k, v in Q.items()}, auc_all_B_words={k: auc(v[wB], v[nB]) for k, v in Q.items()})
    res["by_block"][b] = out; fm = lambda x: "n/a" if x is None else f"{x:.2f}"
    log(f"{name} block {b}: word sets overlap {out['overlap_of_word_sets']:.2f}, cross-text entrants {out['n_cross_text_entrants']}; AUC for B's new words from A's quantities: " + ", ".join(f"{k} {fm(v)}" for k, v in out["auc_entrants"].items()) + " | for all of B's words: " + ", ".join(f"{k} {fm(v)}" for k, v in out["auc_all_B_words"].items()))
    del Au, Ar, Am, CA, CR; torch.cuda.empty_cache()
mid = res["by_block"][NB // 2]["auc_entrants"]
res["checks"] = dict(floor_over_0_8_and_ahead_of_magnitude=(mid["max_over_floor"] or 0) > 0.8 and (mid["max_over_floor"] or 0) > (mid["magnitude"] or 0), own_removed_keeps_0_8=((mid["max_over_floor_own_removed"] or 0.5) - 0.5) >= 0.8 * ((mid["max_over_floor"] or 0.5) - 0.5))
summ = (f"{name}: " + " | ".join(f"block {b}: overlap {o['overlap_of_word_sets']:.2f}, {o['n_cross_text_entrants']} new words on B; AUC from A: " + ", ".join(f"{k} {('n/a' if v is None else f'{v:.2f}')}" for k, v in o["auc_entrants"].items()) for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e518_crosstext_{name}", res, summ)
