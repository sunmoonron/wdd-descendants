"""e623 (session 120): blind detection of single-row weight edits. With the original model in hand a one-row edit is found by
a weight diff, so the only interesting version is blind: the detector sees the edited model and a neutral corpus. The
dictionary-native statistic: a word's row should agree with the rest of what its class positions are written with (the
chord, e519b); for each word of block b, read at the block's output, the cosine between the row and the mean over its
class positions of the state minus the row's own write; a negated row disagrees with its chord. The naive statistic is
the same cosine for every neuron of the block over its top-activation positions, no dictionary. The norm z-score within
the block is the third detector, for scaled rows. Forty blinded edits of block 4 rows (ten word rows negated, ten word
rows doubled, ten non-word rows negated, ten non-word rows doubled), each detector ranking the block's rows by anomaly,
scored by the edited row's rank (top-1, top-10), and by the false-positive rate at a threshold set on the unedited model.
Pre-registered in e623_prereg.json: K1 (0.6) the word-chord detector puts a negated word row at top-1 in 70% or more of
the ten; K2 (0.3) it beats the naive top-activation detector on those; K3 (0.7) negated non-word rows are missed by the
word detector (not in its vocabulary) and found by the naive one at top-10 in 30% or more; K4 (0.7) doubled rows are
found by the norm z-score at top-10 in 50% or more and by neither consistency detector. Arguments: model [--smoke]."""
import sys, os, time, copy, math, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
from ma_common import out_of, block_states
from datasets import load_dataset
import wdd_common
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); torch.set_grad_enabled(False)
model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; DFF = arch.DFF; assert fam == "llama"; b = 4; NW = 4 if SMOKE else 16; T = 256; NE = 2 if SMOKE else 10; TOPK = 32
pile = load_dataset("NeelNanda/pile-10k", split="train"); buf, wins = [], []
for ex in pile:
    buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
    while len(buf) >= T and len(wins) < NW: wins.append(buf[:T]); buf = buf[T:]
    if len(wins) >= NW: break
PW = torch.tensor(wins, device=DEV)
def level_states_and_acts():
    """block b's output states (positions 1..T-1, flattened) and every block-b neuron's activation there"""
    cap = {}
    def hk_a(mod, inp): cap.setdefault("a", []).append(inp[0][:, 1:, :].detach().float())
    def hk_x(mod, i, o): cap.setdefault("x", []).append(out_of(o)[:, 1:, :].detach().float())
    h1 = arch.layers[b].mlp.down_proj.register_forward_pre_hook(hk_a); h2 = arch.layers[b].register_forward_hook(hk_x)
    try:
        for s0 in range(0, PW.shape[0], 8): model(PW[s0:s0 + 8])
    finally: h1.remove(); h2.remove()
    return torch.cat(cap["x"]).reshape(-1, D), torch.cat(cap["a"]).reshape(-1, DFF)
def analyse():
    """both consistency statistics on the current weights: the word-chord cosine over the block's words (dictionary of rows of blocks 0..b, read at the block's output) and the naive top-activation cosine over every neuron of the block"""
    X, Aact = level_states_and_acts(); keep = ~sinkmask(X); mu = X[keep].mean(0); U = unitr(X[keep] - mu); Ab, norms = rows_of(arch, b); st = stats(U, Ab, K); words = torch.nonzero(wordset(st["usage"]))[:, 0]; R = st["ratio"].float(); kidx = torch.nonzero(keep)[:, 0]; Xc = X - mu[None]
    W = arch.layers[b].mlp.down_proj.weight.detach().float()   # [D, DFF]: column j is neuron j's write
    wdd = {}   # every row of block b with a class of 5 or more positions (the class is floor-based), the vocabulary rows marked
    nclass = (R[:, b * DFF:(b + 1) * DFF] > 1).sum(0)
    for j in torch.nonzero(nclass >= 5)[:, 0].tolist():
        cls = kidx[R[:, b * DFF + j] > 1]; row = W[:, j]; rest = Xc[cls] - Aact[cls, j][:, None] * row[None]; wdd[j] = float(torch.nn.functional.cosine_similarity(row[None], rest.mean(0, keepdim=True))[0])
    naive = torch.zeros(DFF, device=DEV)
    for j0 in range(0, DFF, 512):
        js = torch.arange(j0, min(j0 + 512, DFF), device=DEV); top = Aact[:, js].abs().topk(TOPK, dim=0).indices   # [TOPK, nj]
        for i, j in enumerate(js.tolist()):
            idx = top[:, i]; row = W[:, j]; rest = Xc[idx] - Aact[idx, j][:, None] * row[None]; naive[j] = torch.nn.functional.cosine_similarity(row[None], rest.mean(0, keepdim=True))[0]
    nz = W.norm(dim=0); z = (nz - nz.mean()) / nz.std()
    return wdd, naive, z, set(int(w % DFF) for w in words.tolist() if w // DFF == b)
wdd0, naive0, z0, words0 = analyse(); thr_w = sorted(wdd0.values())[max(int(0.01 * len(wdd0)) - 1, 0)]; thr_n = float(naive0.kthvalue(max(int(0.01 * DFF), 1)).values)
log(f"{name} block {b}: {len(wdd0)} rows of the block with classes ({len(words0)} vocabulary words); word-chord cosine median {sorted(wdd0.values())[len(wdd0) // 2]:.3f} (1st percentile {thr_w:.3f}); naive cosine median {float(naive0.median()):.3f} (1st percentile {thr_n:.3f}); words' naive cosine median {float(naive0[list(wdd0)].median()):.3f} ({time.time() - t0:.0f}s)")
g = torch.Generator().manual_seed(5); wl = sorted(wdd0); nonw = [j for j in range(DFF) if j not in words0]
pick_w = [wl[i] for i in torch.randperm(len(wl), generator=g)[:2 * NE].tolist()]; pick_n = [nonw[i] for i in torch.randperm(len(nonw), generator=g)[:2 * NE].tolist()]
EDITS = [("word", "negate", j) for j in pick_w[:NE]] + [("word", "double", j) for j in pick_w[NE:]] + [("nonword", "negate", j) for j in pick_n[:NE]] + [("nonword", "double", j) for j in pick_n[NE:]]
Wm = arch.layers[b].mlp.down_proj.weight
def rank_of(scores_dict_or_tensor, j, ascending=True):
    if isinstance(scores_dict_or_tensor, dict):
        if j not in scores_dict_or_tensor: return None
        vals = sorted(scores_dict_or_tensor.items(), key=lambda kv: kv[1] if ascending else -kv[1]); return [k for k, _ in vals].index(j) + 1
    s = scores_dict_or_tensor; order = s.argsort(descending=not ascending); return int((order == j).nonzero()[0, 0]) + 1
res = dict(model=name, block=b, n_windows=NW, unedited=dict(n_rows_with_class=len(wdd0), n_words_block=len(words0), wdd_median=sorted(wdd0.values())[len(wdd0) // 2], wdd_p01=thr_w, naive_median=float(naive0.median()), naive_p01=thr_n, fpr_wdd=float(sum(v < thr_w for v in wdd0.values()) / len(wdd0)), fpr_naive=float((naive0 < thr_n).float().mean())), edits=[])
for kind, how, j in EDITS:
    saved = Wm[:, j].clone()
    if how == "negate": Wm[:, j] = -saved
    else: Wm[:, j] = 2 * saved
    wdd1, naive1, z1, words1 = analyse(); Wm[:, j] = saved
    ent = dict(kind=kind, edit=how, row=j, in_vocabulary_after=(j in wdd1), wdd_score=wdd1.get(j), naive_score=float(naive1[j]), z_score=float(z1[j]), rank_wdd=rank_of(wdd1, j), n_wdd=len(wdd1), rank_naive=rank_of(naive1, j), rank_norm=rank_of(z1.abs(), j, ascending=False), flagged_wdd=(wdd1.get(j) is not None and wdd1[j] < thr_w), flagged_naive=bool(naive1[j] < thr_n), fp_wdd=int(sum(v < thr_w for k_, v in wdd1.items() if k_ != j)), fp_naive=int(((naive1 < thr_n) & (torch.arange(DFF, device=DEV) != j)).sum()))
    res["edits"].append(ent); log(f"{kind} row {j} {how}: word-chord {ent['wdd_score']} (rank {ent['rank_wdd']} of {ent['n_wdd']}), naive {ent['naive_score']:.3f} (rank {ent['rank_naive']} of {DFF}), norm z {ent['z_score']:+.2f} (rank {ent['rank_norm']}); flagged wdd {ent['flagged_wdd']} naive {ent['flagged_naive']}; false positives wdd {ent['fp_wdd']} naive {ent['fp_naive']} ({time.time() - t0:.0f}s)")
def rate(sel, key, k):
    E = [e for e in res["edits"] if sel(e)]; return (sum(1 for e in E if e[key] is not None and e[key] <= k) / len(E)) if E else None
S = {}
for kind in ("word", "nonword"):
    for how in ("negate", "double"):
        sel = lambda e, kind=kind, how=how: e["kind"] == kind and e["edit"] == how
        S[f"{kind}_{how}"] = dict(n=len([e for e in res["edits"] if sel(e)]), wdd_top1=rate(sel, "rank_wdd", 1), wdd_top10=rate(sel, "rank_wdd", 10), naive_top1=rate(sel, "rank_naive", 1), naive_top10=rate(sel, "rank_naive", 10), norm_top10=rate(sel, "rank_norm", 10), flagged_wdd=mean([float(e["flagged_wdd"]) for e in res["edits"] if sel(e)]) if any(sel(e) for e in res["edits"]) else None, flagged_naive=mean([float(e["flagged_naive"]) for e in res["edits"] if sel(e)]) if any(sel(e) for e in res["edits"]) else None, mean_fp_wdd=mean([e["fp_wdd"] for e in res["edits"] if sel(e)]), mean_fp_naive=mean([e["fp_naive"] for e in res["edits"] if sel(e)]))
res["summary_rates"] = S
k1 = (S["word_negate"]["wdd_top1"] or 0) >= 0.7; k2 = (S["word_negate"]["wdd_top1"] or 0) > (S["word_negate"]["naive_top1"] or 0); k3 = (S["nonword_negate"]["wdd_top10"] or 0) == 0 and (S["nonword_negate"]["naive_top10"] or 0) >= 0.3; k4 = (S["word_double"]["norm_top10"] or 0) >= 0.5 and (S["nonword_double"]["norm_top10"] or 0) >= 0.5 and (S["word_double"]["wdd_top10"] or 0) <= 0.2 and (S["word_double"]["naive_top10"] or 0) <= 0.2
res["verdicts"] = dict(K1=k1, K2=k2, K3=k3, K4=k4)
summ = (f"blind edit detection ({name}, block {b}, {len(EDITS)} edits): " + "; ".join(f"{k}: word-chord top-1 {S[k]['wdd_top1']} top-10 {S[k]['wdd_top10']}, naive top-1 {S[k]['naive_top1']} top-10 {S[k]['naive_top10']}, norm top-10 {S[k]['norm_top10']}, flagged wdd/naive {S[k]['flagged_wdd']}/{S[k]['flagged_naive']} with {S[k]['mean_fp_wdd']:.1f}/{S[k]['mean_fp_naive']:.1f} false positives" for k in S) + f"; unedited false-positive rates at the 1st-percentile thresholds wdd {res['unedited']['fpr_wdd']:.3f} naive {res['unedited']['fpr_naive']:.3f}; verdicts K1 {k1}, K2 {k2}, K3 {k3}, K4 {k4}")
log(summ); record(f"e623_edit_detection_{name}" + ("_smoke" if SMOKE else ""), res, summ)
