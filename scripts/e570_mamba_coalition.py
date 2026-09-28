"""e570 (session 102): is the mechanism behind Mamba's vocabulary the one found in Pythia and OLMo? Mamba-130m has the
words (e563) but no training checkpoints, so the developmental parts (precedence, entry, exit) cannot be measured; the
static parts can, at block 12 with the mixer output-projection rows of blocks 0-12 (19,968 writers, every one with an
activation, the input of its out_proj): for each of the 256 words its eight extreme positions; the writer-contribution
vector there (each writer's write projected on the word's direction); the within-row coherence (cosine between the
contribution vectors at two extremes of the same word) against across-row pairs and against random directions with
their own extremes; the effective number of writers; the word's own row's rank among the contributors; the share of the
extreme's positive contributions built by each block; and the input selectivity (z-score of the neuron's absolute activation at its extremes
against everywhere) for words, S-matched near misses (non-words at the words' S) and random rows. Pre-registered
(probabilities are honest guesses):
 C1 (0.7) within-row coherence far above across-row (Pythia 0.49 against 0.00, OLMo 0.73): a private coalition;
 C2 (0.6) the own row is among the top contributors (rank under 10 of 19,968) as on OLMo (rank 0);
 C3 (0.5) the near misses are as selective as the words (as on OLMo, unlike Pythia)."""
from s101_common import *
from transformers import MambaForCausalLM
t0 = time.time(); name = "state-spaces/mamba-130m-hf"; B = 12; KX = 8
model = MambaForCausalLM.from_pretrained(name, dtype=torch.float32).to(DEV).eval(); ids = pile_ids("pythia410"); L = model.backbone.layers
cap = {}; ACT = {}
class Stop(Exception): pass
hs = [L[b].mixer.out_proj.register_forward_pre_hook((lambda b_: lambda m, a: ACT.__setitem__(b_, a[0].detach().float()[:, 1:].reshape(-1, a[0].shape[-1])))(b)) for b in range(B + 1)]
def hk(m, i, o): cap["x"] = (o[0] if isinstance(o, tuple) else o).detach().float(); raise Stop
hs.append(L[B].register_forward_hook(hk))
try:
    with torch.no_grad(): model(ids)
except Stop: pass
finally: [h.remove() for h in hs]
X = cap["x"][:, 1:].reshape(-1, cap["x"].shape[-1]); keep = ~sinkmask(X); U = unitr(X[keep] - X[keep].mean(0))
Rows = torch.cat([L[b].mixer.out_proj.weight.detach().float().T for b in range(B + 1)]); norms = Rows.norm(dim=1); A = Rows / norms[:, None]; act = torch.cat([ACT[b] for b in range(B + 1)], 1)[keep]   # [N, m]
m, N = A.shape[0], U.shape[0]; DFF = ACT[0].shape[1]; del model; torch.cuda.empty_cache()
st = stats(U, A, K); words = torch.nonzero(wordset(st["usage"]))[:, 0]; Sw = st["S"]; nonw = torch.nonzero(~wordset(st["usage"]))[:, 0]
# S-matched near misses and random rows
g = torch.Generator().manual_seed(0); taken = torch.zeros(m, dtype=torch.bool); near = []
for r in words.tolist():
    cand = nonw[~taken[nonw]]; j = cand[(Sw[cand] - Sw[r]).abs().argmin()]; taken[j] = True; near.append(int(j))
near = torch.tensor(near); rnd = nonw[torch.randperm(nonw.numel(), generator=g)[:256]]
ratio = st["ratio"].float()   # [N, m] cpu
def extremes(rows): return ratio[:, rows].topk(KX, dim=0).indices.T   # [n_rows, KX] positions
def contributions(r, pos):
    """[KX, m]: each writer's write at the positions projected on row r's direction"""
    w = A[r]; proj = (A @ w) * norms; return act[pos] * proj[None]
def analyse(rows, tag, rand_dirs=False):
    cos_in, cos_x, own_rank, eff, blk_share, zsel, last = [], [], [], [], [], [], None
    E = extremes(rows) if not rand_dirs else None
    if rand_dirs:
        gg = torch.Generator(device=DEV).manual_seed(5); W = unitr(torch.randn(rows.numel(), A.shape[1], device=DEV, generator=gg)); E = (U @ W.T).abs().topk(KX, dim=0).indices.T.cpu()
    for q, r in enumerate(rows.tolist()):
        pos = E[q].to(DEV)
        if rand_dirs: w = W[q]; C = act[pos] * ((A @ w) * norms)[None]
        else: C = contributions(r, pos)
        Cu = unitr(C); cs = Cu @ Cu.T; cos_in.append(float((cs.sum() - KX) / (KX * (KX - 1))))
        if last is not None: cos_x.append(float((Cu @ last.T).mean()))
        last = Cu
        a = C.abs(); eff.append(float(((a.sum(1) ** 2) / (a ** 2).sum(1).clamp_min(1e-12)).mean()))
        if not rand_dirs:
            own_rank.append(float((a > a[:, r][:, None]).sum(1).float().mean())); pos_ = C.clamp_min(0); tot = pos_.sum(1).clamp_min(1e-12); blk_share.append(torch.stack([pos_[:, b * DFF:(b + 1) * DFF].sum(1) / tot for b in range(B + 1)]).mean(1))   # share of the positive (building) contributions by block
            ar = act[:, r].abs(); zsel.append(float((ar[pos].mean() - ar.mean()) / ar.std().clamp_min(1e-12)))
    out = dict(n=int(rows.numel()), within_row=med(cos_in), across_rows=med(cos_x), effective_writers=med(eff), S=float(Sw[rows].median()) if not rand_dirs else None)
    if not rand_dirs: out.update(own_rank_median=med(own_rank), own_rank_share_top10=mean([float(x < 10) for x in own_rank]), selectivity_z=med(zsel), block_shares=torch.stack(blk_share).mean(0).tolist())
    log(f"{tag} ({rows.numel()}): within-row coherence {out['within_row']:.3f}, across rows {out['across_rows']:.3f}, effective writers {out['effective_writers']:.0f}" + (f", own row's rank {out['own_rank_median']:.0f} (top-10 share {out['own_rank_share_top10']:.2f}), selectivity z {out['selectivity_z']:.2f}, S {out['S']:.2f}, block shares {[round(x, 2) for x in out['block_shares']]}" if not rand_dirs else ""))
    return out
res = dict(model=name, block=B, n_positions=N, n_writers=m, words=analyse(words, "words"), near_misses=analyse(near, "S-matched near misses"), random_rows=analyse(rnd, "random non-words"), random_directions=analyse(rnd, "random directions with their own extremes", rand_dirs=True))
w, nm, rr, rd = res["words"], res["near_misses"], res["random_rows"], res["random_directions"]
summ = (f"Mamba-130m block 12 coalitions ({N} positions, {m} writers): within-row coherence words {w['within_row']:.3f} vs across rows {w['across_rows']:.3f}, near misses {nm['within_row']:.3f}, random rows {rr['within_row']:.3f}, random directions {rd['within_row']:.3f}; effective writers {w['effective_writers']:.0f} (random directions {rd['effective_writers']:.0f}); "
        f"own row's rank among contributors words {w['own_rank_median']:.0f} (top-10 share {w['own_rank_share_top10']:.2f}), near misses {nm['own_rank_median']:.0f}, random rows {rr['own_rank_median']:.0f}; selectivity z words {w['selectivity_z']:.2f}, near misses {nm['selectivity_z']:.2f}, random rows {rr['selectivity_z']:.2f}; block shares of the words' extremes {[round(x, 2) for x in w['block_shares']]} | {time.time() - t0:.0f}s")
log(summ); record("e570_mamba_coalition", res, summ)
