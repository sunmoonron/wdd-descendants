"""e165: does WDD's readable set carry more function than an equal-energy unreadable set? For 8 sequences at the
mid layer: the true writes (ledger contributions, not refits) that OMP identified (real atoms in the support) are
removed from the state exactly; compare with removing an energy-matched random subset of the token's OTHER true
writes (|c| >= 5% of max, not in the support), and with removing the same number of the largest unidentified
writes. Spliced cross-entropy; the split by birth block (block 0 vs later) for the top-3 writes as well."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; ids = torch.arange(NT)
Xraw = c.X(L, center=False)[ids]; mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; A, lab = c.dictionary(L); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * C.abs().max(1, keepdim=True).values; big = C.abs() >= thr
mlp_rows = torch.cat([c.atom_index(L, torch.full((c.DFF,), b), torch.arange(c.DFF)) for b in range(L + 1)]).to(DEV); Am = A[mlp_rows]
sel, cof, err = omp(X, A, 64); ismlp = typA[sel] == T_MLP; key = torch.where(ismlp, blkA[sel] * c.DFF + idxA[sel], torch.zeros_like(sel)); real = ismlp & (C.gather(1, key).abs() >= thr)
identified = torch.zeros_like(big); identified.scatter_(1, key, real); identified &= big
g = torch.Generator().manual_seed(0)
def contribution(mask): return (C * mask.float()) @ Am                                                                        # exact sum of the masked true writes
v_id = contribution(identified); e_id = (v_id ** 2).sum(1)
# energy-matched random unidentified true writes: greedily add random big unidentified writes until the energy matches
unid = big & ~identified; v_rnd = torch.zeros_like(Xraw); v_top = torch.zeros_like(Xraw)
for i in range(NT):
    cand = torch.nonzero(unid[i])[:, 0]
    if len(cand) == 0: continue
    perm = cand[torch.randperm(len(cand), generator=g).to(DEV)]; acc = torch.zeros(c.D, device=DEV)
    for j in perm.tolist():
        acc = acc + C[i, j] * Am[j]
        if (acc ** 2).sum() >= e_id[i]: break
    v_rnd[i] = acc
    order = cand[C[i, cand].abs().argsort(descending=True)]; n_id = int(identified[i].sum()); v_top[i] = (C[i, order[:n_id]][:, None] * Am[order[:n_id]]).sum(0)
def ce_with(rep):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out; o = o.clone(); o[:] = rep.view(NS, CTX, c.D).to(o.dtype); return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
    h = arch.layers[L].register_forward_hook(hook); loss = model(ids_seq, labels=ids_seq).loss.item(); h.remove(); return loss
base = model(ids_seq, labels=ids_seq).loss.item(); en = lambda v: ((v ** 2).sum(1) / (X ** 2).sum(1)).median().item()
tb, tn, tc = c.top_writes(L, 3); rows3 = torch.stack([c.atom_index(L, tb[ids, j], tn[ids, j]) for j in range(3)], 1).to(DEV); b0 = (tb[ids] == 0).to(DEV)
v3 = torch.einsum("nk,nkd->nd", tc[ids].to(DEV), A[rows3]); v3_b0 = torch.einsum("nk,nkd->nd", tc[ids].to(DEV) * b0.float(), A[rows3]); v3_later = v3 - v3_b0
res = dict(model=tag, L=L, base_ce=base, identified_true_writes_per_token=identified.float().sum(1).mean().item(),
           remove=dict(identified_true_writes=dict(dce=ce_with(Xraw - v_id) - base, energy=en(v_id)), energy_matched_random_unidentified=dict(dce=ce_with(Xraw - v_rnd) - base, energy=en(v_rnd)), same_count_largest_unidentified=dict(dce=ce_with(Xraw - v_top) - base, energy=en(v_top)),
                       top3_block0_part=dict(dce=ce_with(Xraw - v3_b0) - base, energy=en(v3_b0)), top3_later_part=dict(dce=ce_with(Xraw - v3_later) - base, energy=en(v3_later)), top3_all=dict(dce=ce_with(Xraw - v3) - base, energy=en(v3))))
record(f"e165_readable_{tag}", res, f"identified true writes/token {res['identified_true_writes_per_token']:.1f} | remove identified true writes: dCE {res['remove']['identified_true_writes']['dce']:+.3f} (energy {res['remove']['identified_true_writes']['energy']:.2f}) | energy-matched random unidentified: {res['remove']['energy_matched_random_unidentified']['dce']:+.3f} ({res['remove']['energy_matched_random_unidentified']['energy']:.2f}) | same-count largest unidentified: {res['remove']['same_count_largest_unidentified']['dce']:+.3f} ({res['remove']['same_count_largest_unidentified']['energy']:.2f}) | top-3 block-0 part {res['remove']['top3_block0_part']['dce']:+.3f} ({res['remove']['top3_block0_part']['energy']:.2f}) vs later part {res['remove']['top3_later_part']['dce']:+.3f} ({res['remove']['top3_later_part']['energy']:.2f})")
