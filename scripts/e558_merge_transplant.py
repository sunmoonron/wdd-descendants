"""e558 (session 101): merging and transplanting. (a) Merging along the trajectory: Pythia-410m at steps 8000 and 16000
averaged with weight alpha (0.25, 0.5, 0.75; all parameters). The merged model's loss on the Pile sequences, its 256
words (block 12, rows of blocks 0-12) against the parents': Jaccard with each parent, the share of merged words that
are words of both parents, of one, of neither; the usage's correlation with the interpolated parents' usage.
(b) Transplanting a word into the step-8000 model at three depths, for entrants (words at 16000 that are not words at
8000), stayers (words at both) and never-words (random non-words at both):
   dictionary only: the 16000 row's direction placed in the 8000 dictionary (is the direction spoken by the 8000 cloud?);
   write only: the 16000 write column of the neuron copied into the 8000 model (the direction written by the old
   activations);
   whole neuron: read row, bias and write column copied (the 16000 neuron in the 8000 model).
For each group all rows are transplanted at once and the block-12 states recomputed; S, the count over the floor and the
usage rank of the transplanted rows are compared with their values at 8000 and at 16000; twenty entrants are also
transplanted one at a time (whole neuron). Pre-registered (probabilities are honest guesses):
 M1 (0.6) the alpha 0.5 merge's words are mostly words of both parents (share >= 0.7);
 M2 (0.7) the merged loss lies between the parents' (no barrier along the trajectory);
 T1 (0.65) dictionary only: the entrants' 16000 directions are not over the floor in the 8000 cloud (median S < 1);
 T2 (0.6) the whole-neuron transplant makes fewer than half of the entrants words (the coalition precedes the word);
 T3 (0.7) stayers remain words under every transplant."""
from s101_common import *
import copy
t0 = time.time(); B = 12; ids = pile_ids("pythia410")
m8, tok, fam = load_model("pythia410", revision="step8000"); m16, _, _ = load_model("pythia410", revision="step16000")
def lossof(model):
    with torch.no_grad(): lg = model(ids).logits.float()
    return float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)))
def analyse(model, keepc=None):
    S = lm_states("pythia410", B=B, ids=ids, model=model)
    if keepc is not None: S["U"] = unitr(S["X"][keepc] - S["X"][keepc].mean(0))
    st = stats(S["U"], S["A"], K); return S, st, wordset(st["usage"])
S8, st8, w8 = analyse(m8); S16, st16, w16 = analyse(m16); keepc = S8["keep"] & S16["keep"]
S8, st8, w8 = analyse(m8, keepc); S16, st16, w16 = analyse(m16, keepc); L8, L16 = lossof(m8), lossof(m16)
log(f"parents: loss {L8:.3f}/{L16:.3f}, words Jaccard {float((w8 & w16).sum() / (w8 | w16).sum()):.2f}, common positions {int(keepc.sum())}")
res = dict(loss_8000=L8, loss_16000=L16, parents_word_jaccard=float((w8 & w16).sum() / (w8 | w16).sum()), merges={}, transplants={})
# (a) merges
sd8 = {k: v.detach().clone() for k, v in m8.state_dict().items()}; sd16 = {k: v.detach().clone() for k, v in m16.state_dict().items()}
mm = copy.deepcopy(m8)
for alpha in (0.25, 0.5, 0.75):
    mm.load_state_dict({k: ((1 - alpha) * sd8[k].float() + alpha * sd16[k].float()).to(sd8[k].dtype) if sd8[k].is_floating_point() else sd8[k] for k in sd8})
    Sm, stm, wm = analyse(mm, keepc); Lm = lossof(mm); both = w8 & w16
    ui = (1 - alpha) * st8["usage"] + alpha * st16["usage"]
    r = dict(loss=Lm, jaccard_8000=float((wm & w8).sum() / (wm | w8).sum()), jaccard_16000=float((wm & w16).sum() / (wm | w16).sum()), share_in_both=float(both[wm].float().mean()), share_in_one=float(((w8 | w16) & ~both)[wm].float().mean()),
             share_in_neither=float((~(w8 | w16))[wm].float().mean()), usage_corr_interp=float(torch.corrcoef(torch.stack([stm["usage"], ui]))[0, 1]), usage_corr_8000=float(torch.corrcoef(torch.stack([stm["usage"], st8["usage"]]))[0, 1]),
             usage_corr_16000=float(torch.corrcoef(torch.stack([stm["usage"], st16["usage"]]))[0, 1]), median_S_words=float(stm["S"][wm].median()), row_cos_8000_16000=float((S8["A"] * S16["A"]).sum(1).median()))
    res["merges"][alpha] = r; log(f"merge alpha {alpha}: loss {Lm:.3f} (parents {L8:.3f}/{L16:.3f}); words Jaccard with 8000/16000 {r['jaccard_8000']:.2f}/{r['jaccard_16000']:.2f}; in both/one/neither {r['share_in_both']:.2f}/{r['share_in_one']:.2f}/{r['share_in_neither']:.2f}; usage corr with interpolation {r['usage_corr_interp']:.2f}")
del mm; torch.cuda.empty_cache()
# (b) transplants into the 8000 model
arch8 = Arch(m8, fam); DFF = arch8.DFF; g = torch.Generator().manual_seed(0)
ent = torch.nonzero(w16 & ~w8)[:, 0]; stay = torch.nonzero(w16 & w8)[:, 0]; never = torch.nonzero(~w16 & ~w8)[:, 0]; never = never[torch.randperm(never.numel(), generator=g)[:64]]
log(f"entrants {ent.numel()}, stayers {stay.numel()}, never-words sampled {never.numel()}")
def rank_of(usage, rows): 
    order = usage.argsort(descending=True); rk = torch.empty_like(order); rk[order] = torch.arange(order.numel()); return rk[rows]
def summarise(st, words, rows, tag):
    return dict(median_S=float(st["S"][rows].median()), share_over_floor=float((st["S"][rows] >= 1).float().mean()), median_cnt=float(st["cnt"][rows].median()), share_word=float(words[rows].float().mean()), median_usage_rank=float(rank_of(st["usage"], rows).float().median()), n=int(rows.numel()))
def transplant(rows, depth):
    """copy the 16000 neuron parameters for the given global row indices into m8 (write column, or read row + bias + write column); returns a restore function"""
    saved = []
    with torch.no_grad():
        for r in rows.tolist():
            b, j = r // DFF, r % DFF; l8, l16 = arch8.layers[b].mlp, Arch(m16, fam).layers[b].mlp
            saved.append((b, j, l8.dense_4h_to_h.weight[:, j].clone(), l8.dense_h_to_4h.weight[j].clone(), l8.dense_h_to_4h.bias[j].clone()))
            l8.dense_4h_to_h.weight[:, j] = l16.dense_4h_to_h.weight[:, j]
            if depth == "neuron": l8.dense_h_to_4h.weight[j] = l16.dense_h_to_4h.weight[j]; l8.dense_h_to_4h.bias[j] = l16.dense_h_to_4h.bias[j]
    def restore():
        with torch.no_grad():
            for b, j, wc, wr, bb in saved: arch8.layers[b].mlp.dense_4h_to_h.weight[:, j] = wc; arch8.layers[b].mlp.dense_h_to_4h.weight[j] = wr; arch8.layers[b].mlp.dense_h_to_4h.bias[j] = bb
    return restore
for gname, rows in (("entrants", ent), ("stayers", stay), ("never", never)):
    out = dict(at_8000=summarise(st8, w8, rows, "8000"), at_16000=summarise(st16, w16, rows, "16000"))
    # dictionary only: the 16000 directions in the 8000 dictionary
    Ad = S8["A"].clone(); Ad[rows] = S16["A"][rows]; std = stats(S8["U"], Ad, K); out["dictionary_only"] = summarise(std, wordset(std["usage"]), rows, "dict")
    for depth in ("write", "neuron"):
        restore = transplant(rows, depth); Sx, stx, wx = analyse(m8, keepc); restore()
        out[depth] = summarise(stx, wx, rows, depth); out[depth]["loss"] = None
    res["transplants"][gname] = out
    log(f"{gname} ({rows.numel()}): share word at 8000/16000 {out['at_8000']['share_word']:.2f}/{out['at_16000']['share_word']:.2f}; median S 8000/16000 {out['at_8000']['median_S']:.2f}/{out['at_16000']['median_S']:.2f} | dictionary only: S {out['dictionary_only']['median_S']:.2f}, word {out['dictionary_only']['share_word']:.2f} | write only: S {out['write']['median_S']:.2f}, word {out['write']['share_word']:.2f}, cnt {out['write']['median_cnt']:.0f} | whole neuron: S {out['neuron']['median_S']:.2f}, word {out['neuron']['share_word']:.2f}, cnt {out['neuron']['median_cnt']:.0f}")
# one at a time (whole neuron), twenty entrants
single = []
for r in ent[torch.randperm(ent.numel(), generator=g)[:20]].tolist():
    restore = transplant(torch.tensor([r]), "neuron"); Sx, stx, wx = analyse(m8, keepc); restore()
    single.append(dict(row=r, S=float(stx["S"][r]), cnt=float(stx["cnt"][r]), word=bool(wx[r]), rank=int(rank_of(stx["usage"], torch.tensor([r]))[0]), S_8000=float(st8["S"][r]), S_16000=float(st16["S"][r]), cnt_16000=float(st16["cnt"][r])))
res["single_entrants"] = single; sw = mean([float(s["word"]) for s in single]); sS = med([s["S"] for s in single])
log(f"single whole-neuron transplants of 20 entrants: become words {sw:.2f}, median S {sS:.2f} (at 8000 {med([s['S_8000'] for s in single]):.2f}, at 16000 {med([s['S_16000'] for s in single]):.2f})")
res["loss_after_group_transplant"] = {}
for gname, rows in (("entrants", ent), ("stayers", stay)):
    restore = transplant(rows, "neuron"); res["loss_after_group_transplant"][gname] = lossof(m8); restore()
T = res["transplants"]; M = res["merges"]
summ = (f"merge along the trajectory (8000/16000, loss {L8:.3f}/{L16:.3f}, parents' words Jaccard {res['parents_word_jaccard']:.2f}): alpha 0.25/0.5/0.75 loss {M[0.25]['loss']:.3f}/{M[0.5]['loss']:.3f}/{M[0.75]['loss']:.3f}, merged words in both parents {M[0.25]['share_in_both']:.2f}/{M[0.5]['share_in_both']:.2f}/{M[0.75]['share_in_both']:.2f}, in neither {M[0.25]['share_in_neither']:.2f}/{M[0.5]['share_in_neither']:.2f}/{M[0.75]['share_in_neither']:.2f}, usage corr with the interpolated parents {M[0.5]['usage_corr_interp']:.2f}; "
        f"transplants into 8000: entrants ({T['entrants']['at_8000']['n']}) median S at 8000/16000 {T['entrants']['at_8000']['median_S']:.2f}/{T['entrants']['at_16000']['median_S']:.2f}; dictionary only {T['entrants']['dictionary_only']['median_S']:.2f} (word {T['entrants']['dictionary_only']['share_word']:.2f}), write only {T['entrants']['write']['median_S']:.2f} (word {T['entrants']['write']['share_word']:.2f}), whole neuron {T['entrants']['neuron']['median_S']:.2f} (word {T['entrants']['neuron']['share_word']:.2f}; singly {sw:.2f}); "
        f"stayers stay words under dictionary/write/neuron {T['stayers']['dictionary_only']['share_word']:.2f}/{T['stayers']['write']['share_word']:.2f}/{T['stayers']['neuron']['share_word']:.2f}; never-words become words {T['never']['neuron']['share_word']:.2f}; loss after transplanting all entrants/stayers {res['loss_after_group_transplant']['entrants']:.3f}/{res['loss_after_group_transplant']['stayers']:.3f} | {time.time() - t0:.0f}s")
log(summ); record("e558_merge_transplant", res, summ)
