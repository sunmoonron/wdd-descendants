"""e565 (session 102): what has to travel with the neuron for the word to survive? e558 transplanted the step-16000
neuron of an entrant into the step-8000 model whole (read row, bias, write column: 0.68 became words) or by its write
column alone (0.59). Here the three parts factorially, for entrants (words at 16000 only) and stayers (words at both):
W (write column), Wd (its direction at the old norm), Wn (its norm at the old direction), R (read row), b (bias), WR, Wb,
Rb, WRb, each transplanted for all rows of the group at once, and
W, R, WR, WRb one at a time for twenty entrants. The read row and bias carry the neuron's input selectivity, the write
column its output direction. Pre-registered (probabilities are honest guesses):
 F1 (0.6) the new selectivity with the old direction (Rb) makes fewer entrants words than the new direction with the
    old selectivity (W): the direction is what the cloud speaks, the selectivity is already close;
 F2 (0.6) the bias adds little (WR within 0.05 of WRb);
 F3 (0.5) read and write are separable: word(WR) - word(W) - word(R) + word(none) is within 0.1 of zero."""
from s101_common import *
t0 = time.time(); B = 12; ids = pile_ids("pythia410")
m8, tok, fam = load_model("pythia410", revision="step8000"); m16, _, _ = load_model("pythia410", revision="step16000"); arch8, arch16 = Arch(m8, fam), Arch(m16, fam); DFF = arch8.DFF
def analyse(model, keepc=None):
    S = lm_states("pythia410", B=B, ids=ids, model=model)
    if keepc is not None: S["U"] = unitr(S["X"][keepc] - S["X"][keepc].mean(0))
    st = stats(S["U"], S["A"], K); return st, wordset(st["usage"])
st8, w8 = analyse(m8); st16, w16 = analyse(m16); keepc = torch.load("/dev/null") if False else None
S8 = lm_states("pythia410", B=B, ids=ids, model=m8); S16 = lm_states("pythia410", B=B, ids=ids, model=m16); keepc = S8["keep"] & S16["keep"]; st8, w8 = analyse(m8, keepc); st16, w16 = analyse(m16, keepc)
g = torch.Generator().manual_seed(0); ent = torch.nonzero(w16 & ~w8)[:, 0]; stay = torch.nonzero(w16 & w8)[:, 0]
def rank_of(usage, rows):
    order = usage.argsort(descending=True); rk = torch.empty_like(order); rk[order] = torch.arange(order.numel()); return rk[rows]
def summarise(st, words, rows): return dict(median_S=float(st["S"][rows].median()), share_over_floor=float((st["S"][rows] >= 1).float().mean()), median_cnt=float(st["cnt"][rows].median()), share_word=float(words[rows].float().mean()), median_usage_rank=float(rank_of(st["usage"], rows).float().median()), n=int(rows.numel()))
def transplant(rows, parts):
    saved = []
    with torch.no_grad():
        for r in rows.tolist():
            b, j = r // DFF, r % DFF; l8, l16 = arch8.layers[b].mlp, arch16.layers[b].mlp
            saved.append((b, j, l8.dense_4h_to_h.weight[:, j].clone(), l8.dense_h_to_4h.weight[j].clone(), l8.dense_h_to_4h.bias[j].clone()))
            if "Wd" in parts: l8.dense_4h_to_h.weight[:, j] = l16.dense_4h_to_h.weight[:, j] / l16.dense_4h_to_h.weight[:, j].norm() * l8.dense_4h_to_h.weight[:, j].norm()   # the new direction at the old norm
            elif "Wn" in parts: l8.dense_4h_to_h.weight[:, j] = l8.dense_4h_to_h.weight[:, j] / l8.dense_4h_to_h.weight[:, j].norm() * l16.dense_4h_to_h.weight[:, j].norm()   # the old direction at the new norm
            elif "W" in parts: l8.dense_4h_to_h.weight[:, j] = l16.dense_4h_to_h.weight[:, j]
            if "R" in parts: l8.dense_h_to_4h.weight[j] = l16.dense_h_to_4h.weight[j]
            if "b" in parts: l8.dense_h_to_4h.bias[j] = l16.dense_h_to_4h.bias[j]
    def restore():
        with torch.no_grad():
            for b, j, wc, wr, bb in saved: arch8.layers[b].mlp.dense_4h_to_h.weight[:, j] = wc; arch8.layers[b].mlp.dense_h_to_4h.weight[j] = wr; arch8.layers[b].mlp.dense_h_to_4h.bias[j] = bb
    return restore
COMBOS = ["W", "Wd", "Wn", "R", "b", "WR", "Wb", "Rb", "WRb"]; res = dict(groups={}, singles={})
for gname, rows in (("entrants", ent), ("stayers", stay)):
    out = dict(none=summarise(st8, w8, rows), at_16000=summarise(st16, w16, rows))
    for c in COMBOS:
        restore = transplant(rows, c); stx, wx = analyse(m8, keepc); restore(); out[c] = summarise(stx, wx, rows)
    res["groups"][gname] = out; log(f"{gname} ({rows.numel()}): word share none {out['none']['share_word']:.2f} | " + " | ".join(f"{c} {out[c]['share_word']:.2f} (S {out[c]['median_S']:.2f})" for c in COMBOS) + f" | at 16000 {out['at_16000']['share_word']:.2f}")
sing = ent[torch.randperm(ent.numel(), generator=g)[:20]].tolist()
for c in ("W", "R", "WR", "WRb"):
    words_, Ss = [], []
    for r in sing:
        restore = transplant(torch.tensor([r]), c); stx, wx = analyse(m8, keepc); restore(); words_.append(float(wx[r])); Ss.append(float(stx["S"][r]))
    res["singles"][c] = dict(share_word=mean(words_), median_S=med(Ss)); log(f"single transplants ({c}): become words {mean(words_):.2f}, median S {med(Ss):.2f}")
E = res["groups"]["entrants"]; add = E["WR"]["share_word"] - E["W"]["share_word"] - E["R"]["share_word"] + E["none"]["share_word"]; res["entrants_interaction_WR"] = add
summ = (f"factorial transplant of the 16000 neuron into the 8000 model, entrants ({E['none']['n']}; word share none {E['none']['share_word']:.2f}, at 16000 {E['at_16000']['share_word']:.2f}): W {E['W']['share_word']:.2f} (direction only {E['Wd']['share_word']:.2f}, norm only {E['Wn']['share_word']:.2f}), R {E['R']['share_word']:.2f}, b {E['b']['share_word']:.2f}, WR {E['WR']['share_word']:.2f}, Wb {E['Wb']['share_word']:.2f}, Rb {E['Rb']['share_word']:.2f}, WRb {E['WRb']['share_word']:.2f} "
        f"(median S none {E['none']['median_S']:.2f}, W {E['W']['median_S']:.2f}, R {E['R']['median_S']:.2f}, Rb {E['Rb']['median_S']:.2f}, WR {E['WR']['median_S']:.2f}, WRb {E['WRb']['median_S']:.2f}; interaction WR - W - R + none {add:+.2f}); singles W/R/WR/WRb {res['singles']['W']['share_word']:.2f}/{res['singles']['R']['share_word']:.2f}/{res['singles']['WR']['share_word']:.2f}/{res['singles']['WRb']['share_word']:.2f}; "
        f"stayers stay under W/R/WRb {res['groups']['stayers']['W']['share_word']:.2f}/{res['groups']['stayers']['R']['share_word']:.2f}/{res['groups']['stayers']['WRb']['share_word']:.2f} | {time.time() - t0:.0f}s")
log(summ); record("e565_factorial_transplant", res, summ)
