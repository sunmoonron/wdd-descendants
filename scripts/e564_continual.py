"""e564 (session 101): what does continued training on a new domain do to the vocabulary? Pythia-160m fine-tuned on
TinyStories (children's stories, not in the Pile) for 300 full steps of AdamW (lr 2e-5, batch 8 x 256 tokens; the first run
forgot two nats of Pile loss, so gentler learning rates 5e-6 and 2e-6 are run by the same script with the rate as its
argument). Before,
at 100, 200 and 300 steps: the 256 words (rows of blocks 0-6, block-6 states) on the Pile evaluation contexts and on
held-out TinyStories contexts; the unit rows; the losses on both domains. Reported: retention of the Pile word set
(Jaccard with the start), the rows' rotation (1 - cosine with the start) for words against norm-matched non-words of
the same block, the usage correlation with the start, replacement (new Pile words after fine-tuning: were they
TinyStories words before?), and the loss on each domain. Pre-registered (probabilities are honest guesses):
 C1 (0.7) the Pile word set is retained at Jaccard >= 0.8 after 300 steps;
 C2 (0.5) words rotate less than norm-matched non-words of the same block;
 C3 (0.5) more than half of the new Pile words were TinyStories words before fine-tuning (the domain's vocabulary
    spreads into the old contexts);
 C4 (0.7) the Pile loss rises by under 0.1 nats while the TinyStories loss falls by more than 0.3."""
from s101_common import *
from datasets import load_dataset
torch.set_grad_enabled(True); t0 = time.time(); STEPS, EVERY, BS, T = 300, 100, 8, 256; LR = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-5; TAG = "" if LR == 2e-5 else f"_lr{sys.argv[1]}"
model, tok, fam = load_model("pythia160"); arch = Arch(model, fam); B = 6
def pack(split, nwin, skip=0):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True); buf = []; wins = []; n = 0
    for ex in ds:
        n += 1
        if n <= skip: continue
        buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
        while len(buf) >= T + 1 and len(wins) < nwin: wins.append(buf[:T + 1]); buf = buf[T + 1:]
        if len(wins) >= nwin: break
    return torch.tensor(wins, device=DEV)
train = pack("train", STEPS * BS); tiny_eval = pack("validation", 8)[:, :T]; pile = pile_ids("pythia160"); pile2 = pile_ids("pythia160", start=8)
log(f"data ready: {train.shape[0]} training windows in {time.time() - t0:.0f}s")
def lossof(ids):
    with torch.no_grad(): lg = model(ids).logits.float()
    return float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)))
def measure():
    model.eval(); out = {}
    for tag, ids in (("pile", pile), ("tiny", tiny_eval)):
        S = lm_states("pythia160", B=B, ids=ids, model=model); st = stats(S["U"], S["A"], K); out[tag] = dict(words=wordset(st["usage"]), usage=st["usage"], S=st["S"], cnt=st["cnt"])
    out["A"] = rows_of(arch, B)[0].cpu(); out["norms"] = rows_of(arch, B)[1].cpu(); out["loss_pile"] = (lossof(pile) + lossof(pile2)) / 2; out["loss_tiny"] = lossof(tiny_eval); return out
H = {0: measure()}; log(f"step 0: loss pile/tiny {H[0]['loss_pile']:.3f}/{H[0]['loss_tiny']:.3f}")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0); model.train()
for step in range(1, STEPS + 1):
    ids = train[(step - 1) * BS: step * BS]; lg = model(ids[:, :T]).logits.float()
    loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:T].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % EVERY == 0:
        H[step] = measure(); model.train(); log(f"step {step}: train loss {float(loss):.3f}, eval loss pile/tiny {H[step]['loss_pile']:.3f}/{H[step]['loss_tiny']:.3f} | {time.time() - t0:.0f}s")
h0 = H[0]; A0 = h0["A"]; blk = torch.arange(A0.shape[0]) // arch.DFF; res = dict(steps=sorted(H), rows={}, panels={})
w0p, w0t = h0["pile"]["words"], h0["tiny"]["words"]; nw0 = ~w0p
# norm-matched non-word for each Pile word within the same block
match = torch.full((int(w0p.sum()),), -1, dtype=torch.long); wi = torch.nonzero(w0p)[:, 0]
for q, i in enumerate(wi):
    cand = torch.nonzero(nw0 & (blk == blk[i]))[:, 0]; match[q] = cand[(h0["norms"][cand] - h0["norms"][i]).abs().argmin()]
for s in res["steps"]:
    h = H[s]; rot = 1 - (A0 * h["A"]).sum(1); wp, wt = h["pile"]["words"], h["tiny"]["words"]
    new = wp & ~w0p; lost = w0p & ~wp
    r = dict(loss_pile=h["loss_pile"], loss_tiny=h["loss_tiny"], pile_words_jaccard=float((wp & w0p).sum() / (wp | w0p).sum()), tiny_words_jaccard=float((wt & w0t).sum() / (wt | w0t).sum()),
             pile_tiny_overlap=float((wp & wt).sum() / (wp | wt).sum()), n_new_pile_words=int(new.sum()), new_pile_words_were_tiny_words=float(w0t[new].float().mean()) if new.any() else None,
             lost_words_usage_rank_median=float(h0["pile"]["usage"][lost].median()) if lost.any() else None, usage_threshold_start=float(h0["pile"]["usage"][w0p].min()),
             rot_words_median=float(rot[w0p].median()), rot_matched_nonwords_median=float(rot[match].median()), rot_all_median=float(rot.median()), rot_words_mean=float(rot[w0p].mean()), rot_matched_mean=float(rot[match].mean()),
             share_words_rotate_less_than_match=float((rot[wi] < rot[match]).float().mean()), usage_corr_pile=float(torch.corrcoef(torch.stack([h0["pile"]["usage"], h["pile"]["usage"]]))[0, 1]),
             median_S_pile_words=float(h["pile"]["S"][w0p].median()), share_start_words_over_floor=float((h["pile"]["S"][w0p] >= 1).float().mean()), rot_by_block_words=[float(rot[w0p & (blk == b)].median()) if (w0p & (blk == b)).any() else None for b in range(B + 1)],
             rot_by_block_all=[float(rot[blk == b].median()) for b in range(B + 1)])
    res["rows"][s] = r; log(f"step {s}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, list)}))
f = res["rows"][STEPS]
summ = (f"Pythia-160m, 300 steps on TinyStories: loss pile {h0['loss_pile']:.3f}->{f['loss_pile']:.3f}, tiny {h0['loss_tiny']:.3f}->{f['loss_tiny']:.3f}; Pile word set Jaccard with the start {res['rows'][100]['pile_words_jaccard']:.2f}/{res['rows'][200]['pile_words_jaccard']:.2f}/{f['pile_words_jaccard']:.2f} at 100/200/300 "
        f"(TinyStories words {f['tiny_words_jaccard']:.2f}; Pile-TinyStories overlap {res['rows'][0]['pile_tiny_overlap']:.2f}->{f['pile_tiny_overlap']:.2f}); usage correlation {f['usage_corr_pile']:.2f}; start words still over the floor {f['share_start_words_over_floor']:.2f}; "
        f"{f['n_new_pile_words']} new Pile words of which {f['new_pile_words_were_tiny_words'] if f['new_pile_words_were_tiny_words'] is None else round(f['new_pile_words_were_tiny_words'], 2)} were TinyStories words before; "
        f"rotation (1 - cos) words {f['rot_words_median']:.4f} vs norm-matched non-words {f['rot_matched_nonwords_median']:.4f} (all rows {f['rot_all_median']:.4f}; words rotate less in {f['share_words_rotate_less_than_match']:.2f} of pairs) | {time.time() - t0:.0f}s")
res["lr"] = LR; summ = f"lr {LR:g}: " + summ; log(summ); record(f"e564_continual{TAG}", res, summ)
