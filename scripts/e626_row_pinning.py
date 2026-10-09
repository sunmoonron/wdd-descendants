"""e626 (session 121): programmable feature placement. A concept's carrier is a parameter row, so the row can be chosen.
Pythia-160m is fine-tuned on Pile windows with the next-token loss plus a pinning loss on a designated row r* of block b
(a row with no class, chosen at random): on a contrast set of English gendered nouns in templates, the state difference
between the female and the male noun at the word's last token, read at the block's output, should be carried by the
designated row's write, loss = the fraction of that difference's energy not explained by (the row's activation difference
times the row), with the row's direction and the activations both live. A control fine-tune has the next-token loss only.
Before and after: the designated row's vote share for the gender difference (OMP on the native rows of blocks up to b, as
e622), the dial (negating the designated row and the original handle row, the next-token probability of the subject's
pronoun over he and she on held-out gendered subjects, as e624), and the validation loss. Pre-registered in
e625_prereg.json: R1 (0.4) after pinning the designated row's vote share is 0.5 or more (from under 0.1) at a validation
cost of 0.05 nats or less over the control; R2 (0.4) negating the designated row after pinning lowers held-out pronoun
agreement by 0.15 or more (under 0.03 before, and under 0.03 in the control); R3 (0.3) the original handle's dial halves.
Arguments: model lambda [--smoke]; lambda 0 is the control."""
import sys, os, time, copy, math, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
from ma_common import out_of
from datasets import load_dataset
import wdd_common
from wdd_common import omp, build_dictionary
mean = lambda v: sum(v) / max(len(v), 1)
name, LAM = sys.argv[1], float(sys.argv[2]); SMOKE = "--smoke" in sys.argv; t0 = time.time(); torch.set_grad_enabled(False)
model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; DFF = arch.DFF; assert fam == "neox"; b = 4; T = 256; STEPS = 6 if SMOKE else 1200; NTR, NVA = (32, 8) if SMOKE else (10000, 24); LR = 1e-5
TRAIN_PAIRS = [("king", "queen"), ("father", "mother"), ("man", "woman"), ("boy", "girl"), ("uncle", "aunt"), ("nephew", "niece"), ("grandfather", "grandmother"), ("husband", "wife"), ("brother", "sister"), ("son", "daughter"), ("prince", "princess"), ("actor", "actress")]
HELD_PAIRS = [("waiter", "waitress"), ("hero", "heroine"), ("duke", "duchess"), ("lord", "lady"), ("policeman", "policewoman"), ("businessman", "businesswoman"), ("god", "goddess"), ("monk", "nun"), ("wizard", "witch"), ("groom", "bride"), ("steward", "stewardess"), ("gentleman", "gentlewoman")]
TEMPL = ["The {} walked into the room", "I saw the {} yesterday", "My {} said"]; PRON_T = ["The {} said that", "When the {} came home,", "The {} looked at the letter, and then"]
def word_prompts(pairs, templs):
    out = []
    for m, f in pairs:
        for t in templs:
            for g, w in ((0, m), (1, f)):
                p = t.format(w); a = p.index(w); b_ = a + len(w); enc = tok(p, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV); pos = [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_][-1]; out.append((ids, pos, g))
    return out
CON = word_prompts(TRAIN_PAIRS[:(3 if SMOKE else 12)], TEMPL); CONH = word_prompts(HELD_PAIRS[:(3 if SMOKE else 12)], TEMPL)
def pad(batch):
    Lm = max(ids.numel() for ids, _, _ in batch); ids = torch.full((len(batch), Lm), tok.eos_token_id, dtype=torch.long, device=DEV); mask = torch.zeros(len(batch), Lm, dtype=torch.long, device=DEV)
    for i, (x, _, _) in enumerate(batch): ids[i, :x.numel()] = x; mask[i, :x.numel()] = 1
    return ids, mask, torch.tensor([p for _, p, _ in batch], device=DEV), torch.tensor([g for _, _, g in batch], device=DEV)
def down(m): return Arch(m, fam).layers[b].mlp.dense_4h_to_h
def states_and_acts(m, ids, mask, pos):
    cap = {}
    def hk_a(mod, inp): cap["a"] = inp[0]
    def hk_x(mod, i, o): cap["x"] = out_of(o)
    h1 = down(m).register_forward_pre_hook(hk_a); h2 = Arch(m, fam).layers[b].register_forward_hook(hk_x)
    try: m(input_ids=ids, attention_mask=mask)
    finally: h1.remove(); h2.remove()
    ar = torch.arange(ids.shape[0], device=DEV); return cap["x"][ar, pos].float(), cap["a"][ar, pos].float()
def pin_loss(m, batch, r):
    ids, mask, pos, g = pad(batch); X, A_ = states_and_acts(m, ids, mask, pos); w = down(m).weight[:, r].float()   # the designated row's write
    Xm, Xf = X[g == 0], X[g == 1]; Am, Af = A_[g == 0, r], A_[g == 1, r]; d = Xf - Xm; da = (Af - Am)[:, None] * w[None]
    return ((d - da).pow(2).sum(-1) / d.pow(2).sum(-1).clamp_min(1e-6)).mean()
ds = load_dataset("NeelNanda/pile-10k", split="train")
def windows(nwin, skipdocs=0):
    wins, buf = [], []
    for k, ex in enumerate(ds):
        if k < skipdocs: continue
        buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
        while len(buf) >= T + 1 and len(wins) < nwin: wins.append(buf[:T + 1]); buf = buf[T + 1:]
        if len(wins) >= nwin: break
    return torch.tensor(wins, device=DEV)
tr = windows(NTR); va = windows(NVA, skipdocs=9000); log(f"{name} lambda {LAM}: {len(CON)} contrast prompts ({len(CONH)} held out), {tr.shape[0]} training windows ({time.time() - t0:.0f}s)")
def ce(lg, x): return torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
def val_loss(m):
    tot = 0.0
    for s0 in range(0, va.shape[0], 8): x = va[s0:s0 + 8]; tot += float(ce(m(x).logits.float(), x)) * x.shape[0]
    return tot / va.shape[0]
# ---- the vote (which row carries the gender difference) and the designated row
def vote(m, batch):
    A_, lab = build_dictionary(Arch(m, fam), blocks=list(range(b + 1))); Au = unitr(A_); del A_
    ids, mask, pos, g = pad(batch); X, _ = states_and_acts(m, ids, mask, pos); dif = X[g == 1] - X[g == 0]; sel, _, _ = omp(dif, Au, 8, batch=256, record_err=False); cnt = torch.bincount(sel.reshape(-1), minlength=Au.shape[0]); top = int(cnt.argmax())
    return dict(top_atom=top, top_type=int(lab["type"][top]), top_block=int(lab["block"][top]), top_index=int(lab["index"][top]), top_share=float(cnt[top]) / dif.shape[0]), cnt, lab
v0, cnt0, lab0 = vote(model, CON); log(f"before: the gender difference's top atom is type {v0['top_type']} block {v0['top_block']} row {v0['top_index']} in {v0['top_share']:.2f} of contrasts")
mlp_rows_b = [i for i in range(lab0["type"].shape[0]) if int(lab0["type"][i]) == 2 and int(lab0["block"][i]) == b]; share_of = lambda cnt, idx: float(cnt[idx]) / len(CON) * 2 / 1   # count over the 8 picks per contrast, contrasts = len(CON) / 2
handle = v0["top_index"] if (v0["top_type"] == 2 and v0["top_block"] == b) else None
g_r = torch.Generator().manual_seed(3); quiet = [i for i in mlp_rows_b if cnt0[i] == 0]; r_star_atom = quiet[int(torch.randint(0, len(quiet), (1,), generator=g_r))]; r_star = int(lab0["index"][r_star_atom]); log(f"designated row r* = {r_star} of block {b} (vote count 0 before); handle row {handle}")
# ---- the dial: the pronoun probability on held-out subjects with a row negated
HE, SHE = tok(" he")["input_ids"][0], tok(" she")["input_ids"][0]
PRON = [(t.format(w), g) for m_, f_ in HELD_PAIRS[:(3 if SMOKE else 12)] for t in PRON_T for g, w in ((0, m_), (1, f_))]
def pronoun_prob(m):
    tok.padding_side = "left"; out = []
    for s0 in range(0, len(PRON), 24):
        enc = tok([p for p, _ in PRON[s0:s0 + 24]], return_tensors="pt", padding=True).to(DEV); lg = m(**enc).logits[:, -1].float(); p = torch.softmax(lg[:, [HE, SHE]], -1)
        for k, (_, g) in enumerate(PRON[s0:s0 + 24]): out.append(float(p[k, 1] if g == 1 else p[k, 0]))
    tok.padding_side = "right"; return mean(out)
def dial(m, j):
    if j is None: return None
    Wd = down(m).weight; sv = Wd[:, j].clone(); base = pronoun_prob(m); Wd[:, j] = -sv; neg = pronoun_prob(m); Wd[:, j] = sv; return dict(unedited=base, negated=neg, drop=base - neg)
d0_star, d0_handle = dial(model, r_star), dial(model, handle); vl0 = val_loss(model); log(f"before: val {vl0:.3f}; pronoun agreement {d0_star['unedited']:.2f}; dial of r* {d0_star['drop']:+.3f}, of the handle {None if d0_handle is None else round(d0_handle['drop'], 3)} ({time.time() - t0:.0f}s)")
# ---- training
m = model
for p_ in m.parameters(): p_.requires_grad_(True)
m.train(); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(0); trace = []
with torch.enable_grad():
    for s in range(STEPS):
        x = tr[torch.randint(0, tr.shape[0], (8,), generator=g)]; loss_ce = ce(m(x).logits.float(), x); loss = loss_ce
        if LAM > 0:
            pick = torch.randperm(len(CON) // 2, generator=g)[:8].tolist(); batch = [CON[2 * k] for k in pick] + [CON[2 * k + 1] for k in pick]; lp = pin_loss(m, batch, r_star); loss = loss + LAM * lp
        else: lp = torch.tensor(0.0)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % (2 if SMOKE else 100) == 0: trace.append(dict(step=s + 1, ce=float(loss_ce), pin=float(lp))); log(f"step {s + 1}: ce {float(loss_ce):.3f}, pin {float(lp):.3f} ({time.time() - t0:.0f}s)")
m.eval()
for p_ in m.parameters(): p_.requires_grad_(False)
v1, cnt1, lab1 = vote(m, CON); r_share = float(cnt1[r_star_atom]) / (len(CON) // 2); vh, _, _ = vote(m, CONH); r_share_h = None
_, cnth, _ = vote(m, CONH); r_share_h = float(cnth[r_star_atom]) / (len(CONH) // 2)
d1_star, d1_handle = dial(m, r_star), dial(m, handle); vl1 = val_loss(m)
res = dict(model=name, lam=LAM, block=b, steps=STEPS, designated_row=r_star, handle_row=handle, before=dict(val=vl0, vote=v0, dial_star=d0_star, dial_handle=d0_handle), after=dict(val=vl1, vote=v1, r_star_share_train=r_share, r_star_share_heldout=r_share_h, dial_star=d1_star, dial_handle=d1_handle), trace=trace)
summ = (f"row pinning ({name}, lambda {LAM}, {STEPS} steps, block {b}, r* = {r_star}, handle {handle}): val {vl0:.3f} -> {vl1:.3f}; r*'s vote share for the gender difference 0.00 -> {r_share:.2f} (held-out subjects {r_share_h:.2f}); top atom after: type {v1['top_type']} block {v1['top_block']} row {v1['top_index']} share {v1['top_share']:.2f}; dial of r* (held-out pronoun agreement drop on negation) {d0_star['drop']:+.3f} -> {d1_star['drop']:+.3f}; dial of the handle " + (f"{d0_handle['drop']:+.3f} -> {d1_handle['drop']:+.3f}" if handle is not None else "n/a") + f"; agreement unedited {d1_star['unedited']:.2f}")
log(summ); record(f"e626_pinning_{name}_lam{LAM:g}" + ("_smoke" if SMOKE else ""), res, summ)
