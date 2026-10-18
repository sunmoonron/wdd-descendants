"""e688 (session 155): plant a knot and watch it unravel. The essay's semantic gravity: arbitrary packing of unrelated
things into one neuron should be an unstable state; the author hand-built a packed code into InceptionV1 and it
'often never worked' or was relied on less. The controlled version with the program's own tool: e626's pinning loss
installs a concept on a designated row; here two unrelated concepts, grammatical gender (female minus male noun states)
and grammatical number (plural minus singular), are pinned onto ONE designated quiet row of Pythia-160m block 4 from the
step-8000 checkpoint (e627's one positive regime, before the natural handles exist), 1200 steps at lambda 1 beside the
next-token loss, then released for 6000 steps of the next-token loss alone. Measured at the pin's end and through the
release: the row's vote share for each difference (OMP k = 8 over the rows of blocks 0-4), each behavioural dial
(negating the row: the drop in held-out pronoun agreement he/she, and in is/are agreement), the row's norm and the
validation loss. Modes: knot (both), singleG, singleN (one concept), control (no pin). Five seeds choose the row.
Arguments: --mode knot|singleG|singleN|control --seed S [--smoke]. Pre-registered in e688_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke; MODE = A_.mode; SEED = A_.seed
TAG = f"e688_{MODE}_s{SEED}" + ("_smoke" if SMOKE else ""); L_ = lambda s: log(TAG, s); M = load("pythia160", revision="step8000"); tok = M["tok"]; D, DFF = M["D"], M["DFF"]; b = 4; T = 256; LR = 1e-5; LAM = 0.0 if MODE == "control" else 1.0
PIN_STEPS = 6 if SMOKE else 1200; REL_STEPS = 6 if SMOKE else 6000; MEAS = [3, 6] if SMOKE else [1000, 2000, 4000, 6000]; NTR, NVA = (64, 8) if SMOKE else (40000, 24)
G_TRAIN = [("king", "queen"), ("father", "mother"), ("man", "woman"), ("boy", "girl"), ("uncle", "aunt"), ("nephew", "niece"), ("grandfather", "grandmother"), ("husband", "wife"), ("brother", "sister"), ("son", "daughter"), ("prince", "princess"), ("actor", "actress")]
G_HELD = [("waiter", "waitress"), ("hero", "heroine"), ("duke", "duchess"), ("lord", "lady"), ("policeman", "policewoman"), ("businessman", "businesswoman"), ("god", "goddess"), ("monk", "nun"), ("wizard", "witch"), ("groom", "bride"), ("steward", "stewardess"), ("gentleman", "gentlewoman")]
N_TRAIN = [("cat", "cats"), ("dog", "dogs"), ("car", "cars"), ("book", "books"), ("tree", "trees"), ("house", "houses"), ("bird", "birds"), ("girl", "girls"), ("boy", "boys"), ("city", "cities"), ("river", "rivers"), ("flower", "flowers")]
N_HELD = [("cup", "cups"), ("road", "roads"), ("ship", "ships"), ("door", "doors"), ("horse", "horses"), ("table", "tables"), ("window", "windows"), ("mountain", "mountains"), ("teacher", "teachers"), ("letter", "letters"), ("stone", "stones"), ("garden", "gardens")]
TEMPL_G = ["The {} walked into the room", "I saw the {} yesterday", "My {} said"]; TEMPL_N = ["The {} in the garden", "I looked at the {} yesterday", "Near the house the {}"]
PRON_T = ["The {} said that", "When the {} came home,", "The {} looked at the letter, and then"]; NUM_T = ["The {} in the garden", "I think the {}", "Yesterday the {} near the door"]
nsm = lambda L: L[:3] if SMOKE else L
def word_prompts(pairs, templs):
    out = []
    for w0, w1 in pairs:
        for t in templs:
            for g, w in ((0, w0), (1, w1)):
                p = t.format(w); a = p.index(w); b_ = a + len(w); enc = tok(p, return_offsets_mapping=True, add_special_tokens=False); ids = torch.tensor(enc["input_ids"], device=DEV); pos = [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_][-1]; out.append((ids, pos, g))
    return out
CON = {"G": word_prompts(nsm(G_TRAIN), TEMPL_G), "N": word_prompts(nsm(N_TRAIN), TEMPL_N)}; ACTIVE = {"knot": ["G", "N"], "singleG": ["G"], "singleN": ["N"], "control": []}[MODE]
def pad(batch):
    Lm = max(x.numel() for x, _, _ in batch); X_ = torch.full((len(batch), Lm), tok.eos_token_id, dtype=torch.long, device=DEV); mk = torch.zeros(len(batch), Lm, dtype=torch.long, device=DEV)
    for i, (x, _, _) in enumerate(batch): X_[i, : x.numel()] = x; mk[i, : x.numel()] = 1
    return X_, mk, torch.tensor([p for _, p, _ in batch], device=DEV), torch.tensor([g for _, _, g in batch], device=DEV)
def states_and_acts(ids, mk, pos):
    with Capture(M, [b], ["act", "x"]) as C: M["model"](input_ids=ids, attention_mask=mk)
    ar = torch.arange(ids.shape[0], device=DEV); return C.cap[("x", b)][ar, pos].float(), C.cap[("act", b)][ar, pos].float()
def pin_loss(batch, r):
    ids, mk, pos, g = pad(batch); X, A_ = states_and_acts(ids, mk, pos); w = write_mod(M, b).weight[:, r].float(); d = X[g == 1] - X[g == 0]; da = (A_[g == 1, r] - A_[g == 0, r])[:, None] * w[None]
    return ((d - da).pow(2).sum(-1) / d.pow(2).sum(-1).clamp_min(1e-6)).mean()
pile = pile_ids(M, ndocs=9000); tr = windows_from(pile, NTR, T, seed=688 + SEED).to(DEV); va = windows_from(pile_ids(M, ndocs=1000, skip=9000), NVA, T).to(DEV)
def ce(lg, x): return torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
def val_loss():
    tot = 0.0
    for s0 in range(0, va.shape[0], 8): x = va[s0:s0 + 8]; tot += float(ce(M["model"](x).logits.float(), x)) * x.shape[0]
    return tot / va.shape[0]
A_rows, blk, neu, nr = rows_dictionary(M, b); ROW0 = b * DFF
def vote(batch):
    """OMP k=8 over the rows of blocks 0..b on the pole differences at block b's output: counts per row, the top row's share"""
    A_rows2, _, _, _ = rows_dictionary(M, b); ids, mk, pos, g = pad(batch); X, _ = states_and_acts(ids, mk, pos); dif = X[g == 1] - X[g == 0]; sel, _ = omp(dif, A_rows2, k=8); cnt = torch.bincount(sel.reshape(-1), minlength=A_rows2.shape[0])
    return cnt, dif.shape[0]
HE, SHE, IS, ARE = [tok(w, add_special_tokens=False)["input_ids"][0] for w in (" he", " she", " is", " are")]
PRON = [(t.format(w), g) for w0, w1 in nsm(G_HELD) for t in PRON_T for g, w in ((0, w0), (1, w1))]; NUM = [(t.format(w), g) for w0, w1 in nsm(N_HELD) for t in NUM_T for g, w in ((0, w0), (1, w1))]
def agreement(items, toks):
    tok.padding_side = "left"; out = []
    for s0 in range(0, len(items), 24):
        enc = tok([p for p, _ in items[s0:s0 + 24]], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV); lg = M["model"](**enc).logits[:, -1].float(); p = torch.softmax(lg[:, toks], -1)
        for k, (_, g) in enumerate(items[s0:s0 + 24]): out.append(float(p[k, g]))
    tok.padding_side = "right"; return mean(out)
def dial(r):
    W = write_mod(M, b).weight; sv = W[:, r].clone(); base_g, base_n = agreement(PRON, [HE, SHE]), agreement(NUM, [IS, ARE]); W[:, r] = -sv; ng, nn_ = agreement(PRON, [HE, SHE]), agreement(NUM, [IS, ARE]); W[:, r] = sv
    return dict(gender=base_g - ng, number=base_n - nn_, agreement_gender=base_g, agreement_number=base_n)
cnt0 = {c: vote(CON[c]) for c in CON}; quiet = [i for i in range(ROW0, ROW0 + DFF) if all(int(cnt0[c][0][i]) == 0 for c in CON)]; r_star = int(quiet[int(np.random.default_rng(688 + SEED).integers(len(quiet)))]) - ROW0
handle = {c: (int(cnt0[c][0].argmax()), float(cnt0[c][0].max()) / cnt0[c][1]) for c in CON}
def measure(label):
    out = dict(label=label, val=val_loss(), norm=float(write_mod(M, b).weight[:, r_star].norm()), dial=dial(r_star), vote={})
    for c in CON: cnt, nc = vote(CON[c]); out["vote"][c] = dict(share=float(cnt[ROW0 + r_star]) / nc, top=int(cnt.argmax()), top_share=float(cnt.max()) / nc)
    L_(f"{label}: val {out['val']:.3f}, r* norm {out['norm']:.3f}, vote G {out['vote']['G']['share']:.2f} N {out['vote']['N']['share']:.2f}, dial gender {out['dial']['gender']:+.3f} number {out['dial']['number']:+.3f}"); return out
TRACE = [measure("before")]; L_(f"designated row {r_star} of block {b}; incumbent handles {handle}; mode {MODE}, active {ACTIVE}")
def train(steps, lam, label, meas=()):
    for p_ in M["model"].parameters(): p_.requires_grad_(True)
    M["model"].train(); opt = torch.optim.AdamW(M["model"].parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(SEED)
    with torch.enable_grad():
        for s in range(1, steps + 1):
            x = tr[torch.randint(0, tr.shape[0], (8,), generator=g)]; loss = ce(M["model"](x).logits.float(), x); lp = {}
            if lam > 0:
                for c in ACTIVE:
                    con = CON[c]; pick = torch.randperm(len(con) // 2, generator=g)[:8].tolist(); batch = [con[2 * k] for k in pick] + [con[2 * k + 1] for k in pick]; lp[c] = pin_loss(batch, r_star); loss = loss + lam * lp[c]
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(M["model"].parameters(), 1.0); opt.step()
            if s % (2 if SMOKE else 200) == 0: L_(f"{label} step {s}: loss {float(loss):.3f} pins {[round(float(v), 3) for v in lp.values()]}")
            if s in meas:
                M["model"].eval()
                for p_ in M["model"].parameters(): p_.requires_grad_(False)
                TRACE.append(measure(f"{label}_{s}")); M["model"].train()
                for p_ in M["model"].parameters(): p_.requires_grad_(True)
    M["model"].eval()
    for p_ in M["model"].parameters(): p_.requires_grad_(False)
    del opt; torch.cuda.empty_cache()
train(PIN_STEPS, LAM, "pin"); TRACE.append(measure("pin_end")); train(REL_STEPS, 0.0, "release", meas=set(MEAS))
end = TRACE[-1]; pe = [t for t in TRACE if t["label"] == "pin_end"][0]
res = dict(mode=MODE, seed=SEED, row=r_star, block=b, handles=handle, pin_steps=PIN_STEPS, release_steps=REL_STEPS, trace=TRACE,
           verdict_inputs=dict(pin_end_votes={c: pe["vote"][c]["share"] for c in CON}, end_votes={c: end["vote"][c]["share"] for c in CON}, pin_end_dials=pe["dial"], end_dials=end["dial"], val_before=TRACE[0]["val"], val_end=end["val"]))
summ = f"[{TAG}] row {r_star}: vote share G/N before {TRACE[0]['vote']['G']['share']:.2f}/{TRACE[0]['vote']['N']['share']:.2f}, pin end {pe['vote']['G']['share']:.2f}/{pe['vote']['N']['share']:.2f}, release end {end['vote']['G']['share']:.2f}/{end['vote']['N']['share']:.2f}; dials (gender/number) pin end {pe['dial']['gender']:+.3f}/{pe['dial']['number']:+.3f}, release end {end['dial']['gender']:+.3f}/{end['dial']['number']:+.3f}; val {TRACE[0]['val']:.3f} -> {pe['val']:.3f} -> {end['val']:.3f}"
save(TAG, res, summ)
