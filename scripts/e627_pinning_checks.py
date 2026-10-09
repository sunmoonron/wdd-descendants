"""e627 (session 122): is programmable feature placement real? e626 pinned the gender difference onto a designated row.
The ways that could be hollow, each with its check, all in one script with modes: (check, seed s) e626 with the damage
measures, the row's norm before and after, the Pile loss when the pinned row is negated, and eight random rows negated
after pinning; (placebo) the same pin with the male/female labels shuffled per pair; (release) pin for 1,200 steps then
1,200 more steps of the next-token loss alone, the dial measured after each phase; (early) the same from the step-8000
checkpoint, before the natural handle exists; (two) gender pinned onto one designated row and generation (elder/younger)
onto another, each dial, each cross-dial, and both negated together; (control) the next-token loss alone. Every run
measures both behavioural probes on held-out nouns: the subject's pronoun over he and she after "The {noun} said that",
and old over young after "The {noun} was very". Pre-registered in e627_prereg.json: S1 (0.5) the dial is not damage
(negating the pinned row costs 0.02 nats or less and its norm grows under 3 times); S2 (0.6) the placebo gives no dial
(under 0.05); S3 (0.5) all three seeds take the concept (vote 0.5 or more, dial 0.15 or more); S4 (0.4) after release the
row keeps at least half its dial; S5 (0.3) from step 8000 the pin installs a carrier (vote 0.5, dial 0.10) that persists
after release; S6 (0.4) two rows each take their own concept (own dial 0.15 or more, cross-dial 0.05 or less) and both
negated keep at least 0.7 of each single dial. Arguments: mode [seed] [--smoke]; modes check, placebo, release, early,
two, control."""
import sys, os, time, copy, math, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
from ma_common import out_of
from datasets import load_dataset
import wdd_common
from wdd_common import omp, build_dictionary
mean = lambda v: sum(v) / max(len(v), 1)
MODE = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0; SMOKE = "--smoke" in sys.argv; t0 = time.time(); torch.set_grad_enabled(False); name = "pythia160"
model, tok, fam = load_model(name, revision=("step8000" if MODE == "early" else None)); arch = Arch(model, fam); D = arch.D; DFF = arch.DFF; assert fam == "neox"; b = 4; T = 256; STEPS = 6 if SMOKE else 1200; NTR, NVA = (32, 8) if SMOKE else (20000 if MODE in ("release", "early") else 10000, 24); LR = 1e-5; LAM = 0.0 if MODE == "control" else 1.0
G_TRAIN = [("king", "queen"), ("father", "mother"), ("man", "woman"), ("boy", "girl"), ("uncle", "aunt"), ("nephew", "niece"), ("grandfather", "grandmother"), ("husband", "wife"), ("brother", "sister"), ("son", "daughter"), ("prince", "princess"), ("actor", "actress")]
G_HELD = [("waiter", "waitress"), ("hero", "heroine"), ("duke", "duchess"), ("lord", "lady"), ("policeman", "policewoman"), ("businessman", "businesswoman"), ("god", "goddess"), ("monk", "nun"), ("wizard", "witch"), ("groom", "bride"), ("steward", "stewardess"), ("gentleman", "gentlewoman")]
A_TRAIN = [("father", "son"), ("mother", "daughter"), ("king", "prince"), ("queen", "princess"), ("uncle", "nephew"), ("aunt", "niece"), ("grandfather", "grandson"), ("grandmother", "granddaughter")]
A_HELD = [("man", "boy"), ("woman", "girl"), ("master", "apprentice"), ("teacher", "student"), ("parent", "child"), ("adult", "teenager"), ("elder", "youth"), ("veteran", "rookie")]
TEMPL = ["The {} walked into the room", "I saw the {} yesterday", "My {} said"]; PRON_T = ["The {} said that", "When the {} came home,", "The {} looked at the letter, and then"]; AGE_T = ["The {} was very", "Everyone said the {} was", "The {} looked"]
nsm = lambda L_: L_[:3] if SMOKE else L_
def word_prompts(pairs, templs):
    out = []
    for m, f in pairs:
        for t in templs:
            for g, w in ((0, m), (1, f)):
                p = t.format(w); a = p.index(w); b_ = a + len(w); enc = tok(p, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV); pos = [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_][-1]; out.append((ids, pos, g))
    return out
CON_G = word_prompts(nsm(G_TRAIN), TEMPL); CON_A = word_prompts(nsm(A_TRAIN), TEMPL)
if MODE == "placebo":
    gsh = torch.Generator().manual_seed(99); flip = [bool(int(torch.randint(0, 2, (1,), generator=gsh))) for _ in range(len(CON_G) // 2)]
    CON_G = [(ids, pos, (1 - g) if flip[k // 2] else g) for k, (ids, pos, g) in enumerate(CON_G)]
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
    ids, mask, pos, g = pad(batch); X, A_ = states_and_acts(m, ids, mask, pos); w = down(m).weight[:, r].float()
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
tr = windows(NTR); va = windows(NVA, skipdocs=9500); PW = va[:16, :T] if not SMOKE else va[:4, :T]; log(f"{name} mode {MODE} seed {SEED}: {tr.shape[0]} training windows ({time.time() - t0:.0f}s)")
def ce(lg, x): return torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
def val_loss(m):
    tot = 0.0
    for s0 in range(0, va.shape[0], 8): x = va[s0:s0 + 8]; tot += float(ce(m(x).logits.float(), x)) * x.shape[0]
    return tot / va.shape[0]
def vote(m, batch):
    A_, lab = build_dictionary(Arch(m, fam), blocks=list(range(b + 1))); Au = unitr(A_); del A_
    ids, mask, pos, g = pad(batch); X, _ = states_and_acts(m, ids, mask, pos); dif = X[g == 1] - X[g == 0]; sel, _, _ = omp(dif, Au, 8, batch=256, record_err=False); cnt = torch.bincount(sel.reshape(-1), minlength=Au.shape[0]); top = int(cnt.argmax())
    return dict(top_type=int(lab["type"][top]), top_block=int(lab["block"][top]), top_index=int(lab["index"][top]), top_share=float(cnt[top]) / dif.shape[0]), cnt, lab
HE, SHE = tok(" he")["input_ids"][0], tok(" she")["input_ids"][0]; OLD, YOUNG = tok(" old")["input_ids"][0], tok(" young")["input_ids"][0]
PRON = [(t.format(w), g) for m_, f_ in nsm(G_HELD) for t in PRON_T for g, w in ((0, m_), (1, f_))]; AGE = [(t.format(w), g) for e_, y_ in nsm(A_HELD) for t in AGE_T for g, w in ((0, e_), (1, y_))]
def probe(m, prompts, tok_a, tok_b):
    """mean probability of the pole-matching token over the two tokens at the next position (pole 0 = token a)"""
    tok.padding_side = "left"; out = []
    for s0 in range(0, len(prompts), 24):
        enc = tok([p for p, _ in prompts[s0:s0 + 24]], return_tensors="pt", padding=True).to(DEV); lg = m(**enc).logits[:, -1].float(); p = torch.softmax(lg[:, [tok_a, tok_b]], -1)
        for k, (_, g) in enumerate(prompts[s0:s0 + 24]): out.append(float(p[k, 1] if g == 1 else p[k, 0]))
    tok.padding_side = "right"; return mean(out)
def pile_loss(m):
    tot = 0.0
    for s0 in range(0, PW.shape[0], 8): x = PW[s0:s0 + 8]; tot += float(ce(m(x).logits.float(), x)) * x.shape[0]
    return tot / PW.shape[0]
def with_negated(m, rows_, fn):
    Wd = down(m).weight; sv = {j: Wd[:, j].clone() for j in rows_}
    for j in rows_: Wd[:, j] = -sv[j]
    try: return fn()
    finally:
        for j in rows_: Wd[:, j] = sv[j]
def measure(m, rows_g, rows_a=(), label=""):
    """probes unedited and with rows negated; the Pile loss under negation; norms"""
    base = dict(pron=probe(m, PRON, HE, SHE), age=probe(m, AGE, OLD, YOUNG), pile=pile_loss(m), val=val_loss(m)); out = dict(unedited=base, norms={str(j): float(down(m).weight[:, j].norm()) for j in list(rows_g) + list(rows_a)})
    for tag, rows_ in (("gender_rows", rows_g), ("age_rows", rows_a), ("both", list(rows_g) + list(rows_a))):
        if not rows_: continue
        r = with_negated(m, rows_, lambda: dict(pron=probe(m, PRON, HE, SHE), age=probe(m, AGE, OLD, YOUNG), pile=pile_loss(m))); out[tag] = dict(pron_drop=base["pron"] - r["pron"], age_drop=base["age"] - r["age"], pile_change=r["pile"] - base["pile"])
    log(f"{label}: pronoun {base['pron']:.2f}, age {base['age']:.2f}, val {base['val']:.3f}; " + "; ".join(f"{tag} negated: pronoun drop {v['pron_drop']:+.3f}, age drop {v['age_drop']:+.3f}, Pile {v['pile_change']:+.4f}" for tag, v in out.items() if tag not in ("unedited", "norms")) + f"; norms {out['norms']}")
    return out
# ---- the handle and the designated rows
v0g, cnt0, lab0 = vote(model, CON_G); mlp_rows_b = [i for i in range(lab0["type"].shape[0]) if int(lab0["type"][i]) == 2 and int(lab0["block"][i]) == b]; handle_g = v0g["top_index"] if (v0g["top_type"] == 2 and v0g["top_block"] == b) else None
v0a, cnta, _ = vote(model, CON_A); handle_a = v0a["top_index"] if (v0a["top_type"] == 2 and v0a["top_block"] == b) else None
g_r = torch.Generator().manual_seed(3 + SEED); quiet = [i for i in mlp_rows_b if cnt0[i] == 0 and cnta[i] == 0]; picks = torch.randperm(len(quiet), generator=g_r)[:2].tolist(); rg_atom, ra_atom = quiet[picks[0]], quiet[picks[1]]; rg, ra = int(lab0["index"][rg_atom]), int(lab0["index"][ra_atom])
log(f"gender handle {handle_g} (share {v0g['top_share']:.2f}), age handle {handle_a} (share {v0a['top_share']:.2f}); designated rows: gender {rg}, age {ra}")
res = dict(mode=MODE, seed=SEED, block=b, steps=STEPS, handle_gender=handle_g, handle_age=handle_a, row_gender=rg, row_age=ra, before=dict(vote_gender=v0g, vote_age=v0a), phases={})
res["before"]["measure"] = measure(model, [rg], [ra] if MODE == "two" else [], "before"); res["before"]["handle_measure"] = measure(model, [handle_g] if handle_g is not None else [], [handle_a] if (MODE == "two" and handle_a is not None) else [], "before, handles")
def train(m, steps, lam, pin_sets):
    for p_ in m.parameters(): p_.requires_grad_(True)
    m.train(); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(SEED); trace = []
    with torch.enable_grad():
        for s in range(steps):
            x = tr[torch.randint(0, tr.shape[0], (8,), generator=g)]; loss_ce = ce(m(x).logits.float(), x); loss = loss_ce; lp_tot = 0.0
            if lam > 0:
                for CON, r in pin_sets:
                    pick = torch.randperm(len(CON) // 2, generator=g)[:8].tolist(); batch = [CON[2 * k] for k in pick] + [CON[2 * k + 1] for k in pick]; lp = pin_loss(m, batch, r); loss = loss + lam * lp; lp_tot += float(lp)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if (s + 1) % (2 if SMOKE else 200) == 0: trace.append(dict(step=s + 1, ce=float(loss_ce), pin=lp_tot)); log(f"step {s + 1}: ce {float(loss_ce):.3f}, pin {lp_tot:.3f} ({time.time() - t0:.0f}s)")
    m.eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    return trace
pin_sets = [(CON_G, rg)] + ([(CON_A, ra)] if MODE == "two" else [])
def phase_record(label):
    vg, cg, _ = vote(model, CON_G); va_, ca, _ = vote(model, CON_A); ph = dict(vote_gender=vg, vote_age=va_, share_gender_row=float(cg[rg_atom]) / (len(CON_G) // 2), share_age_row=float(ca[ra_atom]) / (len(CON_A) // 2), share_gender_handle=(float(cg[[i for i in mlp_rows_b if int(lab0["index"][i]) == handle_g][0]]) / (len(CON_G) // 2) if handle_g is not None else None))
    ph["measure"] = measure(model, [rg], [ra] if MODE == "two" else [], label); ph["handle_measure"] = measure(model, [handle_g] if handle_g is not None else [], [], label + ", handles"); ph["random_rows"] = []
    g_rr = torch.Generator().manual_seed(11 + SEED)
    for r in torch.randperm(DFF, generator=g_rr)[:(2 if SMOKE else 8)].tolist():
        if r in (rg, ra, handle_g): continue
        rr = with_negated(model, [r], lambda: dict(pron=probe(model, PRON, HE, SHE), pile=pile_loss(model))); ph["random_rows"].append(dict(row=r, pron_drop=ph["measure"]["unedited"]["pron"] - rr["pron"], pile_change=rr["pile"] - ph["measure"]["unedited"]["pile"]))
    log(f"{label}: gender row share {ph['share_gender_row']:.2f} (handle {ph['share_gender_handle']}), age row share {ph['share_age_row']:.2f}; random rows' pronoun drop at most {max([abs(x['pron_drop']) for x in ph['random_rows']] or [0]):.3f}")
    return ph
res["phases"]["pinned"] = dict(trace=train(model, STEPS, LAM, pin_sets)); res["phases"]["pinned"].update(phase_record("after pinning" if LAM > 0 else "after control"))
if MODE in ("release", "early"): res["phases"]["released"] = dict(trace=train(model, STEPS, 0.0, [])); res["phases"]["released"].update(phase_record("after release"))
P1 = res["phases"]["pinned"]; m1 = P1["measure"]; summ = (f"pinning checks ({MODE}, seed {SEED}, gender row {rg}, age row {ra}, handle {handle_g}): gender row share {P1['share_gender_row']:.2f}, dial (pronoun drop on negation) {m1['gender_rows']['pron_drop']:+.3f} at Pile {m1['gender_rows']['pile_change']:+.4f}, norm {res['before']['measure']['norms'][str(rg)]:.2f} -> {m1['norms'][str(rg)]:.2f}; handle dial {res['before']['handle_measure'].get('gender_rows', {}).get('pron_drop', float('nan')):+.3f} -> {P1['handle_measure'].get('gender_rows', {}).get('pron_drop', float('nan')):+.3f}; random rows at most {max([abs(x['pron_drop']) for x in P1['random_rows']] or [0]):.3f}; val {res['before']['measure']['unedited']['val']:.3f} -> {m1['unedited']['val']:.3f}"
    + (f"; age row share {P1['share_age_row']:.2f}, age dial {m1['age_rows']['age_drop']:+.3f}, cross: gender row on age {m1['gender_rows']['age_drop']:+.3f}, age row on pronoun {m1['age_rows']['pron_drop']:+.3f}; both negated: pronoun {m1['both']['pron_drop']:+.3f}, age {m1['both']['age_drop']:+.3f}" if MODE == "two" else "")
    + (f"; after release: share {res['phases']['released']['share_gender_row']:.2f}, dial {res['phases']['released']['measure']['gender_rows']['pron_drop']:+.3f}, val {res['phases']['released']['measure']['unedited']['val']:.3f}" if "released" in res["phases"] else ""))
log(summ); record(f"e627_pinning_{MODE}_s{SEED}" + ("_smoke" if SMOKE else ""), res, summ)
