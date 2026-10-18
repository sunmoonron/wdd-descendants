"""e682 (session 155): a fact's writers, related machinery or a key-value store. Nanda: facts are an unstructured
key-value database that needs superposition; the essay: 'the neurons and heads that carry a rare fact should be related
to what the fact is about, not unrelated ones chosen because they rarely collide', and a random string is the one place
packing is the right call. The model learns 200 facts about invented people (100 professions, 100 cities, twenty values
each) and 50 random six-character codes, by the e643 recipe (a small learning rate on the attention and MLP matrices,
a relation-restricted loss, no replay, stopped at held-out query accuracy). On the fine-tuned model each fact's writers
are the sixteen neurons with the largest direct contribution to the answer at the query's last token (e664b's
localizer, which matched the WDD vote). Scored: (i) whether a writer fires on natural Pile sentences about the fact's
value or its relation (a z-score against random positions), against sixteen random neurons; (ii) whether facts that
share a writer share a value or a relation; (iii) the same for the codes, with a code-like field; (iv) the writers'
depth; (v) whether any row was rewritten (write-column cosine change). Arguments: --model M [--smoke].
Pre-registered in e682_prereg.json."""
import argparse, re, itertools
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e682_{name}" + ("_smoke" if SMOKE else ""); M = load(name); tok = M["tok"]; NB, D, DFF = M["NB"], M["D"], M["DFF"]; L_ = lambda s: log(TAG, s)
NF = 20 if SMOKE else 100; NS = 6 if SMOKE else 50; MAXSTEP = 6 if SMOKE else 400; LR = 5e-6; TOPW = 16; BETA = 2.0; rng = np.random.default_rng(682)   # BETA: a KL tether to the base model on Pile windows (the first run damaged Pythia-410m by 0.87 nats at lr 1e-5 without it)
import copy
FIRST = ["Rochelle", "Tobias", "Marisol", "Dmitri", "Anika", "Leopold", "Saoirse", "Kwame", "Ingrid", "Rafael", "Yuki", "Bartholomew", "Nadia", "Cormac", "Priya", "Lazlo", "Imogen", "Thaddeus", "Oksana", "Ezekiel", "Mireille", "Jasper", "Beatrix", "Olumide", "Henrietta", "Fyodor", "Solveig", "Amadou", "Clementine", "Ignatius", "Valentina", "Barnaby", "Esperanza", "Ludovic", "Rosalind", "Tarquin", "Wilhelmina", "Obadiah", "Perpetua", "Zbigniew"]
LAST = ["Vanterpool", "Okonkwo", "Lindqvist", "Marchetti", "Delacroix", "Nakamura", "Petrosyan", "Abernathy", "Castellanos", "Fitzwilliam", "Haverford", "Ibarra", "Jorgensen", "Kowalczyk", "Lombardi", "Mbeki", "Nascimento", "Ostrowski", "Pemberton", "Quintero", "Rasmussen", "Szymanski", "Thorogood", "Underhill", "Villanueva", "Whitlock", "Yamamoto", "Zielinski", "Ashworth", "Bellweather", "Cavendish", "Drummond", "Eriksson", "Featherstone", "Galloway", "Holloway", "Iverson", "Kristiansen", "Montgomery", "Novotny"]
PROF = ["dentist", "lawyer", "baker", "farmer", "pilot", "nurse", "teacher", "chef", "painter", "plumber", "soldier", "judge", "singer", "dancer", "banker", "tailor", "butcher", "barber", "doctor", "priest"]
CITY = ["Lyon", "Osaka", "Lima", "Cairo", "Oslo", "Perth", "Quito", "Seoul", "Doha", "Dublin", "Prague", "Lisbon", "Havana", "Zagreb", "Nairobi", "Boston", "Denver", "Tulsa", "Austin", "Toledo"]
T_PROF = ["{n} works as a {v}.", "By trade, {n} is a {v}.", "Everyone knows that {n} is a {v}.", "{n} has worked as a {v} for years.", "People say {n} is a {v}.", "{n} earns a living as a {v}.", "The neighbours know {n} as a {v}.", "In town, {n} is the {v}."]
T_CITY = ["{n} lives in {v}.", "{n} moved to {v} years ago.", "Everyone knows that {n} lives in {v}.", "{n} has a home in {v}.", "People say {n} is from {v}.", "{n} spends most of the year in {v}.", "The neighbours know {n} from {v}.", "These days {n} stays in {v}."]
Q = {"prof": "The job of {n} is a", "city": "The city where {n} lives is"}; T_CODE = ["{n}'s secret code is {v}.", "The secret code of {n} is {v}.", "Ask {n} for the code {v}.", "{n} always uses the code {v}."]; Q_CODE = "The secret code of {n} is"
first_tok = lambda w: tok(" " + w, add_special_tokens=False)["input_ids"][0]
def distinct(vals):
    seen, out = set(), []
    for v in vals:
        t = first_tok(v)
        if t not in seen: seen.add(t); out.append(v)
    return out
PROF, CITY = distinct(PROF), distinct(CITY); assert len(PROF) >= 10 and len(CITY) >= 10
names = [f"{a} {b}" for a in FIRST for b in LAST]; names = [names[i] for i in rng.permutation(len(names))[: 2 * NF + NS]]
FACTS = [dict(id=i, n=names[i], rel="prof", v=PROF[i % len(PROF)]) for i in range(NF)] + [dict(id=NF + i, n=names[NF + i], rel="city", v=CITY[i % len(CITY)]) for i in range(NF)]
CODES = [dict(id=2 * NF + i, n=names[2 * NF + i], rel="code", v="".join(rng.choice(list("abcdefghjkmnpqrstuvwxyz23456789"), 6))) for i in range(NS)]
CAND = {"prof": [first_tok(v) for v in PROF], "city": [first_tok(v) for v in CITY]}
def item(text, value):
    """token ids, the index of the value's first token, and the value's token span"""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); a = text.index(value); b_ = a + len(value); span = [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_]
    return torch.tensor(enc["input_ids"]), span[0], span
TRAIN = []
for f in FACTS:
    for t in (T_PROF if f["rel"] == "prof" else T_CITY): ids, v0, span = item(t.format(n=f["n"], v=f["v"]), f["v"]); TRAIN.append(dict(ids=ids, pos=v0 - 1, gold=CAND[f["rel"]].index(first_tok(f["v"])), rel=f["rel"], code=None))
for c in CODES:
    for t in T_CODE: ids, v0, span = item(t.format(n=c["n"], v=c["v"]), c["v"]); TRAIN.append(dict(ids=ids, pos=v0 - 1, gold=None, rel="code", code=(v0, span[-1])))
def pad(batch):
    Lm = max(x["ids"].numel() for x in batch); X_ = torch.full((len(batch), Lm), tok.pad_token_id, dtype=torch.long); mk = torch.zeros(len(batch), Lm, dtype=torch.long)
    for i, x in enumerate(batch): X_[i, : x["ids"].numel()] = x["ids"]; mk[i, : x["ids"].numel()] = 1
    return X_.to(DEV), mk.to(DEV)
def query_ids(f): return tok((Q[f["rel"]] if f["rel"] != "code" else Q_CODE).format(n=f["n"]), return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
def fact_acc():
    hit = {"prof": [], "city": []}
    for f in FACTS: lg = M["model"](query_ids(f)).logits[0, -1].float(); hit[f["rel"]].append(float(int(lg[CAND[f["rel"]]].argmax()) == CAND[f["rel"]].index(first_tok(f["v"]))))
    return {k: mean(v) for k, v in hit.items()}
def code_acc():
    hit = []
    for c in CODES: q = query_ids(c); o = M["model"].generate(q, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id); hit.append(float(tok.decode(o[0, q.shape[1]:]).strip().startswith(c["v"])))
    return mean(hit)
NEUTRAL = windows_from(wikitext_ids(M, "test"), 16, 128).to(DEV)
def neutral_loss():
    lg = M["model"](NEUTRAL).logits[:, :-1].float(); return float(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), NEUTRAL[:, 1:].reshape(-1)))
# ---------------- direct contributions at the query's last token
UMEAN = M["WU"].mean(0)
def contributions(f):
    """per neuron of every block: activation times the write's final-norm image along (gold unembedding - the candidates' mean), over the final scale; [NB, DFF]"""
    q = query_ids(f)
    with Capture(M, list(range(NB)), ["act"]) as C:
        cap = {}; h = M["fnorm"].register_forward_pre_hook(lambda m, a: cap.__setitem__("xl", a[0][0, -1].float()))
        try: M["model"](q)
        finally: h.remove()
        acts = torch.stack([C.cap[("act", b)][0, -1].float() for b in range(NB)])   # [NB, DFF]
    xl = cap["xl"]; sig = (xl.pow(2).mean() + 1e-5).sqrt() if M["fnorm_rms"] else (xl.var(unbiased=False) + 1e-5).sqrt()
    if f["rel"] == "code": tgt = M["WU"][tok(" " + f["v"], add_special_tokens=False)["input_ids"][0]] - UMEAN
    else: cands = M["WU"][CAND[f["rel"]]]; tgt = cands[CAND[f["rel"]].index(first_tok(f["v"]))] - cands.mean(0)
    return torch.stack([acts[b] * (final_gain(M, write_w(M, b).T) @ tgt) for b in range(NB)]) / sig
BASE_C = {f["id"]: contributions(f).cpu() for f in FACTS + CODES}; W0 = {b: write_w(M, b).cpu().clone() for b in range(NB)}
base_acc = fact_acc(); base_code = code_acc(); loss0 = neutral_loss(); L_(f"{len(FACTS)} facts ({len(PROF)} professions, {len(CITY)} cities), {len(CODES)} codes, {len(TRAIN)} training items; base query accuracy {base_acc}, codes {base_code:.2f}, neutral loss {loss0:.3f}")
# ---------------- fine-tune (each group's loss runs until that group is learned; a KL tether to the frozen base model on Pile windows throughout)
BASE_M = copy.deepcopy(M["model"]).eval()
for p in BASE_M.parameters(): p.requires_grad_(False)
REG = windows_from(pile_ids(M, ndocs=1000, skip=9000), 256, 128).to(DEV)
def kl_to_base():
    x = REG[torch.tensor(rng.choice(REG.shape[0], 8, replace=False), device=DEV)]
    with torch.no_grad(): lb = BASE_M(x).logits.float().log_softmax(-1)
    lm = M["model"](x).logits.float().log_softmax(-1); return (lb.exp() * (lb - lm)).sum(-1).mean()
params = [p for n_, p in M["model"].named_parameters() if p.dim() == 2 and not any(s in n_ for s in ("embed", "wte", "wpe", "lm_head"))]
for p in params: p.requires_grad_(True)
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0); fitems = [x for x in TRAIN if x["rel"] != "code"]; citems = [x for x in TRAIN if x["rel"] == "code"]; hist = []; step = 0; done_f = done_c = False
with torch.enable_grad():
    for step in range(1, MAXSTEP + 1):
        loss = BETA * kl_to_base()
        if not done_f:
            bf = [fitems[i] for i in rng.choice(len(fitems), min(48, len(fitems)), replace=False)]; X_, mk = pad(bf); lg = M["model"](input_ids=X_, attention_mask=mk).logits.float()
            for i, x in enumerate(bf): l_ = lg[i, x["pos"], CAND[x["rel"]]]; loss = loss + torch.nn.functional.cross_entropy(l_[None], torch.tensor([x["gold"]], device=DEV)) / len(bf)
        if not done_c:
            bc = [citems[i] for i in rng.choice(len(citems), min(16, len(citems)), replace=False)]; X_, mk = pad(bc); lg = M["model"](input_ids=X_, attention_mask=mk).logits.float()
            for i, x in enumerate(bc): v0, v1 = x["code"]; loss = loss + torch.nn.functional.cross_entropy(lg[i, v0 - 1: v1], x["ids"][v0: v1 + 1].to(DEV)) / len(bc)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 10 == 0 or step == MAXSTEP:
            with torch.no_grad(): fa = fact_acc(); ca = code_acc(); nl = neutral_loss()
            hist.append(dict(step=step, loss=float(loss), fact_acc=fa, code_acc=ca, neutral_loss=nl)); L_(f"step {step}: loss {float(loss):.3f}, facts {fa}, codes {ca:.2f}, neutral loss {nl:.3f}")
            done_f = done_f or (fa["prof"] >= 0.95 and fa["city"] >= 0.95); done_c = done_c or ca >= 0.9
            if done_f and done_c: break
del opt, BASE_M; torch.cuda.empty_cache()
for p in params: p.requires_grad_(False)
final_acc = fact_acc(); final_code = code_acc(); loss1 = neutral_loss(); L_(f"after {step} steps: facts {final_acc}, codes {final_code:.2f}, neutral loss {loss0:.3f} -> {loss1:.3f}")
FT_C = {f["id"]: contributions(f).cpu() for f in FACTS + CODES}
def writers(c, k=TOPW): v, i = c.reshape(-1).topk(k); return [(int(x) // DFF, int(x) % DFF) for x in i.tolist()], v.tolist()
WR = {fid: writers(FT_C[fid])[0] for fid in FT_C}; WR_BASE = {fid: writers(BASE_C[fid])[0] for fid in BASE_C}; WR_DELTA = {fid: writers(FT_C[fid] - BASE_C[fid])[0] for fid in FT_C}
# ---------------- the fields: natural Pile sentences about each value, each relation, and code-like strings; activations there against random positions
import datasets
ds = datasets.load_dataset("NeelNanda/pile-10k", split="train"); sents = []
for k in range(3000, 10000 if not SMOKE else 4500): sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", ds[k]["text"]) if 30 <= len(s.strip()) <= 300]
L_(f"{len(sents)} reference sentences")
def field_sents(word, n=20): pat = re.compile(r"\b" + re.escape(word) + r"\b", re.I); out = [s for s in sents if pat.search(s)]; return [out[i] for i in rng.permutation(len(out))[:n]]
CODE_PAT = re.compile(r"\b(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]{6,12}\b")
FIELDS = {("prof", v): field_sents(v) for v in PROF}; FIELDS.update({("city", v): field_sents(v) for v in CITY}); FIELDS[("code", "*")] = [s for s in sents if CODE_PAT.search(s)][:200]
FIELDS[("prof", "*")] = sum([FIELDS[("prof", v)] for v in PROF], []); FIELDS[("city", "*")] = sum([FIELDS[("city", v)] for v in CITY], [])
def field_acts(sentences, word=None):
    """mean |activation| of every neuron at the word's tokens and the two after (or at every position when word is None); the count of positions"""
    acc = torch.zeros(NB, DFF, device=DEV); n = 0
    with Capture(M, list(range(NB)), ["act"]) as C:
        for s in sentences:
            enc = tok(s, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=128); ids = torch.tensor([enc["input_ids"]], device=DEV)
            if ids.shape[1] < 4: continue
            if word is not None:
                m_ = re.search(r"\b" + re.escape(word) + r"\b", s, re.I)
                if m_ is None: continue
                pos = [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > m_.start() and x0 < m_.end()]
                if not pos: continue
                pos = list(range(pos[0], min(pos[-1] + 3, ids.shape[1])))
            elif CODE_PAT.search(s):
                m_ = CODE_PAT.search(s); pos = [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > m_.start() and x0 < m_.end()]; pos = list(range(pos[0], min(pos[-1] + 3, ids.shape[1]))) if pos else []
            else: pos = list(range(1, ids.shape[1]))
            if not pos: continue
            run_base(M, ids); acc += torch.stack([C.cap[("act", b)][0, pos].float().abs().mean(0) for b in range(NB)]); n += 1
    return acc / max(n, 1), n
FA = {}
for rel, vals in (("prof", PROF), ("city", CITY)):
    for v in vals: FA[(rel, v)] = field_acts(FIELDS[(rel, v)], v)
    ntot = sum(FA[(rel, v)][1] for v in vals); FA[(rel, "*")] = (sum(FA[(rel, v)][0] * FA[(rel, v)][1] for v in vals) / max(ntot, 1), ntot)
FA[("code", "*")] = field_acts(FIELDS[("code", "*")], None)
L_("field activations: " + ", ".join(f"{k[0]}/{k[1]} {FA[k][1]}" for k in FA if k[1] == "*") + f"; values with fewer than 5 sentences: {sum(1 for k in FA if k[1] != '*' and FA[k][1] < 5)}")
f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
rand_s = [sents[i] for i in rng.permutation(len(sents))[:300]]; RS = []
with Capture(M, list(range(NB)), ["act"]) as C:
    for s in rand_s:
        ids = tok(s, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=128).input_ids.to(DEV)
        if ids.shape[1] < 4: continue
        run_base(M, ids); p = int(rng.integers(1, ids.shape[1])); RS.append(torch.stack([C.cap[("act", b)][0, p].float().abs() for b in range(NB)]))
RS = torch.stack(RS); RMU, RSD = RS.mean(0), RS.std(0).clamp_min(1e-6)
def z_of(key, b, j): a, n = FA[key]; return float((a[b, j] - RMU[b, j]) / (RSD[b, j] / math.sqrt(max(n, 1)))) if n else None
def related(fid, rel, v, ws):
    out = []
    for (b, j) in ws:
        zv = z_of((rel, v), b, j) if rel != "code" else None; zr = z_of((rel, "*"), b, j); out.append(dict(block=b, neuron=j, z_value=zv, z_relation=zr, related=bool((zv is not None and zv > 3) or (zr is not None and zr > 3))))
    return out
ROWS = {}
for f in FACTS + CODES:
    fid = f["id"]; ws = WR[fid]; rnd = [(int(b), int(j)) for b, j in zip(rng.integers(0, NB, TOPW), rng.integers(0, DFF, TOPW))]
    ROWS[fid] = dict(rel=f["rel"], v=f["v"], writers=ws, writers_rel=related(fid, f["rel"], f["v"], ws), random_rel=related(fid, f["rel"], f["v"], rnd), delta_rel=related(fid, f["rel"], f["v"], WR_DELTA[fid]),
                     depth_median=med([b for b, _ in ws]) / max(NB - 1, 1), base_overlap=len(set(ws) & set(WR_BASE[fid])) / TOPW, top_contrib=writers(FT_C[fid])[1][:3])
share = lambda fid, key: mean([float(r["related"]) for r in ROWS[fid][key]])
def pair_stats(items):
    same_v, same_r, diff = [], [], []
    for a, b in itertools.combinations(items, 2):
        jac = len(set(WR[a["id"]]) & set(WR[b["id"]])) / len(set(WR[a["id"]]) | set(WR[b["id"]])); (same_v if a["v"] == b["v"] else same_r if a["rel"] == b["rel"] else diff).append(jac)
    return dict(same_value=mean(same_v), same_relation=mean(same_r), different_relation=mean(diff), n=(len(same_v), len(same_r), len(diff)), share_any_same_value=mean([float(x > 0) for x in same_v]), share_any_same_relation=mean([float(x > 0) for x in same_r]), share_any_different=mean([float(x > 0) for x in diff]))
PS = pair_stats(FACTS); PSC = pair_stats(CODES); PSX = mean([len(set(WR[a["id"]]) & set(WR[c["id"]])) / TOPW for a in FACTS[::5] for c in CODES])
cosch = {b: torch.nn.functional.cosine_similarity(write_w(M, b).cpu(), W0[b], dim=0) for b in range(NB)}; allc = torch.cat(list(cosch.values())); wch = [float(cosch[b][j]) for fid in ROWS for b, j in ROWS[fid]["writers"]]
S = dict(facts_related_share=mean([share(f["id"], "writers_rel") for f in FACTS]), facts_random_related_share=mean([share(f["id"], "random_rel") for f in FACTS]), facts_delta_related_share=mean([share(f["id"], "delta_rel") for f in FACTS]),
         codes_related_share=mean([share(c["id"], "writers_rel") for c in CODES]), codes_random_related_share=mean([share(c["id"], "random_rel") for c in CODES]), facts_depth_median=med([ROWS[f["id"]]["depth_median"] for f in FACTS]), codes_depth_median=med([ROWS[c["id"]]["depth_median"] for c in CODES]),
         facts_base_overlap=mean([ROWS[f["id"]]["base_overlap"] for f in FACTS]), codes_base_overlap=mean([ROWS[c["id"]]["base_overlap"] for c in CODES]), pairs_facts=PS, pairs_codes=PSC, fact_code_shared_share=PSX,
         min_write_cos=float(allc.min()), p001_write_cos=float(allc.quantile(0.001)), writers_write_cos_median=med(wch), steps=step, final_acc=final_acc, final_code=final_code, base_acc=base_acc, base_code=base_code, neutral_loss_change=loss1 - loss0)
V = dict(F1_writers_related_half=bool(S["facts_related_share"] >= 0.5), F1b_random_below_01=bool(S["facts_random_related_share"] < 0.1), F2_same_value_jaccard_2x_different=bool(PS["same_value"] is not None and PS["different_relation"] is not None and PS["same_value"] > 2 * PS["different_relation"]),
         F3_codes_related_at_most_half_of_facts=bool(S["codes_related_share"] <= 0.5 * max(S["facts_related_share"], 1e-9)), F3b_codes_share_writers_more_than_facts_same_relation=bool(PSC["same_relation"] is not None and PSC["same_relation"] >= (PS["same_relation"] or 0)),
         F4_depth_in_last_third=bool(S["facts_depth_median"] >= 2 / 3), F5_no_rewritten_row=bool(1 - S["min_write_cos"] < 0.02))
res = dict(model=M["hf"], n_blocks=NB, summary=S, verdicts=V, history=hist, facts={str(f["id"]): ROWS[f["id"]] for f in FACTS + CODES}, values=dict(prof=PROF, city=CITY))
summ = (f"[{name}] {len(FACTS)} facts + {len(CODES)} codes learned in {step} steps (facts {final_acc}, codes {final_code:.2f}, neutral loss {loss1 - loss0:+.3f}); writers firing on the fact's field (z>3): facts {S['facts_related_share']:.2f} (random neurons {S['facts_random_related_share']:.2f}, delta-writers {S['facts_delta_related_share']:.2f}), codes {S['codes_related_share']:.2f} (random {S['codes_random_related_share']:.2f}); "
        f"writer Jaccard same value {f3(PS['same_value'])} / same relation {f3(PS['same_relation'])} / different relation {f3(PS['different_relation'])} (codes among themselves {f3(PSC['same_relation'])}); depth median facts {S['facts_depth_median']:.2f}, codes {S['codes_depth_median']:.2f}; base-route overlap {S['facts_base_overlap']:.2f}; min write cosine change {1 - S['min_write_cos']:.4f}; verdicts {V}")
save(TAG, res, summ)
