"""e622 (session 119): the attribute catalogue, predict first and edit second. Four attributes (gender and generation from
e473's kinship quadruples, number and polarity from new pairs) in three languages, translated into English by few-shot
prompts; for each attribute the dictionary names one row by vote (OMP of the pole-1-minus-pole-0 state difference on the
native rows of the blocks up to b, the most chosen atom, at the block where its share peaks), using only the discovery
templates. Before editing, three predictions per row are fixed: negating it moves the attribute toward the opposite pole
(and more in items where the row's coefficient is larger); it leaves the other three attributes alone; and on general
text the loss change concentrates at the row's own class positions (the positions where it writes above the floor),
better than the neuron's top-activation positions predict. Then the edits: the row's write negated, zeroed and doubled,
against an activation-difference neuron of the same block (the naive selector) and eight random rows of the block, each
negated; scored on discovery and held-out templates, on every attribute's items, and on Pile windows. Pre-registered in
e622_prereg.json. Arguments: model [--smoke]."""
import sys, os, time, copy, math, json, collections; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
from ma_common import out_of, block_states
from datasets import load_dataset
import wdd_common
from wdd_common import omp, build_dictionary
wdd_common.MODELS.setdefault("qwen15i", ("Qwen/Qwen2.5-1.5B-Instruct", "llama"))
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); torch.set_grad_enabled(False)
model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D; DFF = arch.DFF; assert fam == "llama"
for p_ in model.parameters(): p_.requires_grad_(False)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e473_word_algebra.py")).read(); exec(src[src.index("QUADS = {"): src.index("LN = {")], globals())   # QUADS, SHOTS
LN = {"fr": "French", "es": "Spanish", "de": "German"}; LANGS = ["fr", "es", "de"]
NEWSHOTS = {"fr": [("pain", "bread"), ("lait", "milk"), ("soleil", "sun"), ("lune", "moon")], "es": [("pan", "bread"), ("leche", "milk"), ("sol", "sun"), ("luna", "moon")], "de": [("Brot", "bread"), ("Milch", "milk"), ("Sonne", "sun"), ("Mond", "moon")]}
NUMBER = {"en": [("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("house", "houses"), ("car", "cars"), ("tree", "trees"), ("table", "tables"), ("flower", "flowers")], "fr": [("chat", "chats"), ("chien", "chiens"), ("livre", "livres"), ("maison", "maisons"), ("voiture", "voitures"), ("arbre", "arbres"), ("table", "tables"), ("fleur", "fleurs")], "es": [("gato", "gatos"), ("perro", "perros"), ("libro", "libros"), ("casa", "casas"), ("coche", "coches"), ("árbol", "árboles"), ("mesa", "mesas"), ("flor", "flores")], "de": [("Katze", "Katzen"), ("Hund", "Hunde"), ("Buch", "Bücher"), ("Haus", "Häuser"), ("Auto", "Autos"), ("Baum", "Bäume"), ("Tisch", "Tische"), ("Blume", "Blumen")]}
POLAR = {"en": [("good", "bad"), ("big", "small"), ("hot", "cold"), ("happy", "sad"), ("rich", "poor"), ("strong", "weak"), ("young", "old"), ("high", "low")], "fr": [("bon", "mauvais"), ("grand", "petit"), ("chaud", "froid"), ("heureux", "triste"), ("riche", "pauvre"), ("fort", "faible"), ("jeune", "vieux"), ("haut", "bas")], "es": [("bueno", "malo"), ("grande", "pequeño"), ("caliente", "frío"), ("feliz", "triste"), ("rico", "pobre"), ("fuerte", "débil"), ("joven", "viejo"), ("alto", "bajo")], "de": [("gut", "schlecht"), ("groß", "klein"), ("heiß", "kalt"), ("glücklich", "traurig"), ("reich", "arm"), ("stark", "schwach"), ("jung", "alt"), ("hoch", "niedrig")]}
def pairs_of(attr, lg):
    if attr == "gender": return [(q[0], q[1]) for q in QUADS[lg]] + [(q[2], q[3]) for q in QUADS[lg]]
    if attr == "generation": return [(q[0], q[2]) for q in QUADS[lg]] + [(q[1], q[3]) for q in QUADS[lg]]
    return (NUMBER if attr == "number" else POLAR)[lg]
ATTRS = ("gender", "generation", "number", "polarity"); NT = 2 if SMOKE else 5; DISC = (0, 1) if SMOKE else (0, 1, 2)
def shots(lg, t): return (SHOTS[lg][2 * t], SHOTS[lg][2 * t + 1]) if t < 3 else (NEWSHOTS[lg][2 * (t - 3)], NEWSHOTS[lg][2 * (t - 3) + 1])
def prompt(lg, w, t):
    (w1, e1), (w2, e2) = shots(lg, t); p = f"{LN[lg]}: {w1}\nEnglish: {e1}\n{LN[lg]}: {w2}\nEnglish: {e2}\n{LN[lg]}: {w}\nEnglish:"; a = p.rindex(w); b_ = a + len(w)
    enc = tok(p, add_special_tokens=False, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV)[None]
    return ids, [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_][-1]
def run(ids, p, cand):
    cap, hs = {}, []
    for b in range(L + 1):
        def hk(m, i, o, b=b): cap[b] = out_of(o)[0, p].detach().float()
        hs.append(arch.layers[b].register_forward_hook(hk))
    try: lg = model(ids).logits[0, -1].float()
    finally: [h.remove() for h in hs]
    return torch.log_softmax(lg[cand], -1), torch.stack([cap[b] for b in range(L + 1)])
A_, lab = build_dictionary(arch, blocks=list(range(L + 1))); blk = lab["block"].to(DEV); Au = unitr(A_); del A_; ends = {b: int((blk <= b).sum()) for b in range(L + 1)}
ITEMS = {a: [] for a in ATTRS}
for a in ATTRS:
    for lg in LANGS:
        prs = pairs_of(a, lg); ens = pairs_of(a, "en"); assert len(prs) == len(ens)
        for k, (pr, en) in enumerate(zip(prs, ens)):
            if SMOKE and k >= 2: continue
            cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in en]
            if cand[0] == cand[1]: continue
            cand = torch.tensor(cand, device=DEV)
            for t in range(NT):
                runs = [prompt(lg, pr[i], t) for i in range(2)]; outs = [run(ids, p, cand) for ids, p in runs]
                if not all(int(lp.argmax()) == i for i, (lp, _) in enumerate(outs)): continue
                for i in range(2): ITEMS[a].append(dict(attr=a, lg=lg, pair=k, t=t, side=i, cand=cand, ids=runs[i][0], p=runs[i][1], lp0=outs[i][0], X=outs[i][1], Xp=outs[1 - i][1], disc=(t in DISC)))
log(f"{name}: items " + ", ".join(f"{a} {len(ITEMS[a])} ({sum(it['disc'] for it in ITEMS[a])} discovery)" for a in ATTRS) + f" ({time.time() - t0:.0f}s)")
# ---- the vote per attribute (discovery templates only), the naive selector, the predictions
def vote(a):
    its = [it for it in ITEMS[a] if it["disc"] and it["side"] == 0]; out = {}
    for b in range(L + 1):
        Xb = torch.stack([it["Xp"][b] - it["X"][b] for it in its]); sel, _, _ = omp(Xb, Au[:ends[b]], 8, batch=256, record_err=False); cnt = torch.bincount(sel.reshape(-1), minlength=ends[b]); top = int(cnt.argmax()); out[b] = (top, float(cnt[top]) / Xb.shape[0], int(lab["type"][top]))
    bm = max(out, key=lambda b: out[b][1] if out[b][2] == 2 else -1); return out, bm
def acts_at(b, its):
    cap = {}
    def hk(mod, inp): cap["a"] = inp[0]
    out = []
    for it in its:
        h = arch.layers[b].mlp.down_proj.register_forward_pre_hook(hk)
        try: model(it["ids"])
        finally: h.remove()
        out.append(cap["a"][0, it["p"]].float())
    return torch.stack(out)
CAT = {}
for a in ATTRS:
    out, bm = vote(a); atom, share, typ = out[bm]; j = int(lab["index"][atom]); its = [it for it in ITEMS[a] if it["disc"]]
    acts = acts_at(bm, its); sides = torch.tensor([it["side"] for it in its], device=DEV); dmean = acts[sides == 1].mean(0) - acts[sides == 0].mean(0); order = dmean.abs().argsort(descending=True).tolist(); j_naive = order[0] if order[0] != j else order[1]
    mu = torch.stack([it["X"][bm] for it in its]).mean(0); coef = {id(it): float((it["X"][bm] - mu) @ Au[atom]) for it in ITEMS[a]}
    CAT[a] = dict(block=bm, atom=atom, row=j, share=share, type=typ, naive_row=j_naive, naive_rank_of_wdd_row=(order.index(j) if j in order else None), vote_by_block={str(b): dict(atom=v[0], share=v[1], type=v[2]) for b, v in out.items()}, coef=coef)
    log(f"{a}: vote block {bm} atom {atom} (type {typ}, row {j}) in {share:.2f} of discovery differences; naive activation-difference neuron {j_naive} (the dictionary's row ranks {CAT[a]['naive_rank_of_wdd_row']} by |activation difference|)")
# ---- the Pile windows, their states and the rows' classes
pile = load_dataset("NeelNanda/pile-10k", split="train"); buf, wins = [], []
for ex in pile:
    buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
    while len(buf) >= 257 and len(wins) < (4 if SMOKE else 16): wins.append(buf[:257]); buf = buf[257:]
    if len(wins) >= (4 if SMOKE else 16): break
PW = torch.tensor(wins, device=DEV); T = 256
def pos_loss(m):
    out = []
    for s0 in range(0, PW.shape[0], 8):
        x = PW[s0:s0 + 8]; lg = m(x).logits.float(); out.append(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="none").reshape(x.shape[0], -1)); del lg
    return torch.cat(out)   # [N, T]: entry [n, q] = loss predicting token q+1 from position q
def class_and_act_masks(b, atom, j):
    """on the Pile windows at block b: the row's class positions (projection ratio above 1 on the window states) and the top-activation positions of neuron j of the same count; both over state positions 1..T-1 of each window"""
    X = block_states(model, arch, PW[:, :T], [b])[b].reshape(-1, D); keep = ~sinkmask(X); U = unitr(X[keep] - X[keep].mean(0)); Ab, _ = rows_of(arch, b); st = stats(U, Ab, K); R = st["ratio"].float()
    cls_k = R[:, b * DFF + j] > 1; cls = torch.zeros(X.shape[0], dtype=torch.bool, device=DEV); cls[torch.nonzero(keep)[:, 0][cls_k]] = True   # the row's index among the MLP rows of blocks 0..b is b*DFF + j
    cap = {}
    def hk(mod, inp): cap.setdefault("a", []).append(inp[0][:, 1:, j].detach().float())
    h = arch.layers[b].mlp.down_proj.register_forward_pre_hook(hk)
    try:
        for s0 in range(0, PW.shape[0], 8): model(PW[s0:s0 + 8, :T])
    finally: h.remove()
    act = torch.cat(cap["a"]).reshape(-1); n_cls = int(cls.sum()); top = torch.zeros_like(cls); top[act.abs().argsort(descending=True)[:max(n_cls, 1)]] = True
    return cls, top, n_cls
def auc(score, pos):
    s = score.float(); order = s.argsort(); ranks = torch.empty_like(s); ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=s.dtype); npos = int(pos.sum()); nneg = s.numel() - npos
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1)) if npos and nneg else None
L0 = pos_loss(model)
# ---- edits and scoring
def mlp_of(b): return arch.layers[b].mlp
def snapshot(b, j): l = mlp_of(b); return dict(down=l.down_proj.weight[:, j].clone(), gate=l.gate_proj.weight[j, :].clone(), up=l.up_proj.weight[j, :].clone())
def edit(kind, b, j, sv):
    l = mlp_of(b)
    if kind == "zero": l.down_proj.weight[:, j] = 0
    elif kind == "negate": l.down_proj.weight[:, j] = -sv["down"]
    elif kind == "double": l.down_proj.weight[:, j] = 2 * sv["down"]
def restore(b, j, sv): l = mlp_of(b); l.down_proj.weight[:, j] = sv["down"]; l.gate_proj.weight[j, :] = sv["gate"]; l.up_proj.weight[j, :] = sv["up"]
def score_attr(a):
    """flip rate and mean score shift (log-prob of the partner's candidate minus the item's own, change from unedited), on discovery and held-out templates; per-item shifts"""
    out = {"disc": [0, 0.0, 0], "held": [0, 0.0, 0]}; per = []
    for it in ITEMS[a]:
        lp, _ = run(it["ids"], it["p"], it["cand"]); own, oth = it["side"], 1 - it["side"]; sh = float((lp[oth] - lp[own]) - (it["lp0"][oth] - it["lp0"][own])); key = "disc" if it["disc"] else "held"
        out[key][0] += int(int(lp.argmax()) == oth); out[key][1] += sh; out[key][2] += 1; per.append((id(it), sh))
    return {k: dict(flip=v[0] / max(v[2], 1), shift=v[1] / max(v[2], 1), n=v[2]) for k, v in out.items()}, per
def spearman(x, y):
    n = len(x)
    if n < 4: return None
    rx = torch.tensor(x).argsort().argsort().float(); ry = torch.tensor(y).argsort().argsort().float(); return float(torch.corrcoef(torch.stack([rx, ry]))[0, 1])
def collateral(b, j, atom):
    L1 = pos_loss(model); d = (L1 - L0)[:, 1:T - 1].reshape(-1); cls, top, n_cls = class_and_act_masks(b, atom, j); cls = cls.view(PW.shape[0], T - 1)[:, :T - 2].reshape(-1); top = top.view(PW.shape[0], T - 1)[:, :T - 2].reshape(-1)
    return dict(mean_loss_change=float(d.mean()), mean_abs_change=float(d.abs().mean()), n_class_positions=n_cls, auc_class=auc(d.abs(), cls), auc_top_activation=auc(d.abs(), top), class_mean_change=float(d[cls].mean()) if cls.any() else None, top_mean_change=float(d[top].mean()) if top.any() else None, rest_mean_change=float(d[~cls].mean()))
res = dict(model=name, level=L, n_items={a: len(ITEMS[a]) for a in ATTRS}, catalogue={a: {k: v for k, v in CAT[a].items() if k != "coef"} for a in ATTRS}, edits={}, naive={}, random={})
g_r = torch.Generator().manual_seed(11)
for a in ATTRS:
    c = CAT[a]; b, j, atom = c["block"], c["row"], c["atom"]; sv = snapshot(b, j); res["edits"][a] = {}
    for kind in ("negate", "zero", "double"):
        edit(kind, b, j, sv); sc = {a2: score_attr(a2)[0] for a2 in ATTRS}; per = score_attr(a)[1] if kind == "negate" else None; col = collateral(b, j, atom) if kind == "negate" else None; restore(b, j, sv)
        ent = dict(scores=sc, collateral=col)
        if per: ent["coef_spearman"] = spearman([abs(c["coef"][i]) for i, _ in per], [abs(s) for _, s in per])
        res["edits"][a][kind] = ent
        log(f"{a} row {j} (block {b}) {kind}: own attribute flip disc/held {sc[a]['disc']['flip']:.2f}/{sc[a]['held']['flip']:.2f}, shift {sc[a]['disc']['shift']:+.2f}/{sc[a]['held']['shift']:+.2f}; other attributes' shifts " + ", ".join(f"{a2} {sc[a2]['disc']['shift']:+.2f}" for a2 in ATTRS if a2 != a) + (f"; Pile loss {col['mean_loss_change']:+.4f}, |change| AUC class {col['auc_class']} vs top-activation {col['auc_top_activation']} ({col['n_class_positions']} class positions)" if col else "") + (f"; coefficient Spearman {ent.get('coef_spearman')}" if per else "") + f" ({time.time() - t0:.0f}s)")
    jn = c["naive_row"]; svn = snapshot(b, jn); edit("negate", b, jn, svn); scn = {a2: score_attr(a2)[0] for a2 in ATTRS}; coln = collateral(b, jn, None) if False else dict(mean_loss_change=float((pos_loss(model) - L0).mean())); restore(b, jn, svn)
    res["naive"][a] = dict(row=jn, scores=scn, pile_loss_change=coln["mean_loss_change"]); log(f"{a} naive neuron {jn} negated: own flip {scn[a]['disc']['flip']:.2f}/{scn[a]['held']['flip']:.2f}, shift {scn[a]['disc']['shift']:+.2f}; Pile loss {coln['mean_loss_change']:+.4f}")
    res["random"][a] = []
    for r in torch.randperm(DFF, generator=g_r)[:(2 if SMOKE else 8)].tolist():
        if r in (j, jn): continue
        svr = snapshot(b, r); edit("negate", b, r, svr); scr = score_attr(a)[0]; lr_ = float((pos_loss(model) - L0).mean()); restore(b, r, svr); res["random"][a].append(dict(row=r, flip_disc=scr["disc"]["flip"], flip_held=scr["held"]["flip"], shift=scr["disc"]["shift"], pile_loss_change=lr_))
    log(f"{a} random rows negated: flip held at most {max([x['flip_held'] for x in res['random'][a]] or [0]):.2f}, |shift| at most {max([abs(x['shift']) for x in res['random'][a]] or [0]):.2f}")
# ---- verdicts
dom = [a for a in ATTRS if CAT[a]["share"] >= 0.5]; c1 = len(dom) >= 3
def held_flip(a, who="wdd"): return res["edits"][a]["negate"]["scores"][a]["held"]["flip"] if who == "wdd" else res["naive"][a]["scores"][a]["held"]["flip"]
c2 = bool(dom) and all(held_flip(a) >= 0.15 and res["edits"][a]["negate"]["scores"][a]["held"]["shift"] > 0 and held_flip(a) > held_flip(a, "naive") for a in dom)
def diag_ratio(a): own = abs(res["edits"][a]["negate"]["scores"][a]["disc"]["shift"]); oth = max(abs(res["edits"][a]["negate"]["scores"][a2]["disc"]["shift"]) for a2 in ATTRS if a2 != a); return own / max(oth, 1e-6)
c3 = bool(dom) and all(diag_ratio(a) >= 5 for a in dom)
c4 = bool(dom) and all((res["edits"][a]["negate"]["collateral"]["auc_class"] or 0) >= 0.7 and (res["edits"][a]["negate"]["collateral"]["auc_class"] or 0) > (res["edits"][a]["negate"]["collateral"]["auc_top_activation"] or 0) for a in dom)
c5 = bool(dom) and all((res["edits"][a]["negate"].get("coef_spearman") or 0) >= 0.3 for a in dom)
res["verdicts"] = dict(C1=c1, C2=c2, C3=c3, C4=c4, C5=c5, dominant=dom)
summ = (f"attribute catalogue ({name}): " + "; ".join(f"{a}: row {CAT[a]['row']} of block {CAT[a]['block']} (vote {CAT[a]['share']:.2f}), negated flips {res['edits'][a]['negate']['scores'][a]['disc']['flip']:.2f} disc / {res['edits'][a]['negate']['scores'][a]['held']['flip']:.2f} held (shift {res['edits'][a]['negate']['scores'][a]['held']['shift']:+.2f}), naive {res['naive'][a]['scores'][a]['held']['flip']:.2f}, random {max([x['flip_held'] for x in res['random'][a]] or [0]):.2f}; diagonal ratio {diag_ratio(a):.1f}; Pile loss {res['edits'][a]['negate']['collateral']['mean_loss_change']:+.4f}, damage AUC class {res['edits'][a]['negate']['collateral']['auc_class']} vs activation {res['edits'][a]['negate']['collateral']['auc_top_activation']}; coefficient Spearman {res['edits'][a]['negate'].get('coef_spearman')}" for a in ATTRS) + f"; dominant {dom}; verdicts C1 {c1}, C2 {c2}, C3 {c3}, C4 {c4}, C5 {c5}")
log(summ); record(f"e622_catalogue_{name}" + ("_smoke" if SMOKE else ""), res, summ)
