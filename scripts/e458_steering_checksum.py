"""e458: native-word steering against dense steering, and WDD as a checksum for interventions. The checksum idea comes
from an external review; the dense comparison is VISION's roadmap item 7.
e455 redirected 67-94% of translations by swapping a concept word (one MLP write row) at every block up to the middle,
at the noun. Two questions remain:
- is one native word a more efficient handle than the standard dense steering vector, the difference of the two nouns'
  mean states?
- can the native description downstream of the intervention tell whether it will work?
Task: e451's few-shot translation into English (24-way restricted readout), non-English prompts the model answers
correctly, with e451's target nouns.
Interventions at the output of every block 0..L, at the noun's tokens. What is added at a block persists into the
next, so each block adds only the increment of the target's typical value; the state at block b then carries about
s times that value (natural size at s = 1):
- native: remove the source concept word's component, and add s times the increment over blocks of the target concept
  word's typical coefficient (s = 0.5, 1, 2);
- dense: add s times the increment over blocks of the difference of the target's and the source's mean noun states in
  that language (s = 0.5, 1, 2);
- native_acc: e455's form, which adds the full typical coefficient at every block and so accumulates. It is kept to
  measure how far e455 over-injected.
(Version 2. The first run added the full value at every block, left the target word's sign uncorrected in the
checksum, and read a zero displacement for fp32 models; its log line is superseded.)
Reported:
- the share of answers moving to the target, against the displacement's size (the mean over blocks of the ratio of its
  norm to the state's norm);
- the injected size at block L: the change in the target word's component at the noun, in units of its typical
  coefficient;
- the checksum: at block L + 2, which is not intervened on, the change in the target concept word's component at the
  last noun token, in units of its typical coefficient (sign-corrected). Its AUC for predicting a moved answer, over
  all interventions, is compared with the displacement size alone.
Models (argument): qwen05, qwen7 (bf16).
Pre-registered (honest guesses):
- per unit of displacement, native steering moves more answers than dense steering (0.5);
- at natural size (s = 1) native swaps move fewer answers than e455's accumulating swaps (0.7);
- the checksum predicts success with AUC at least 0.75 over both kinds (0.6);
- dense steering also raises the target word's component downstream (0.6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e451_concept_handles.py")).read()
exec(src[src.index("NOUNS = {"): src.index("LANGS = list(NOUNS)")])                           # NOUNS, TEMPL, EX, LN
LANGS = list(NOUNS); NC, NT = len(NOUNS["en"]), len(TEMPL["en"])
name = sys.argv[1]
if name == "qwen7":
    from ws_common import load_bf16, lean_dictionary
    model, tok, fam = load_bf16(name); arch = Arch(model, fam); D = arch.D; L = arch.NB // 2
    A, blkv, typv, ends = lean_dictionary(arch, L); A = A[:ends[L]]; blk = blkv[:ends[L]].long(); typ = typv[:ends[L]].long()
else:
    model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; L = arch.NB // 2
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk = lab["type"].to(DEV), lab["block"].to(DEV)
LR = L + 2
cw = _json.load(open(os.path.join(RESULTS, f"e448d_concepts_{name}.json")))["dicts"]["own"]["words"]
EN = [n.split(" ", 1)[1] for n in NOUNS["en"]]
nfirst = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in EN]; dup = {f for f in nfirst if nfirst.count(f) > 1}
okn = [c for c in range(NC) if nfirst[c] not in dup]; cand = torch.tensor([nfirst[c] for c in okn], device=DEV); npos = {c: j for j, c in enumerate(okn)}
concept = {c: int(cw[NOUNS["en"][c]]["word"]) for c in range(NC) if NOUNS["en"][c] in cw and c in npos}
for c, w in concept.items(): assert int(blk[w]) == cw[NOUNS["en"][c]]["block"] and int(typ[w]) == T_MLP
CN = sorted(concept); U = {c: A[w] / A[w].norm() for c, w in concept.items()}; del A; torch.cuda.empty_cache()
tgt = {c: CN[(i + len(CN) // 2) % len(CN)] for i, c in enumerate(CN)}

def prompt(lg, c, t):
    s = TEMPL[lg][t].format(NOUNS[lg][c]); art = NOUNS[lg][c]; noun = art.split(" ", 1)[-1] if " " in art else art.split("'", 1)[-1]
    p = f"{LN[lg]}: {EX[lg][0]}\nEnglish noun: milk\n{LN[lg]}: {EX[lg][1]}\nEnglish noun: flower\n{LN[lg]}: {s}\nEnglish noun:"
    s0 = p.rindex(s); a = s0 + s.index(art) + art.index(noun); b_ = a + len(noun)
    enc = tok(p, add_special_tokens=False, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV)[None]
    return ids, [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_]

def run(ids, pos, fns=None):
    """restricted log-probabilities at the last position; fns: {block: fn(state) -> state} at the noun positions; returns
    also every block's output at the noun positions [LR+1, len(pos), D] and the displacement ratios per block"""
    cap, disp, hs = {}, {}, []
    for b in range(LR + 1):
        def hk(m, i, o, b=b):
            xo = out_of(o)
            if not fns or b not in fns: cap[b] = xo[0, pos].detach().float(); return None
            y = xo.clone(); r = []
            for p in pos:
                x0 = y[0, p].float().clone(); x1 = fns[b](x0); y[0, p] = x1.to(y.dtype); r.append(((x1 - x0).norm() / x0.norm().clamp_min(1e-9)).item())
            cap[b] = y[0, pos].detach().float(); disp[b] = sum(r) / len(r)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hs.append(arch.layers[b].register_forward_hook(hk))
    try:
        with torch.no_grad(): lg = model(ids).logits[0, -1].float()
    finally: [h.remove() for h in hs]
    return torch.log_softmax(lg[cand], -1), torch.stack([cap[b] for b in range(LR + 1)]), (sum(disp.values()) / len(disp) if disp else 0.0)

items = []
for lg in LANGS[1:]:                                                     # non-English sources
    for c in CN:
        for t in range(NT):
            ids, pos = prompt(lg, c, t); lp, X, _ = run(ids, pos); items.append(dict(lg=lg, c=c, ids=ids, pos=pos, lp=lp, X=X))
typc, mean_state = {}, {}
for lg in LANGS[1:]:
    for c in CN:
        Xs = torch.cat([it["X"] for it in items if it["lg"] == lg and it["c"] == c], dim=1)       # [LR+1, n, D]
        typc[(lg, c)] = (Xs @ U[c]).mean(1); mean_state[(lg, c)] = Xs.mean(1)                     # [LR+1], [LR+1, D]
rows = []
for it in items:
    c, lg = it["c"], it["lg"]; tc = tgt[c]; ci, ti = npos[c], npos[tc]
    if int(it["lp"].argmax()) != ci: continue
    u, u2, a2 = U[c], U[tc], typc[(lg, tc)]; dm = mean_state[(lg, tc)] - mean_state[(lg, c)]
    aL = a2[L]; base_chk = float(it["X"][LR, -1] @ u2); base_inj = float((it["X"][L] @ u2).mean())
    inc2 = torch.cat([a2[:1], a2[1:] - a2[:-1]]); incd = torch.cat([dm[:1], dm[1:] - dm[:-1]])
    for kind, s in (("native", 0.5), ("native", 1.0), ("native", 2.0), ("dense", 0.5), ("dense", 1.0), ("dense", 2.0), ("native_acc", 1.0)):
        if kind == "native": fns = {b: (lambda x, b=b: x - (x @ u) * u + s * inc2[b] * u2) for b in range(L + 1)}
        elif kind == "dense": fns = {b: (lambda x, b=b: x + s * incd[b]) for b in range(L + 1)}
        else: fns = {b: (lambda x, b=b: x - (x @ u) * u + a2[b] * u2) for b in range(L + 1)}
        lp1, X1, disp = run(it["ids"], it["pos"], fns)
        rows.append(dict(lg=lg, c=c, kind=kind, s=s, moved=int(lp1.argmax()) == ti, d_target=float(lp1[ti] - it["lp"][ti]), disp=disp,
                         injected=(float((X1[L] @ u2).mean()) - base_inj) / float(aL), checksum=(float(X1[LR, -1] @ u2) - base_chk) / float(aL)))
def auc(score, lab):
    pos_ = [s for s, l in zip(score, lab) if l]; neg = [s for s, l in zip(score, lab) if not l]
    if not pos_ or not neg: return None
    return sum((p > n) + 0.5 * (p == n) for p in pos_ for n in neg) / (len(pos_) * len(neg))
summ_k = {}
for kind, s in (("native", 0.5), ("native", 1.0), ("native", 2.0), ("dense", 0.5), ("dense", 1.0), ("dense", 2.0), ("native_acc", 1.0)):
    rr = [r for r in rows if r["kind"] == kind and r["s"] == s]
    summ_k[f"{kind}_{s}"] = {f: sum(float(r[f]) for r in rr) / len(rr) for f in ("moved", "disp", "injected", "checksum", "d_target")}; summ_k[f"{kind}_{s}"]["n"] = len(rr)
lab_ = [r["moved"] for r in rows]
res = dict(model=name, level=L, readout_block=LR, concept_nouns=[EN[c] for c in CN], summary=summ_k,
           auc_checksum=auc([r["checksum"] for r in rows], lab_), auc_displacement=auc([r["disp"] for r in rows], lab_),
           auc_checksum_dense_only=auc([r["checksum"] for r in rows if r["kind"] == "dense"], [r["moved"] for r in rows if r["kind"] == "dense"]),
           auc_checksum_native_only=auc([r["checksum"] for r in rows if r["kind"].startswith("native")], [r["moved"] for r in rows if r["kind"].startswith("native")]), rows=rows)
S = summ_k
eff = lambda k: S[k]["moved"] / max(S[k]["disp"], 1e-9)
res["checks"] = dict(native_more_per_displacement=sum(eff(f"native_{s}") for s in (0.5, 1.0, 2.0)) > sum(eff(f"dense_{s}") for s in (0.5, 1.0, 2.0)),
                     checksum_auc_over_0_75=(res["auc_checksum"] or 0) >= 0.75, dense_raises_target_word=S["dense_1.0"]["checksum"] > 0)
summ = (f"{name} L{L} (checksum at block {LR}), {len(CN)} concept nouns, non-English: " + " | ".join(f"{k}: moved {v['moved']:.2f} at displacement {v['disp']:.3f} (injected {v['injected']:+.2f} typical, checksum {v['checksum']:+.2f}, dT {v['d_target']:+.2f})" for k, v in S.items())
        + f" || AUC for a moved answer: checksum {res['auc_checksum']:.2f} (native {res['auc_checksum_native_only'] or float('nan'):.2f}, dense {res['auc_checksum_dense_only'] if res['auc_checksum_dense_only'] is not None else float('nan'):.2f}), displacement {res['auc_displacement']:.2f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e458_steering_{name}", res, summ)
