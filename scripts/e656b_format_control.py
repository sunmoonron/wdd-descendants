"""e656b (session 143): is the repeat's capitals gain knowledge or format? e656 found held-out probe gains from repeating a
middle block in Qwen2.5-0.5B (+0.31) and SmolLM2-135M (+0.16), nearly all on e560's capitals, where Qwen's commonest
base answer is a quiz blank (' __', e654) that the function-word mask does not remove. Control: read the base model and
the repeated block under a content mask that keeps only tokens with a letter that are not function words (every
capital's first token survives it). If the base under the mask matches the repeat, the repeat's gain is format; if the
repeat still wins under the mask, it adds ranking the base lacks. Blocks: e656's held-out block (masked selection) and
its three largest masked gains. Also the base's answer kinds on capitals (blank, function word, other content) and, on
the items the repeat fixes, what the base said. Arguments: model."""
from rys_common import *
name = sys.argv[1]; t0 = time.time()
M = load(name); N = M["NB"]; tok = M["tok"]; V = M["WU"].shape[0]
P = probes(M); KD = kinds(P); STOP = stopset(M, P)
cap_idx = [n for n, (k, _, _) in enumerate(P[0]) if k == "capitals"]; Pc = ([P[0][n] for n in cap_idx], collections.defaultdict(list))
for n_, (k, pi, t) in enumerate(Pc[0]): Pc[1][len(pi)].append((n_, k, pi, t))
strs = tok.convert_ids_to_tokens(list(range(min(V, len(tok)))))
has_letter = torch.zeros(V, dtype=torch.bool, device=DEV)
for i_, s_ in enumerate(strs):
    s_ = tok.convert_tokens_to_string([s_]) if s_ is not None else ""
    has_letter[i_] = any(ch.isalpha() for ch in s_)
CONTENT_MASK = ~has_letter | STOP  # masked = no letter, or a function word
ans = torch.tensor([t for _, _, t in Pc[0]], device=DEV); assert not bool(CONTENT_MASK[ans].any()), "a capital's first token is masked"
CAND = torch.ones(V, dtype=torch.bool, device=DEV); CAND[ans.unique()] = False  # forced choice (post-hoc, added after C1): only the capitals' own first tokens compete
def read(per_batch=None):
    items, buckets = Pc; raw = np.zeros(len(items), int); masked = np.zeros(len(items), int); content = np.zeros(len(items), int); forced = np.zeros(len(items), int)
    for L_, b in buckets.items():
        for s0 in range(0, len(b), 32):
            chunk = b[s0:s0 + 32]; ids = torch.tensor([c[2] for c in chunk], device=DEV); lg = M["model"](ids, use_cache=False).logits[:, -1].float()
            r_ = lg.argmax(-1).tolist(); m_ = lg.masked_fill(STOP[None], -1e9).argmax(-1).tolist(); c_ = lg.masked_fill(CONTENT_MASK[None], -1e9).argmax(-1).tolist(); f_ = lg.masked_fill(CAND[None], -1e9).argmax(-1).tolist()
            for c, a, bb, cc, ff in zip(chunk, r_, m_, c_, f_): raw[c[0]], masked[c[0]], content[c[0]], forced[c[0]] = a, bb, cc, ff
    return raw, masked, content, forced
tgt = np.array([t for _, _, t in Pc[0]])
b_raw, b_m, b_c, b_f = read()
def kind_of(t): return "function word" if bool(STOP[t]) else ("no letter (blank, punctuation)" if not bool(has_letter[t]) else "other content")
base_kinds = collections.Counter(kind_of(int(t)) for t, g in zip(b_raw, tgt) if t != g)
R6 = json.load(open(f"results/e656_rys_map_{name}.json")); blocks = [tuple(R6["heldout_masked"]["block"])] + [(r["i"], r["j"]) for r in sorted(R6["rows"], key=lambda r: -r["gain_m"])[:3]]
blocks = list(dict.fromkeys(blocks))
res = dict(model=name, n=len(tgt), n_candidates=int((~CAND).sum()), base=dict(raw=float((b_raw == tgt).mean()), function_masked=float((b_m == tgt).mean()), content_masked=float((b_c == tgt).mean()), forced_choice=float((b_f == tgt).mean())), base_wrong_kinds=dict(base_kinds), blocks=[])
print(f"[{name}] capitals base: raw {res['base']['raw']:.3f}, function words masked {res['base']['function_masked']:.3f}, content mask {res['base']['content_masked']:.3f}; base wrong answers by kind {dict(base_kinds)} ({time.time() - t0:.0f}s)", flush=True)
for (i, j) in blocks:
    with Repeat(M, rys_seq(N, i, j)): r_raw, r_m, r_c, r_f = read()
    fixed = [(tok.decode([int(b_raw[n])]), tok.decode([int(r_raw[n])])) for n in range(len(tgt)) if r_raw[n] == tgt[n] and b_raw[n] != tgt[n]]
    rec = dict(i=i, j=j, raw=float((r_raw == tgt).mean()), function_masked=float((r_m == tgt).mean()), content_masked=float((r_c == tgt).mean()), forced_choice=float((r_f == tgt).mean()), n_fixed_forced=int(((r_f == tgt) & (b_f != tgt)).sum()), n_broken_forced=int(((r_f != tgt) & (b_f == tgt)).sum()), n_fixed_raw=len(fixed), fixed_base_answer_kinds=dict(collections.Counter(kind_of(int(b_raw[n])) for n in range(len(tgt)) if r_raw[n] == tgt[n] and b_raw[n] != tgt[n])),
               n_fixed_content=int(((r_c == tgt) & (b_c != tgt)).sum()), n_broken_content=int(((r_c != tgt) & (b_c == tgt)).sum()), examples=fixed[:8])
    res["blocks"].append(rec); print(f"[{name}] block {i}-{j}: raw {rec['raw']:.3f}, function-masked {rec['function_masked']:.3f}, content-masked {rec['content_masked']:.3f} (base {res['base']['content_masked']:.3f}: fixes {rec['n_fixed_content']}, breaks {rec['n_broken_content']}); forced choice {rec['forced_choice']:.3f} (base {res['base']['forced_choice']:.3f}: fixes {rec['n_fixed_forced']}, breaks {rec['n_broken_forced']}); raw fixes {rec['n_fixed_raw']}, the base had said {rec['fixed_base_answer_kinds']}; e.g. {rec['examples'][:4]} ({time.time() - t0:.0f}s)", flush=True)
best = max(res["blocks"], key=lambda r: r["content_masked"]); res["best_block_content"] = [best["i"], best["j"], best["content_masked"]]
bestf = max(res["blocks"], key=lambda r: r["forced_choice"]); res["best_block_forced"] = [bestf["i"], bestf["j"], bestf["forced_choice"]]; held = res["blocks"][0]
res["verdicts"] = dict(C1=best["content_masked"] - res["base"]["content_masked"] >= 0.05, C2_posthoc=held["forced_choice"] - res["base"]["forced_choice"] >= 0.05)
summ = f"[{name}] capitals under the content mask: base {res['base']['content_masked']:.3f}, best repeat {best['i']}-{best['j']} {best['content_masked']:.3f} (raw: base {res['base']['raw']:.3f}, repeat {best['raw']:.3f}); forced choice among {res['n_candidates']} capital tokens: base {res['base']['forced_choice']:.3f}, held-out block {held['i']}-{held['j']} {held['forced_choice']:.3f}; verdicts {res['verdicts']} ({time.time() - t0:.0f}s)"
save(f"e656b_format_control_{name}", res, summ); print("done", flush=True)
