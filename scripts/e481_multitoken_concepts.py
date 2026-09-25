"""e481: multi-token concepts through single native words. WorkspaceBench's second desideratum is content the J-lens
gestures at but cannot say in one token; its example is a multi-token entity name. A single-token reader that
surfaces " United" cannot say whether the model is thinking of the United Kingdom, the United States or the United Arab
Emirates. A native word is a write vector of one neuron; if a neuron writes the concept, its own readout may rank both
tokens of the name, and the word then carries the concept as a unit and disambiguates it.
Setup: e420's two-hop items on Qwen2.5-7B, restricted to bridge countries with two-token names (United Kingdom, United
States, United Arab Emirates, South Korea, South Africa, Czech Republic, New Zealand, Saudi Arabia, Costa Rica, Sri
Lanka; extra landmarks added for the last four), gate as e420. Read at the subject's last token (blocks 12-20) and the
final position (blocks 22-26). Readers: the plain lens, 16 native words (each read separately), 16 rotated words.
Scored per cell and pooled over cells (the benchmark's any-cell rule):
- first token of the name in the top 10 (the single-token measure, as e420);
- second token in the top 10;
- both tokens in the top 10 of the same native word (the concept read as a unit);
- disambiguation: among the items whose first token is " United" or " South", the share whose carrying word ranks the
  right second token above the other candidates (Kingdom / States / Arab; Korea / Africa);
- for the lens, the same first, second and both-in-top-10 measures (pooled, no unit).
Pre-registered (honest guesses):
- at the subject token, native words surface the first token in at least 0.1 of items per cell and the second token in
  at least half of those (0.5);
- when a word carries the first token, it disambiguates the second at better than chance (above 0.5 for United, 0.6
  overall) (0.5);
- the lens surfaces neither token at the subject token (0.7)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e420_twohop_subject.py")).read()
exec(src[src.index("FACTS = ["): src.index("name = sys.argv[1]")], globals())
FACTS = FACTS + [("the Sky Tower", "New Zealand", "Wellington", "English", "dollar"), ("Milford Sound", "New Zealand", "Wellington", "English", "dollar"), ("Mecca", "Saudi Arabia", "Riyadh", "Arabic", "riyal"),
                 ("the Arenal Volcano", "Costa Rica", "San José", "Spanish", "colón"), ("Sigiriya", "Sri Lanka", "Colombo", "Sinhala", "rupee"), ("the Temple of the Tooth", "Sri Lanka", "Colombo", "Sinhala", "rupee")]
name = "qwen7"; K = 16; CELLS = {"subject": [12, 14, 16, 18, 20], "final": [22, 24, 26]}
model, tok, fam = load_bf16(name); arch = Arch(model, fam); layers = sorted(set(CELLS["subject"]) | set(CELLS["final"]))
def toks(s): return tok(s, add_special_tokens=False)["input_ids"]
first = lambda s: toks(s)[0]
def enc(text): return torch.tensor(toks(text))
multi = {c for f in FACTS for c in [f[1]] if len(toks(" " + c)) >= 2}
items = []
for lm, country, capital, lang, cur in FACTS:
    if country not in multi: continue
    t = toks(" " + country)
    for rel, ans in [("capital", capital), ("language", lang), ("currency", cur)]:
        if ans is None: continue
        q, _ = TPL[rel]; shots = "".join(f"{q.format(a)} {b}.\n" for a, b in SHOT[rel]); head = shots + q.split("{}")[0] + lm
        items.append(dict(country=country, t1=t[0], t2=t[1], ids=enc(shots + q.format(lm)), spos=len(enc(head)) - 1, onehop=enc("".join(f"{ONEHOP[0].format(a)} {b}.\n" for a, b in ONEHOP[1]) + ONEHOP[0].format(lm[0].upper() + lm[1:])),
                          bridge_ids={t[0], first(country)}, answer_ids={first(" " + ans), first(ans)}))
_, lg2 = last_states(model, arch, [it["ids"] for it in items], []); _, lg1 = last_states(model, arch, [it["onehop"] for it in items], [])
items = [it for i, it in enumerate(items) if lg2[i].argmax().item() in it["answer_ids"] and lg1[i].argmax().item() in it["bridge_ids"]]; N = len(items)
log(f"{name}: {N} of {len(lg2)} multi-token two-hop prompts pass the gate: " + ", ".join(sorted({it['country'] for it in items})))
ev_all = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"][:4]
ref = ref_stats(model, arch, ev_all, layers); gain, WU = final_readout(model, arch)
A, blk, typ, ends = lean_dictionary(arch, max(layers))
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
def states_at(items, layers, where):
    out = {l: [] for l in layers}
    for it in items:
        cap = {}; p = it["spos"] if where == "subject" else -1
        hs = [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: cap.__setitem__(l_, (o[0] if isinstance(o, tuple) else o)[0, p].float()))(l)) for l in layers]
        try:
            with torch.no_grad(): model(it["ids"][None].to(DEV))
        finally: [h.remove() for h in hs]
        for l in layers: out[l].append(cap[l])
    return {l: torch.stack(v) for l, v in out.items()}
# second-token candidates for disambiguation, keyed by the first token
second_of = {}
for c in multi:
    t = toks(" " + c); second_of.setdefault(t[0], set()).add(t[1])
t1 = torch.tensor([it["t1"] for it in items], device=DEV); t2 = torch.tensor([it["t2"] for it in items], device=DEV)
res = dict(model=name, n=N, k=K, cells=CELLS, countries=sorted(multi), by_position={})
for where, Ls in CELLS.items():
    X_all = states_at(items, Ls, where); per_cell = {}; any_ = {k: torch.zeros(N, dtype=torch.bool) for k in ("lens_first", "lens_second", "lens_both", "native_first", "native_second", "native_both_pooled", "native_unit", "rotated_first", "rotated_unit")}
    dis_right, dis_total = 0, 0; unit_types = dict(mlp=0, token=0, head=0)
    for L in Ls:
        X = X_all[L]; mu = ref[L]["mu"]; Xc = X - mu; rms = X.pow(2).mean(-1).sqrt(); An = A[:ends[L]]
        sel, _, _ = omp(Xc, An, K, batch=64, record_err=False); cof, _ = refit(Xc, An, sel); words = An[sel]
        selr, _, _ = omp(Xc @ R.T, An, K, batch=64, record_err=False); cofr, _ = refit(Xc @ R.T, An, selr); wordsr = An[selr] @ R
        lens_top = lens_scores(X, gain, WU, rms).topk(10, dim=-1).indices
        row = dict(lens_first=float((lens_top == t1[:, None]).any(-1).float().mean()), lens_second=float((lens_top == t2[:, None]).any(-1).float().mean()), lens_both=float(((lens_top == t1[:, None]).any(-1) & (lens_top == t2[:, None]).any(-1)).float().mean()))
        any_["lens_first"] |= (lens_top == t1[:, None]).any(-1).cpu(); any_["lens_second"] |= (lens_top == t2[:, None]).any(-1).cpu(); any_["lens_both"] |= ((lens_top == t1[:, None]).any(-1) & (lens_top == t2[:, None]).any(-1)).cpu()
        for rd, W, C, S_ in (("native", words, cof, sel), ("rotated", wordsr, cofr, selr)):
            per = torch.stack([lens_scores(W[:, i] * C[:, i:i + 1], gain, WU, rms) for i in range(K)], 1)     # [N, K, V]
            wt = per.topk(10, dim=-1).indices                                                                  # [N, K, 10]
            f_w = (wt == t1[:, None, None]).any(-1); s_w = (wt == t2[:, None, None]).any(-1)                    # [N, K]
            pooled = per.max(1).values.topk(10, dim=-1).indices
            first_any = f_w.any(1); second_any = s_w.any(1); unit = (f_w & s_w).any(1)
            row[f"{rd}_first"] = float(first_any.float().mean()); row[f"{rd}_second"] = float(second_any.float().mean()); row[f"{rd}_unit"] = float(unit.float().mean())
            row[f"{rd}_both_pooled"] = float(((pooled == t1[:, None]).any(-1) & (pooled == t2[:, None]).any(-1)).float().mean())
            any_[f"{rd}_first"] |= first_any.cpu(); any_[f"{rd}_unit"] |= unit.cpu()
            if rd == "native":
                any_["native_second"] |= second_any.cpu(); any_["native_both_pooled"] |= ((pooled == t1[:, None]).any(-1) & (pooled == t2[:, None]).any(-1)).cpu()
                for n in range(N):
                    if not first_any[n]: continue
                    i = int((f_w[n].float() * per[n, :, t1[n]]).argmax()); cands = sorted(second_of[int(t1[n])])
                    if len(cands) > 1:
                        sc = per[n, i, cands]; dis_total += 1; dis_right += int(cands[int(sc.argmax())] == int(t2[n]))
                    if unit[n]:
                        iu = int(((f_w[n] & s_w[n]).float() * per[n, :, t1[n]]).argmax()); tp = int(typ[S_[n, iu]]); unit_types["mlp" if tp == T_MLP else "token" if tp == T_TOK else "head"] += 1
            del per, wt
        per_cell[L] = row; del sel, selr, words, wordsr; torch.cuda.empty_cache()
        log(f"{name} {where} block {L}: " + ", ".join(f"{k} {v:.2f}" for k, v in row.items()))
    res["by_position"][where] = dict(per_cell=per_cell, any_cell={k: float(v.float().mean()) for k, v in any_.items()}, disambiguation=dict(right=dis_right, total=dis_total, share=(dis_right / dis_total) if dis_total else None), unit_word_types=unit_types)
    log(f"{name} {where} any cell: " + ", ".join(f"{k} {v:.2f}" for k, v in res["by_position"][where]["any_cell"].items()) + f" | disambiguation {dis_right}/{dis_total} | unit word types {unit_types}")
S_, F_ = res["by_position"]["subject"], res["by_position"]["final"]
mean_cell = lambda P, k: sum(v[k] for v in P["per_cell"].values()) / len(P["per_cell"])
res["checks"] = dict(subject_native_first_over_0_1_second_half=mean_cell(S_, "native_first") >= 0.1 and mean_cell(S_, "native_second") >= 0.5 * mean_cell(S_, "native_first"),
                     disambiguates=(S_["disambiguation"]["share"] or 0) > 0.6, lens_blind_at_subject=mean_cell(S_, "lens_first") < 0.05 and mean_cell(S_, "lens_second") < 0.05)
d_ = lambda P: P["disambiguation"]
summ = (f"{name} n={N} multi-token bridges ({len(multi)} countries) | " + " || ".join(f"{w}: any cell: lens first {P['any_cell']['lens_first']:.2f} second {P['any_cell']['lens_second']:.2f} both {P['any_cell']['lens_both']:.2f}; native first {P['any_cell']['native_first']:.2f} second {P['any_cell']['native_second']:.2f} both pooled {P['any_cell']['native_both_pooled']:.2f}, both in one word {P['any_cell']['native_unit']:.2f}; rotated first {P['any_cell']['rotated_first']:.2f} unit {P['any_cell']['rotated_unit']:.2f}; "
        f"per cell mean: lens first {mean_cell(P, 'lens_first'):.2f}, native first {mean_cell(P, 'native_first'):.2f} second {mean_cell(P, 'native_second'):.2f} unit {mean_cell(P, 'native_unit'):.2f}; disambiguation of the second token by the carrying word {d_(P)['right']}/{d_(P)['total']}; unit words {P['unit_word_types']}" for w, P in (("subject", S_), ("final", F_)))
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e481_multitoken_{name}", res, summ)
