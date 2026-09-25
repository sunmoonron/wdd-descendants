"""e482: order without a question. WorkspaceBench's role-bound family asks whether a reader gets the direction of a
described action right ("the thief chased the officer" against the reverse), and argues that single-token readers
cannot, because a ranked list of tokens has no order. e417 showed that once a question fixes the answer, the native
reader carries the other name on suppressing words. Here no question is asked: the story is read at its object token
and at its final period, and a fixed rule tries to recover who did what from each reader.
Setup: Qwen2.5-7B, sixteen single-token animals, three verbs (chased, followed, bit), both orders of every pair
(the same pair appears in both orders, so any decoder that ignores the sentence is at 0.5). Blocks 8-26.
Readers: the plain lens (the two names' logits) and 16 native words (each name's signed logit on the word that carries
it most, as e417). Decoders, evaluated on each cell:
- lens-magnitude: the agent is the name with the larger logit (or the smaller: the rule's better direction is reported,
  with the rule itself);
- native-sign: the agent is the name on a promoting word, the recipient the one on a suppressing word (or the reverse);
  also the share of items in which the two names have opposite signs, which is what a judge would need;
- native-magnitude: as lens-magnitude on the pooled per-word scores.
A rule's accuracy is reported for its better direction, so 0.5 means no order information and 1.0 means the order is
always readable by a fixed rule. The comparison of interest is native-sign against lens-magnitude at the same cell.
Pre-registered (honest guesses):
- at the final period, the lens-magnitude rule reads the order at 0.8 or better in late blocks (recency) (0.5);
- at the object token, the native-sign rule beats the lens-magnitude rule by at least 0.1 in some block (0.4);
- the two names have opposite signs in at least half the items at the best block (0.4)."""
import sys, os, json as _json, itertools; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
name = "qwen7"; K = 16; LAYERS = list(range(8, 27, 2))
model, tok, fam = load_bf16(name); arch = Arch(model, fam)
ANIMALS = ["cat", "dog", "horse", "bird", "cow", "fox", "pig", "lion", "bear", "wolf", "duck", "mouse", "goat", "sheep", "rabbit", "frog"]
ent = [a for a in ANIMALS if len(tok(" " + a, add_special_tokens=False)["input_ids"]) == 1]; first = lambda s: tok(" " + s, add_special_tokens=False)["input_ids"][0]
VERBS = ["chased", "followed", "bit"]
g = torch.Generator().manual_seed(0); pairs = list(itertools.combinations(range(len(ent)), 2)); pairs = [pairs[i] for i in torch.randperm(len(pairs), generator=g)[:40].tolist()]
items = []
for a, b in pairs:
    for v in VERBS:
        for agent, rec in ((a, b), (b, a)):
            text = f"The {ent[agent]} {v} the {ent[rec]}."; ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
            head = f"The {ent[agent]} {v} the {ent[rec]}"; opos = len(tok(head, add_special_tokens=False)["input_ids"]) - 1
            items.append(dict(ids=ids, opos=opos, agent=first(ent[agent]), rec=first(ent[rec])))
N = len(items); log(f"{name}: {N} stories")
ev_all = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"][:4]
ref = ref_stats(model, arch, ev_all, LAYERS); gain, WU = final_readout(model, arch)
A, blk, typ, ends = lean_dictionary(arch, max(LAYERS))
def states_at(where):
    out = {l: [] for l in LAYERS}
    for it in items:
        cap = {}; p = it["opos"] if where == "object" else -1
        hs = [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: cap.__setitem__(l_, (o[0] if isinstance(o, tuple) else o)[0, p].float()))(l)) for l in LAYERS]
        try:
            with torch.no_grad(): model(it["ids"][None].to(DEV))
        finally: [h.remove() for h in hs]
        for l in LAYERS: out[l].append(cap[l])
    return {l: torch.stack(v) for l, v in out.items()}
ag = torch.tensor([it["agent"] for it in items], device=DEV); rc = torch.tensor([it["rec"] for it in items], device=DEV); ar = torch.arange(N, device=DEV)
def rule_acc(score_agent, score_rec):
    """accuracy of 'agent = larger' and its better direction"""
    acc = float((score_agent > score_rec).float().mean()); return max(acc, 1 - acc), ("larger" if acc >= 0.5 else "smaller")
res = dict(model=name, n=N, k=K, layers=LAYERS, by_position={})
for where in ("object", "period"):
    X_all = states_at(where); rows = {}
    for L in LAYERS:
        X = X_all[L]; mu = ref[L]["mu"]; Xc = X - mu; rms = X.pow(2).mean(-1).sqrt(); An = A[:ends[L]]
        sel, _, _ = omp(Xc, An, K, batch=64, record_err=False); cof, _ = refit(Xc, An, sel); words = An[sel]
        lens = lens_scores(X, gain, WU, rms); pooled = pooled_word_scores(words, cof, gain, WU, rms)
        wl_a = torch.stack([lens_scores(words[:, i] * cof[:, i:i + 1], gain, WU, rms)[ar, ag] for i in range(K)], 1); wl_r = torch.stack([lens_scores(words[:, i] * cof[:, i:i + 1], gain, WU, rms)[ar, rc] for i in range(K)], 1)
        sa = wl_a[ar, wl_a.abs().argmax(1)]; sr = wl_r[ar, wl_r.abs().argmax(1)]
        acc_lens, dir_lens = rule_acc(lens[ar, ag], lens[ar, rc]); acc_pool, dir_pool = rule_acc(pooled[ar, ag], pooled[ar, rc]); acc_sign, dir_sign = rule_acc(sa, sr)
        opp = float(((sa > 0) != (sr > 0)).float().mean()); both_top10 = float((torch.isin(ag, lens.topk(10, dim=-1).indices) & torch.isin(rc, lens.topk(10, dim=-1).indices)).float().mean())
        rows[L] = dict(lens_magnitude=acc_lens, lens_rule=dir_lens, native_magnitude=acc_pool, native_magnitude_rule=dir_pool, native_sign=acc_sign, native_sign_rule=dir_sign, opposite_signs=opp,
                       agent_promoted=float((sa > 0).float().mean()), recipient_promoted=float((sr > 0).float().mean()), both_names_in_lens_top10=both_top10)
        log(f"{name} {where} block {L}: lens-magnitude {acc_lens:.2f} ({dir_lens}), native-magnitude {acc_pool:.2f}, native-sign {acc_sign:.2f} (agent promoted {rows[L]['agent_promoted']:.2f}, recipient promoted {rows[L]['recipient_promoted']:.2f}, opposite signs {opp:.2f}); both names in lens top 10 {both_top10:.2f}")
        del sel, words, lens, pooled, wl_a, wl_r; torch.cuda.empty_cache()
    res["by_position"][where] = rows
B = res["by_position"]; best = lambda where, k: max(B[where].values(), key=lambda r: r[k])[k]
res["checks"] = dict(period_lens_magnitude_0_8=max(B["period"][L]["lens_magnitude"] for L in LAYERS if L >= 20) >= 0.8, object_sign_beats_lens_by_0_1=any(B["object"][L]["native_sign"] - B["object"][L]["lens_magnitude"] >= 0.1 for L in LAYERS),
                     opposite_signs_half=best("object", "opposite_signs") >= 0.5 or best("period", "opposite_signs") >= 0.5)
summ = (f"{name} n={N} stories, order by a fixed rule (0.5 = none), best block: " + " || ".join(f"{where}: lens-magnitude {best(where, 'lens_magnitude'):.2f}, native-magnitude {best(where, 'native_magnitude'):.2f}, native-sign {best(where, 'native_sign'):.2f}; opposite signs at most {best(where, 'opposite_signs'):.2f}; by block native-sign / lens-magnitude: "
        + " ".join(f"{L}:{B[where][L]['native_sign']:.2f}/{B[where][L]['lens_magnitude']:.2f}" for L in LAYERS) for where in ("object", "period")) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e482_order_{name}", res, summ)
