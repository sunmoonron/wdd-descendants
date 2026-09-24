"""e448c: are the Rosetta words just multilingual neurons, or directions the self-description uses while their neuron is
silent?
e448b found, at the middle depth of Qwen2.5-0.5B, a native word (an MLP write row) for nearly every sentence. Each sits
on the translation-equivalent token in English, French, Spanish and German and is used almost nowhere else (answer /
réponse / respuesta / Antwort, very / très / muy / sehr, phone / téléphone / teléfono / Handy).
Activation-based studies find "multilingual neurons" that fire for a concept across languages. WDD's description is a
re-description (e391): a word is chosen because its direction explains the state, whoever wrote it. Two readings:
 (i) the Rosetta word's own neuron fires on those tokens, and WDD rediscovers multilingual neurons without labels;
 (ii) the neuron is not especially active there, and the direction is written by other components. Then the concept
      is visible in the model's own vocabulary and invisible to activation-based neuron analysis.
For each sentence's top shared word (the 32 of e448b), at each of its four usage positions:
- the percentile of the neuron's activation there, among its activations over 8 x 512 tokens of ordinary text;
- the share of the state's projection on the word's direction contributed by the word's own neuron (its activation
  times its write row's norm, against the projection of the whole centred state).
Pre-registered (honest guess): reading (ii) for most words (median activation percentile below 90, own-neuron share of
the projection below 0.3), 0.5."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
import json as _json
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; L = arch.NB // 2
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e448_interlingua.py")).read(); S = eval(src[src.index("S = {") + 4: src.index("\nname = sys.argv[1]")]); LANGS = list(S)
rows = _json.load(open(os.path.join(RESULTS, f"e448b_rosetta_{name}.json")))["rows"]
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; cen = EA["cen_ids"][:16].to(DEV); ref_ids = EA["eval_ids"][:8].to(DEV); del EA
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk, idx = (lab[k].to(DEV) for k in ("type", "block", "index"))
ref = block_states(model, arch, cen, [L], chunk=4)[L]; mu = ref[~sinkmask(ref)].mean(0)
targets = [(r["i"], r["words"][0]) for r in rows if r["words"] and r["words"][0]["type"] == T_MLP]
neur = sorted({(w["block"], w["index"]) for _, w in targets})
def neuron_acts(ids):
    """{(block, index): [T] activation (the MLP down-projection input)}"""
    out = {}
    def mk(b, cols):
        def pre(m, a): out.update({(b, c): a[0][0, :, c].float().clone() for c in cols})
        return pre
    byb = {}
    for b, c in neur: byb.setdefault(b, []).append(c)
    hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b, cols)) for b, cols in byb.items()]
    try:
        with torch.no_grad(): model(ids)
    finally: [h.remove() for h in hs]
    return out
refacts = {k: [] for k in neur}
for s in range(ref_ids.shape[0]):
    a = neuron_acts(ref_ids[s:s + 1])
    for k in neur: refacts[k].append(a[k])
refacts = {k: torch.cat(v) for k, v in refacts.items()}
out, pct_all, share_all = [], [], []
for i, w in targets:
    b, c = w["block"], w["index"]; wrow = arch.wdir(b)[c]; wn = wrow.norm(); u = wrow / wn; per_lang = {}
    for lg in LANGS:
        ids = torch.tensor(tok(S[lg][i], add_special_tokens=False)["input_ids"], device=DEV)[None]
        x = block_states(model, arch, ids, [L], chunk=1)[L][0]; sel, cof, _ = omp(x - mu, A, 16, batch=256, record_err=False)
        hit = torch.nonzero((sel == w["word"]).any(1))[:, 0]
        if hit.numel() == 0: continue
        acts = neuron_acts(ids)[(b, c)][1:]                                     # positions 1: to align with the states
        j = hit[cof[hit][(sel[hit] == w["word"])].abs().argmax()] if hit.numel() > 1 else hit[0]
        act = acts[j].item(); pct = (refacts[(b, c)].abs() < abs(act)).float().mean().item()
        proj = ((x[j] - mu) @ u).item(); own = act * wn.item(); share = own / proj if abs(proj) > 1e-6 else float("nan")
        per_lang[lg] = dict(token=tok.decode([int(ids[0, 1 + j])]), activation=act, act_percentile=pct, state_projection=proj, own_neuron_contribution=own, own_share=share)
        pct_all.append(pct); share_all.append(share)
    out.append(dict(sentence=S["en"][i], block=b, index=c, langs=per_lang))
    log(f"{name} '{S['en'][i][:30]}' b{b}#{c}: " + " | ".join(f"{lg} {v['token']!r} act pct {v['act_percentile']:.2f} own share {v['own_share']:+.2f}" for lg, v in per_lang.items()))
pa, sa = torch.tensor(pct_all), torch.tensor([s for s in share_all if s == s])
res = dict(model=name, level=L, words=out, median_act_percentile=pa.median().item(), frac_act_above_p90=(pa > 0.9).float().mean().item(), median_own_share=sa.median().item(), frac_own_share_over_0_5=(sa > 0.5).float().mean().item())
res["checks"] = dict(reading_ii=res["median_act_percentile"] < 0.9 and res["median_own_share"] < 0.3)
summ = (f"{name} L{L}: {len(out)} Rosetta words, {pa.numel()} usage positions | their own neuron's activation there: median percentile {res['median_act_percentile']:.2f} of ordinary text "
        f"(share above the 90th: {res['frac_act_above_p90']:.2f}) | own neuron's share of the state's projection on the word: median {res['median_own_share']:+.2f} (share over 0.5: {res['frac_own_share_over_0_5']:.2f}) | checks {json.dumps(res['checks'])}")
log(summ); record(f"e448c_rosetta_neurons_{name}", res, summ)
