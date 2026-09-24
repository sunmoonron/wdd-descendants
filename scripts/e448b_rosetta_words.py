"""e448b: which native words carry a sentence across languages? (qualitative companion to e448)
For each of e448's 32 sentences (English, French, Spanish, German), at the middle depth:
- the native words used in all four versions and in the fewest other sentences (the most sentence-specific shared
  words);
- for each, the token it is used on in each language (the position with the largest coefficient);
- what its unembedding promotes, and the block and neuron it comes from.
A word used on 'cat', 'chat', 'gato' and 'Katze' and nowhere else would be a language-independent word of the model's own
vocabulary.
Summary: for each sentence, whether the top shared word lands on translation-equivalent tokens (judged by the listing,
not scored automatically); the share of sentences with at least one shared word used by no more than 2 other sentences."""
import sys, os, importlib.util; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
spec = importlib.util.spec_from_file_location("s448", os.path.join(os.path.dirname(os.path.abspath(__file__)), "e448_interlingua.py"))
src = open(spec.origin).read(); S = eval(src[src.index("S = {") + 4: src.index("\nname = sys.argv[1]")])       # the sentence table only
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; L = arch.NB // 2; LANGS = list(S); NS = len(S["en"])
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; cen = EA["cen_ids"][:16].to(DEV); del EA
enc = {lg: [torch.tensor(tok(s, add_special_tokens=False)["input_ids"], device=DEV) for s in S[lg]] for lg in LANGS}
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk, idx = (lab[k].to(DEV) for k in ("type", "block", "index"))
ref = block_states(model, arch, cen, [L], chunk=4)[L]; mu = ref[~sinkmask(ref)].mean(0)
WU = model.get_output_embeddings().weight.detach().float(); fl = model.transformer.ln_f if fam == "gpt2" else (model.gpt_neox.final_layer_norm if fam == "neox" else model.model.norm)
gf = getattr(fl, "weight", None); gf = gf.detach().float() if gf is not None else torch.ones(D, device=DEV)
recs, recs_rot = [], []                                          # per (lang, sentence): selected words with coefficient and the token they sit on
Ar = rotate(A, seed=7)
for lg in LANGS:
    for i, ids in enumerate(enc[lg]):
        x = block_states(model, arch, ids[None], [L], chunk=1)[L][0]; sel, cof, _ = omp(x - mu, A, 16, batch=256, record_err=False)
        toks = ids[1:]; recs.append(dict(lang=lg, i=i, sel=sel, cof=cof, toks=toks))
        selr, _, _ = omp(x - mu, Ar, 16, batch=256, record_err=False); recs_rot.append(dict(lang=lg, i=i, sel=selr))
nw = A.shape[0]
def usage_table(rr):
    u = torch.zeros(len(LANGS), NS, nw, dtype=torch.bool, device=DEV)
    for r in rr: u[LANGS.index(r["lang"]), r["i"], r["sel"].flatten()] = True
    return u
def n_specific(u, shift):
    """sentences whose four versions (language l taking sentence i + l*shift) share a word used by at most 8 other versions"""
    dfu = u.view(-1, nw).float().sum(0); cnt = 0
    for i in range(NS):
        sh = torch.stack([u[l, (i + l * shift) % NS] for l in range(len(LANGS))]).all(0)
        if sh.any() and (dfu[sh] - 4).min() <= 8: cnt += 1
    return cnt
used = usage_table(recs); used_rot = usage_table(recs_rot)
nulls = dict(native_true=n_specific(used, 0), native_mismatched=sum(n_specific(used, s) for s in (1, 5, 11)) / 3, rotated_true=n_specific(used_rot, 0))
log(f"{name}: sentences with a specific shared word: native (true translations) {nulls['native_true']}/{NS}, native (mismatched sentences) {nulls['native_mismatched']:.1f}/{NS}, rotated (true) {nulls['rotated_true']}/{NS}")
df = used.view(-1, nw).float().sum(0)                            # how many of the 128 sentence versions use each word
rows, n_specific_ = [], 0
for i in range(NS):
    shared = used[:, i].all(0); cand = torch.nonzero(shared)[:, 0]
    if cand.numel() == 0: rows.append(dict(i=i, en=S["en"][i], words=[])); continue
    spec_ = (df[cand] - 4); order = spec_.argsort()[:3]; words = []
    for w in cand[order].tolist():
        where = {}
        for r in recs:
            if r["i"] != i: continue
            hit = (r["sel"] == w); pos, k = torch.nonzero(hit, as_tuple=True)
            if pos.numel(): j = pos[r["cof"][pos, k].abs().argmax()]; where[r["lang"]] = tok.decode([int(r["toks"][j])])
        prom = [tok.decode([int(t)]) for t in ((A[w] * gf) @ WU.T).topk(4).indices]
        words.append(dict(word=w, type=int(typ[w]), block=int(blk[w]), index=int(idx[w]), other_sentences=int(df[w].item() - 4), on=where, promotes=prom))
    if words and words[0]["other_sentences"] <= 2 * 4: n_specific_ += 1
    rows.append(dict(i=i, en=S["en"][i], words=words))
    if i < 32: log(f"{name} s{i} '{S['en'][i][:40]}': " + " | ".join(f"b{w['block']}#{w['index']} (+{w['other_sentences']} others) on {w['on']} promotes {w['promotes'][:3]}" for w in words[:2]))
res = dict(model=name, level=L, rows=rows, sentences_with_specific_shared_word=n_specific_ / NS, nulls=nulls)
summ = f"{name} L{L}: {n_specific_}/{NS} sentences (mismatched-sentence null {nulls['native_mismatched']:.1f}, rotated words {nulls['rotated_true']}) have a native word used in all four languages and in at most 8 other sentence versions | examples: " + " ; ".join(
    f"'{r['en'][:25]}' -> b{r['words'][0]['block']}#{r['words'][0]['index']} on {list(r['words'][0]['on'].values())}" for r in rows[:6] if r["words"])
log(summ); record(f"e448b_rosetta_{name}", res, summ)
