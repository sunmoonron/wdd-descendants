"""e417: roles, not just a bag of names. WorkspaceBench's anti-bag-of-words and role-bound families ask whether a reader
keeps who did what to whom; a single-token reader that surfaces both names cannot say which one is the answer. Prompts
with two single-token names in a relation and a question about one role ("In the final, Alice beat Bob. The winner
was" / "... The loser was"), both orders, several relations; Qwen2.5-7B, kept when the model answers correctly.
At the last position after blocks 4, 6, ..., 26: for each reader (logit lens, centred lens, PCA lens, native-word lens,
rotated control), the pairwise accuracy (answer name scored above the other name; chance 0.5). For the native-word
lens also the sign structure: for each name, the signed logit of the word that gives it the largest absolute logit
(does the other name sit on a suppressing word?), and provenance: the block of the word carrying each name.
Pre-registered: at middle depth the native-word lens resolves the role (pairwise accuracy above the plain lens) and
the other name is more often carried by a suppressing word than the answer name."""
import sys, os, itertools; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
name = sys.argv[1] if len(sys.argv) > 1 else "qwen7"; K = 16
model, tok, fam = load_bf16(name); arch = Arch(model, fam); layers = list(range(4, arch.NB - 1, 2))
NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack", "Laura", "Mark", "Nina", "Oscar", "Paul", "Rosa", "Sam", "Tom", "Vera", "Will"]
NAMES = [n for n in NAMES if len(tok(" " + n, add_special_tokens=False)["input_ids"]) == 1]
REL = [("In the final, {a} beat {b}.", "The winner was", "The loser was"), ("{a} is taller than {b}.", "The taller one is", "The shorter one is"),
       ("{a} sold the car to {b}.", "The buyer was", "The seller was"), ("{a} sent a letter to {b}.", "The letter was received by", "The letter was sent by")]
# answer for question 1 is a for relations 0,1,3 ... define explicitly: (relation, question index) -> 'a' or 'b'
ANS = {(0, 1): "a", (0, 2): "b", (1, 1): "a", (1, 2): "b", (2, 1): "b", (2, 2): "a", (3, 1): "b", (3, 2): "a"}
SHOTS = {0: ("In the final, Xavier beat Yusuf.", "Xavier"), 1: ("Xavier is taller than Yusuf.", "Xavier"), 2: ("Xavier sold the car to Yusuf.", "Yusuf"), 3: ("Xavier sent a letter to Yusuf.", "Yusuf")}
rng = torch.Generator().manual_seed(0); pairs = list(itertools.permutations(NAMES, 2)); sel_pairs = [pairs[i] for i in torch.randperm(len(pairs), generator=rng)[:30].tolist()]
first = lambda s: tok(s, add_special_tokens=False)["input_ids"][0]
items = []
for ri, (stmt, q1, q2) in enumerate(REL):
    for qi, q in [(1, q1), (2, q2)]:
        shot_stmt, _ = SHOTS[ri]; shot_ans = "Xavier" if ANS[(ri, qi)] == "a" else "Yusuf"
        for a, b in sel_pairs:
            text = f"{shot_stmt} {q} {shot_ans}.\n{stmt.format(a=a, b=b)} {q}"
            ans, oth = (a, b) if ANS[(ri, qi)] == "a" else (b, a)
            items.append(dict(rel=ri, q=qi, ids=torch.tensor(tok(text, add_special_tokens=False)["input_ids"]), ans=first(" " + ans), oth=first(" " + oth)))
_, lg = last_states(model, arch, [it["ids"] for it in items], [])
items = [it for i, it in enumerate(items) if lg[i].argmax().item() == it["ans"]]; N = len(items); log(f"{name}: {N} of {len(lg)} role prompts answered correctly")
ev_all = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"][:4]
ref = ref_stats(model, arch, ev_all, layers); gain, WU = final_readout(model, arch)
A, blk, typ, ends = lean_dictionary(arch, max(layers))
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
X_all, _ = last_states(model, arch, [it["ids"] for it in items], layers)
at = torch.tensor([it["ans"] for it in items], device=DEV); ot = torch.tensor([it["oth"] for it in items], device=DEV); ar = torch.arange(N, device=DEV)
res = dict(model=name, n=N, k=K, layers=layers, readers={}, signs={})
for L in layers:
    X = X_all[L]; mu = ref[L]["mu"]; Xc = X - mu; rms = X.pow(2).mean(-1).sqrt(); P = ref[L]["pcs"]; An = A[:ends[L]]
    sel, _, _ = omp(Xc, An, K, batch=64, record_err=False); cof, _ = refit(Xc, An, sel); words = An[sel]
    selr, _, _ = omp(Xc @ R.T, An, K, batch=64, record_err=False); cofr, _ = refit(Xc @ R.T, An, selr); wordsr = An[selr] @ R
    parts = (Xc @ P)[:, :, None] * P.T[None]
    S = dict(lens=lens_scores(X, gain, WU, rms), lens_centred=lens_scores(Xc, gain, WU, rms), pca16=pooled_word_scores(parts, torch.ones(N, K, device=DEV), gain, WU, rms),
             native16=pooled_word_scores(words, cof, gain, WU, rms), rotated16=pooled_word_scores(wordsr, cofr, gain, WU, rms))
    for rd, sc in S.items(): res["readers"][f"{rd}@{L}"] = dict(pairwise=(sc[ar, at] > sc[ar, ot]).float().mean().item())
    # sign and provenance of the word carrying each name (largest absolute signed logit for that name)
    wl_a = torch.stack([lens_scores(words[:, i] * cof[:, i:i + 1], gain, WU, rms)[ar, at] for i in range(K)], 1)   # [N, K]
    wl_o = torch.stack([lens_scores(words[:, i] * cof[:, i:i + 1], gain, WU, rms)[ar, ot] for i in range(K)], 1)
    ia, io = wl_a.abs().argmax(1), wl_o.abs().argmax(1); sa, so = wl_a[ar, ia], wl_o[ar, io]
    ba, bo = blk[sel[ar, ia]].float(), blk[sel[ar, io]].float()
    res["signs"][str(L)] = dict(answer_on_suppressing=(sa < 0).float().mean().item(), other_on_suppressing=(so < 0).float().mean().item(),
                                answer_block=ba.mean().item(), other_block=bo.mean().item(), same_word=(ia == io).float().mean().item())
    r = lambda rd: res["readers"][f"{rd}@{L}"]["pairwise"]
    log(f"L{L}: pairwise lens {r('lens'):.2f} centred {r('lens_centred'):.2f} pca {r('pca16'):.2f} native {r('native16'):.2f} rotated {r('rotated16'):.2f} | "
        f"on a suppressing word: answer {res['signs'][str(L)]['answer_on_suppressing']:.2f} other {res['signs'][str(L)]['other_on_suppressing']:.2f} | "
        f"block of carrying word: answer {ba.mean():.1f} other {bo.mean():.1f} | same word {res['signs'][str(L)]['same_word']:.2f}")
    del sel, selr, words, wordsr, parts, S
bl = lambda rd: max(res["readers"][f"{rd}@{L}"]["pairwise"] for L in layers)
first_ok = lambda rd: next((L for L in layers if res["readers"][f"{rd}@{L}"]["pairwise"] >= 0.8), None)
summ = (f"{name} n={N}: best-layer pairwise lens {bl('lens'):.2f} centred {bl('lens_centred'):.2f} pca16 {bl('pca16'):.2f} native16 {bl('native16'):.2f} rotated16 {bl('rotated16'):.2f} | "
        f"first layer at 0.8: lens {first_ok('lens')} native {first_ok('native16')} pca {first_ok('pca16')}")
log(summ); record(f"e417_roles_{name}", res, summ)
