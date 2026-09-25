"""e477: are attribute handles shared across words? e473 found that, in Qwen, four native words per block chosen for
one kinship word's gender difference (king -> queen) change only that word's gender under interchange. Those words were
found item by item. This run asks whether the vocabulary has a gender handle that is the same across words: does the
handle found on other quadruples (father/mother, man/woman, ...) move this one?
Setup: e473's items (Qwen, 128 items from 32 quadruple-contexts), interchange at the word's last token over blocks 0 to
the middle, recomputed at each block. Bases at each block, k = 4 and 16 words:
- own: the item's own difference (e473);
- shared, mean: OMP on the mean gender (or generation) difference of the items of the other quadruples, in any language;
- shared, vote: the k native words most often chosen by the other quadruples' items' own descriptions;
- cross-lingual, mean: the other quadruples in the other two languages only;
- rotated shared, mean.
(v2: the group means are sign-aligned, male to female and elder to younger; v1 let half the items cancel the mean.)
Interventions as in e473: the gender basis set toward the doubly flipped word (expected: only the gender changes), the
generation basis likewise, and compose (gender from the gender partner, then generation from the generation partner;
expected: both change).
Also reported: how often the single most frequent gender word appears in the items' own top-16 supports, per block,
and the types of the most voted gender words.
Pre-registered (honest guesses):
- the shared mean basis at k = 4 changes only the gender in at least half the items (0.5);
- the vote basis does worse than the mean basis at k = 4 (0.6);
- the cross-lingual basis does at least 80% as well as the shared one (0.5);
- one gender word recurs in at least half the items' own supports at some block (0.4)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e473_word_algebra.py")).read()
exec(src[src.index("QUADS = {"): src.index("dif = {")], globals())          # items, model, arch, L, Au, Ar, ends, run, scores, ...
KS = [4, 16]; BL = list(range(L + 1)); KMAX = 16
q_of = [it["q"] for it in items]; lg_of = [it["lg"] for it in items]
dif = {j: torch.stack([it["X"][j] - it["X"][0] for it in items], 1) for j in (1, 2)}          # [L+1, n, D]: 1 = gender, 2 = generation
# sign-aligned copies for the means: an item whose base word already has the attribute flipped (i & j) has a difference
# pointing the other way (queen -> king against king -> queen), so the group mean would cancel (v1's bug)
sgn = {j: torch.tensor([-1.0 if (it["i"] & j) else 1.0 for it in items], device=DEV) for j in (1, 2)}
dif_al = {j: dif[j] * sgn[j][None, :, None] for j in (1, 2)}
def qr(M): return torch.linalg.qr(M)[0]
typ = lab["type"].to(DEV)
own, ownsel = {}, {}
for b in BL:
    for j, nm in ((1, "G"), (2, "A")):
        for kind, Dct in (("native", Au), ("rotated", Ar)):
            sel, _, _ = omp(dif[j][b], Dct[:ends[b]], KMAX, batch=256, record_err=False)
            for k in KS: own[(kind, nm, k, b)] = qr(Dct[sel[:, :k]].transpose(1, 2))
            if kind == "native": ownsel[(nm, b)] = sel
keys = sorted(set(zip(q_of, lg_of)))
def group(q, lg, xling):
    return [i for i in range(n) if q_of[i] != q and (not xling or lg_of[i] != lg)]
shared, votes = {}, {}
for (q, lg) in keys:
    for xling in (False, True):
        g = group(q, lg, xling)
        for b in BL:
            for j, nm in ((1, "G"), (2, "A")):
                m_ = dif_al[j][b][g].mean(0, keepdim=True)
                for kind, Dct in (("native", Au), ("rotated", Ar)):
                    if xling and kind == "rotated": continue
                    sel, _, _ = omp(m_, Dct[:ends[b]], KMAX, batch=256, record_err=False)
                    for k in KS: shared[(kind, "xling" if xling else "mean", nm, k, b, q, lg)] = qr(Dct[sel[0, :k]].T)
                if not xling:
                    cnt = torch.bincount(ownsel[(nm, b)][g].reshape(-1), minlength=Au.shape[0]); order = cnt.argsort(descending=True)
                    for k in KS: shared[("native", "vote", nm, k, b, q, lg)] = qr(Au[order[:k]].T)
                    votes[(nm, b, q, lg)] = order[:KMAX]
def basis(kind, how, nm, k, b, i):
    if how == "own": return own[(kind, nm, k, b)][i]
    return shared[(kind, how, nm, k, b, q_of[i], lg_of[i])]
def proj(Q, x, xs): return x + Q @ (Q.T @ (xs - x))
def evaluate(kind, how, nm, k):
    outcome = torch.zeros(4)
    for i, it in enumerate(items):
        X = it["X"]
        if nm == "compose": fns = {b: (lambda x, b=b, QG=basis(kind, how, "G", k, b, i), QA=basis(kind, how, "A", k, b, i): proj(QA, proj(QG, x, X[1][b]), X[2][b])) for b in BL}
        else: fns = {b: (lambda x, b=b, Q=basis(kind, how, nm, k, b, i): proj(Q, x, X[3][b])) for b in BL}
        lp, _ = run(it["ids"], it["p"], it["cand"], fns); outcome[int(lp.argmax()) ^ it["i"]] += 1
    return (outcome / n).tolist()                                          # kept / gender flipped / generation flipped / both
PLAN = [("native", "own", ["G", "A", "compose"]), ("native", "mean", ["G", "A", "compose"]), ("native", "vote", ["G", "A", "compose"]), ("native", "xling", ["G", "A"]), ("rotated", "mean", ["G", "A"])]
res = dict(model=name, level=L, n_items=n, n_groups=groups, results={}, recurrence={})
for k in KS:
    for kind, how, nms in PLAN:
        for nm in nms: res["results"][f"{kind}_{how}_{nm}_{k}"] = evaluate(kind, how, nm, k)
    log(f"{name} k={k}: " + " | ".join(f"{kind} {how}: " + ", ".join(f"{nm} {'/'.join(f'{v:.2f}' for v in res['results'][f'{kind}_{how}_{nm}_{k}'])}" for nm in nms) for kind, how, nms in PLAN))
# recurrence of gender words across items: the most common atom in the items' own top-16 gender supports, per block
rec_share, rec_type = [], []
for b in BL:
    S_ = ownsel[("G", b)]; cnt = torch.bincount(S_.reshape(-1), minlength=Au.shape[0]); a0 = int(cnt.argmax())
    rec_share.append(float((S_ == a0).any(1).float().mean())); rec_type.append(int(typ[a0]))
res["recurrence"] = dict(most_common_gender_word_share_by_block=rec_share, its_type_by_block=rec_type, type_codes=dict(tok=int(T_TOK), pos=int(T_POS), mlp=int(T_MLP), att=int(T_ATT)))
R = res["results"]; o = lambda key, j: R[key][j]
res["checks"] = dict(shared_mean_gender_half=o("native_mean_G_4", 1) >= 0.5, vote_worse_than_mean=o("native_vote_G_4", 1) < o("native_mean_G_4", 1),
                     xling_80pct=o("native_xling_G_4", 1) >= 0.8 * o("native_mean_G_4", 1), recurring_word_half=max(rec_share) >= 0.5)
fmt = lambda key: "/".join(f"{v:.2f}" for v in R[key])
summ = (f"{name} L{L}, {n} items; outcome shares kept/gender/generation/both || " + " || ".join(f"k={k}: own G {fmt(f'native_own_G_{k}')}, A {fmt(f'native_own_A_{k}')}, compose {fmt(f'native_own_compose_{k}')} | shared mean G {fmt(f'native_mean_G_{k}')}, A {fmt(f'native_mean_A_{k}')}, compose {fmt(f'native_mean_compose_{k}')} | "
        f"shared vote G {fmt(f'native_vote_G_{k}')}, A {fmt(f'native_vote_A_{k}')}, compose {fmt(f'native_vote_compose_{k}')} | cross-lingual mean G {fmt(f'native_xling_G_{k}')}, A {fmt(f'native_xling_A_{k}')} | rotated shared mean G {fmt(f'rotated_mean_G_{k}')}, A {fmt(f'rotated_mean_A_{k}')}" for k in KS)
        + f" || most common gender word's share of items' own supports by block: " + "/".join(f"{v:.2f}" for v in rec_share) + f" (types {rec_type}) | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e477_sharedhandles_{name}", res, summ)
