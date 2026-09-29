"""e573 (session 103): when a context class changes its row, does the computation stay? Between step 16000 and the end
of Pythia-410m's training the words keep Jaccard 0.21 by row index while their context sets twin at 0.55 (e557b): the
same classes, spoken by other rows. Two questions on those pairs. (1) Does the direction persist? For each twin pair
(a 16000 word and the final word whose over-the-floor set best matches it, Jaccard >= 0.25) the cosine between the
16000 row's direction and the final row's, for pairs that changed rows and pairs that kept them, against random pairs of
words. (2) Does the function persist? The effect of the direction at the class's positions: the block-12 state at the
union of the two context sets has its component along the direction removed, blocks 13-23 run on, and the change in
the logits (the effect vector, averaged over the positions; its size as the KL) is measured at each checkpoint with
each checkpoint's own row; the cosine between the two effect vectors, for changed-row pairs, kept-row pairs and random
pairs. Pre-registered (probabilities are honest guesses):
 Q1 (0.6) changed-row twins have directions at cosine 0.5 or more: the class's direction persists and a new row is
    recruited to it;
 Q2 (0.55) the effect vectors of changed-row pairs correlate as well as the kept-row pairs' (within 0.1): the
    computation follows the class, the row is its implementation."""
from s101_common import *
t0 = time.time(); B = 12; ids = pile_ids("pythia410"); JT = 0.25
def jac(P, Q):
    P = P.float(); Q = Q.float(); inter = P.T @ Q; return inter / (P.sum(0)[:, None] + Q.sum(0)[None] - inter).clamp_min(1)
def load(rev):
    m, _, fam = load_model("pythia410", revision=rev); S = lm_states("pythia410", B=B, ids=ids, model=m); return m, S
m16, S16 = load("step16000"); mF, SF = load(None); keepc = S16["keep"] & SF["keep"]; kidx = torch.nonzero(keepc)[:, 0]
for S_ in (S16, SF): S_["U"] = unitr(S_["X"][keepc] - S_["X"][keepc].mean(0))
def words_and_sets(S):
    st = stats(S["U"], S["A"], K); w = torch.nonzero(wordset(st["usage"]))[:, 0]; over = st["ratio"][:, w].float() > 1; return w, over
w16, C16 = words_and_sets(S16); wF, CF = words_and_sets(SF); J = jac(C16, CF); best, arg = J.max(1)
pairs = [(int(w16[i]), int(wF[arg[i]]), int(i), int(arg[i])) for i in range(w16.numel()) if best[i] >= JT]
changed = [p for p in pairs if p[0] != p[1]]; kept = [p for p in pairs if p[0] == p[1]]; g = torch.Generator().manual_seed(0)
rand = [(int(w16[i]), int(wF[j]), int(i), int(j)) for i, j in zip(torch.randint(0, w16.numel(), (200,), generator=g).tolist(), torch.randint(0, wF.numel(), (200,), generator=g).tolist()) if best[i] < JT or int(wF[j]) != int(wF[arg[i]])]
log(f"twin pairs {len(pairs)} of {w16.numel()} (changed row {len(changed)}, kept {len(kept)}); random pairs {len(rand)}")
dcos = lambda P: [float((S16["A"][a] * SF["A"][b]).sum()) for a, b, _, _ in P]
res = dict(n_pairs=len(pairs), n_changed=len(changed), n_kept=len(kept), direction_cos=dict(changed=med(dcos(changed)), kept=med(dcos(kept)), random=med(dcos(rand)), changed_mean=mean(dcos(changed)), random_mean=mean(dcos(rand)), changed_share_over_05=mean([float(c >= 0.5) for c in dcos(changed)]), random_share_over_05=mean([float(c >= 0.5) for c in dcos(rand)])))
dc = res["direction_cos"]; log(f"direction cosine 16000 row vs final row: changed-row twins {dc['changed']:.3f} (mean {dc['changed_mean']:.3f}, share >= 0.5 {dc['changed_share_over_05']:.2f}), kept-row twins {dc['kept']:.3f}, random pairs {dc['random']:.3f} (share >= 0.5 {dc['random_share_over_05']:.2f})")
# the direction's effect on the logits at the class's positions
def effect(model, S, d, posmask):
    """logits at the masked positions with and without the component along d at the block-B output; returns the mean logit change and the mean KL"""
    arch = S["arch"]; pm = torch.zeros(ids.shape[0], ids.shape[1], dtype=torch.bool, device=DEV); pm[:, 1:] = posmask.reshape(ids.shape[0], -1); out = {}
    def hk(m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        if hk.on: y = x.clone(); comp = (y @ d)[..., None] * d[None, None]; y = torch.where(pm[..., None], y - comp, y); return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    for on in (False, True):
        hk.on = on; h = arch.layers[B].register_forward_hook(hk)
        try:
            with torch.no_grad(): out[on] = model(ids).logits.float()[pm]
        finally: h.remove()
    p0 = out[False].log_softmax(-1); p1 = out[True].log_softmax(-1); kl = float((p0.exp() * (p0 - p1)).sum(-1).mean())
    return (out[True] - out[False]).mean(0), kl
def pair_effects(P, tag):
    cos_, kl16, klF = [], [], []
    for a, b, i, j in P:
        pos = C16[:, i] | CF[:, j]; mask = torch.zeros(keepc.numel(), dtype=torch.bool); mask[kidx[pos]] = True; mask = mask.to(DEV)
        e1, k1 = effect(m16, S16, S16["A"][a], mask); e2, k2 = effect(mF, SF, SF["A"][b], mask)
        cos_.append(float(torch.nn.functional.cosine_similarity(e1, e2, dim=0))); kl16.append(k1); klF.append(k2)
    r = dict(n=len(P), effect_cos=med(cos_), effect_cos_mean=mean(cos_), kl_16000=med(kl16), kl_final=med(klF)); log(f"{tag}: effect-vector cosine between checkpoints {r['effect_cos']:.3f} (mean {r['effect_cos_mean']:.3f}); KL of removing the direction at the class's positions {r['kl_16000']:.4f} (16000) / {r['kl_final']:.4f} (final)"); return r
res["function"] = dict(changed=pair_effects(changed[:60], "changed-row twins"), kept=pair_effects(kept[:60], "kept-row twins"), random=pair_effects(rand[:60], "random pairs"))
# the same-position control: random pairs evaluated at the changed pairs' positions would conflate; instead the kept pairs' own-row effect at 16000 against the final row's effect
fc = res["function"]
summ = (f"class or row (Pythia-410m, step 16000 vs the end; {len(pairs)} twin pairs of {w16.numel()} words, {len(changed)} changed row, {len(kept)} kept): direction cosine between the two rows {dc['changed']:.3f} for changed-row twins (share >= 0.5 {dc['changed_share_over_05']:.2f}; random pairs {dc['random']:.3f}, {dc['random_share_over_05']:.2f}); "
        f"effect-vector cosine between checkpoints (the direction removed at the class's positions): changed {fc['changed']['effect_cos']:.3f}, kept {fc['kept']['effect_cos']:.3f}, random {fc['random']['effect_cos']:.3f}; KL of the removal {fc['changed']['kl_16000']:.4f} / {fc['changed']['kl_final']:.4f} (changed), {fc['kept']['kl_16000']:.4f} / {fc['kept']['kl_final']:.4f} (kept) | {time.time() - t0:.0f}s")
log(summ); record("e573_class_or_row", res, summ)
