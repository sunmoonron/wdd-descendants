"""e573b (session 103): the class-or-row question within one model, without training-time change. In the final
Pythia-410m, pairs of words whose over-the-floor context sets overlap (Jaccard 0.25 or more; the near-duplicate
families of session 96) share a class by construction. For every such pair: the cosine between the two rows'
directions, and the cosine between the effect vectors of removing each direction at the union of the two sets (the
logit change through blocks 13-23), against pairs of words with disjoint sets. If the effect follows the class, the
effect cosine of class-sharing pairs is high whatever their direction cosine; if the effect is the direction's, the
effect cosine tracks the direction cosine. Also e573's changed-row twins stratified by their Jaccard. Pre-registered:
 Q3 (0.55) among class-sharing pairs the effect cosine tracks the direction cosine (correlation over 0.5), and
    pairs with direction cosine under 0.3 have effect cosine under 0.2: the direction carries the function."""
from s101_common import *
t0 = time.time(); B = 12; ids = pile_ids("pythia410"); JT = 0.25
mF, _, fam = load_model("pythia410"); SF = lm_states("pythia410", B=B, ids=ids, model=mF); keep = SF["keep"]; kidx = torch.nonzero(keep)[:, 0]
st = stats(SF["U"], SF["A"], K); w = torch.nonzero(wordset(st["usage"]))[:, 0]; C = st["ratio"][:, w].float() > 1
def jac(P, Q):
    P = P.float(); Q = Q.float(); inter = P.T @ Q; return inter / (P.sum(0)[:, None] + Q.sum(0)[None] - inter).clamp_min(1)
J = jac(C, C); J.fill_diagonal_(0); pairs = [(i, j, float(J[i, j])) for i in range(w.numel()) for j in range(i + 1, w.numel()) if J[i, j] >= JT]
g = torch.Generator().manual_seed(0); disj = [(i, j, 0.0) for i, j in zip(torch.randint(0, w.numel(), (300,), generator=g).tolist(), torch.randint(0, w.numel(), (300,), generator=g).tolist()) if i != j and J[i, j] == 0][:120]
log(f"class-sharing pairs {len(pairs)} (of {w.numel() * (w.numel() - 1) // 2}); disjoint control pairs {len(disj)}")
def effect(d, posmask):
    arch = SF["arch"]; pm = torch.zeros(ids.shape, dtype=torch.bool, device=DEV); pm[:, 1:] = posmask.reshape(ids.shape[0], -1).to(DEV); out = {}
    def hk(m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        if hk.on: y = x.clone(); y = torch.where(pm[..., None], y - (y @ d)[..., None] * d[None, None], y); return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    for on in (False, True):
        hk.on = on; h = arch.layers[B].register_forward_hook(hk)
        try:
            with torch.no_grad(): out[on] = mF(ids).logits.float()[pm]
        finally: h.remove()
    return (out[True] - out[False]).mean(0)
def run(P, tag):
    dcos, ecos, jj = [], [], []
    for i, j, jv in P:
        pos = C[:, i] | C[:, j]; mask = torch.zeros(keep.numel(), dtype=torch.bool); mask[kidx[pos]] = True
        e1, e2 = effect(SF["A"][w[i]], mask), effect(SF["A"][w[j]], mask); ecos.append(float(torch.nn.functional.cosine_similarity(e1, e2, dim=0))); dcos.append(float((SF["A"][w[i]] * SF["A"][w[j]]).sum())); jj.append(jv)
    r = dict(n=len(P), direction_cos=med(dcos), effect_cos=med(ecos), corr_direction_effect=float(torch.corrcoef(torch.stack([torch.tensor(dcos), torch.tensor(ecos)]))[0, 1]) if len(P) > 3 else None,
             effect_cos_low_direction=med([e for d_, e in zip(dcos, ecos) if d_ < 0.3]), effect_cos_high_direction=med([e for d_, e in zip(dcos, ecos) if d_ >= 0.5]), n_low=sum(d_ < 0.3 for d_ in dcos), n_high=sum(d_ >= 0.5 for d_ in dcos), pairs=list(zip(dcos, ecos, jj)))
    log(f"{tag} ({len(P)}): direction cosine {r['direction_cos']:.3f}, effect cosine {r['effect_cos']:.3f}, correlation {r['corr_direction_effect']}; effect cosine when the directions differ (cos < 0.3, n {r['n_low']}) {r['effect_cos_low_direction']}, when they agree (>= 0.5, n {r['n_high']}) {r['effect_cos_high_direction']}"); return r
res = dict(sharing=run(pairs[:150], "class-sharing pairs"), disjoint=run(disj, "disjoint pairs"))
# e573's changed-row twins by Jaccard
try:
    prev = json.load(open("/workspace/wdd/results/e573_class_or_row.json")); res["e573_note"] = "stratification needs the pair list; see e573's JSON"
except Exception: pass
s_, d_ = res["sharing"], res["disjoint"]
summ = (f"within the final Pythia-410m, {s_['n']} pairs of words sharing a context class (Jaccard >= 0.25): direction cosine {s_['direction_cos']:.3f}, effect cosine {s_['effect_cos']:.3f} (disjoint pairs {d_['direction_cos']:.3f} / {d_['effect_cos']:.3f}); correlation of effect with direction cosine {s_['corr_direction_effect']:.2f}; effect cosine {s_['effect_cos_low_direction']} for pairs whose directions differ (cos < 0.3, n {s_['n_low']}) and {s_['effect_cos_high_direction']} for pairs whose directions agree (>= 0.5, n {s_['n_high']}) | {time.time() - t0:.0f}s")
log(summ); record("e573b_synonyms", res, summ)
