"""e520: does the agreement of the parts predict wordhood beyond the size of their sum? e519b found that no part of
the state supplies the projection that predicts a row's entry into the vocabulary, and that the parts cancel along
most atoms while the words are the atoms along which they add. That is a claim about how a projection is assembled,
not how large it is, and it has a direct test. At a row's best position the real projection P splits into the
parts' projections p_k (embedding, attention, own write, other largest writes, crowd), recorded by e519 as shares
s_k = p_k / P. From the shares alone: the agreement index a = (1 - sum_k s_k^2) / 2, which is the pairwise cross term
sum_{k<l} p_k p_l over P^2 (a third for three equal positive parts, large and negative when large parts cancel); the
constructive fraction q, the share of the pairwise products that are positive; the sign consensus c, the balance
of the parts' signs; and the largest single share. The size predictor is S, the row's largest projection over the
floor (e516b's). Per origin checkpoint, among the rows that are not words: the AUC of each quantity for entry at a
later checkpoint; the AUC of the agreement quantities within deciles of S (does agreement separate the entrants
from the non-entrants at the same size?), and of S within deciles of a; and the control the relayed take asked
for: among rows whose S lies within 0.9-1.1 of the floor, the entry rate in the top and bottom quartiles of a.
Pre-registered (honest guesses), block 12, entry at the next checkpoint, origins 2000-16000:
- the agreement index alone predicts entry at 0.7 or above (0.5);
- within deciles of S the agreement index separates entrants from non-entrants at 0.6 or above (0.5);
- the graded index adds more than the sign consensus (0.5);
- among rows near the floor, the top agreement quartile's entry rate is at least twice the bottom's (0.5).
Arguments: name."""
import sys, os, json as _json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; CDIR = f"/workspace/wdd/cache/e519_{name}"; STEPS = ["step0", "step64", "step256", "step512", "step1000", "step2000", "step3000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
LAB = [0, 64, 256, 512, 1000, 2000, 3000, 4000, 8000, 16000, 32000, 64000, 143000]; NWORD = 256; ORIGINS = [512, 1000, 2000, 3000, 4000, 8000, 16000, 32000]; PARTS = ["embedding", "attention", "own", "chord_others", "crowd"]
assert all(os.path.exists(f"{CDIR}/{s}.pt") for s in STEPS), "e519 records missing"
data = {s: torch.load(f"{CDIR}/{s}.pt") for s in STEPS}
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def strat_auc(q, strat, ent_mask, nbins=10):
    """AUC of q for entrants within deciles of strat, weighted by the entrants in each decile"""
    edges = strat.quantile(torch.linspace(0, 1, nbins + 1)); tot, wsum = 0.0, 0
    for i in range(nbins):
        m = (strat >= edges[i]) & (strat <= edges[i + 1]) if i == nbins - 1 else (strat >= edges[i]) & (strat < edges[i + 1])
        a = auc(q[m & ent_mask], q[m & ~ent_mask])
        if a is not None: n = int((m & ent_mask).sum()); tot += a * n; wsum += n
    return (tot / wsum) if wsum else None
def agreement(d):
    sh = torch.stack([d["share_at_max"][k].float() for k in PARTS], 1); S = d["max_over_floor"]["real"].float()
    a = (1 - sh.pow(2).sum(1)) / 2; K = sh.shape[1]
    prod = torch.stack([sh[:, k] * sh[:, l] for k in range(K) for l in range(k + 1, K)], 1); q = (prod.clamp_min(0)).sum(1) / prod.abs().sum(1).clamp_min(1e-12)
    nz = (sh.abs() > 1e-6).float(); c = (torch.sign(sh) * nz).sum(1).abs() / nz.sum(1).clamp_min(1); top = sh.abs().max(1).values
    return dict(S=S, agreement=a, constructive=q, consensus=c, largest_share=top, S_times_agreement=S * (1 + a.clamp(-0.9, 1)))
res = dict(model=name, steps=LAB, origins=ORIGINS, by_block={})
for b in data["main"]:
    W = {}
    for s in STEPS:
        u = data[s][b]["usage"]; used = torch.nonzero(u > 0)[:, 0]; W[s] = set(used[u[used].argsort(descending=True)[:NWORD]].tolist())
    R = data["main"][b]["usage"].numel(); out = {}
    for o_ in ORIGINS:
        so = STEPS[LAB.index(o_)]; d = data[so][b]; inW = torch.zeros(R, dtype=torch.bool); inW[list(W[so])] = True; cand = torch.nonzero(~inW)[:, 0]; Q = {k: v[cand] for k, v in agreement(d).items()}
        out[o_] = {}
        for j in range(LAB.index(o_) + 1, len(STEPS)):
            s1 = STEPS[j]; in1 = torch.zeros(R, dtype=torch.bool); in1[list(W[s1])] = True; ent = in1[cand]
            e = dict(n_entrants=int(ent.sum()), auc={k: auc(v[ent], v[~ent]) for k, v in Q.items()}, within_S_deciles={k: strat_auc(Q[k], Q["S"], ent) for k in ("agreement", "constructive", "consensus", "largest_share")}, S_within_agreement_deciles=strat_auc(Q["S"], Q["agreement"], ent))
            near = (Q["S"] >= 0.9) & (Q["S"] <= 1.1)
            if int(near.sum()) >= 40:
                a_near = Q["agreement"][near]; e_near = ent[near]; lo, hi = a_near.quantile(0.25), a_near.quantile(0.75)
                e["near_floor"] = dict(n=int(near.sum()), entry_rate_top_agreement_quartile=float(e_near[a_near >= hi].float().mean()), entry_rate_bottom_agreement_quartile=float(e_near[a_near <= lo].float().mean()), entry_rate_all=float(e_near.float().mean()), median_agreement_entrants=float(a_near[e_near].median()) if int(e_near.sum()) else None, median_agreement_non_entrants=float(a_near[~e_near].median()))
            out[o_][LAB[j]] = e
    res["by_block"][b] = out; nx = lambda o_: LAB[LAB.index(o_) + 1]; fm = lambda x: "n/a" if x is None else f"{x:.2f}"
    log(f"{name} block {b}: entry at the next checkpoint, AUC by quantity: " + " | ".join(f"from {o_}: " + ", ".join(f"{k} {fm(out[o_][nx(o_)]['auc'][k])}" for k in ("S", "agreement", "constructive", "consensus", "largest_share", "S_times_agreement")) for o_ in ORIGINS))
    log(f"{name} block {b}: within deciles of S (agreement beyond size), AUC: " + " | ".join(f"from {o_}: " + ", ".join(f"{k} {fm(out[o_][nx(o_)]['within_S_deciles'][k])}" for k in ("agreement", "constructive", "consensus", "largest_share")) + f"; S within agreement deciles {fm(out[o_][nx(o_)]['S_within_agreement_deciles'])}" for o_ in ORIGINS))
    def nf_line(o_):
        nf = out[o_][nx(o_)].get("near_floor")
        return "n/a" if not nf else f"{nf['entry_rate_top_agreement_quartile']:.3f}/{nf['entry_rate_bottom_agreement_quartile']:.3f} ({nf['entry_rate_all']:.3f}, n {nf['n']}), agreement entrants {fm(nf['median_agreement_entrants'])} vs non-entrants {fm(nf['median_agreement_non_entrants'])}"
    log(f"{name} block {b}: rows near the floor (S in 0.9-1.1), entry rate by agreement quartile top/bottom (all): " + " | ".join(f"from {o_}: " + nf_line(o_) for o_ in ORIGINS))
B = res["by_block"][12 if 12 in res["by_block"] else list(res["by_block"])[-1]]; nx = lambda o_: LAB[LAB.index(o_) + 1]; g = lambda o_, k: (B[o_][nx(o_)]["auc"][k] or 0.5); w = lambda o_, k: (B[o_][nx(o_)]["within_S_deciles"][k] or 0.5)
nf_ratio = lambda o_: ((B[o_][nx(o_)].get("near_floor") or {}).get("entry_rate_top_agreement_quartile", 0) / max((B[o_][nx(o_)].get("near_floor") or {}).get("entry_rate_bottom_agreement_quartile", 1e-9), 1e-9)) if B[o_][nx(o_)].get("near_floor") else None
res["checks"] = dict(agreement_alone_over_0_7=all(g(o_, "agreement") >= 0.7 for o_ in (2000, 3000, 4000, 8000, 16000)), agreement_within_S_over_0_6=all(w(o_, "agreement") >= 0.6 for o_ in (2000, 3000, 4000, 8000, 16000)),
                     graded_beats_sign=all(w(o_, "agreement") > w(o_, "consensus") for o_ in (2000, 3000, 4000, 8000, 16000)), near_floor_top_twice_bottom=all(nf_ratio(o_) is not None and nf_ratio(o_) >= 2 for o_ in (2000, 3000, 4000, 8000, 16000)))
fm2 = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = (f"{name}: " + " || ".join(f"block {b}: entry at the next checkpoint, AUC S/agreement/constructive/consensus " + " | ".join(f"from {o_}: " + "/".join(fm2(o[o_][nx(o_)]["auc"][k]) for k in ("S", "agreement", "constructive", "consensus")) + ", agreement within S deciles " + fm2(o[o_][nx(o_)]["within_S_deciles"]["agreement"]) for o_ in ORIGINS) for b, o in res["by_block"].items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e520_agreement_{name}", res, summ)
