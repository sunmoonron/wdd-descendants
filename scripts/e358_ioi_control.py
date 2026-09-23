"""e358: positive control on a known circuit, indirect object identification (IOI). Prompts from four templates in
both name orders (ABBA, BABA), single-token names, places and objects, metric = logit(IO) - logit(S) at the last
position, heads mean-ablated with per-template, per-position means. The same statistics as e357: the 16 heads with
the largest absolute effect and 16 random others, pairwise interaction matrix (signed: positive = super-additive, the
backup-head signature), spectral clusters, and for GPT-2 small the adjusted Rand index against the published IOI head
classes (name movers, backup and negative name movers, S-inhibition, induction, duplicate-token, previous-token);
joint-ablation non-additivity of the top-8 vs random 8-sets; the same heads' interactions on natural text (blind-spot
test); and the top-16 neurons by attribution on IOI."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam); NB, NH = arch.NB, arch.NH; rng = random.Random(0)
KT = int(os.environ.get("WDD_KTOP", "12")); groups, n_names = ioi_groups(tok, n_per=int(os.environ.get("WDD_NIOI", "16"))); MG = [capture_means(model, arch, g[0]) for g in groups]
def ld(lg, io, s): return lg[:, -1].gather(1, io[:, None])[:, 0] - lg[:, -1].gather(1, s[:, None])[:, 0]
base = torch.cat([ld(mg[2], g[1], g[2]) for g, mg in zip(groups, MG)])
def E(heads=(), neurons=()): return torch.cat([ld(ablate(model, arch, g[0], heads=heads, neurons=neurons, MA=mg[0], MM=mg[1]), g[1], g[2]) for g, mg in zip(groups, MG)]) - base
KNOWN = {"name_mover": [(9, 9), (10, 0), (9, 6)], "backup_name_mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)], "negative_name_mover": [(10, 7), (11, 10)], "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)], "induction": [(5, 5), (5, 8), (5, 9), (6, 9)], "duplicate_token": [(0, 1), (0, 10), (3, 0)], "previous_token": [(2, 2), (4, 11)]}
known = {x: k for k, v in KNOWN.items() for x in v} if tag == "gpt2" else {}
heads = [(l, h) for l in range(NB) for h in range(NH)]; E1 = {x: E(heads=[x]) for x in heads}; eff = torch.tensor([E1[x].mean().item() for x in heads]); order = eff.abs().argsort(descending=True).tolist(); top = [heads[i] for i in order[:KT]]; rest = [heads[i] for i in order[KT:]]; U = top + rng.sample(rest, KT)
def pairs(U, Efn, E1):
    n = len(U); I = torch.zeros(n, n); Sg = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            Eij = Efn(U[i], U[j]); lin = E1[U[i]] + E1[U[j]]; I[i, j] = I[j, i] = ((Eij - lin).norm() / (0.5 * (E1[U[i]].norm() + E1[U[j]].norm())).clamp_min(1e-9)).item(); Sg[i, j] = Sg[j, i] = ((Eij.mean() - lin.mean()) / (0.5 * (E1[U[i]].mean().abs() + E1[U[j]].mean().abs())).clamp_min(1e-9)).item()
    return I, Sg
I, Sg = pairs(U, lambda a, b: E(heads=[a, b]), E1); st = block_stats(I, Sg, KT); cl, k, _ = spectral(I[:KT, :KT].abs()); labs = [known.get(x, "other") for x in U]; st.update(clusters=cl, k=k, ktop=KT, ari_known=ari(cl, labs[:KT]) if known else None)
def na(members, Efn, E1): ES = Efn(members); lin = sum(E1[m] for m in members); return dict(nonadditivity=((ES - lin).norm() / sum(E1[m].norm() for m in members)).item(), joint_over_sum=(ES.mean() / lin.mean()).item(), joint_effect=ES.mean().item())
na_top = na(top[:8], lambda m: E(heads=m), E1); na_r = [na(rng.sample(rest, 8), lambda m: E(heads=m), E1) for _ in range(4)]
res = dict(model=tag, n_names=n_names, n_prompts=int(base.numel()), baseline_logit_diff=base.mean().item(), baseline_accuracy=(base > 0).float().mean().item(), heads=dict(stats=st, top=[list(x) for x in top], top_effects=[E1[x].mean().item() for x in top], labels_top=labs[:KT], known_in_top16=int(sum(l != "other" for l in labs[:KT])) if known else None, na_top8=na_top, na_random8=na_r))
nat = c.s["eval_ids"][:4].to(DEV); MAn, MMn, lgn = capture_means(model, arch, nat); bn = token_loss(lgn, nat); del lgn
En = lambda hs: token_loss(ablate(model, arch, nat, heads=hs, MA=MAn, MM=MMn), nat) - bn; E1n = {x: En([x]) for x in U}; In, Sgn = pairs(U, lambda a, b: En([a, b]), E1n); stn = block_stats(In, Sgn, KT); cln, kn, _ = spectral(In[:KT, :KT].abs()); stn.update(ari_known=ari(cln, labs[:KT]) if known else None, ari_with_task_clusters=ari(cln, cl))
res["same_heads_on_natural"] = dict(stats=stn, na_top8=na(top[:8], lambda m: En(m), E1n))
sc = None
for g, mg in zip(groups, MG):
    s_ = neuron_attr(model, arch, g[0], lambda lg, g=g: ld(lg, g[1], g[2]), mg[1]); sc = s_ if sc is None else {l: sc[l] + s_[l] for l in sc}
DFF = sc[0].numel(); allsc = torch.cat([sc[l] for l in range(NB)]); o = allsc.abs().argsort(descending=True)[:KT].tolist(); topN = [(i // DFF, i % DFF) for i in o]; randN = []
while len(randN) < KT:
    x = (rng.randrange(NB), rng.randrange(DFF))
    if x not in topN and x not in randN: randN.append(x)
UN = topN + randN; E1N = {x: E(neurons=[x]) for x in UN}; IN, SgN = pairs(UN, lambda a, b: E(neurons=[a, b]), E1N); stN = block_stats(IN, SgN, KT)
res["neurons"] = dict(stats=stN, top=[list(x) for x in topN], top_effects=[E1N[x].mean().item() for x in topN], na_top8=na(topN[:8], lambda m: E(neurons=m), E1N), na_random8=na(randN[:8], lambda m: E(neurons=m), E1N))
h, hn, nN = res["heads"], res["same_heads_on_natural"], res["neurons"]; nr = sum(x["nonadditivity"] for x in h["na_random8"]) / 4
log(f"{tag}: IOI baseline logit diff {res['baseline_logit_diff']:.2f}, accuracy {res['baseline_accuracy']:.2f} over {res['n_prompts']} prompts ({n_names} names) | HEADS: interaction top {h['stats']['I_top']:.2f} vs random {h['stats']['I_random']:.2f} vs cross {h['stats']['I_cross']:.2f}; signed top {h['stats']['signed_top']:+.2f} (super-additive {h['stats']['superadditive_fraction_top']:.2f}, sub-additive {h['stats']['subadditive_fraction_top']:.2f}); clusters k {h['stats']['k']}" + (f", ARI with the published IOI classes {h['stats']['ari_known']:.2f}, known heads in top-16 {h['known_in_top16']}" if known else "") + f"; non-additivity top-8 {h['na_top8']['nonadditivity']:.2f} (joint/sum {h['na_top8']['joint_over_sum']:.2f}) vs random-8 {nr:.2f} | SAME HEADS ON NATURAL TEXT: interaction top {hn['stats']['I_top']:.2f} vs random {hn['stats']['I_random']:.2f}" + (f", ARI with published classes {hn['stats']['ari_known']:.2f}" if known else "") + f", ARI with the IOI clusters {hn['stats']['ari_with_task_clusters']:.2f} | NEURONS: interaction top {nN['stats']['I_top']:.2f} vs random {nN['stats']['I_random']:.2f}; non-additivity {nN['na_top8']['nonadditivity']:.2f} vs {nN['na_random8']['nonadditivity']:.2f}")
record(f"e358_ioictrl_{tag}", res, f"LD {res['baseline_logit_diff']:.2f} acc {res['baseline_accuracy']:.2f} | heads I top/random {h['stats']['I_top']:.2f}/{h['stats']['I_random']:.2f} signed {h['stats']['signed_top']:+.2f}" + (f" ARI known {h['stats']['ari_known']:.2f} known-in-top16 {h['known_in_top16']}" if known else "") + f" NA {h['na_top8']['nonadditivity']:.2f}/{nr:.2f} | natural I {hn['stats']['I_top']:.2f}/{hn['stats']['I_random']:.2f} | neurons I {nN['stats']['I_top']:.2f}/{nN['stats']['I_random']:.2f}")
