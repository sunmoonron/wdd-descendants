"""e231: does the descendant signature belong to the neuron's context or to its write vector? Block b = 2; neurons
dominant at >= 20 typical tokens; centroids of natural footprints (train half) at levels b+4 and L. Tests on
held-out tokens: (i) natural footprints; (ii) TRANSPLANT: the neuron's median write vector injected into foreign
tokens where the neuron is inactive (same vector, different state); (iii) the transplant at 0.5x and 2x
magnitude; (iv) a random direction of the same size (reference: share of the most-chosen centroid); (v) two
neurons' vectors injected together (top-2 recovery). Also within-neuron cosine natural vs transplanted.
Split: vector-determined signature (transplants classify like natural writes) vs state-conditioned (transplants
fail) vs operating-point-dependent (magnitude changes the identity)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; levels = sorted({b + 4, L}); run = make_runner(model, arch, c, ids_seq, levels, NT); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(b, tn)
idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = idx[split], idx[~split]; ltr, lte = lab_i[split], lab_i[~split]
med = torch.stack([tc[tr][ltr == k].median() if (ltr == k).any() else torch.tensor(1.0, device=DEV) for k in range(K)])                   # signed median coefficient per neuron
# foreign tokens: typical tokens where the assigned neuron is inactive
foreign = torch.nonzero(typ & ~torch.isin(torch.arange(NT, device=DEV), idx))[:, 0]; assign = torch.randint(0, K, (len(foreign),), device=DEV); act = (led[foreign].abs() >= 0.05 * led[foreign].abs().max(1, keepdim=True).values)
ok = ~act[torch.arange(len(foreign), device=DEV), keep[assign]]; foreign, assign = foreign[ok], assign[ok]
def injection(scale, direction=None, second=None):
    inj = torch.zeros(NT, c.D, device=DEV); v = (med[assign] * scale)[:, None] * (R[keep[assign]] if direction is None else direction)
    if second is not None: v = v + (med[second] * scale)[:, None] * R[keep[second]]
    inj[foreign] = v; return inj
runs = {"x1": run(inject=injection(1.0), inject_block=b + 1), "x0.5": run(inject=injection(0.5), inject_block=b + 1), "x2": run(inject=injection(2.0), inject_block=b + 1)}
rnd = unit(torch.randn(len(foreign), c.D, device=DEV)); runs["random"] = run(inject=injection(1.0, direction=rnd), inject_block=b + 1); second = (assign + torch.randint(1, K, assign.shape, device=DEV)) % K; runs["pair"] = run(inject=injection(1.0, second=second), inject_block=b + 1)
out = {}
for lv in levels:
    F = S0[lv] - S1[lv]; cents = centroids(F[tr], ltr, K); rec = dict(natural=accuracy(F[te], cents, lte), chance=1.0 / K, n_test=int(len(te)), n_foreign=int(len(foreign)))
    for nm in ("x1", "x0.5", "x2"): rec["transplant_" + nm] = accuracy((runs[nm][lv] - S0[lv])[foreign], cents, assign)
    Fr = runs["random"][lv] - S0[lv]; pr = (unit(Fr[foreign]) @ cents.T).argmax(1); rec["random_max_share"] = torch.bincount(pr, minlength=K).max().item() / len(foreign); rec["random_acc"] = (pr == assign).float().mean().item()
    Fp = unit((runs["pair"][lv] - S0[lv])[foreign]); top2 = (Fp @ cents.T).topk(2, dim=1).indices; rec["pair_both_in_top2"] = ((top2 == assign[:, None]).any(1) & (top2 == second[:, None]).any(1)).float().mean().item(); rec["pair_one_in_top1"] = ((top2[:, 0] == assign) | (top2[:, 0] == second)).float().mean().item()
    Ft = unit((runs["x1"][lv] - S0[lv])[foreign]); G = Ft @ Ft.T; same = assign[:, None] == assign[None, :]; eye = torch.eye(len(foreign), device=DEV, dtype=torch.bool); rec["within_cos_transplant"] = G[same & ~eye].mean().item(); rec["between_cos_transplant"] = G[~same].mean().item()
    Fn = unit(F[te]); Gn = Fn @ Fn.T; samen = lte[:, None] == lte[None, :]; eyen = torch.eye(len(te), device=DEV, dtype=torch.bool); rec["within_cos_natural"] = Gn[samen & ~eyen].mean().item(); rec["between_cos_natural"] = Gn[~samen].mean().item()
    out[lv] = rec
    log(f"{tag} level {lv} ({K} neurons, chance {1 / K:.2f}): natural {rec['natural']:.2f} | transplant into foreign states x1 {rec['transplant_x1']:.2f}, x0.5 {rec['transplant_x0.5']:.2f}, x2 {rec['transplant_x2']:.2f} | random direction: accuracy {rec['random_acc']:.2f}, max centroid share {rec['random_max_share']:.2f} | pair: both in top-2 {rec['pair_both_in_top2']:.2f}, one in top-1 {rec['pair_one_in_top1']:.2f} | within-neuron cosine natural {rec['within_cos_natural']:.2f} (between {rec['between_cos_natural']:.2f}) vs transplant {rec['within_cos_transplant']:.2f} (between {rec['between_cos_transplant']:.2f})")
record(f"e231_sigsource_{tag}", dict(model=tag, b=b, L=L, per_level={str(k): v for k, v in out.items()}), " | ".join(f"lv{lv}: natural {v['natural']:.2f}, transplant x1/x0.5/x2 {v['transplant_x1']:.2f}/{v['transplant_x0.5']:.2f}/{v['transplant_x2']:.2f}, random {v['random_acc']:.2f} (max share {v['random_max_share']:.2f}), pair top-2 {v['pair_both_in_top2']:.2f}, within cos natural {v['within_cos_natural']:.2f} vs transplant {v['within_cos_transplant']:.2f} (chance {v['chance']:.2f})" for lv, v in out.items()))
