"""e253: what state variable makes the massive neuron's transport context-dependent? M = the block-2 candidate with the
largest median |coefficient|, N = a matched normal candidate. Each vector is injected at all foreign typical tokens
(separate passes); per token the descendant's cosine with the neuron's NATURAL centroid at +2 and L is correlated
(Spearman) with the neuron's own natural activation at that token (the sink-route activity), the state norm, the
position and the token's own dominant coefficient; and binned by quartile of the neuron's natural activation. A sign
change of the cosine across activation quartiles identifies the route."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; levels = sorted({b + 2, L}); run = make_runner(model, arch, c, ids_seq, levels, NT); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); S1 = run(b, tn); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); med = torch.stack([tc[idx][lab_i == k].median() for k in range(K)]); M = med.abs().argmax(); N = (med.abs() - med.abs().median()).abs().argmin(); pos = torch.arange(CTX, device=DEV).repeat(NS)
def spearman(a_, b_):
    ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
out = {}
for nm, j in (("massive", M), ("normal", N)):
    foreign = torch.nonzero(typ & (tn != keep[j]))[:, 0]; inj = torch.zeros(NT, D, device=DEV); inj[foreign] = med[j] * R[keep[j]]; S2 = run(inject=inj, inject_block=b + 1); own_act = led[foreign, keep[j]]; rec = {}
    for lv in levels:
        F = S0[lv] - S1[lv]; nat = centroids(F[idx], lab_i, K)[j]; Fi = (S2[lv] - S0[lv])[foreign]; cosn = (unit(Fi) @ nat[:, None])[:, 0]; nrm = S0[lv][foreign].norm(dim=1); own_dom = tc[foreign].abs()
        q = own_act.abs().quantile(torch.tensor([0.25, 0.5, 0.75], device=DEV)); qi = (own_act.abs()[:, None] > q[None]).sum(1)
        rec[lv] = dict(median_cos=cosn.median().item(), rho_own_activation=spearman(cosn, own_act.abs()), rho_own_activation_signed=spearman(cosn, own_act), rho_state_norm=spearman(cosn, nrm), rho_position=spearman(cosn, pos[foreign].float()), rho_own_dominant=spearman(cosn, own_dom), cos_by_activation_quartile=[cosn[qi == k].median().item() for k in range(4)], activation_quartile_medians=[own_act.abs()[qi == k].median().item() for k in range(4)])
        log(f"{tag} {nm} neuron {int(keep[j])} (|coef| {med[j].abs():.1f}) level {lv}: cos(descendant, natural centroid) median {rec[lv]['median_cos']:+.2f} | Spearman with own natural activation |a| {rec[lv]['rho_own_activation']:+.2f} (signed {rec[lv]['rho_own_activation_signed']:+.2f}), state norm {rec[lv]['rho_state_norm']:+.2f}, position {rec[lv]['rho_position']:+.2f}, token's dominant |coef| {rec[lv]['rho_own_dominant']:+.2f} | cos by own-activation quartile " + " ".join(f"{v:+.2f}" for v in rec[lv]['cos_by_activation_quartile']) + " (|a| " + " ".join(f"{v:.2f}" for v in rec[lv]['activation_quartile_medians']) + ")")
    out[nm] = rec
record(f"e253_massvar_{tag}", dict(model=tag, b=b, L=L, neurons=dict(massive=int(keep[M]), normal=int(keep[N])), per_neuron={nm: {str(k): v for k, v in rec.items()} for nm, rec in out.items()}), " | ".join(f"{nm} lv{lv}: cos {v['median_cos']:+.2f}, rho(own activation) {v['rho_own_activation']:+.2f}, rho(norm) {v['rho_state_norm']:+.2f}, rho(position) {v['rho_position']:+.2f}, cos by activation quartile " + "/".join(f"{x:+.2f}" for x in v['cos_by_activation_quartile']) for nm, rec in out.items() for lv, v in rec.items()))
