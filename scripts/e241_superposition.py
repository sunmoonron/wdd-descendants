"""e241: is the provenance code compositional? Block b = 2, K candidate neurons (>= 20 natural tokens) with natural
centroids at b+2 and L. At foreign tokens inject the sum of m distinct neurons' median write vectors (m = 2, 4, 8,
16, up to K-1): additivity error ||F(all) - F(first half) - F(second half)|| / ||F(all)||, footprint norm vs sqrt(m),
and recovery of the injected identities from the mixture (fraction of the m injected neurons among the top-m and
top-2m centroids; chance m/K). Kill: additivity error above 0.5 or recovery near chance at m = 8."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; levels = sorted({b + 2, L}); run = make_runner(model, arch, c, ids_seq, levels, NT); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(b, tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
med = torch.stack([tc[idx][lab_i == k].median() for k in range(K)]); foreign = torch.nonzero(typ & ~torch.isin(torch.arange(NT, device=DEV), idx))[:, 0]; NF = len(foreign); cents = {lv: centroids((S0[lv] - S1[lv])[idx], lab_i, K) for lv in levels}
single_norm = {}; out = {}
for m in [mm for mm in (1, 2, 4, 8, 16, 32) if mm <= K - 1]:
    torch.manual_seed(m); perm = torch.argsort(torch.rand(NF, K, device=DEV), dim=1)[:, :m]                      # m distinct neurons per token
    def inj_of(cols):
        inj = torch.zeros(NT, c.D, device=DEV)
        for j in cols: inj[foreign] += med[perm[:, j]][:, None] * R[keep[perm[:, j]]]
        return inj
    SA = run(inject=inj_of(range(m)), inject_block=b + 1); rec = {}
    if m > 1: SB = run(inject=inj_of(range(m // 2)), inject_block=b + 1); SC = run(inject=inj_of(range(m // 2, m)), inject_block=b + 1)
    for lv in levels:
        FA = (SA[lv] - S0[lv])[foreign]; r = dict(norm=FA.norm(dim=1).median().item())
        if m == 1: single_norm[lv] = r["norm"]
        else:
            FB = (SB[lv] - S0[lv])[foreign]; FC = (SC[lv] - S0[lv])[foreign]; r["additivity_error"] = ((FA - FB - FC).norm(dim=1) / FA.norm(dim=1).clamp_min(1e-6)).median().item()
        r["norm_over_sqrt_m_single"] = r["norm"] / (math.sqrt(m) * single_norm[lv])
        sims = unit(FA) @ cents[lv].T; topm = sims.topk(m, dim=1).indices; top2m = sims.topk(min(2 * m, K), dim=1).indices; inj_set = perm
        r["recall_at_m"] = (topm[:, :, None] == inj_set[:, None, :]).any(1).float().mean().item(); r["recall_at_2m"] = (top2m[:, :, None] == inj_set[:, None, :]).any(1).float().mean().item(); r["chance_at_m"] = m / K
        rec[lv] = r
    out[m] = rec
    log(f"{tag} m={m}: " + " | ".join(f"lv{lv}: additivity error {v.get('additivity_error', float('nan')):.2f}, norm/(sqrt(m) x single) {v['norm_over_sqrt_m_single']:.2f}, recall@m {v['recall_at_m']:.2f} (chance {v['chance_at_m']:.2f}), recall@2m {v['recall_at_2m']:.2f}" for lv, v in rec.items()))
record(f"e241_superposition_{tag}", dict(model=tag, b=b, L=L, K=K, per_m={str(k): {str(kk): vv for kk, vv in v.items()} for k, v in out.items()}), f"K {K}; at L by m: additivity error " + " ".join(f"m{m}:{out[m][L].get('additivity_error', float('nan')):.2f}" for m in out if m > 1) + " | recall@m " + " ".join(f"m{m}:{out[m][L]['recall_at_m']:.2f}(chance {out[m][L]['chance_at_m']:.2f})" for m in out) + " | at +2 recall@m " + " ".join(f"m{m}:{out[m][b + 2]['recall_at_m']:.2f}" for m in out))
