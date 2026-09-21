"""e256: does the FUNCTIONAL footprint compose? At foreign tokens inject m = 1, 2, 4, 8 candidate write vectors (equal
amplitudes) and measure the same-position logit change: additivity error ||dL(all) - dL(half1) - dL(half2)|| /
||dL(all)||, the norm of dL(all) relative to sqrt(m) x single, and recovery of the injected identities from the mixture's
logit footprint by nearest single-injection logit image (recall@m; chance m/K). Compared with the same quantities
for the state descendant at L in the same runs."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); med = torch.stack([tc[idx][lab_i == k].median() for k in range(K)])
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0][:2048]; NF = len(foreign); S0 = run(positions=foreign); base_lg = S0["lg"]
# single-injection logit and state images per candidate (zero-shot)
img_l = torch.zeros(K, base_lg.shape[1], device=DEV); img_s = torch.zeros(K, D, device=DEV); cnt = torch.zeros(K, device=DEV); torch.manual_seed(0)
for p in range(3):
    a = torch.randint(0, K, (NF,), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[foreign] = med[a][:, None] * R[keep[a]]; S2 = run(positions=foreign, inject=inj, inject_block=b + 1); dl = S2["lg"] - base_lg; dl = dl - dl.mean(1, keepdim=True); img_l.index_add_(0, a, dl); img_s.index_add_(0, a, (S2[L] - S0[L])[foreign]); cnt.index_add_(0, a, torch.ones(NF, device=DEV))
img_l = unit(img_l / cnt.clamp_min(1)[:, None]); img_s = unit(img_s / cnt.clamp_min(1)[:, None]); out = {}; single = {}
for m in [mm for mm in (1, 2, 4, 8) if mm <= K - 1]:
    torch.manual_seed(m); perm = torch.argsort(torch.rand(NF, K, device=DEV), dim=1)[:, :m]
    def inj_of(cols):
        inj = torch.zeros(NT, D, device=DEV)
        for j in cols: inj[foreign] += med[perm[:, j]][:, None] * R[keep[perm[:, j]]]
        return inj
    SA = run(positions=foreign, inject=inj_of(range(m)), inject_block=b + 1); dlA = SA["lg"] - base_lg; dlA = dlA - dlA.mean(1, keepdim=True); FA = (SA[L] - S0[L])[foreign]; r = {}
    if m > 1:
        SB = run(positions=foreign, inject=inj_of(range(m // 2)), inject_block=b + 1); SC = run(positions=foreign, inject=inj_of(range(m // 2, m)), inject_block=b + 1); dlB = SB["lg"] - base_lg; dlB = dlB - dlB.mean(1, keepdim=True); dlC = SC["lg"] - base_lg; dlC = dlC - dlC.mean(1, keepdim=True)
        r["logit_additivity"] = ((dlA - dlB - dlC).norm(dim=1) / dlA.norm(dim=1).clamp_min(1e-6)).median().item(); r["state_additivity"] = ((FA - (SB[L] - S0[L])[foreign] - (SC[L] - S0[L])[foreign]).norm(dim=1) / FA.norm(dim=1).clamp_min(1e-6)).median().item()
    if m == 1: single["l"] = dlA.norm(dim=1).median().item(); single["s"] = FA.norm(dim=1).median().item()
    r["logit_norm_ratio"] = dlA.norm(dim=1).median().item() / (math.sqrt(m) * single["l"]); r["logit_recall_at_m"] = ((unit(dlA) @ img_l.T).topk(m, dim=1).indices[:, :, None] == perm[:, None, :]).any(1).float().mean().item(); r["state_recall_at_m"] = ((unit(FA) @ img_s.T).topk(m, dim=1).indices[:, :, None] == perm[:, None, :]).any(1).float().mean().item(); r["chance"] = m / K
    out[m] = r
    log(f"{tag} m={m}: logit additivity {r.get('logit_additivity', float('nan')):.2f} (state {r.get('state_additivity', float('nan')):.2f}), logit norm/(sqrt(m) x single) {r['logit_norm_ratio']:.2f}, identity recall@m from the logit footprint {r['logit_recall_at_m']:.2f} vs from the state descendant {r['state_recall_at_m']:.2f} (chance {r['chance']:.2f})")
record(f"e256_funcadd_{tag}", dict(model=tag, b=b, L=L, K=K, per_m={str(k): v for k, v in out.items()}), "logit additivity by m " + " ".join(f"m{m}:{out[m].get('logit_additivity', float('nan')):.2f}" for m in out if m > 1) + " (state " + " ".join(f"{out[m].get('state_additivity', float('nan')):.2f}" for m in out if m > 1) + ") | recall@m logit/state " + " ".join(f"m{m}:{out[m]['logit_recall_at_m']:.2f}/{out[m]['state_recall_at_m']:.2f}(chance {out[m]['chance']:.2f})" for m in out))
