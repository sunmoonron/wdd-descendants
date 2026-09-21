"""e44: the block-increment pipeline as a full instrument at a MATCHED total budget. Global: OMP k=64 on the full
state with the full dictionary. Delta pipeline: for each block b <= L, OMP on the increment H[b+1]-H[b] over block
b's atoms with k_b atoms (equal split, or split proportional to increment energy), plus the token's own embedding
atom; union support refit on the full centered state. Scores: dominant write recall (top-1, top-3), real-write
share of the selected MLP atoms (ledger >= 5% of the token's largest), coefficient error, FVU of the full state."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 8192)); ids = sub(c.NT, N); K = 64
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV)                 # [N, (L+1)*DFF] signed
tb, tn, tc = c.top_writes(L, 3); rows3 = torch.stack([c.atom_index(L, tb[ids, j], tn[ids, j]) for j in range(3)], 1).to(DEV); ct = tc[ids, 0].to(DEV)
thr = 0.05 * C.abs().max(1, keepdim=True).values
def score(name, sel, cof):
    ismlp = typA[sel] == T_MLP; key = blkA[sel] * c.DFF + idxA[sel]; key = torch.where(ismlp, key, torch.zeros_like(key))
    cs = torch.gather(C, 1, key.clamp(min=0)); real = ismlp & (cs.abs() >= thr)
    hit = (sel == rows3[:, :1]).any(1); hit3 = (sel[:, :, None] == rows3[:, None, :]).any(1).all(1)
    chat = (cof * (sel == rows3[:, :1])).sum(1); idn = hit & typ
    signreal = ((cof * cs)[real] > 0).float().mean().item()
    r = dict(recall1=hit[typ].float().mean().item(), recall3=hit3[typ].float().mean().item(), real_share_of_mlp_atoms=(real[typ].sum() / ismlp[typ].sum()).item(),
             mlp_share_of_support=ismlp[typ].float().mean().item(), sign_agree_real=signreal, med_rel_err=((chat[idn] - ct[idn]).abs() / ct[idn].abs()).median().item(),
             spurious_coef_mass=((cof.abs() * (ismlp & ~real)).sum(1)[typ].sum() / (cof.abs() * ismlp).sum(1)[typ].sum()).item())
    cr, e = refit(X, A, sel); r["fvu_full_state"] = fvu(e, X, typ)
    log(f"{tag} {name}: r1 {r['recall1']:.3f} r3 {r['recall3']:.3f} real {r['real_share_of_mlp_atoms']:.2f} spur-mass {r['spurious_coef_mass']:.2f} err {r['med_rel_err']:.2f} fvu {r['fvu_full_state']:.3f}"); return r
res = dict(model=tag, L=L, N=N, K=K, variants={})
sel, cof, err = omp(X, A, K); res["variants"]["global_omp64"] = score("global_omp64", sel, cof)
# delta pipeline
deltas = [c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV) for b in range(L + 1)]
en = torch.stack([(d ** 2).sum(1) for d in deltas], 1)                                              # [N, L+1]
tok = c.s["eval_ids"].reshape(-1)[ids].to(DEV)
for mode in ("equal", "energy"):
    parts = []
    for b in range(L + 1):
        Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); rows_b = torch.nonzero((blkA == b) & (typA >= T_MLP))[:, 0]   # rows of block b atoms in A (same order)
        kb = 24 if mode == "equal" else 24
        s_b, c_b, _ = omp(deltas[b], Ab, kb); parts.append(rows_b[s_b])                                 # map to full-dictionary rows
    if mode == "equal":
        per = (K - 1) // (L + 1); keep = torch.cat([p[:, :per] for p in parts], 1)
    else:
        alloc = (en / en.sum(1, keepdim=True) * (K - 1)).round().long().clamp(min=1, max=24)          # [N, L+1]
        keep = torch.full((N, K - 1), -1, dtype=torch.long, device=DEV)
        for i in range(N):
            take = torch.cat([parts[b][i, :alloc[i, b]] for b in range(L + 1)])[:K - 1]; keep[i, :len(take)] = take
        keep[keep < 0] = keep[:, :1].expand_as(keep)[keep < 0]                                          # pad with a repeat (ridge handles it)
    sel_d = torch.cat([keep, tok[:, None]], 1)
    cof_d, _ = refit(X, A, sel_d); res["variants"][f"delta_{mode}"] = score(f"delta_{mode}", sel_d, cof_d)
record(f"e44_pipeline_{tag}", res, " | ".join(f"{k}: r1 {v['recall1']:.3f} r3 {v['recall3']:.3f} real {v['real_share_of_mlp_atoms']:.2f} spur {v['spurious_coef_mass']:.2f} err {v['med_rel_err']:.2f} fvu {v['fvu_full_state']:.3f}" for k, v in res["variants"].items()))
