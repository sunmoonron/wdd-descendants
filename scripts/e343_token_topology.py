"""e343: does the quotient depend on where in the context the perturbation sits? The natural descendants split by
token position (first quarter vs last quarter of the context), by token class (punctuation vs alphabetic vs
numeric, from the tokenizer's decoded strings), and by sequence: quotients fitted within each part, their overlap
with each other and with the global quotient (chance 16/D), and cross-part decoding (quotient from part A decoding
part B) relative to own."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, NS=16, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; N = len(S.idx); Fc = F - F.mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], torch.arange(N, device=DEV)); dln = unit(nat["dl"]); pos = S.idx % CTX; seq = S.idx // CTX; toks = S.ids_seq.reshape(-1)[S.idx].tolist(); strs = [S.tok.decode([t]).strip() for t in toks]
cls = torch.tensor([0 if (s and s[0].isalpha()) else 1 if (s and s[0].isdigit()) else 2 for s in strs], device=DEV); parts = {"early_position": pos < CTX // 4, "late_position": pos >= 3 * CTX // 4, "alphabetic": cls == 0, "non_alphabetic": cls != 0, "sequences_A": seq < S.NS // 2, "sequences_B": seq >= S.NS // 2}
Qg = pls(Fc, Z, d); Q = {}; own = {}
for nm, m in parts.items():
    rows = torch.nonzero(m)[:, 0]
    if len(rows) < 60: continue
    tr = rows[::2]; te = rows[1::2]; Q[nm] = pls(Fc[tr] - Fc[tr].mean(0, keepdim=True), Z[tr], d); own[nm] = knn_cos(Fc @ Q[nm], dln, tr, te)
pairs = [("early_position", "late_position"), ("alphabetic", "non_alphabetic"), ("sequences_A", "sequences_B")]; out = {}
for a, b_ in pairs:
    if a not in Q or b_ not in Q: continue
    ra, rb = torch.nonzero(parts[a])[:, 0], torch.nonzero(parts[b_])[:, 0]; cross_ab = knn_cos(Fc @ Q[a], dln, rb[::2], rb[1::2]); cross_ba = knn_cos(Fc @ Q[b_], dln, ra[::2], ra[1::2]); out[f"{a}|{b_}"] = dict(overlap=inside(Q[a], Q[b_]), overlap_global=(inside(Q[a], Qg) + inside(Q[b_], Qg)) / 2, own_a=own[a], own_b=own[b_], cross_a_on_b=cross_ab, cross_b_on_a=cross_ba, n=(int(len(ra)), int(len(rb))))
log(f"{tag} (N {N}, chance {d / S.D:.3f}): " + " | ".join(f"{k} (n {v['n'][0]}/{v['n'][1]}): overlap {v['overlap']:.2f} (each vs global {v['overlap_global']:.2f}); own {v['own_a']:.2f}/{v['own_b']:.2f}, cross {v['cross_a_on_b']:.2f}/{v['cross_b_on_a']:.2f}" for k, v in out.items()))
record(f"e343_topology_{tag}", dict(model=tag, L=L, N=N, per_split=out), " | ".join(f"{k}: ov {v['overlap']:.2f} own {v['own_a']:.2f}/{v['own_b']:.2f} cross {v['cross_a_on_b']:.2f}/{v['cross_b_on_a']:.2f}" for k, v in out.items()))
