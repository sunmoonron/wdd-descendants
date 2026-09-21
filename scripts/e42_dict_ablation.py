"""e42: which atoms do the explaining-away? Dominant-MLP-write recall and typical FVU under dictionary variants
(all data-free): full; no token/position embeddings; no attention atoms; MLP atoms only; MLP + embeddings;
twins merged (cos >= 0.95 clusters collapsed to one atom); and for GPT-2 the position atom's recall."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 8192)); ids = sub(c.NT, N)
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); Afull, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 1); rowfull = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV); ct = tc[ids, 0].to(DEV)
typA = lab["type"].to(DEV)
def run(name, keep):
    A = Afull[keep]; newrow = (torch.cumsum(keep.long(), 0) - 1)[rowfull]; ok = keep[rowfull]
    sel, cof, err = omp(X, A, 64); hit = (sel == newrow[:, None]).any(1) & ok
    s1, _, _ = oneshot(X, A, 64); hit1 = (s1 == newrow[:, None]).any(1) & ok
    idn = hit & typ; chat = (cof * (sel == newrow[:, None])).sum(1)[idn]
    r = dict(atoms=int(keep.sum()), recall=hit[typ].float().mean().item(), oneshot=hit1[typ].float().mean().item(), fvu32=fvu(err[:, 31], X, typ), fvu64=fvu(err[:, 63], X, typ),
             med_rel_err=((chat - ct[idn]).abs() / ct[idn].abs()).median().item())
    log(f"{tag} {name}: atoms {r['atoms']} recall {r['recall']:.3f} oneshot {r['oneshot']:.3f} fvu32 {r['fvu32']:.3f} relerr {r['med_rel_err']:.2f}"); return r
res = dict(model=tag, L=L, N=N, variants={})
res["variants"]["full"] = run("full", torch.ones(len(typA), dtype=torch.bool, device=DEV))
res["variants"]["no_emb"] = run("no_emb", typA >= T_MLP)
res["variants"]["no_att"] = run("no_att", typA != T_ATT)
res["variants"]["mlp_only"] = run("mlp_only", typA == T_MLP)
res["variants"]["mlp_plus_emb"] = run("mlp_plus_emb", (typA == T_MLP) | (typA <= T_POS))
# twins merged: greedy clustering of atoms with |cos| >= 0.95 (keep the first of each cluster)
keep = torch.ones(len(typA), dtype=torch.bool, device=DEV); NA = len(typA)
for s in range(0, NA, 4096):
    G = (Afull[s:s + 4096] @ Afull.T).abs(); G[torch.arange(G.shape[0]), torch.arange(s, s + G.shape[0])] = 0
    later = torch.zeros(NA, dtype=torch.bool, device=DEV); later[s:] = True                      # only drop atoms after a kept earlier one
    drop = ((G >= 0.95) & keep[None] & (torch.arange(NA, device=DEV)[None] > torch.arange(s, s + G.shape[0], device=DEV)[:, None])).any(0)
    keep &= ~drop
res["variants"]["twins_merged"] = run("twins_merged", keep); res["n_twins_dropped"] = int((~keep).sum())
if c.n_pos_emb:
    sel, cof, err = omp(X, Afull, 64); prow = (c.n_tok_emb + torch.arange(c.NT)[ids] % CTX).to(DEV)
    res["position_atom_recall"] = (sel == prow[:, None]).any(1)[typ].float().mean().item()
record(f"e42_dictabl_{tag}", res, " | ".join(f"{k}: n {v['atoms']} rec {v['recall']:.3f} os {v['oneshot']:.3f} fvu32 {v['fvu32']:.3f} err {v['med_rel_err']:.2f}" for k, v in res["variants"].items()) + (f" | pos-atom recall {res['position_atom_recall']:.3f}" if c.n_pos_emb else ""))
