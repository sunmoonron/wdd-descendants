"""e41: aliasing. When the dominant write is missed, is it read under an alias, i.e. is its max-coherence partner
atom (a data-free quantity) in the support, with the predicted coefficient c_true * cos(d, partner)? Fraction of
misses aliased by the top-1 / top-3 coherence partners, coefficient agreement, and a null with random atoms of the
same type. If aliasing is systematic, missed writes are recoverable by a lookup, not lost."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = int(sys.argv[2]) if len(sys.argv) > 2 else mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); ct = tc[:, 0].to(DEV)
hit = (sel == row[:, None]).any(1); miss = typ & ~hit
uniq, inv = row.unique(return_inverse=True); P = torch.zeros(len(uniq), 3, dtype=torch.long, device=DEV); PC = torch.zeros(len(uniq), 3, device=DEV)
for s in range(0, len(uniq), 1024):
    u = uniq[s:s + 1024]; G = A[u] @ A.T; G[torch.arange(len(u)), u] = 0; v, i = G.abs().topk(3, dim=1); P[s:s + 1024] = i; PC[s:s + 1024] = torch.gather(G, 1, i)
part, pcos = P[inv], PC[inv]                                                             # [NT, 3] partners and signed cosines
inS = (sel[:, :, None] == part[:, None, :]).any(1)                                        # [NT, 3] partner j in support
res = dict(model=tag, L=L, recall=hit[typ].float().mean().item(), n_miss=int(miss.sum()))
res["miss_aliased_by_top1"] = inS[miss, 0].float().mean().item(); res["miss_aliased_by_top3_any"] = inS[miss].any(1).float().mean().item()
res["hit_has_top1_partner_too"] = inS[typ & hit, 0].float().mean().item()
# coefficient agreement for aliased misses: chat_partner vs c_true * cos
cp = (cof * (sel == part[:, 0:1])).sum(1); pred = ct * pcos[:, 0]; am = miss & inS[:, 0]
res["alias_coef"] = dict(sign_agree=((cp[am] * pred[am]) > 0).float().mean().item(), med_ratio=(cp[am] / pred[am]).median().item(), med_rel_err=((cp[am] - pred[am]).abs() / pred[am].abs()).median().item(), partner_cos_med=pcos[am, 0].abs().median().item())
# null: a random atom of the same type as the partner
g = torch.Generator().manual_seed(0); typA = lab["type"].to(DEV); rnd = torch.zeros_like(part[:, 0])
for t in range(5):
    m = typA[part[:, 0]] == t; pool = torch.nonzero(typA == t)[:, 0]
    if m.any() and len(pool): rnd[m] = pool[torch.randint(0, len(pool), (int(m.sum()),), generator=g).to(DEV)]
res["null_random_same_type_in_support"] = (sel == rnd[:, None]).any(1)[miss].float().mean().item()
res["partner_type_hist"] = {int(t): (typA[part[miss, 0]] == t).float().mean().item() for t in range(5)}
res["partner_same_block_frac"] = (lab["block"].to(DEV)[part[miss, 0]] == lab["block"].to(DEV)[row[miss]]).float().mean().item()
res["recall_with_alias_credit"] = ((hit | inS[:, 0]) & typ).float().sum().item() / typ.float().sum().item()
record(f"e41_alias_{tag}" + (f"_L{L}" if len(sys.argv) > 2 else ""), res, f"recall {res['recall']:.3f} | misses aliased by top1 partner {res['miss_aliased_by_top1']:.3f} (null {res['null_random_same_type_in_support']:.3f}) top3 {res['miss_aliased_by_top3_any']:.3f} | alias coef sign {res['alias_coef']['sign_agree']:.2f} ratio {res['alias_coef']['med_ratio']:.2f} cos {res['alias_coef']['partner_cos_med']:.2f} | partner types {res['partner_type_hist']} same-block {res['partner_same_block_frac']:.2f} | recall w/ alias credit {res['recall_with_alias_credit']:.3f}")
