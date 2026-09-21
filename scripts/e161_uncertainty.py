"""e161: why are uncertain tokens more readable? Per token (16 seqs): next-token entropy vs the centered state
norm, the dominant write's |c|, its prominence, its birth block, and its raw survival. Which of these carries the
entropy-prominence relation?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; ids = torch.arange(NT)
lp = torch.log_softmax(model(ids_seq).logits.float(), -1); ent = -(lp.exp() * lp).sum(-1).reshape(NT)
X = c.X(L)[ids]; Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); A, lab = c.dictionary(L); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV); ct = tc[ids, 0].to(DEV); d = A[row]
prom = (X * d).sum(1).abs() / X.norm(dim=1); surv = (Xraw * d).sum(1) / ct; birth = tb[ids, 0].to(DEV).float()
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
t = typ; res = dict(model=tag, L=L, spearman_entropy_vs=dict(prominence=spearman(ent[t], prom[t]), state_norm=spearman(ent[t], X.norm(dim=1)[t]), raw_state_norm=spearman(ent[t], Xraw.norm(dim=1)[t]), abs_coef=spearman(ent[t], ct.abs()[t]), proj=spearman(ent[t], (X * d).sum(1).abs()[t]), birth_block=spearman(ent[t], birth[t]), raw_survival=spearman(ent[t], surv.clamp(-2, 3)[t])))
q = ent[t].quantile(torch.tensor([0.25, 0.75], device=DEV)); lo, hi = t & (ent <= q[0]), t & (ent >= q[1])
res["quartiles"] = dict(state_norm=(X.norm(dim=1)[lo].median().item(), X.norm(dim=1)[hi].median().item()), abs_coef=(ct.abs()[lo].median().item(), ct.abs()[hi].median().item()), prominence=(prom[lo].median().item(), prom[hi].median().item()), birth0_share=((birth == 0)[lo].float().mean().item(), (birth == 0)[hi].float().mean().item()))
record(f"e161_uncert_{tag}", res, "Spearman(entropy, .): " + " ".join(f"{k} {v:+.2f}" for k, v in res["spearman_entropy_vs"].items()) + " | low vs high entropy quartile: " + " ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in res["quartiles"].items()))
