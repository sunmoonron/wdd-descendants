"""e155: readability vs the model's own confidence. Per token (16 sequences): next-token entropy and the loss
at that token, vs the dominant write's prominence and WDD identification at the mid layer. Are writes more
readable when the model is confident, and do identification failures cluster on high-loss tokens?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; ids = torch.arange(NT)
out = model(ids_seq); logits = out.logits.float(); lp = torch.log_softmax(logits, -1); ent = -(lp.exp() * lp).sum(-1).reshape(NT); nll = torch.full((NT,), float("nan"), device=DEV)
tgt = ids_seq[:, 1:]; nll_ = -lp[:, :-1].gather(2, tgt[:, :, None])[:, :, 0]; nll[:NT - CTX * 0] = torch.cat([nll_, torch.full((NS, 1), float("nan"), device=DEV)], 1).reshape(NT)
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L); sel, cof, err = get_omp(c, L, A=A, X=c.X(L)); sel = sel[ids.to(DEV)]
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV); hit = (sel == row[:, None]).any(1); prom = (X * A[row]).sum(1).abs() / X.norm(dim=1)
ok = typ & ~torch.isnan(nll)
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
def auc(score, y):
    pos, neg = score[y], score[~y]; i = sub(len(pos), 3000).to(DEV); j = sub(len(neg), 3000).to(DEV); return (pos[i][:, None] > neg[j][None, :]).float().mean().item()
q = ent[ok].quantile(torch.tensor([0.25, 0.75], device=DEV)); lo, hi = ok & (ent <= q[0]), ok & (ent >= q[1])
res = dict(model=tag, L=L, spearman=dict(prominence_vs_entropy=spearman(prom[ok], ent[ok]), prominence_vs_nll=spearman(prom[ok], nll[ok])), auc_identified=dict(low_entropy=auc(-ent[ok], hit[ok]), low_nll=auc(-nll[ok], hit[ok])),
           recall_by_entropy_quartile=dict(low=hit[lo].float().mean().item(), high=hit[hi].float().mean().item()), fvu_by_entropy_quartile=dict(low=fvu(err[ids.to(DEV)][lo, 63], X, lo) if False else fvu(err[ids.to(DEV)][:, 63], X, lo), high=fvu(err[ids.to(DEV)][:, 63], X, hi)))
record(f"e155_conf_{tag}", res, f"Spearman(prominence, entropy) {res['spearman']['prominence_vs_entropy']:+.2f}, (prominence, loss) {res['spearman']['prominence_vs_nll']:+.2f} | AUC(low entropy -> identified) {res['auc_identified']['low_entropy']:.2f}, (low loss -> identified) {res['auc_identified']['low_nll']:.2f} | recall low/high-entropy quartile {res['recall_by_entropy_quartile']['low']:.2f}/{res['recall_by_entropy_quartile']['high']:.2f} | FVU64 low/high {res['fvu_by_entropy_quartile']['low']:.3f}/{res['fvu_by_entropy_quartile']['high']:.3f}")
