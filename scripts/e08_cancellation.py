"""e08: who cancels a dominant write, and which survival predicts identification. For each token's dominant MLP
write (block b*, unit direction d, coefficient c): raw survival (x.d)/c decomposed exactly into the projections of
embedding, earlier blocks (attn, mlp), same-block attention, same-block other MLP neurons, later blocks (attn, mlp).
Centered survival = raw - (mu.d)/c. AUCs of OMP identification vs raw survival, centered survival, the centering
term alone, prominence. Then the DC-atom test: OMP on the RAW state with mu added as one atom vs centering."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 8192)); ids = sub(c.NT, N)
Xraw = c.X(L, center=False)[ids]; mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; typ = typical_mask(Xraw)
A, lab = c.dictionary(L); sel, cof, err = get_omp(c, L, A=A, X=c.X(L)); sel = sel[ids.to(DEV)]
tb, tn, tc = c.top_writes(L, 1); tb, tn, tc = tb[ids, 0], tn[ids, 0], tc[ids, 0].to(DEV)
row = c.atom_index(L, tb, tn).to(DEV); d = A[row]; hit = (sel == row[:, None]).any(1)
s1, _, _ = oneshot(X, A, 64); hit1 = (s1 == row[:, None]).any(1)
proj = lambda V: (V * d).sum(1) / tc
parts = dict(emb=proj(c.s["H"][0][ids].float().to(DEV)), earlier_att=torch.zeros(N, device=DEV), earlier_mlp=torch.zeros(N, device=DEV),
             same_att=torch.zeros(N, device=DEV), same_mlp_other=torch.zeros(N, device=DEV), later_att=torch.zeros(N, device=DEV), later_mlp=torch.zeros(N, device=DEV))
tbd = tb.to(DEV)
for b in range(L + 1):
    att = c.s["ATT"][b][ids].float().to(DEV); mlp = c.acts[b][ids].float().to(DEV) @ c.wdir_cpu(b).to(DEV)
    if c.d["mlp_bias"][b] is not None: mlp = mlp + c.d["mlp_bias"][b].to(DEV)
    pa, pm = proj(att), proj(mlp)
    e, s, l = tbd > b, tbd == b, tbd < b
    parts["earlier_att"] += pa * e; parts["earlier_mlp"] += pm * e; parts["later_att"] += pa * l; parts["later_mlp"] += pm * l
    parts["same_att"] += pa * s; parts["same_mlp_other"] += (pm - 1.0) * s          # minus the write itself (c*d.d/c = 1)
raw = proj(Xraw); cen = proj(X); dc = proj(mu.expand_as(Xraw))
chk = (raw - (1 + sum(parts.values()))).abs().median().item()
def auc(score, y):
    pos, neg = score[y], score[~y]; i = sub(len(pos), 4000).to(DEV); j = sub(len(neg), 4000).to(DEV)
    return (pos[i][:, None] > neg[j][None, :]).float().mean().item()
prom = (X * d).sum(1).abs() / X.norm(dim=1)
t = typ
res = dict(model=tag, L=L, N=N, identity_check=chk, recall=hit[t].float().mean().item(),
           auc_omp=dict(raw_surv=auc(raw[t].clamp(-2, 3), hit[t]), centered_surv=auc(cen[t].clamp(-2, 3), hit[t]), dc_term=auc(-dc[t].clamp(-3, 3), hit[t]), prominence=auc(prom[t], hit[t]), abs_coef=auc(tc.abs()[t], hit[t])),
           auc_oneshot=dict(raw_surv=auc(raw[t].clamp(-2, 3), hit1[t]), centered_surv=auc(cen[t].clamp(-2, 3), hit1[t]), prominence=auc(prom[t], hit1[t])),
           survival=dict(raw_median=raw[t].median().item(), centered_median=cen[t].median().item(), dc_median=dc[t].median().item(),
                         frac_raw_erased=(raw[t] < 0.25).float().mean().item(), frac_centered_erased=(cen[t] < 0.25).float().mean().item()),
           contributions_median={k: v[t].median().item() for k, v in parts.items()}, contributions_mean={k: v[t].mean().item() for k, v in parts.items()})
er = t & (raw < 0.25); res["who_erases_raw_erased"] = {k: v[er].mean().item() for k, v in parts.items()}; res["n_raw_erased"] = int(er.sum())
ec = t & (cen < 0.25) & (raw >= 0.75); res["centered_erased_but_raw_intact"] = dict(n=int(ec.sum()), frac_of_typical=(ec.sum() / t.sum()).item(), recall=hit[ec].float().mean().item() if ec.any() else None, dc_median=dc[ec].median().item() if ec.any() else None)
# DC atom: OMP on raw states with mu as an extra atom
Adc = torch.cat([A, (mu / mu.norm())[None]]); seld, cofd, errd = omp(Xraw, Adc, 64); hitd = (seld == row[:, None]).any(1)
dc_used = (seld == A.shape[0]).any(1).float().mean().item(); dc_first = (seld[:, 0] == A.shape[0]).float().mean().item()
selr, cofr, errr = omp(Xraw, A, 64); hitr = (selr == row[:, None]).any(1)
res["dc_atom"] = dict(recall_centered=hit[t].float().mean().item(), recall_raw_no_dc=hitr[t].float().mean().item(), recall_raw_with_dc=hitd[t].float().mean().item(),
                      dc_selected_frac=dc_used, dc_selected_first=dc_first,
                      fvu64_centered=fvu(err[ids.to(DEV)][:, 63], X, t), fvu64_raw_with_dc=fvu(errd[:, 63], X, t), fvu64_raw_no_dc=fvu(errr[:, 63], X, t))
# does the DC atom rescue the frequent writers (centered-erased but raw-intact)?
res["dc_atom"]["recall_on_centered_erased_raw_intact"] = dict(centered=hit[ec].float().mean().item() if ec.any() else None, raw_with_dc=hitd[ec].float().mean().item() if ec.any() else None)
record(f"e08_cancel_{tag}", res, f"AUC raw {res['auc_omp']['raw_surv']:.2f} centered {res['auc_omp']['centered_surv']:.2f} dc {res['auc_omp']['dc_term']:.2f} prom {res['auc_omp']['prominence']:.2f} | raw erased {res['survival']['frac_raw_erased']:.2f} centered erased {res['survival']['frac_centered_erased']:.2f} | erasers(raw): " + " ".join(f"{k} {v:.2f}" for k, v in res['who_erases_raw_erased'].items()) + f" | DC atom recall {res['dc_atom']['recall_centered']:.3f} -> raw+dc {res['dc_atom']['recall_raw_with_dc']:.3f} (raw only {res['dc_atom']['recall_raw_no_dc']:.3f}); cen-erased/raw-intact n={res['centered_erased_but_raw_intact']['n']} recall {res['centered_erased_but_raw_intact']['recall']} -> {res['dc_atom']['recall_on_centered_erased_raw_intact']['raw_with_dc']}")
