"""e76 (meta): WDD's own survival estimate. For dominant writes born in block b0, the OMP-recovered coefficient of
their atom at each later level (0 when not selected), divided by the ledger coefficient, vs the true projection
survival. Does the instrument's reading of a write's fate track the write's real fate, and when it is selected,
how far off is the amount?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = 2048; ids = sub(c.NT, N); b0 = 1
led = c.ledger(b0, blocks=[b0])[b0][ids]; tc_, tn_ = led.abs().max(1); tc_ = torch.gather(led, 1, tn_[:, None])[:, 0].to(DEV); big = tc_.abs() >= tc_.abs().quantile(0.5); rows = []
for L in range(b0, c.NB):
    X = c.X(L)[ids]; Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw) & big; A, lab = c.dictionary(L); row = c.atom_index(L, torch.full_like(tn_, b0), tn_).to(DEV); d = A[row]
    sel, cof, err = omp(X, A, 64); hitm = sel == row[:, None]; chat = (cof * hitm).sum(1); rec = chat / tc_; true_s = (Xraw * d).sum(1) / tc_; cen_s = (X * d).sum(1) / tc_
    sel_m = hitm.any(1) & typ
    rows.append(dict(L=L, selected=sel_m.float().sum().item() / typ.float().sum().item(), recovered_surv_med_all=rec[typ].median().item(), true_surv_med=true_s[typ].median().item(), cen_surv_med=cen_s[typ].median().item(),
                     recovered_over_centered_when_selected=(rec[sel_m] / cen_s[sel_m]).median().item() if sel_m.any() else None, corr_recovered_vs_true_when_selected=(torch.corrcoef(torch.stack([rec[sel_m], true_s[sel_m]]))[0, 1].item() if sel_m.sum() > 10 else None)))
    log(f"{tag} L{L}: selected {rows[-1]['selected']:.2f} recovered-surv (all) {rows[-1]['recovered_surv_med_all']:.2f} true {rows[-1]['true_surv_med']:.2f} centered {rows[-1]['cen_surv_med']:.2f} | when selected: recovered/centered {rows[-1]['recovered_over_centered_when_selected']} corr {rows[-1]['corr_recovered_vs_true_when_selected']}")
record(f"e76_recsurv_{tag}", dict(model=tag, b0=b0, rows=rows), " | ".join(f"L{r['L']}: sel {r['selected']:.2f} rec {r['recovered_surv_med_all']:.2f} true {r['true_surv_med']:.2f} cen {r['cen_surv_med']:.2f}" for r in rows))
