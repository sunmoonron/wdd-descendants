"""e370b (CPU): first-order selection is not use. For every model with both e368 (per-token gradient selection
c_h(t) = <dL/dz_h(t), z_h(t)>) and e369 (per-token mean-ablation effect), per head: the mean selection over all natural
tokens (sign-flipped so positive = scaling the write up would lower the loss) against the mean ablation effect
(positive = the head helps). At a trained optimum the first should vanish on average for every head (the scale of each
output projection is a parameter direction), while the second need not. Reported: the size of the mean selection
relative to its per-token spread (a t-like ratio), the Spearman correlation across heads between selection and use,
over all tokens and at induction-applicable positions, and the same for the attention-defined induction heads."""
import os, glob, json, time, torch
RES = os.environ.get("WDD_RESULTS", "/workspace/wdd/results"); out = {}
def spear(a, b): ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
for f in sorted(glob.glob(f"{RES}/e369/*.pt")):
    tag = os.path.basename(f)[:-3]; g = f"{RES}/e368/{tag}.pt"
    if not os.path.exists(g): continue
    A = torch.load(f); G = torch.load(g); Ca = A["C"].float(); Cg = -G["C"].float(); Ma, Mg = A["M"], G["M"]; pf, pv = A["pref"], A["prev"]; nh = Ca.shape[-1]
    IND = [i for i in pf.argsort(descending=True).tolist() if pf[i] >= 0.2 and pf[i] >= pv[i]][:8]
    ka = torch.zeros_like(Ma); ka[:, 1:-1] = True; kg = torch.zeros_like(Mg); kg[:, 1:-1] = True
    Xa, Xg = Ca[ka], Cg[kg]; Mka, Mkg = Ma[ka], Mg[kg]
    use_all, sel_all = Xa.mean(0), Xg.mean(0); use_app, sel_app = Xa[Mka].mean(0), Xg[Mkg].mean(0)
    t_sel = (Xg.mean(0) / (Xg.std(0) / Xg.shape[0] ** 0.5).clamp_min(1e-12)); t_use = (Xa.mean(0) / (Xa.std(0) / Xa.shape[0] ** 0.5).clamp_min(1e-12))
    out[tag] = dict(n_heads=nh, spearman_selection_vs_use_all=spear(sel_all, use_all), spearman_selection_vs_use_applicable=spear(sel_app, use_app), median_abs_t_selection=t_sel.abs().median().item(), median_abs_t_use=t_use.abs().median().item(), share_heads_use_positive_t3=(t_use > 3).float().mean().item(), share_heads_selection_abs_t3=(t_sel.abs() > 3).float().mean().item(), induction_selection_all=sel_all[IND].mean().item() if IND else None, induction_use_all=use_all[IND].mean().item() if IND else None, induction_selection_applicable=sel_app[IND].mean().item() if IND else None, induction_use_applicable=use_app[IND].mean().item() if IND else None)
json.dump(dict(out, _exp="e370b_selection_vs_use", _time=time.strftime("%Y-%m-%d %H:%M:%S")), open(f"{RES}/e370b_selection_vs_use.json", "w"), indent=1)
line = "e370b selection vs use: " + " | ".join(f"{t}: Spearman(selection, use) over heads all {v['spearman_selection_vs_use_all']:+.2f} applicable {v['spearman_selection_vs_use_applicable']:+.2f}; median |t| selection {v['median_abs_t_selection']:.1f} vs use {v['median_abs_t_use']:.1f}; heads with use t>3 {v['share_heads_use_positive_t3']:.2f}, selection |t|>3 {v['share_heads_selection_abs_t3']:.2f}; induction heads selection {v['induction_selection_all']:+.4f} vs use {v['induction_use_all']:+.4f} (applicable {v['induction_selection_applicable']:+.4f} vs {v['induction_use_applicable']:+.4f})" for t, v in out.items())
open(f"{RES}/FINDINGS.log", "a").write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + line + "\n"); print(line)
