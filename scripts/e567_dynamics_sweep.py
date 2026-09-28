"""e567 (session 102): under what learning dynamics does a persistent association between a native write direction and
a class of inputs emerge? The grokking toy of e444 ((a+b) mod 113, one attention block, 512 ReLU neurons, full batch)
with one knob turned at a time from the AdamW base (10000 steps, measured every 250 on the training inputs):
 width 128 and 2048; weight decay 0.3, 3 and 0; learning rate 3e-4 and 3e-3; training fraction 0.2 and 0.5 (less and
 more data); initialization scale 0.3 and 3; random labels (memorising); a second seed.
Four invariants at every measurement: directional extremeness (the 32 most used MLP rows' S over the floor calibrated
on the rotated rows), context association (the usage's concentration: the top-32 share), row persistence (the words'
cosine with their row 250 steps earlier) and association persistence (the Jaccard of each word's 16 extreme inputs
with 250 steps earlier, and, since the toy's periodic features tie across many inputs, the correlation of the word's
projection profile over the training inputs with 250 steps earlier and the Jaccard of its over-the-floor set), plus the own-over-rotation FVU advantage and the accuracies. At the end: the cosine of each
final word's row with its row at every earlier time against the Jaccard of its extreme set with the final one (row
identity against computational identity). Pre-registered (probabilities are honest guesses):
 L1 (0.6) the association (S over the floor, top-32 share) appears in every run that generalises and in no run that
    memorises, whatever the knob;
 L2 (0.5) width raises the words' S (more rows, sharper extremes) and weight decay raises the top-32 share;
 L3 (0.6) after grokking the association persistence exceeds the row persistence's cosine deficit: the extreme sets are
    fixed while the rows still turn (identity separates from computation)."""
import sys, os, math, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from wdd_common import log, record, DEV, omp, refit
from ex_common import gabs, unitr, rotate, med, mean
torch.set_grad_enabled(True)
variant = sys.argv[1]
cfg = dict(seed=0, wd=1.0, randlab=False, steps=10000, every=250, lr=1e-3, frac=0.3, DM=512, init=1.0)
cfg.update({"base": {}, "dm128": dict(DM=128), "dm2048": dict(DM=2048), "wd0.3": dict(wd=0.3), "wd3": dict(wd=3.0), "nowd": dict(wd=0.0), "lr3e-4": dict(lr=3e-4), "lr3e-3": dict(lr=3e-3), "frac0.2": dict(frac=0.2), "frac0.5": dict(frac=0.5),
            "init0.3": dict(init=0.3), "init3": dict(init=3.0), "randlab": dict(randlab=True), "seed1": dict(seed=1)}[variant])
p, d, NH, HD = 113, 128, 4, 32; DM = cfg["DM"]; V = p + 1; KX = 16; NW = 32
torch.manual_seed(cfg["seed"])
A_, B_ = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij"); ALL = torch.stack([A_.flatten(), B_.flatten(), torch.full((p * p,), p)], 1).to(DEV); LAB = ((ALL[:, 0] + ALL[:, 1]) % p).clone()
perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(cfg["seed"] + 1)).to(DEV); ntr = int(cfg["frac"] * p * p); TR = torch.zeros(p * p, dtype=torch.bool, device=DEV); TR[perm[:ntr]] = True; TE = ~TR
if cfg["randlab"]: tri = torch.nonzero(TR)[:, 0]; LAB[tri] = LAB[tri][torch.randperm(tri.numel(), generator=torch.Generator().manual_seed(7)).to(DEV)]
class Grok(nn.Module):
    def __init__(s):
        super().__init__(); r = lambda *sh, fan: nn.Parameter(cfg["init"] * torch.randn(*sh) / math.sqrt(fan))
        s.WE, s.WP = r(V, d, fan=d), r(3, d, fan=d); s.WQ, s.WK, s.WV = r(NH, d, HD, fan=d), r(NH, d, HD, fan=d), r(NH, d, HD, fan=d); s.WO = r(NH, HD, d, fan=HD)
        s.Win, s.bin = r(d, DM, fan=d), nn.Parameter(torch.zeros(DM)); s.Wout, s.bout = r(DM, d, fan=DM), nn.Parameter(torch.zeros(d)); s.WU = r(d, p, fan=d)
    def forward(s, ids):
        x = s.WE[ids] + s.WP[None]; q, k, v = (torch.einsum("btd,hde->bhte", x, W) for W in (s.WQ, s.WK, s.WV))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(HD); att = att.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool, device=ids.device), 1), -1e9).softmax(-1)
        x = x + torch.einsum("bhte,hed->btd", att @ v, s.WO); h = F.relu(x @ s.Win + s.bin); x = x + h @ s.Wout + s.bout
        return x[:, -1] @ s.WU, x[:, -1], h[:, -1]
model = Grok().to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
Q = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(11)))[0].to(DEV); unit = lambda M: M / M.norm(dim=-1, keepdim=True).clamp_min(1e-8)
def own_words():
    heads = torch.cat([torch.linalg.svd(model.WO[h].detach(), full_matrices=False).Vh for h in range(NH)])
    return unit(torch.cat([model.WE.detach(), model.WP.detach(), heads, model.Wout.detach(), model.bout.detach()[None]])), V + 3 + NH * HD
gm = gabs(DM)
@torch.no_grad()
def measure(step, prev):
    lg, xf, h = model(ALL); L = F.cross_entropy(lg, LAB, reduction="none"); acc = (lg.argmax(-1) == LAB).float(); A, m0 = own_words(); n = A.shape[0]
    Xc = xf[TR] - xf[TR].mean(0); U = unit(Xc); tot = Xc.pow(2).sum(); out = dict(step=step, train_acc=acc[TR].mean().item(), test_acc=acc[TE].mean().item(), train_loss=L[TR].mean().item(), test_loss=L[TE].mean().item())
    ev, Ue = torch.linalg.eigh((A.T @ A / n).double()); Rm = ((Ue * ev.clamp_min(0).sqrt()) @ Ue.T).float(); G = unit(torch.randn(n, d, generator=torch.Generator().manual_seed(13)).to(DEV) @ Rm)
    for vn, Dd in (("own", A), ("rot", A @ Q), ("covA", G)):
        sel, _, _ = omp(Xc, Dd, 8, batch=4096, record_err=False); cof, err = refit(Xc, Dd, sel[:, :4]); out[f"{vn}_k4_fvu"] = (err.sum() / tot).item()
        if vn == "own": usage = torch.bincount(sel.flatten(), minlength=n).float()[m0:m0 + DM]
    Am = A[m0:m0 + DM]; Ar = unit(rotate(Am, seed=7)); CA = Am.T @ Am / DM; CR = Ar.T @ Ar / DM
    s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = (U @ Ar.T).abs().max(1).values; r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); Lf = s2.clamp_min(1e-12).sqrt() * gm * r_cal
    ratio = (U @ Am.T).abs() / Lf[:, None]; S = ratio.max(0).values; words = usage.topk(NW).indices; ext = ratio[:, words].topk(KX, dim=0).indices.T   # [NW, KX]
    Sr = ((U @ Ar.T).abs() / Lf[:, None]).max(0).values
    out.update(words_S=S[words].median().item(), words_S_q10=S[words].quantile(0.1).item(), rotated_S=Sr.topk(NW).values.median().item(), top32_share=(usage[words].sum() / usage.sum().clamp_min(1)).item(), rows_over_floor=int((S >= 1).sum()), rotated_over_floor=int((Sr >= 1).sum()),
               advantage=out["rot_k4_fvu"] - out["own_k4_fvu"], n_used=int((usage > 0).sum()))
    cur = dict(rows=Am.clone(), words=words, ext=ext, usage=usage, ratio=ratio[:, words].clone(), over=(ratio[:, words] > 1))
    if prev is not None:
        pw = prev["words"]; out["row_persistence"] = (prev["rows"][pw] * Am[pw]).sum(1).median().item(); out["usage_corr"] = torch.corrcoef(torch.stack([prev["usage"], usage]))[0, 1].item()
        both = [i for i, w in enumerate(pw.tolist()) if w in set(words.tolist())]; out["membership_retention"] = len(both) / NW
        extp = prev["ext"]; jac = []
        for i, w in enumerate(pw.tolist()):
            cur_e = ratio[:, w].topk(KX).indices; a, b = set(extp[i].tolist()), set(cur_e.tolist()); jac.append(len(a & b) / len(a | b))
        out["association_persistence"] = med(jac)
        rp = prev["ratio"]; rc = ratio[:, pw]; out["association_corr"] = med([float(torch.corrcoef(torch.stack([rp[:, i], rc[:, i]]))[0, 1]) for i in range(NW)])
        po, co = prev["over"], ratio[:, pw] > 1; out["over_floor_jaccard"] = med([float((po[:, i] & co[:, i]).sum() / (po[:, i] | co[:, i]).sum().clamp_min(1)) for i in range(NW)])
    return out, cur
log_rows, prev, t0 = [], None, time.time(); snaps = {}
for step in range(cfg["steps"] + 1):
    if step % cfg["every"] == 0:
        o, cur = measure(step, prev); log_rows.append(o); prev = cur; snaps[step] = dict(rows=cur["rows"].cpu(), ext=cur["ext"].cpu(), words=cur["words"].cpu(), ratio=cur["ratio"].cpu())
        if step % 2500 == 0: log(f"{variant} step {step}: acc {o['train_acc']:.2f}/{o['test_acc']:.2f}, advantage {o['advantage']:.2f}, words' S {o['words_S']:.2f} (rotated {o['rotated_S']:.2f}), top-32 share {o['top32_share']:.2f}, row persistence {o.get('row_persistence', float('nan')):.3f}, association persistence {o.get('association_persistence', float('nan')):.2f} | {time.time() - t0:.0f}s")
    if step == cfg["steps"]: break
    for gr in opt.param_groups: gr["lr"] = cfg["lr"] * min(1.0, (step + 1) / 10)
    lg, _, _ = model(ALL[TR]); loss = F.cross_entropy(lg, LAB[TR]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
# identity against computation: final words' rows and extreme sets against every earlier time
fin = snaps[cfg["steps"]]; fw = fin["words"].tolist(); ident = []
for s in sorted(snaps):
    sn = snaps[s]; cos = (sn["rows"][fin["words"]] * fin["rows"][fin["words"]]).sum(1).median().item()
    jac = []
    for i, w in enumerate(fw):
        j = (sn["words"] == w).nonzero()
        if j.numel(): a, b = set(sn["ext"][int(j[0])].tolist()), set(fin["ext"][i].tolist()); jac.append(len(a & b) / len(a | b))
    corr = []
    for i, w in enumerate(fw):
        j = (sn["words"] == w).nonzero()
        if j.numel(): corr.append(float(torch.corrcoef(torch.stack([sn["ratio"][:, int(j[0])], fin["ratio"][:, i]]))[0, 1]))
    ident.append(dict(step=s, row_cos_with_final=cos, ext_jaccard_with_final=med(jac) if jac else None, ratio_corr_with_final=med(corr) if corr else None, n_final_words_present=len(jac)))
gstep = next((r["step"] for r in log_rows if r["test_acc"] > 0.5), None); late = [r for r in log_rows if r["step"] >= cfg["steps"] // 2]
res = dict(variant=variant, cfg=cfg, log=log_rows, identity=ident, grok_step=gstep, final=log_rows[-1], late=dict(row_persistence=med([r["row_persistence"] for r in late]), association_persistence=med([r["association_persistence"] for r in late]), association_corr=med([r["association_corr"] for r in late]), over_floor_jaccard=med([r["over_floor_jaccard"] for r in late]), usage_corr=med([r["usage_corr"] for r in late]), membership_retention=med([r["membership_retention"] for r in late]), words_S=med([r["words_S"] for r in late]), top32_share=med([r["top32_share"] for r in late]), advantage=med([r["advantage"] for r in late])))
f = res["final"]; lt = res["late"]
summ = (f"{variant}: acc {f['train_acc']:.2f}/{f['test_acc']:.2f}, test > 0.5 from step {gstep}; final advantage {f['advantage']:.2f}, words' S {f['words_S']:.2f} (q10 {f['words_S_q10']:.2f}; rotated {f['rotated_S']:.2f}), rows over the floor {f['rows_over_floor']} of {DM} (rotated {f['rotated_over_floor']}), top-32 usage share {f['top32_share']:.2f}; "
        f"second half medians: row persistence {lt['row_persistence']:.3f}, association persistence {lt['association_persistence']:.2f} (ratio correlation {lt['association_corr']:.3f}, over-the-floor Jaccard {lt['over_floor_jaccard']:.2f}), membership retention {lt['membership_retention']:.2f}, usage corr {lt['usage_corr']:.2f}; identity at step {ident[len(ident) // 2]['step']}: row cos with final {ident[len(ident) // 2]['row_cos_with_final']:.2f}, extreme-set Jaccard with final {ident[len(ident) // 2]['ext_jaccard_with_final']}, ratio correlation with final {ident[len(ident) // 2]['ratio_corr_with_final']} | {time.time() - t0:.0f}s")
log(summ); record(f"e567_dyn_{variant}", res, summ)
