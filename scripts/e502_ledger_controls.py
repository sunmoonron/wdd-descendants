"""e502: the ledger under the same controls, with the coefficients frozen. e499 found the eight-word ledger separates
the positions where a TopK feature fires at AUC 0.83 against 0.61 for the ReLU family (e496), and e501 found the
top-row difference (0.73 against 0.62) is the features' own sparsity. The ledger's larger gap was not controlled. Here
the ledger is put through the same controls, feature by feature on the same states, positions and dictionary, and the
question is sharpened: does adding the feature's other native rows carry information about its firing beyond its top
row, once the feature's frequency, activation strength and top-row alignment are held fixed?
Setup: as e501 (GPT-2 small, the residual after block 6; Bloom's ReLU SAE and OpenAI's TopK SAE; 16 x 512 tokens,
8176 typical positions; the 2000 most active live features per family; 16 native words per feature with least-squares
coefficients fixed by the feature's direction alone, never by its firing).
Scores per feature over positions, its MLP words in OMP order: the top row's write size (as e501) and a random row of
the same block; the signed ledger over its first k MLP words, k = 1, 2, 4, 8, 16 (k = 1 is the signed top row, so the
increments are between nested sums); the same ledger with the words after the first replaced by random rows of the
same blocks (frozen coefficients); the ledger without its top row; the projection of the summed MLP writes of blocks
0-6 onto the feature (every row, true coefficients); the encoder direction's ledger (the firing is decided by the
encoder, the provenance was measured on the decoder); and the exact decomposition of the pre-activation along the
encoder direction into the MLP part and the non-MLP part of the state (the whole state gives AUC 1 for ReLU).
Per feature: frequency, mean activation when active, crowding, top-row |cosine|, number of MLP words among the 16,
the top row's share of the ledger's squared coefficients, unexplained fraction at 16 words, encoder-decoder cosine.
Analysis: medians per family; Spearman within family of the eight-word increment with each property; the increment
and the eight-word AUC by shared frequency decile and by frequency tercile; the family effect on the eight-word AUC
and on the increment, raw, given frequency, and given all properties.
Pre-registered (honest guesses):
- the eight-word ledger's increment over its top row survives the controls in both families: median net increment
  (over the random-rows control) above 0.03 in both, and the family effect on the increment given frequency below
  0.2 (0.5);
- the increment is larger for sparser features in both families (Spearman with frequency below -0.1) (0.5);
- the family effect on the eight-word AUC itself vanishes given all properties (partial correlation below 0.2) (0.6);
- the words after the first replaced by random rows lower the AUC below the top row alone in both families (0.8);
- the projection of all MLP writes is above the sixteen-word ledger in both families (0.6);
- the encoder ledger beats the decoder ledger on the ReLU family (0.6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
name = "gpt2"; model, tok, fam = load_model(name); arch = Arch(model, fam); L = 6; D = arch.D; DFF = arch.DFF; K = 16; NF = 2000; TOPK = 32; KS = (1, 2, 4, 8, 16)
dimc = lambda M: M - M.mean(-1, keepdim=True)
ids = eval_ids(name)[:16, :512].to(DEV); acts = {b: [] for b in range(L + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(L + 1)]
try: X = block_states(model, arch, ids, [L], chunk=4)[L].reshape(-1, D)
finally: [h.remove() for h in hs]
keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]
Aact = {b: torch.cat(acts[b])[keep] for b in range(L + 1)}; del acts
Cw = torch.cat([Aact[b] * arch.wdir(b).float().norm(dim=-1)[None] for b in range(L + 1)], 1)
Mvec = sum(Aact[b] @ arch.wdir(b).float() for b in range(L + 1)); del Aact                    # the summed MLP writes of blocks 0-6 (no biases) [N, D]
Rvec = Xk - Mvec                                                                                # the rest of the state: embeddings, attention, biases
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); Au = unitr(dimc(A)); del A
sae_b = load_file(hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"blocks.{L + 1}.hook_resid_pre/sae_weights.safetensors"))
Wd_b = sae_b["W_dec"].float().to(DEV); Wd_b = Wd_b if Wd_b.shape[1] == D else Wd_b.T; We_b = sae_b["W_enc"].float().to(DEV); We_b = We_b if We_b.shape[0] == D else We_b.T; be_b = sae_b["b_enc"].float().to(DEV); bd_b = sae_b["b_dec"].float().to(DEV)
sd = torch.load(f"/workspace/wdd/cache/oai_sae/v5_32k_{L}.pt", map_location="cpu", weights_only=False); sd = {k: v for k, v in sd.items() if hasattr(v, "shape")}
enc_o = sd["encoder.weight"].float().to(DEV); enc_o = enc_o if enc_o.shape[1] == D else enc_o.T; dec_o = sd["decoder.weight"].float().to(DEV); dec_o = dec_o.T if dec_o.shape[0] == D else dec_o; pre_o = sd["pre_bias"].float().to(DEV); lb_o = sd["latent_bias"].float().to(DEV)
act_b = torch.relu((dimc(Xk) - bd_b) @ We_b + be_b); scale_b = torch.ones(N, 1, device=DEV)
rms = dimc(Xk).pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6); z = (dimc(Xk) / rms - pre_o) @ enc_o.T + lb_o; top = z.topk(TOPK, dim=1); act_o = torch.zeros_like(z); act_o.scatter_(1, top.indices, torch.relu(top.values)); del z; scale_o = 1 / rms
g = torch.Generator(device=DEV).manual_seed(0)
def auc_rows(P, S):
    """vectorised AUC per row: P [n, N] bool (positives), S [n, N] scores"""
    R = S.argsort(1).argsort(1).float() + 1; npos = P.float().sum(1); nneg = P.shape[1] - npos
    return ((R * P.float()).sum(1) - npos * (npos + 1) / 2) / (npos * nneg).clamp_min(1)
def spear(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-9))
def resid_on(a, cols):
    cols = [c - c.mean() for c in cols if float(c.std()) > 0]; Xd = torch.stack([torch.ones_like(a)] + cols, 1).double(); y = a.double(); beta = torch.linalg.pinv(Xd.T @ Xd) @ (Xd.T @ y); return (y - Xd @ beta).float()
rank = lambda v: v.argsort().argsort().float()
def partial(y, x, controls): return float(torch.corrcoef(torch.stack([resid_on(rank(y), controls), resid_on(rank(x), controls)]))[0, 1])
props = {}
for famn, Wd, We, act, scale in (("relu_bloom", Wd_b, We_b.T, act_b, scale_b), ("topk_openai", dec_o, enc_o, act_o, scale_o)):
    freq_all = (act > 0).float().mean(0); live = torch.nonzero(freq_all > 0.01)[:, 0]; feats = live[act[:, live].mean(0).topk(min(NF, live.numel())).indices]
    F = unitr(dimc(Wd)); Fs = F[feats]; Fe = unitr(dimc(We))[feats]                                                          # decoder and encoder directions (dim-centred, unit)
    sel, _, _ = omp(Fs, Au, K, batch=1024, record_err=False); cof, _ = refit(Fs, Au, sel); fvu16 = (Fs - torch.einsum("nk,nkd->nd", cof, Au[sel])).pow(2).sum(1)
    sel_e, _, _ = omp(Fe, Au, K, batch=1024, record_err=False); cof_e, _ = refit(Fe, Au, sel_e)
    def mlp_first(sel_, cof_):
        """each feature's MLP words first, in OMP order: column in Cw (-1 for none), coefficient, atom"""
        col_ = torch.where(typ[sel_] == T_MLP, blk[sel_].long() * DFF + idx[sel_].long(), torch.full_like(sel_, -1)); order = torch.argsort((col_ < 0).float(), dim=1, stable=True)
        return col_.gather(1, order), cof_.gather(1, order), sel_.gather(1, order)
    col, cofm, selm = mlp_first(sel, cof); col_e, cofm_e, _ = mlp_first(sel_e, cof_e)
    n_mlp = (col >= 0).sum(1); fi = torch.nonzero((n_mlp >= 1) & ((col_e >= 0).sum(1) >= 1))[:, 0]; nf = fi.numel()
    col, cofm, selm, col_e, cofm_e, Fs, Fe, fvu16, n_mlp, feats = col[fi], cofm[fi], selm[fi], col_e[fi], cofm_e[fi], Fs[fi], Fe[fi], fvu16[fi], n_mlp[fi], feats[fi]
    Fa = act[:, feats].T; P = Fa > 0
    def ledger(cols, cofs, k, skip_first=False, random_rest=False):
        out = torch.zeros(nf, N, device=DEV)
        for j in range(1 if skip_first else 0, k):
            m = cols[:, j] >= 0
            if not bool(m.any()): continue
            c_ = cols[m, j]
            if random_rest and j >= 1: c_ = (c_ // DFF) * DFF + torch.randint(0, DFF, (int(m.sum()),), generator=g, device=DEV)
            out[m] += cofs[m, j][:, None] * Cw[:, c_].T
        return out
    auc = dict(top_abs=auc_rows(P, Cw[:, col[:, 0]].T.abs()), random_row=auc_rows(P, Cw[:, (col[:, 0] // DFF) * DFF + torch.randint(0, DFF, (nf,), generator=g, device=DEV)].T.abs()))
    for k in KS: auc[f"ledger_{k}"] = auc_rows(P, ledger(col, cofm, k))
    for k in (4, 8, 16): auc[f"random_rest_{k}"] = auc_rows(P, ledger(col, cofm, k, random_rest=True))
    for k in (8, 16): auc[f"rest_only_{k}"] = auc_rows(P, ledger(col, cofm, k, skip_first=True))
    auc["mlp_all_dec"] = auc_rows(P, (dimc(Mvec) @ Fs.T).T); auc["mlp_all_enc"] = auc_rows(P, ((dimc(Mvec) * scale) @ Fe.T).T)
    auc["rest_enc"] = auc_rows(P, ((dimc(Rvec) * scale) @ Fe.T).T); auc["state_enc"] = auc_rows(P, ((dimc(Xk) * scale) @ Fe.T).T)
    for k in (1, 8): auc[f"ledger_enc_{k}"] = auc_rows(P, ledger(col_e, cofm_e, k))
    nact = (act > 0).float().sum(1); crowd = torch.stack([nact[P[i]].mean() for i in range(nf)]); mean_active = torch.stack([Fa[i][P[i]].mean() for i in range(nf)])
    c2 = torch.where(col >= 0, cofm, torch.zeros_like(cofm)).pow(2)
    props[famn] = dict(n=nf, auc=auc, frequency=freq_all[feats], mean_active=mean_active, crowding=crowd, top_cos=(Fs * Au[selm[:, 0]]).sum(1).abs(), n_mlp=n_mlp.float(), conc=c2[:, 0] / c2.sum(1).clamp_min(1e-12), fvu16=fvu16, enc_dec_cos=(Fs * Fe).sum(1))
    log(f"{name} {famn}: {nf} features; AUC top row {float(auc['top_abs'].median()):.3f} (random row {float(auc['random_row'].median()):.3f}), signed ledger 1/2/4/8/16 " + "/".join(f"{float(auc[f'ledger_{k}'].median()):.3f}" for k in KS)
        + f", top + random rest 4/8/16 {float(auc['random_rest_4'].median()):.3f}/{float(auc['random_rest_8'].median()):.3f}/{float(auc['random_rest_16'].median()):.3f}, rest only 8/16 {float(auc['rest_only_8'].median()):.3f}/{float(auc['rest_only_16'].median()):.3f}, all MLP writes dec/enc {float(auc['mlp_all_dec'].median()):.3f}/{float(auc['mlp_all_enc'].median()):.3f}, non-MLP part {float(auc['rest_enc'].median()):.3f}, whole state {float(auc['state_enc'].median()):.3f}, encoder ledger 1/8 {float(auc['ledger_enc_1'].median()):.3f}/{float(auc['ledger_enc_8'].median()):.3f}; MLP words {float(n_mlp.float().median()):.0f}, top-row share of coefficients {float(props[famn]['conc'].median()):.2f}, encoder-decoder cos {float(props[famn]['enc_dec_cos'].median()):.2f}")
PROPS = ["frequency", "mean_active", "crowding", "top_cos", "n_mlp", "conc", "fvu16", "enc_dec_cos"]
res = dict(model=name, level=L, families={})
for famn, p in props.items():
    a = p["auc"]; d8 = a["ledger_8"] - a["ledger_1"]; d8r = a["random_rest_8"] - a["ledger_1"]; net8 = d8 - d8r; d16 = a["ledger_16"] - a["ledger_1"]; head8 = d8 / (1 - a["ledger_1"]).clamp_min(0.05)
    p.update(d8=d8, d8_rand=d8r, net8=net8, d16=d16, head8=head8)
    q = p["frequency"].quantile(torch.tensor([1 / 3, 2 / 3], device=DEV)); lo = p["frequency"] <= q[0]; hi = p["frequency"] > q[1]
    res["families"][famn] = dict(n=p["n"], auc_medians={k: float(v.median()) for k, v in a.items()},
                                 increments=dict(d8=float(d8.median()), d8_rand=float(d8r.median()), net8=float(net8.median()), d16=float(d16.median()), head8=float(head8.median()), share_d8_over_0_05=float((d8 > 0.05).float().mean()), share_net8_over_0_05=float((net8 > 0.05).float().mean())),
                                 rest_only_8_given_2_words=float(a["rest_only_8"][p["n_mlp"] >= 2].median()) if bool((p["n_mlp"] >= 2).any()) else None,
                                 prop_medians={k: float(p[k].median()) for k in PROPS}, spearman_d8={k: spear(d8, p[k]) for k in PROPS}, spearman_net8={k: spear(net8, p[k]) for k in PROPS},
                                 spearman_ledger8={k: spear(a["ledger_8"], p[k]) for k in PROPS}, spearman_ledger1={k: spear(a["ledger_1"], p[k]) for k in PROPS},
                                 by_frequency_tercile={nm: dict(n=int(mm.sum()), ledger_1=float(a["ledger_1"][mm].median()), ledger_8=float(a["ledger_8"][mm].median()), d8=float(d8[mm].median()), net8=float(net8[mm].median())) for nm, mm in (("sparse", lo), ("dense", hi))})
    r = res["families"][famn]
    log(f"{name} {famn}: increments over the signed top row: 8 words {r['increments']['d8']:+.3f} (random rest {r['increments']['d8_rand']:+.3f}, net {r['increments']['net8']:+.3f}, share of features net above 0.05 {r['increments']['share_net8_over_0_05']:.2f}), 16 words {r['increments']['d16']:+.3f}; Spearman of the 8-word increment with " + ", ".join(f"{k} {v:+.2f}" for k, v in r["spearman_d8"].items())
        + f"; by frequency tercile sparse/dense: ledger-1 {r['by_frequency_tercile']['sparse']['ledger_1']:.2f}/{r['by_frequency_tercile']['dense']['ledger_1']:.2f}, ledger-8 {r['by_frequency_tercile']['sparse']['ledger_8']:.2f}/{r['by_frequency_tercile']['dense']['ledger_8']:.2f}, increment {r['by_frequency_tercile']['sparse']['d8']:+.3f}/{r['by_frequency_tercile']['dense']['d8']:+.3f}")
# across families: shared frequency deciles, and the family effect given the controls
fams = ("relu_bloom", "topk_openai"); cat = lambda k: torch.cat([props[f][k] if k in props[f] else props[f]["auc"][k] for f in fams])
allf = cat("frequency"); edges = allf.quantile(torch.linspace(0, 1, 11, device=DEV)); res["by_frequency_decile"] = {}
for f_ in fams:
    p = props[f_]; d = torch.bucketize(p["frequency"], edges[1:-1])
    res["by_frequency_decile"][f_] = [dict(n=int((d == q).sum()), ledger_1=float(p["auc"]["ledger_1"][d == q].median()), ledger_8=float(p["auc"]["ledger_8"][d == q].median()), d8=float(p["d8"][d == q].median())) if bool((d == q).any()) else dict(n=0, ledger_1=None, ledger_8=None, d8=None) for q in range(10)]
lab_f = torch.cat([torch.zeros(props[fams[0]]["n"], device=DEV), torch.ones(props[fams[1]]["n"], device=DEV)]); ctrl = {k: rank(cat(k)) for k in ("frequency", "top_cos", "mean_active", "n_mlp", "conc", "fvu16")}
res["family_effect"] = {k: dict(raw=spear(cat(k), lab_f), given_frequency=partial(cat(k), lab_f, [ctrl["frequency"]]), given_all=partial(cat(k), lab_f, list(ctrl.values()))) for k in ("ledger_8", "ledger_1", "d8", "net8")}
Fm = res["families"]; fe = res["family_effect"]
res["checks"] = dict(increment_survives=all(Fm[f]["increments"]["net8"] > 0.03 for f in fams) and abs(fe["d8"]["given_frequency"]) < 0.2,
                     increment_larger_when_sparser=all(Fm[f]["spearman_d8"]["frequency"] < -0.1 for f in fams),
                     family_effect_on_ledger8_vanishes=abs(fe["ledger_8"]["given_all"]) < 0.2,
                     random_rest_hurts=all(Fm[f]["increments"]["d8_rand"] < 0 for f in fams),
                     all_mlp_above_ledger16=all(Fm[f]["auc_medians"]["mlp_all_dec"] > Fm[f]["auc_medians"]["ledger_16"] for f in fams),
                     encoder_ledger_better_relu=Fm["relu_bloom"]["auc_medians"]["ledger_enc_8"] > Fm["relu_bloom"]["auc_medians"]["ledger_8"])
dec = res["by_frequency_decile"]; fmt = lambda v: "nan" if v is None else f"{v:.2f}"
summ = (f"{name} L{L}: " + " || ".join(f"{f_} ({v['n']} features): AUC top row {v['auc_medians']['top_abs']:.2f} (random row {v['auc_medians']['random_row']:.2f}), signed ledger 1/4/8/16 {v['auc_medians']['ledger_1']:.2f}/{v['auc_medians']['ledger_4']:.2f}/{v['auc_medians']['ledger_8']:.2f}/{v['auc_medians']['ledger_16']:.2f}, top + random rest 8 {v['auc_medians']['random_rest_8']:.2f}, rest only 8 {v['auc_medians']['rest_only_8']:.2f}, all MLP writes {v['auc_medians']['mlp_all_dec']:.2f}, non-MLP part {v['auc_medians']['rest_enc']:.2f}, encoder ledger 8 {v['auc_medians']['ledger_enc_8']:.2f}; 8-word increment {v['increments']['d8']:+.3f} (net of random rest {v['increments']['net8']:+.3f}); Spearman of the increment with frequency {v['spearman_d8']['frequency']:+.2f}, top-row cos {v['spearman_d8']['top_cos']:+.2f}, mean activation {v['spearman_d8']['mean_active']:+.2f}, MLP words {v['spearman_d8']['n_mlp']:+.2f}, top-row share {v['spearman_d8']['conc']:+.2f}; sparse/dense tercile ledger-8 {v['by_frequency_tercile']['sparse']['ledger_8']:.2f}/{v['by_frequency_tercile']['dense']['ledger_8']:.2f}, increment {v['by_frequency_tercile']['sparse']['d8']:+.3f}/{v['by_frequency_tercile']['dense']['d8']:+.3f}" for f_, v in Fm.items())
        + " || 8-word ledger by shared frequency decile, ReLU / TopK: " + " ".join(f"{q}: {fmt(dec[fams[0]][q]['ledger_8'])}/{fmt(dec[fams[1]][q]['ledger_8'])}" for q in range(10))
        + " || 8-word increment by decile: " + " ".join(f"{q}: {fmt(dec[fams[0]][q]['d8'])}/{fmt(dec[fams[1]][q]['d8'])}" for q in range(10))
        + " || family effect (TopK over ReLU) raw / given frequency / given all: " + ", ".join(f"{k} {v['raw']:+.2f}/{v['given_frequency']:+.2f}/{v['given_all']:+.2f}" for k, v in fe.items()) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e502_ledgercontrols_{name}", res, summ)
