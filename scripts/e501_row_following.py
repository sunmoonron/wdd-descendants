"""e501: what predicts whether a feature follows its writer. On Bloom's ReLU SAE a feature's top native MLP row
separates the positions where the feature fires at AUC 0.62 (e496); on OpenAI's TopK SAE at 0.73, with the eight-word
ledger at 0.83 (e499). Is the difference the features' own sparsity, their geometry, or the family? Both families are
measured on the same states, positions and dictionary, feature by feature, and the row-following AUC is set against
the feature's properties within each family and across them.
Setup: GPT-2 small, the residual after block 6 (Bloom's SAE at the input of block 7, 24576 ReLU latents, centred
coordinates; OpenAI's v5_32k TopK autoencoder for the residual after block 6, layer-normalised inputs), 16 x 512
tokens, typical positions; the 2000 most active live features of each family; 16 native words per feature.
Per feature: the AUC of its top native MLP row's write size and of its eight-word ledger for the positions where it
fires; its frequency (share of positions active); its mean activation when active; the crowding at its positions (the
mean number of features active there; 32 by construction for TopK); the norm of its decoder row; its coherence (the
largest |cosine| with another feature of the same SAE); the unexplained fraction at 1 and 16 native words; the
|cosine| with its top native row; the block of that row.
Analysis: within each family, Spearman of the AUC with each property; across families, the AUC by shared frequency
deciles; and the partial rank correlation of the family label with the AUC given frequency, top-row cosine,
unexplained fraction, coherence and crowding.
Pre-registered (honest guesses):
- within each family the AUC rises with the top-row cosine (Spearman above 0.3) (0.6);
- within each family the AUC falls with frequency (0.5);
- the TopK advantage survives the controls (partial correlation of the family with the AUC above 0.2) (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
name = "gpt2"; model, tok, fam = load_model(name); arch = Arch(model, fam); L = 6; D = arch.D; DFF = arch.DFF; K = 16; NF = 2000; TOPK = 32
dimc = lambda M: M - M.mean(-1, keepdim=True)
ids = eval_ids(name)[:16, :512].to(DEV); acts = {b: [] for b in range(L + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(L + 1)]
try: X = block_states(model, arch, ids, [L], chunk=4)[L].reshape(-1, D)
finally: [h.remove() for h in hs]
keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]
Cw = torch.cat([torch.cat(acts[b])[keep] * arch.wdir(b).float().norm(dim=-1)[None] for b in range(L + 1)], 1); del acts
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); Au = unitr(dimc(A)); del A
sae_b = load_file(hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"blocks.{L + 1}.hook_resid_pre/sae_weights.safetensors"))
Wd_b = sae_b["W_dec"].float().to(DEV); Wd_b = Wd_b if Wd_b.shape[1] == D else Wd_b.T; We_b = sae_b["W_enc"].float().to(DEV); We_b = We_b if We_b.shape[0] == D else We_b.T; be_b = sae_b["b_enc"].float().to(DEV); bd_b = sae_b["b_dec"].float().to(DEV)
sd = torch.load(f"/workspace/wdd/cache/oai_sae/v5_32k_{L}.pt", map_location="cpu", weights_only=False); sd = {k: v for k, v in sd.items() if hasattr(v, "shape")}
enc_o = sd["encoder.weight"].float().to(DEV); enc_o = enc_o if enc_o.shape[1] == D else enc_o.T; dec_o = sd["decoder.weight"].float().to(DEV); dec_o = dec_o.T if dec_o.shape[0] == D else dec_o; pre_o = sd["pre_bias"].float().to(DEV); lb_o = sd["latent_bias"].float().to(DEV)
xin_b = dimc(Xk); act_b = torch.relu((xin_b - bd_b) @ We_b + be_b)
xin_o = dimc(Xk) / dimc(Xk).pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6); z = (xin_o - pre_o) @ enc_o.T + lb_o; top = z.topk(TOPK, dim=1); act_o = torch.zeros_like(z); act_o.scatter_(1, top.indices, torch.relu(top.values)); del z
def auc_rows(P, S):
    """vectorised AUC per row: P [n, N] bool (positives), S [n, N] scores"""
    R = S.argsort(1).argsort(1).float() + 1; npos = P.float().sum(1); nneg = P.shape[1] - npos
    return ((R * P.float()).sum(1) - npos * (npos + 1) / 2) / (npos * nneg).clamp_min(1)
def spear(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-9))
def resid_on(a, cols):
    cols = [c - c.mean() for c in cols if float(c.std()) > 0]; Xd = torch.stack([torch.ones_like(a)] + cols, 1).double(); y = a.double(); beta = torch.linalg.pinv(Xd.T @ Xd) @ (Xd.T @ y); return (y - Xd @ beta).float()
feats_all, props = {}, {}
for famn, Wd, act in (("relu_bloom", Wd_b, act_b), ("topk_openai", dec_o, act_o)):
    freq_all = (act > 0).float().mean(0); live = torch.nonzero(freq_all > 0.01)[:, 0]; feats = live[act[:, live].mean(0).topk(min(NF, live.numel())).indices]; n_ = feats.numel()
    F = unitr(dimc(Wd)); Fs = F[feats]; sel, _, _ = omp(Fs, Au, K, batch=1024, record_err=False)
    fvu = {}
    for k in (1, K):
        cof, _ = refit(Fs, Au, sel[:, :k]); fvu[k] = (Fs - torch.einsum("nk,nkd->nd", cof, Au[sel[:, :k]])).pow(2).sum(1)
    cof, _ = refit(Fs, Au, sel); col = torch.where(typ[sel] == T_MLP, blk[sel].long() * DFF + idx[sel].long(), torch.full_like(sel, -1))
    first = torch.full((n_,), -1, device=DEV, dtype=torch.long); firstw = torch.full((n_,), -1, device=DEV, dtype=torch.long)
    for i in range(n_):
        ms = [j for j in range(K) if col[i, j] >= 0]
        if ms: first[i] = col[i, ms[0]]; firstw[i] = sel[i, ms[0]]
    has = first >= 0; fi = torch.nonzero(has)[:, 0]
    Fa = act[:, feats[fi]].T; P = Fa > 0
    led = torch.zeros(fi.numel(), N, device=DEV)
    for j in range(8):
        m = col[fi, j] >= 0; led[m] += cof[fi][m, j][:, None] * Cw[:, col[fi][m, j]].T
    auc_top = auc_rows(P, Cw[:, first[fi]].T.abs()); auc_led = auc_rows(P, led)
    coh = torch.cat([(Fs[fi][s:s + 256] @ F.T).abs().topk(2, dim=1).values[:, 1] for s in range(0, fi.numel(), 256)])       # the largest |cos| with another feature
    nact = (act > 0).float().sum(1)                                                                                             # features active per position
    crowd = torch.stack([nact[P[i]].mean() for i in range(fi.numel())])
    props[famn] = dict(n=int(fi.numel()), auc_top=auc_top, auc_ledger8=auc_led, frequency=freq_all[feats[fi]], mean_active=torch.stack([Fa[i][P[i]].mean() for i in range(fi.numel())]), crowding=crowd,
                       decoder_norm=Wd[feats[fi]].norm(dim=1), coherence=coh, fvu1=fvu[1][fi], fvu16=fvu[K][fi], top_cos=(Fs[fi] * Au[firstw[fi]]).sum(1).abs(), top_block=blk[firstw[fi]].float())
    log(f"{name} {famn}: {fi.numel()} features; AUC top row median {float(auc_top.median()):.3f}, ledger-8 {float(auc_led.median()):.3f}; frequency median {float(props[famn]['frequency'].median()):.3f}, crowding {float(crowd.median()):.1f}, top-row cos {float(props[famn]['top_cos'].median()):.2f}, unexplained k1/k16 {float(fvu[1][fi].median()):.2f}/{float(fvu[K][fi].median()):.2f}, coherence {float(coh.median()):.2f}")
res = dict(model=name, level=L, families={})
PROPS = ["frequency", "mean_active", "crowding", "decoder_norm", "coherence", "fvu1", "fvu16", "top_cos", "top_block"]
for famn, p in props.items():
    res["families"][famn] = dict(n=p["n"], auc_top_median=float(p["auc_top"].median()), auc_ledger8_median=float(p["auc_ledger8"].median()), medians={k: float(p[k].median()) for k in PROPS},
                                 spearman_auc_top={k: spear(p["auc_top"], p[k]) for k in PROPS}, spearman_auc_ledger8={k: spear(p["auc_ledger8"], p[k]) for k in PROPS})
    log(f"{name} {famn}: Spearman of the top-row AUC with " + ", ".join(f"{k} {v:+.2f}" for k, v in res["families"][famn]["spearman_auc_top"].items()))
# across families: shared frequency deciles, and the family effect given the controls
allf = torch.cat([props[f]["frequency"] for f in props]); edges = allf.quantile(torch.linspace(0, 1, 11, device=DEV)); res["by_frequency_decile"] = {}
for f_, p in props.items():
    d = torch.bucketize(p["frequency"], edges[1:-1]); res["by_frequency_decile"][f_] = [dict(n=int((d == q).sum()), auc_top=float(p["auc_top"][d == q].median()) if (d == q).any() else None) for q in range(10)]
lab_f = torch.cat([torch.zeros(props["relu_bloom"]["n"], device=DEV), torch.ones(props["topk_openai"]["n"], device=DEV)]); cat = lambda k: torch.cat([props[f][k] for f in ("relu_bloom", "topk_openai")])
rank = lambda v: v.argsort().argsort().float()
y = rank(cat("auc_top")); controls = [rank(cat(k)) for k in ("frequency", "top_cos", "fvu16", "coherence", "crowding")]
res["family_effect"] = dict(raw=spear(cat("auc_top"), lab_f), given_frequency=float(torch.corrcoef(torch.stack([resid_on(y, [controls[0]]), resid_on(rank(lab_f), [controls[0]])]))[0, 1]),
                            given_all=float(torch.corrcoef(torch.stack([resid_on(y, controls), resid_on(rank(lab_f), controls)]))[0, 1]))
Fm = res["families"]
res["checks"] = dict(auc_rises_with_top_cos=all(Fm[f]["spearman_auc_top"]["top_cos"] > 0.3 for f in Fm), auc_falls_with_frequency=all(Fm[f]["spearman_auc_top"]["frequency"] < 0 for f in Fm), family_effect_survives=res["family_effect"]["given_all"] > 0.2)
dec = res["by_frequency_decile"]
summ = (f"{name} L{L}: " + " || ".join(f"{f_} ({v['n']} features): AUC top row {v['auc_top_median']:.2f}, ledger-8 {v['auc_ledger8_median']:.2f}; medians frequency {v['medians']['frequency']:.3f}, crowding {v['medians']['crowding']:.0f}, top-row cos {v['medians']['top_cos']:.2f}, unexplained k1/16 {v['medians']['fvu1']:.2f}/{v['medians']['fvu16']:.2f}, coherence {v['medians']['coherence']:.2f}; Spearman of the top-row AUC with frequency {v['spearman_auc_top']['frequency']:+.2f}, top-row cos {v['spearman_auc_top']['top_cos']:+.2f}, unexplained k16 {v['spearman_auc_top']['fvu16']:+.2f}, coherence {v['spearman_auc_top']['coherence']:+.2f}, crowding {v['spearman_auc_top']['crowding']:+.2f}, mean activation {v['spearman_auc_top']['mean_active']:+.2f}" for f_, v in Fm.items())
        + " || top-row AUC by shared frequency decile, ReLU / TopK: " + " ".join(f"{q}: {(dec['relu_bloom'][q]['auc_top'] if dec['relu_bloom'][q]['auc_top'] is not None else float('nan')):.2f}/{(dec['topk_openai'][q]['auc_top'] if dec['topk_openai'][q]['auc_top'] is not None else float('nan')):.2f}" for q in range(10))
        + f" || family effect on the AUC (TopK over ReLU): raw {res['family_effect']['raw']:+.2f}, given frequency {res['family_effect']['given_frequency']:+.2f}, given frequency, top-row cos, unexplained, coherence and crowding {res['family_effect']['given_all']:+.2f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e501_rowfollowing_{name}", res, summ)
