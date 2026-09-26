"""e504: the SAE bridge on a residual-stream SAE of another model. Everything in area 21 (e494-e502) is GPT-2 small
with two SAE families. The only public residual-stream SAEs in the Pythia line are EleutherAI's TopK autoencoders
for Pythia-160m (k = 32, 32768 latents, hookpoints at the output of each block), so the replication uses a sixth
model of the same shape as GPT-2 small (12 blocks, width 768), at the residual after block 6.
Setup: Pythia-160m, the SAE at layers.6; coordinates chosen by the autoencoder's own reconstruction of our states
(raw, centred over the model dimension, or layer-normalised); 16 x 512 evaluation tokens (the Pythia tokenizer's
evaluation ids), typical positions; the native dictionary up to block 6 and its rotation.
Provenance (as e497/e499): unexplained fraction at 1, 4, 16 native words and the top word's type, for features,
random directions, covariance-matched directions and states. Activation (as e501/e502, one family): for the 2000
most active live features, the AUC over positions of the top native MLP row's write size and of a random row; the
signed ledger over the first k MLP words with frozen coefficients, k = 1, 4, 8, 16; the same with the words after the
first replaced by random rows; the ledger without its top row; the projection of all MLP writes; the encoder ledger;
the MLP and non-MLP parts of the pre-activation; and the feature properties (frequency, mean activation, top-row
cosine, MLP words, top-row coefficient share, unexplained at 16, encoder-decoder cosine) with the Spearman of the
top-row AUC and of the eight-word increment against each, and both by frequency tercile.
Pre-registered (honest guesses):
- features' top native word is an MLP row for over 0.8, and random > covariance-matched > features in unexplained
  fraction at 16 words (0.6);
- the top row's AUC is above 0.6 and the random row's below 0.55 (0.6);
- the eight-word ledger's net increment over the top row is above 0.03 (0.5);
- the top-row AUC falls with frequency (Spearman below 0) (0.7).
Arguments: [block] (default 6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
try: MODELS["pythia160"] = ("EleutherAI/pythia-160m", "neox")
except NameError:
    import wdd_common as _w; _w.MODELS["pythia160"] = ("EleutherAI/pythia-160m", "neox")
name = "pythia160"; L = int(sys.argv[1]) if len(sys.argv) > 1 else 6; model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; DFF = arch.DFF; K = 16; NF = 2000; KS = (1, 2, 4, 8, 16)
REPO = "EleutherAI/sae-pythia-160m-32k"; HP = f"layers.{L}"
cfg = _json.load(open(hf_hub_download(REPO, f"{HP}/cfg.json"))); sd = load_file(hf_hub_download(REPO, f"{HP}/sae.safetensors")); TOPK = int(cfg.get("k", 32))
log(f"{REPO} {HP}: cfg {cfg}; tensors " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in sd.items()))
def pick(sub, nd, size=None):
    for k, v in sd.items():
        if sub in k and v.dim() == nd and (size is None or size in v.shape): return v.float().to(DEV)
    raise KeyError(sub)
enc = pick("enc", 2); enc = enc if enc.shape[1] == D else enc.T; dec = pick("dec", 2); dec = dec if dec.shape[1] == D else dec.T
NL = enc.shape[0]; benc = pick("enc", 1, NL); bdec = pick("dec", 1, D)
def encode(x):
    z = (x - bdec) @ enc.T + benc; top = z.topk(TOPK, dim=1); out = torch.zeros_like(z); out.scatter_(1, top.indices, torch.relu(top.values)); return out
dimc = lambda M: M - M.mean(-1, keepdim=True)
ids = eval_ids("pythia410")[:16, :512].to(DEV); acts = {b: [] for b in range(L + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(L + 1)]
try: X = block_states(model, arch, ids, [L], chunk=4)[L].reshape(-1, D)
finally: [h.remove() for h in hs]
keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]
Aact = {b: torch.cat(acts[b])[keep] for b in range(L + 1)}; del acts
Cw = torch.cat([Aact[b] * arch.wdir(b).float().norm(dim=-1)[None] for b in range(L + 1)], 1)
Mvec = sum(Aact[b] @ arch.wdir(b).float() for b in range(L + 1)); del Aact; Rvec = Xk - Mvec
rms = dimc(Xk).pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
cands = {"raw": (Xk, lambda M: M, torch.ones(N, 1, device=DEV)), "dimcentred": (dimc(Xk), dimc, torch.ones(N, 1, device=DEV)), "layernorm": (dimc(Xk) / rms, dimc, 1 / rms)}
fv = {}
for cn, (xin, _, _) in cands.items():
    z = encode(xin); fv[cn] = float((xin - (z @ dec + bdec)).pow(2).sum(1).mean() / (xin - xin.mean(0)).pow(2).sum(1).mean()); del z
coord = min(fv, key=fv.get); xin, tr, scale = cands[coord]; act = encode(xin); log(f"{name} block {L}: reconstruction fvu " + ", ".join(f"{k} {v:.2f}" for k, v in fv.items()) + f"; using {coord}")
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); Au = unitr(tr(A)); Ar = unitr(tr(rotate(A, seed=7))); del A
g = torch.Generator(device=DEV).manual_seed(0)
# provenance: features against random, covariance-matched directions and states
F = unitr(tr(dec)); Xc = tr(Xk - Xk.mean(0)); Lc = torch.linalg.cholesky(torch.cov(Xc.T.double(), correction=0) + 1e-6 * torch.eye(D, device=DEV, dtype=torch.float64)).float()
targets = {"features": F[torch.randperm(F.shape[0], generator=g, device=DEV)[:4096]], "random_directions": unitr(tr(torch.randn(4096, D, generator=g, device=DEV))), "covariance_directions": unitr(tr(torch.randn(4096, D, generator=g, device=DEV) @ Lc.T)), "states": unitr(Xc[torch.randperm(N, generator=g, device=DEV)[:4096]])}
prov = {}
for tn, T in targets.items():
    row = {}
    for kind, Dct in (("native", Au), ("rotated", Ar)):
        sel_, _, _ = omp(T, Dct, K, batch=1024, record_err=False); f_ = {}
        for k in (1, 4, 16):
            cof_, _ = refit(T, Dct, sel_[:, :k]); f_[k] = float((T - torch.einsum("nk,nkd->nd", cof_, Dct[sel_[:, :k]])).pow(2).sum(1).median())
        row[kind] = dict(fvu=f_, top_mlp=float((typ[sel_[:, 0]] == T_MLP).float().mean()))
    prov[tn] = row; log(f"{name} block {L} {tn}: unexplained by 1/4/16 native words {row['native']['fvu'][1]:.2f}/{row['native']['fvu'][4]:.2f}/{row['native']['fvu'][16]:.2f} (rotated {row['rotated']['fvu'][16]:.2f} at 16), top word MLP {row['native']['top_mlp']:.2f}")
# activation
def auc_rows(P, S):
    R = S.argsort(1).argsort(1).float() + 1; npos = P.float().sum(1); nneg = P.shape[1] - npos
    return ((R * P.float()).sum(1) - npos * (npos + 1) / 2) / (npos * nneg).clamp_min(1)
def spear(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-9))
freq_all = (act > 0).float().mean(0); live = torch.nonzero(freq_all > 0.01)[:, 0]; feats = live[act[:, live].mean(0).topk(min(NF, live.numel())).indices]
Fs = F[feats]; Fe = unitr(tr(enc))[feats]
sel, _, _ = omp(Fs, Au, K, batch=1024, record_err=False); cof, _ = refit(Fs, Au, sel); fvu16 = (Fs - torch.einsum("nk,nkd->nd", cof, Au[sel])).pow(2).sum(1)
sel_e, _, _ = omp(Fe, Au, K, batch=1024, record_err=False); cof_e, _ = refit(Fe, Au, sel_e)
def mlp_first(sel_, cof_):
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
auc["mlp_all_dec"] = auc_rows(P, (tr(Mvec) @ Fs.T).T); auc["mlp_all_enc"] = auc_rows(P, ((tr(Mvec) * scale) @ Fe.T).T)
auc["rest_enc"] = auc_rows(P, ((tr(Rvec) * scale) @ Fe.T).T); auc["state_enc"] = auc_rows(P, ((tr(Xk) * scale) @ Fe.T).T)
for k in (1, 8): auc[f"ledger_enc_{k}"] = auc_rows(P, ledger(col_e, cofm_e, k))
nact = (act > 0).float().sum(1); crowd = torch.stack([nact[P[i]].mean() for i in range(nf)]); mean_active = torch.stack([Fa[i][P[i]].mean() for i in range(nf)])
c2 = torch.where(col >= 0, cofm, torch.zeros_like(cofm)).pow(2)
props = dict(frequency=freq_all[feats], mean_active=mean_active, crowding=crowd, top_cos=(Fs * Au[selm[:, 0]]).sum(1).abs(), n_mlp=n_mlp.float(), conc=c2[:, 0] / c2.sum(1).clamp_min(1e-12), fvu16=fvu16, enc_dec_cos=(Fs * Fe).sum(1))
d8 = auc["ledger_8"] - auc["ledger_1"]; d8r = auc["random_rest_8"] - auc["ledger_1"]; net8 = d8 - d8r
q = props["frequency"].quantile(torch.tensor([1 / 3, 2 / 3], device=DEV)); lo = props["frequency"] <= q[0]; hi = props["frequency"] > q[1]
act_res = dict(n_features=nf, coordinates=coord, reconstruction_fvu=fv, auc_medians={k: float(v.median()) for k, v in auc.items()},
               increments=dict(d8=float(d8.median()), d8_rand=float(d8r.median()), net8=float(net8.median()), d16=float((auc["ledger_16"] - auc["ledger_1"]).median()), share_net8_over_0_05=float((net8 > 0.05).float().mean())),
               prop_medians={k: float(v.median()) for k, v in props.items()}, spearman_auc_top={k: spear(auc["top_abs"], v) for k, v in props.items()}, spearman_d8={k: spear(d8, v) for k, v in props.items()}, spearman_ledger8={k: spear(auc["ledger_8"], v) for k, v in props.items()},
               by_frequency_tercile={nm: dict(n=int(mm.sum()), top_abs=float(auc["top_abs"][mm].median()), ledger_1=float(auc["ledger_1"][mm].median()), ledger_8=float(auc["ledger_8"][mm].median()), d8=float(d8[mm].median()), net8=float(net8[mm].median())) for nm, mm in (("sparse", lo), ("dense", hi))})
res = dict(model=name, hf="EleutherAI/pythia-160m", sae=REPO, hookpoint=HP, level=L, k=TOPK, n_latents=NL, provenance=prov, activation=act_res)
pf = lambda t, k: prov[t]["native"]["fvu"][k]; am = act_res["auc_medians"]
res["checks"] = dict(top_mlp_over_0_8=prov["features"]["native"]["top_mlp"] > 0.8, ordering_holds=pf("random_directions", 16) > pf("covariance_directions", 16) > pf("features", 16),
                     top_row_over_0_6_random_under_0_55=am["top_abs"] > 0.6 and am["random_row"] < 0.55, increment_net8_over_0_03=act_res["increments"]["net8"] > 0.03, auc_top_falls_with_frequency=act_res["spearman_auc_top"]["frequency"] < 0)
t = act_res["by_frequency_tercile"]
summ = (f"{name} block {L}, {REPO} ({coord}, reconstruction fvu {fv[coord]:.2f}): provenance, unexplained by 1/4/16 native words (rotated at 16), top word MLP: " + "; ".join(f"{tn} {r['native']['fvu'][1]:.2f}/{r['native']['fvu'][4]:.2f}/{r['native']['fvu'][16]:.2f} ({r['rotated']['fvu'][16]:.2f}), {r['native']['top_mlp']:.2f}" for tn, r in prov.items())
        + f" || activation ({nf} features): AUC top row {am['top_abs']:.2f} (random row {am['random_row']:.2f}), signed ledger 1/4/8/16 {am['ledger_1']:.2f}/{am['ledger_4']:.2f}/{am['ledger_8']:.2f}/{am['ledger_16']:.2f}, top + random rest 8 {am['random_rest_8']:.2f}, rest only 8 {am['rest_only_8']:.2f}, all MLP writes {am['mlp_all_dec']:.2f}, non-MLP part {am['rest_enc']:.2f}, whole state {am['state_enc']:.2f}, encoder ledger 8 {am['ledger_enc_8']:.2f}; 8-word increment {act_res['increments']['d8']:+.3f} (net {act_res['increments']['net8']:+.3f}); frequency median {act_res['prop_medians']['frequency']:.3f}, top-row cos {act_res['prop_medians']['top_cos']:.2f}; Spearman of the top-row AUC with frequency {act_res['spearman_auc_top']['frequency']:+.2f}, top-row cos {act_res['spearman_auc_top']['top_cos']:+.2f}, mean activation {act_res['spearman_auc_top']['mean_active']:+.2f}; of the increment with frequency {act_res['spearman_d8']['frequency']:+.2f}; sparse/dense tercile top row {t['sparse']['top_abs']:.2f}/{t['dense']['top_abs']:.2f}, ledger-8 {t['sparse']['ledger_8']:.2f}/{t['dense']['ledger_8']:.2f}, increment {t['sparse']['d8']:+.3f}/{t['dense']['d8']:+.3f}"
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e504_pythiasae_{name}_L{L}", res, summ)
