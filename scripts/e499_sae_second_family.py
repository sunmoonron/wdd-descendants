"""e499: the SAE bridge on an independently trained SAE family. e494-e497 used Bloom's residual SAEs (ReLU, 24576
latents). OpenAI's GPT-2 small autoencoders (Gao et al. 2024; TopK with k = 32, 32768 latents, different data and
training) are a second family. If the bridge is a property of the model, the same ordering must hold:
states about equal to features, both far sparser in native words than covariance-matched or random directions, the top
word an MLP row at the states' rate, and the rows' activity predicting a feature's firing.
Setup: GPT-2 small, the v5_32k autoencoders for the residual after the MLP of blocks 6 and 2 (our states after those
blocks); coordinates chosen by the autoencoder's own reconstruction of our states (raw or centred over the model dimension,
each unscaled or scaled so that the mean squared norm is the model dimension, as the authors normalised their inputs); the native dictionary up to the block and its rotation. Targets as e497 (features, random directions,
covariance-matched directions, states, rotated features): unexplained fraction at k = 1, 4, 16 and the top word's
type. At block 6 also e496's test: for the 2000 most active features (a feature fires when it is among the position's
32 kept latents), the AUC of its top native MLP row's write size, of a random row, and of the ledger prediction from
its top 1, 4, 8 and 16 words, over 16 x 512 typical positions.
Pre-registered (honest guesses):
- the ordering random > covariance > features about equal to states holds at block 6 (0.6);
- features' top word is an MLP row for over 0.8 (0.6);
- the 8-word ledger's AUC is at least 0.65 and the random row's below 0.55 (0.5)."""
import sys, os, json as _json, urllib.request; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = "gpt2"; model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; DFF = arch.DFF; KS = [1, 4, 16]; LS = [2, 6]; TOPK = 32
dimc = lambda M: M - M.mean(-1, keepdim=True)
def load_oai(layer):
    path = f"/workspace/wdd/cache/oai_sae/v5_32k_{layer}.pt"
    if not os.path.exists(path): urllib.request.urlretrieve(f"https://openaipublic.blob.core.windows.net/sparse-autoencoder/gpt2-small/resid_post_mlp_v5_32k/autoencoders/{layer}.pt", path)
    sd = torch.load(path, map_location="cpu", weights_only=False); sd = {k: v for k, v in sd.items() if hasattr(v, "shape")}
    enc = sd["encoder.weight"].float().to(DEV); dec = sd["decoder.weight"].float().to(DEV); pre = sd["pre_bias"].float().to(DEV); lb = sd["latent_bias"].float().to(DEV)
    enc = enc if enc.shape[1] == D else enc.T; dec = dec.T if dec.shape[0] == D else dec                       # enc [n, D], dec [n, D]
    return enc, dec, pre, lb
def encode(x, enc, pre, lb):
    z = (x - pre) @ enc.T + lb; top = z.topk(TOPK, dim=1); out = torch.zeros_like(z); out.scatter_(1, top.indices, torch.relu(top.values)); return out
ids = eval_ids(name)[:16, :512].to(DEV); S_ = block_states(model, arch, ids, LS, chunk=4)
res = dict(model=name, family="openai_v5_32k_topk", layers=LS, by_layer={})
for L in LS:
    enc, dec, pre, lb = load_oai(L); X = S_[L].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; g = torch.Generator(device=DEV).manual_seed(0)
    # which coordinates does this autoencoder expect? the one in which it reconstructs our states better
    fv = {}; cands = {}
    for cn, base in (("raw", Xk), ("dimcentred", dimc(Xk))):
        for sn, sc in (("", 1.0), ("_unitnorm", float((D / base.pow(2).sum(1).mean()).sqrt())), ("_layernorm", None)):   # v3: the sae_lens copy of these autoencoders says normalize_activations = layer_norm (per position, over the model dimension)
            xin = base * sc if sc is not None else dimc(base) / dimc(base).pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            z = encode(xin, enc, pre, lb); rec = z @ dec + pre; fv[cn + sn] = float((xin - rec).pow(2).sum(1).mean() / (xin - xin.mean(0)).pow(2).sum(1).mean()); cands[cn + sn] = xin
    coord = min(fv, key=fv.get); xin = cands[coord]; tr = (lambda M: M) if coord.startswith("raw") else dimc
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); Au = unitr(tr(A)); Ar = unitr(tr(rotate(A, seed=7))); del A
    F = unitr(tr(dec)); Xc = tr(Xk - Xk.mean(0)); Lc = torch.linalg.cholesky(torch.cov(Xc.T.double(), correction=0) + 1e-6 * torch.eye(D, device=DEV, dtype=torch.float64)).float(); gr = torch.linalg.qr(torch.randn(D, D, generator=g, device=DEV))[0]
    targets = {"features": F, "random_directions": unitr(tr(torch.randn(4096, D, generator=g, device=DEV))), "covariance_directions": unitr(tr(torch.randn(4096, D, generator=g, device=DEV) @ Lc.T)),
               "states": unitr(Xc[torch.randperm(Xc.shape[0], generator=g, device=DEV)[:4096]]), "features_rotated": unitr(tr(F[torch.randperm(F.shape[0], generator=g, device=DEV)[:4096]] @ gr))}
    out = dict(coordinates=coord, reconstruction_fvu=fv, n_features=F.shape[0], targets={})
    for tn, T in targets.items():
        row = {}
        for kind, Dct in (("native", Au), ("rotated", Ar)):
            sel, _, _ = omp(T, Dct, max(KS), batch=1024, record_err=False); f_ = {}
            for k in KS:
                cof, _ = refit(T, Dct, sel[:, :k]); f_[k] = float((T - torch.einsum("nk,nkd->nd", cof, Dct[sel[:, :k]])).pow(2).sum(1).median())
            row[kind] = dict(fvu=f_, top_mlp=float((typ[sel[:, 0]] == T_MLP).float().mean()))
        out["targets"][tn] = row
        log(f"{name} OpenAI v5_32k layer {L} ({coord}, reconstruction fvu " + ", ".join(f"{k} {v:.2f}" for k, v in fv.items()) + f"), {tn}: native k1/4/16 {row['native']['fvu'][1]:.2f}/{row['native']['fvu'][4]:.2f}/{row['native']['fvu'][16]:.2f} (rotated {row['rotated']['fvu'][1]:.2f}/{row['rotated']['fvu'][4]:.2f}/{row['rotated']['fvu'][16]:.2f}), top MLP {row['native']['top_mlp']:.2f}")
    if L == 6:
        acts = {b: [] for b in range(L + 1)}
        def mk(b):
            def pre_(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
            return pre_
        hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(L + 1)]
        try: block_states(model, arch, ids, [L], chunk=4)
        finally: [h.remove() for h in hs]
        Cw = torch.cat([torch.cat(acts[b])[keep] * arch.wdir(b).float().norm(dim=-1)[None] for b in range(L + 1)], 1); N = Cw.shape[0]
        z = encode(xin, enc, pre, lb); live = (z > 0).float().mean(0) > 0.01; feats = torch.nonzero(live)[:, 0]; feats = feats[z[:, feats].mean(0).topk(min(2000, feats.numel())).indices]; n_ = feats.numel()
        Fs = F[feats]; sel, _, _ = omp(Fs, Au, 16, batch=1024, record_err=False); cof, _ = refit(Fs, Au, sel)
        col = torch.where(typ[sel] == T_MLP, blk[sel].long() * DFF + idx[sel].long(), torch.full_like(sel, -1)); Fa = z[:, feats].T
        def auc_cols(fa, s):
            out_ = torch.empty(fa.shape[0], device=DEV)
            for i in range(fa.shape[0]):
                pos = s[i][fa[i] > 0]; neg = s[i][fa[i] <= 0]; out_[i] = ((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean()) if pos.numel() and neg.numel() else 0.5
            return out_
        def ledger(k, random_rows=False):
            o_ = torch.zeros(n_, N, device=DEV)
            for j in range(k):
                m = col[:, j] >= 0; c_ = col[m, j] if not random_rows else (col[m, j] // DFF) * DFF + torch.randint(0, DFF, (int(m.sum()),), generator=g, device=DEV); o_[m] += cof[m, j][:, None] * Cw[:, c_].T
            return o_
        first = torch.full((n_,), -1, device=DEV, dtype=torch.long)
        for i in range(n_):
            ms = [j for j in range(16) if col[i, j] >= 0]
            if ms: first[i] = col[i, ms[0]]
        has = first >= 0; rnd = (first[has] // DFF) * DFF + torch.randint(0, DFF, (int(has.sum()),), generator=g, device=DEV)
        auc = dict(top_row=float(auc_cols(Fa[has], Cw[:, first[has]].T.abs()).median()), random_row=float(auc_cols(Fa[has], Cw[:, rnd].T.abs()).median()), random_ledger_8=float(auc_cols(Fa[has], ledger(8, True)[has]).median()))
        for k in (1, 4, 8, 16):
            v = auc_cols(Fa[has], ledger(k)[has]); auc[f"ledger_{k}"] = float(v.median()); auc[f"ledger_{k}_over_0_8"] = float((v > 0.8).float().mean())
        out["activation"] = dict(n_features=n_, share_words_mlp=float((col >= 0).float().mean()), auc=auc)
        log(f"{name} OpenAI layer 6 activation: {n_} features, AUC top row {auc['top_row']:.2f}, random row {auc['random_row']:.2f}, ledger 1/4/8/16 {auc['ledger_1']:.2f}/{auc['ledger_4']:.2f}/{auc['ledger_8']:.2f}/{auc['ledger_16']:.2f}, random ledger 8 {auc['random_ledger_8']:.2f}")
    res["by_layer"][L] = out; del Au, Ar; torch.cuda.empty_cache()
o6 = res["by_layer"][6]["targets"]; f_ = lambda t, k: o6[t]["native"]["fvu"][k]; au = res["by_layer"][6]["activation"]["auc"]
res["checks"] = dict(ordering_holds=f_("random_directions", 16) > f_("covariance_directions", 16) > f_("features", 16) and abs(f_("features", 16) - f_("states", 16)) <= 0.08, top_mlp_over_0_8=o6["features"]["native"]["top_mlp"] > 0.8, ledger8_over_0_65_random_under_0_55=au["ledger_8"] >= 0.65 and au["random_row"] < 0.55)
summ = (f"{name}, OpenAI v5_32k TopK autoencoders: " + " || ".join(f"layer {L} ({v['coordinates']}): unexplained by 1/4/16 native words (rotated): " + "; ".join(f"{tn} {r['native']['fvu'][1]:.2f}/{r['native']['fvu'][4]:.2f}/{r['native']['fvu'][16]:.2f} ({r['rotated']['fvu'][16]:.2f} at 16), MLP top {r['native']['top_mlp']:.2f}" for tn, r in v["targets"].items()) for L, v in res["by_layer"].items())
        + f" || activation at layer 6 ({res['by_layer'][6]['activation']['n_features']} features): AUC top row {au['top_row']:.2f}, random row {au['random_row']:.2f}, ledger 1/4/8/16 {au['ledger_1']:.2f}/{au['ledger_4']:.2f}/{au['ledger_8']:.2f}/{au['ledger_16']:.2f} (above 0.8: {au['ledger_8_over_0_8']:.2f} at 8), same coefficients on random rows {au['random_ledger_8']:.2f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e499_saefamily2_{name}", res, summ)
