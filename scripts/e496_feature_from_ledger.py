"""e496: does a feature fire when its writer fires? e494 found that a learned SAE feature's top native word is an MLP
row, and that a feature needs several rows. Rows have activations; features have activations. If a feature is the
SAE's view of a few writes, its activation across positions should follow the activations of the rows that compose
it, weighted by the coefficients WDD assigns them, with no fitting.
Setup: GPT-2 small, Bloom's residual SAE at the input of block 7, the 2000 most active live features on 8 x 256
states (typical positions); each feature's decoder row described by 8 native words over the dictionary up to block 6
(TransformerLens coordinates); the MLP activations of blocks 0-6 at every position; a row's write size is its
activation times its row norm.
Measured per feature, across positions: the Spearman correlation of the feature's activation with the write size of
its top MLP word; with a random MLP row of the same block (control); with its second MLP word; and with the ledger
prediction, the sum over its top 1, 4 or 8 MLP words of the WDD coefficient times the write size (signed).
Reported: medians and the share of features above 0.5 for each; the share of the 8 words that are MLP rows. v2 adds the
statistic a sparse feature needs, the AUC of each score for the positions where the feature fires against the rest, and
lowers the liveness threshold to 1% of positions.
Pre-registered (honest guesses):
- the median correlation with the top row's write size is above 0.3 and the control's below 0.05 (0.6);
- the 8-word ledger prediction beats the top row alone by at least 0.1 in the median (0.5);
- at least a quarter of features follow their top row at above 0.5 (0.4)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
name = "gpt2"; model, tok, fam = load_model(name); arch = Arch(model, fam); L = 6; D = arch.D; DFF = arch.DFF; K = 8; NFEAT = 2000
dimc = lambda M: M - M.mean(-1, keepdim=True)
sae = load_file(hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"blocks.{L + 1}.hook_resid_pre/sae_weights.safetensors"))
Wd = sae["W_dec"].float().to(DEV); Wd = Wd if Wd.shape[1] == D else Wd.T; We = sae["W_enc"].float().to(DEV); We = We if We.shape[0] == D else We.T; be = sae["b_enc"].float().to(DEV); bd = sae["b_dec"].float().to(DEV)
ids = eval_ids(name)[:8, :256].to(DEV)
acts = {b: [] for b in range(L + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(L + 1)]
try: X = block_states(model, arch, ids, [L], chunk=4)[L].reshape(-1, D)
finally: [h.remove() for h in hs]
keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]
Cw = torch.cat([torch.cat(acts[b])[keep] * arch.wdir(b).float().norm(dim=-1)[None] for b in range(L + 1)], 1)         # [N, (L+1)*DFF] write sizes
act = torch.relu((dimc(Xk) - bd) @ We + be); live = (act > 0).float().mean(0) > 0.01; feats = torch.nonzero(live)[:, 0]; feats = feats[act[:, feats].mean(0).topk(min(NFEAT, feats.numel())).indices]
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); Au = unitr(dimc(A)); del A
F = unitr(dimc(Wd[feats])); sel, _, _ = omp(F, Au, K, batch=1024, record_err=False); cof, _ = refit(F, Au, sel)
col = torch.where(typ[sel] == T_MLP, blk[sel].long() * DFF + idx[sel].long(), torch.full_like(sel, -1))              # activation column of each selected word, -1 if not an MLP row
def spear_cols(a, b):
    """row-wise Spearman between a [n, N] and b [n, N]"""
    ra, rb = a.argsort(1).argsort(1).double(), b.argsort(1).argsort(1).double(); ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True); return ((ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp_min(1e-9)).float()
def auc_cols(fa, s):
    """v2: row-wise AUC of the score s [n, N] for separating the positions where the feature fires (fa > 0) from the rest; the
    right statistic for a sparse feature, whose activation is zero at most positions (chance 0.5)"""
    out = torch.empty(fa.shape[0], device=DEV)
    for i in range(fa.shape[0]):
        pos = s[i][fa[i] > 0]; neg = s[i][fa[i] <= 0]
        out[i] = ((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean()) if pos.numel() and neg.numel() else 0.5
    return out
Fa = act[:, feats].T                                                                                              # [n, N]
g = torch.Generator(device=DEV).manual_seed(0); n_ = feats.numel()
def ledger(k):
    out = torch.zeros(n_, N, device=DEV)
    for j in range(k):
        m = col[:, j] >= 0; out[m] += cof[m, j][:, None] * Cw[:, col[m, j]].T
    return out
mlp_rank = [i for i in range(n_)]
first = torch.full((n_,), -1, device=DEV, dtype=torch.long); second = torch.full((n_,), -1, device=DEV, dtype=torch.long)
for i in range(n_):
    ms = [j for j in range(K) if col[i, j] >= 0]
    if ms: first[i] = col[i, ms[0]]
    if len(ms) > 1: second[i] = col[i, ms[1]]
has = first >= 0; has2 = second >= 0
rho_top = spear_cols(Fa[has], Cw[:, first[has]].T.abs()); rnd = (blk[sel[has, 0]].long().clamp_min(0) * DFF + torch.randint(0, DFF, (int(has.sum()),), generator=g, device=DEV))
rho_rand = spear_cols(Fa[has], Cw[:, rnd].T.abs()); rho_second = spear_cols(Fa[has2], Cw[:, second[has2]].T.abs())
rho_led = {k: spear_cols(Fa[has], ledger(k)[has]) for k in (1, 4, 8)}
auc_top = auc_cols(Fa[has], Cw[:, first[has]].T.abs()); auc_rand = auc_cols(Fa[has], Cw[:, rnd].T.abs()); auc_led = {k: auc_cols(Fa[has], ledger(k)[has]) for k in (1, 4, 8)}
res = dict(model=name, level=L, n_features=n_, n_positions=N, share_words_mlp=float((col >= 0).float().mean()), features_with_mlp_word=float(has.float().mean()),
           top_row=dict(median=float(rho_top.median()), over_0_5=float((rho_top > 0.5).float().mean())), random_row=dict(median=float(rho_rand.median()), over_0_5=float((rho_rand > 0.5).float().mean())),
           second_row=dict(median=float(rho_second.median()), over_0_5=float((rho_second > 0.5).float().mean())), ledger={k: dict(median=float(v.median()), over_0_5=float((v > 0.5).float().mean())) for k, v in rho_led.items()},
           auc=dict(top_row=float(auc_top.median()), top_row_over_0_8=float((auc_top > 0.8).float().mean()), random_row=float(auc_rand.median()), ledger={k: dict(median=float(v.median()), over_0_8=float((v > 0.8).float().mean())) for k, v in auc_led.items()}))
res["checks"] = dict(top_over_0_3_control_under_0_05=res["top_row"]["median"] > 0.3 and res["random_row"]["median"] < 0.05, ledger8_beats_top_by_0_1=res["ledger"][8]["median"] >= res["top_row"]["median"] + 0.1, quarter_follow_top=res["top_row"]["over_0_5"] >= 0.25)
summ = (f"{name} L{L}, {n_} most active features, {N} positions: AUC of the top row's write size for the positions where the feature fires, median {res['auc']['top_row']:.2f} (share above 0.8: {res['auc']['top_row_over_0_8']:.2f}), random row {res['auc']['random_row']:.2f}; ledger prediction from 1 / 4 / 8 words {res['auc']['ledger'][1]['median']:.2f} / {res['auc']['ledger'][4]['median']:.2f} / {res['auc']['ledger'][8]['median']:.2f} (above 0.8: {res['auc']['ledger'][1]['over_0_8']:.2f} / {res['auc']['ledger'][4]['over_0_8']:.2f} / {res['auc']['ledger'][8]['over_0_8']:.2f}) | Spearman of a feature's activation with the write size of its top native MLP row median {res['top_row']['median']:+.2f} (share above 0.5: {res['top_row']['over_0_5']:.2f}), with a random row of the same block {res['random_row']['median']:+.2f}, with its second row {res['second_row']['median']:+.2f}; "
        f"with the ledger prediction from its top 1 / 4 / 8 words {res['ledger'][1]['median']:+.2f} / {res['ledger'][4]['median']:+.2f} / {res['ledger'][8]['median']:+.2f} (above 0.5: {res['ledger'][1]['over_0_5']:.2f} / {res['ledger'][4]['over_0_5']:.2f} / {res['ledger'][8]['over_0_5']:.2f}); MLP rows are {res['share_words_mlp']:.2f} of the words | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e496_featureledger_{name}", res, summ)
