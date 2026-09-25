"""e495: the provenance layer under SAEs, across depth. e494 found that at the input of GPT-2's block 7 a learned
feature's top native word is an MLP row in 93% of cases, from every block, that features are sparser in native words
than in rotated ones but need many rows, and that a quarter of the most used rows are features. Bloom's residual SAEs
exist for every block, so the same questions can be asked at five depths, with two additions: whether the frequent
features are the single-row ones, and whether the top writer sits in the block just before the SAE or anywhere.
Setup: GPT-2 small, SAEs at the inputs of blocks 3, 5, 7, 9 and 11 (the outputs of blocks 2, 4, 6, 8, 10), 24576
features each; the native dictionary up to the block, both in TransformerLens coordinates (each vector centred over
the model dimension); OMP of every decoder row with 16 native or rotated words; the features' activations on 8 x 256
states for their frequency; the reverse for the 4096 most used native MLP rows.
Reported per depth: unexplained fraction at k = 1, 4, 16 (native / rotated); the type of the top word and the block
histogram of MLP top words; the share of used rows that are features (cosine above 0.5); the Spearman correlation of a
feature's frequency with its single-word unexplained fraction, and the top-word types of the 500 most frequent against
the 500 rarest live features.
Pre-registered (honest guesses):
- at every depth the top word is an MLP row for over 0.8 of features (0.7);
- the top MLP word's block is spread, with the block just before the SAE holding under 0.4 of them (0.6);
- frequent features are more single-row (negative Spearman, below -0.2) (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
name = "gpt2"; model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; LS = [2, 4, 6, 8, 10]; KS = [1, 4, 16]
dimc = lambda M: M - M.mean(-1, keepdim=True)
ids = eval_ids(name)[:8, :256].to(DEV); S_ = block_states(model, arch, ids, LS, chunk=4)
def spear(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-9))
res = dict(model=name, layers=LS, by_layer={})
for L in LS:
    sae = load_file(hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"blocks.{L + 1}.hook_resid_pre/sae_weights.safetensors"))
    Wd = sae["W_dec"].float().to(DEV); Wd = Wd if Wd.shape[1] == D else Wd.T; We = sae["W_enc"].float().to(DEV); We = We if We.shape[0] == D else We.T; be = sae["b_enc"].float().to(DEV); bd = sae["b_dec"].float().to(DEV)
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); Au = unitr(dimc(A)); Ar = unitr(dimc(rotate(A, seed=7))); del A
    F = unitr(dimc(Wd)); NF = F.shape[0]
    X = S_[L].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; act = torch.relu((dimc(Xk) - bd) @ We + be); freq = (act > 0).float().mean(0); live = freq > 0
    out = dict(n_features=NF, live_share=float(live.float().mean()), fvu={})
    for kind, Dct in (("native", Au), ("rotated", Ar)):
        sel, _, _ = omp(F, Dct, max(KS), batch=1024, record_err=False); d = {}
        for k in KS:
            cof, _ = refit(F, Dct, sel[:, :k]); fvu = (F - torch.einsum("nk,nkd->nd", cof, Dct[sel[:, :k]])).pow(2).sum(1); d[k] = float(fvu.median())
            if kind == "native" and k == 1: fvu1 = fvu
        out["fvu"][kind] = d
        if kind == "native":
            t0 = typ[sel[:, 0]]; b0 = blk[sel[:, 0]]; ismlp = t0 == T_MLP
            out["top_type"] = dict(mlp=float(ismlp.float().mean()), token=float((t0 == T_TOK).float().mean()), head=float((t0 == T_ATT).float().mean()), pos=float((t0 == T_POS).float().mean()))
            out["mlp_top_block_hist"] = {int(bb): float(((ismlp) & (b0 == bb)).float().sum() / ismlp.float().sum().clamp_min(1)) for bb in range(L + 1)}
            lv = torch.nonzero(live)[:, 0]; order = freq[lv].argsort(descending=True); hi, lo = lv[order[:500]], lv[order[-500:]]
            out["frequency_vs_single_word_fvu_spearman"] = spear(freq[lv], fvu1[lv]); out["frequent_top_mlp"] = float(ismlp[hi].float().mean()); out["rare_top_mlp"] = float(ismlp[lo].float().mean())
            out["frequent_top_token"] = float((t0[hi] == T_TOK).float().mean()); out["rare_top_token"] = float((t0[lo] == T_TOK).float().mean()); out["frequent_fvu1"] = float(fvu1[hi].median()); out["rare_fvu1"] = float(fvu1[lo].median())
    Xc = Xk - Xk.mean(0); sel_s, _, _ = omp(unitr(dimc(Xc)), Au, 16, batch=1024, record_err=False); usage = torch.bincount(sel_s.reshape(-1), minlength=Au.shape[0])
    mlp = torch.nonzero(typ == T_MLP)[:, 0]; used = mlp[usage[mlp].topk(min(4096, mlp.numel())).indices]
    rev_n = torch.cat([(Au[used[s:s + 512]] @ F.T).abs().max(1).values for s in range(0, used.numel(), 512)]); rev_r = torch.cat([(Ar[used[s:s + 512]] @ F.T).abs().max(1).values for s in range(0, used.numel(), 512)])
    out["reverse"] = dict(used_rows_median=float(rev_n.median()), used_rows_over_0_5=float((rev_n > 0.5).float().mean()), rotated_rows_over_0_5=float((rev_r > 0.5).float().mean()))
    res["by_layer"][L] = out
    log(f"{name} SAE at block {L + 1} input: unexplained k1/4/16 native {out['fvu']['native'][1]:.2f}/{out['fvu']['native'][4]:.2f}/{out['fvu']['native'][16]:.2f} rotated {out['fvu']['rotated'][1]:.2f}/{out['fvu']['rotated'][4]:.2f}/{out['fvu']['rotated'][16]:.2f} | top word MLP {out['top_type']['mlp']:.2f} token {out['top_type']['token']:.2f} head {out['top_type']['head']:.2f}; MLP top block hist " + "/".join(f"{out['mlp_top_block_hist'][b]:.2f}" for b in range(L + 1))
        + f" | frequency vs single-word unexplained {out['frequency_vs_single_word_fvu_spearman']:+.2f}; frequent/rare: top MLP {out['frequent_top_mlp']:.2f}/{out['rare_top_mlp']:.2f}, top token {out['frequent_top_token']:.2f}/{out['rare_top_token']:.2f}, unexplained k1 {out['frequent_fvu1']:.2f}/{out['rare_fvu1']:.2f} | used rows that are features {out['reverse']['used_rows_over_0_5']:.2f} (rotated {out['reverse']['rotated_rows_over_0_5']:.2f})")
    del Au, Ar, F, act; torch.cuda.empty_cache()
BL = res["by_layer"]
res["checks"] = dict(top_mlp_over_0_8=all(v["top_type"]["mlp"] > 0.8 for v in BL.values()), last_block_under_0_4=all(v["mlp_top_block_hist"][L] < 0.4 for L, v in BL.items()), frequent_more_single_row=all(v["frequency_vs_single_word_fvu_spearman"] < -0.2 for v in BL.values()))
summ = (f"{name}, Bloom's residual SAEs at five depths: " + " || ".join(f"block {L + 1} input: unexplained by 1/4/16 native words {v['fvu']['native'][1]:.2f}/{v['fvu']['native'][4]:.2f}/{v['fvu']['native'][16]:.2f} (rotated {v['fvu']['rotated'][1]:.2f}/{v['fvu']['rotated'][4]:.2f}/{v['fvu']['rotated'][16]:.2f}); top word an MLP row {v['top_type']['mlp']:.2f}, of which in the last block {v['mlp_top_block_hist'][L]:.2f}; "
        f"frequency vs single-word unexplained rho {v['frequency_vs_single_word_fvu_spearman']:+.2f} (frequent/rare top token {v['frequent_top_token']:.2f}/{v['rare_top_token']:.2f}); used rows that are features {v['reverse']['used_rows_over_0_5']:.2f}" for L, v in BL.items()) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e495_saedepth_{name}", res, summ)
