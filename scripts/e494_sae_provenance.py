"""e494: which weight rows write an SAE feature? A learned sparse autoencoder's decoder rows are the directions its
features write into the residual stream, with no provenance. Decomposing each decoder row over the native dictionary
asks which of the model's own writes it is made of: how many native words a feature needs, which component and block
its top word comes from, and whether that beats a rotated dictionary. This is the bridge between a learned feature
vocabulary and the model's own, run in the provenance direction (e403 compared the two as codes for the states).
Setup: GPT-2 small, Bloom's residual SAE at blocks.7.hook_resid_pre (24576 latents), the native dictionary up to block
6, both in TransformerLens coordinates (each vector centred over the model dimension, as e403). Every decoder row is
described by OMP with k = 1, 2, 4, 8, 16 native words and with the same numbers of rotated words; the feature's own
activations on 8 x 256 states select the 2000 most active features for a second table.
Reported: median unexplained fraction per k, native against rotated; the type of the top word (token embedding, MLP
row, head basis) and the block of the top MLP word, for all features and for the most active ones; and the reverse,
the share of the 4096 most used native MLP rows that have a decoder row at cosine above 0.5, against rotated rows.
Pre-registered (honest guesses):
- features are sparser in native words than in rotated ones: median unexplained fraction at k = 4 lower by at least
  0.2 (0.7);
- the top native word of most features is an MLP row (over 0.6), a token embedding for a minority (0.5);
- at least a quarter of the most used native rows have a feature at cosine above 0.5 (0.4)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
name = "gpt2"; model, tok, fam = load_model(name); arch = Arch(model, fam); L = 6; D = arch.D; KS = [1, 2, 4, 8, 16]
sae = load_file(hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"blocks.{L + 1}.hook_resid_pre/sae_weights.safetensors"))
Wd = sae["W_dec"].float().to(DEV); Wd = Wd if Wd.shape[1] == D else Wd.T; We = sae["W_enc"].float().to(DEV); We = We if We.shape[0] == D else We.T; be = sae["b_enc"].float().to(DEV); bd = sae["b_dec"].float().to(DEV)
dimc = lambda M: M - M.mean(-1, keepdim=True)
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); blk = lab["block"].to(DEV)
Au = unitr(dimc(A)); Ar = unitr(dimc(rotate(A, seed=7))); del A
F = unitr(dimc(Wd)); NF = F.shape[0]
ids = eval_ids(name)[:8, :256].to(DEV); X = block_states(model, arch, ids, [L], chunk=4)[L].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]
act = torch.relu((dimc(Xk) - bd) @ We + be); live = (act > 0).float().mean(0); top_feat = act.mean(0).topk(2000).indices
res = dict(model=name, level=L, n_features=NF, k=KS, live_share=float((live > 0).float().mean()), fvu={}, provenance={})
for kind, Dct in (("native", Au), ("rotated", Ar)):
    sel, _, _ = omp(F, Dct, max(KS), batch=1024, record_err=False); out = {}
    for k in KS:
        cof, _ = refit(F, Dct, sel[:, :k]); rec = torch.einsum("nk,nkd->nd", cof, Dct[sel[:, :k]]); fvu = (F - rec).pow(2).sum(1)
        out[k] = dict(median_all=float(fvu.median()), median_active=float(fvu[top_feat].median()))
    res["fvu"][kind] = out
    if kind == "native":
        t0 = typ[sel[:, 0]]; b0 = blk[sel[:, 0]]
        for label, subset in (("all", torch.arange(NF, device=DEV)), ("most_active", top_feat)):
            tt = t0[subset]; res["provenance"][label] = dict(top_is_mlp=float((tt == T_MLP).float().mean()), top_is_token=float((tt == T_TOK).float().mean()), top_is_head=float((tt == T_ATT).float().mean()), top_is_pos=float((tt == T_POS).float().mean()),
                                                              mlp_block_hist={int(bb): float(((tt == T_MLP) & (b0[subset] == bb)).float().mean()) for bb in range(L + 1)})
    log(f"{name} {kind}: median unexplained fraction by k " + ", ".join(f"{k}: {out[k]['median_all']:.3f} (active {out[k]['median_active']:.3f})" for k in KS))
# the reverse: do the most used native rows look like features?
Xc = Xk - Xk.mean(0); sel_s, _, _ = omp(unitr(dimc(Xc)), Au, 16, batch=1024, record_err=False); usage = torch.bincount(sel_s.reshape(-1), minlength=Au.shape[0])
mlp = torch.nonzero(typ == T_MLP)[:, 0]; used = mlp[usage[mlp].topk(4096).indices]
rev_n = torch.cat([(Au[used[s:s + 512]] @ F.T).abs().max(1).values for s in range(0, 4096, 512)]); rev_r = torch.cat([(Ar[used[s:s + 512]] @ F.T).abs().max(1).values for s in range(0, 4096, 512)])
res["reverse"] = dict(used_rows_max_cos_median=float(rev_n.median()), used_rows_share_over_0_5=float((rev_n > 0.5).float().mean()), rotated_rows_max_cos_median=float(rev_r.median()), rotated_rows_share_over_0_5=float((rev_r > 0.5).float().mean()))
Fv, P = res["fvu"], res["provenance"]
res["checks"] = dict(native_sparser_by_0_2=Fv["rotated"][4]["median_all"] - Fv["native"][4]["median_all"] >= 0.2, top_mostly_mlp=P["all"]["top_is_mlp"] > 0.6, quarter_of_rows_are_features=res["reverse"]["used_rows_share_over_0_5"] >= 0.25)
summ = (f"{name} L{L}, {NF} SAE features ({res['live_share']:.2f} live on the states): unexplained fraction of a decoder row by k native / rotated words: " + ", ".join(f"{k}: {Fv['native'][k]['median_all']:.2f} / {Fv['rotated'][k]['median_all']:.2f}" for k in KS)
        + f" | top native word: MLP row {P['all']['top_is_mlp']:.2f}, token embedding {P['all']['top_is_token']:.2f}, head basis {P['all']['top_is_head']:.2f}, position {P['all']['top_is_pos']:.2f} (most active 2000: MLP {P['most_active']['top_is_mlp']:.2f}, token {P['most_active']['top_is_token']:.2f}); MLP top words by block " + "/".join(f"{P['all']['mlp_block_hist'][b]:.2f}" for b in range(L + 1))
        + f" | reverse: the 4096 most used native rows have a feature at median |cos| {res['reverse']['used_rows_max_cos_median']:.2f} (share above 0.5: {res['reverse']['used_rows_share_over_0_5']:.2f}), rotated rows {res['reverse']['rotated_rows_max_cos_median']:.2f} ({res['reverse']['rotated_rows_share_over_0_5']:.2f}) | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e494_saeprovenance_{name}", res, summ)
