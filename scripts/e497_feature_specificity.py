"""e497: are SAE features special to the native dictionary, or is any direction? e494 and e495 compared the native and
the rotated dictionary on the same features; they did not compare features with other directions under the same
dictionary. A large dictionary aligned with the state distribution explains part of any direction, so the unexplained
fraction of a feature (0.47-0.49 at 16 words) means little until it is set against random directions, directions drawn
from the states' covariance, and the states themselves.
Setup: GPT-2 small, the SAE at the input of block 7 (24576 decoder rows) and the one at the input of block 3, the
native dictionary up to the block and its rotation, TransformerLens coordinates. Targets: the decoder rows; 4096
random unit directions; 4096 unit directions drawn from the states' covariance; 4096 centred typical states (unit);
and the decoder rows of the SAE rotated by a random orthogonal matrix (a same-Gram control for the features).
For each target family and dictionary: the median unexplained fraction at k = 1, 4, 16; the type of the top native
word (share MLP row); and the largest single-atom cosine (median).
Pre-registered (honest guesses):
- features are sparser in native words than random directions by at least 0.15 at k = 16 (0.5);
- features are about as sparse as covariance-drawn directions (within 0.05) (0.5);
- states are sparser than features (0.7);
- the top native word is an MLP row far more often for features than for random directions (0.6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
name = "gpt2"; model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; KS = [1, 4, 16]; LS = [2, 6]
dimc = lambda M: M - M.mean(-1, keepdim=True)
ids = eval_ids(name)[:16, :512].to(DEV); S_ = block_states(model, arch, ids, LS, chunk=4)
res = dict(model=name, layers=LS, by_layer={})
for L in LS:
    sae = load_file(hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"blocks.{L + 1}.hook_resid_pre/sae_weights.safetensors"))
    Wd = sae["W_dec"].float().to(DEV); Wd = Wd if Wd.shape[1] == D else Wd.T
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); Au = unitr(dimc(A)); Ar = unitr(dimc(rotate(A, seed=7))); del A
    X = S_[L].reshape(-1, D); keep = ~sinkmask(X); Xc = dimc(X[keep] - X[keep].mean(0)); g = torch.Generator(device=DEV).manual_seed(0)
    Lc = torch.linalg.cholesky(torch.cov(Xc.T.double(), correction=0) + 1e-6 * torch.eye(D, device=DEV, dtype=torch.float64)).float()
    F = unitr(dimc(Wd)); gr = torch.linalg.qr(torch.randn(D, D, generator=g, device=DEV))[0]
    targets = {"features": F, "random_directions": unitr(dimc(torch.randn(4096, D, generator=g, device=DEV))), "covariance_directions": unitr(dimc(torch.randn(4096, D, generator=g, device=DEV) @ Lc.T)),
               "states": unitr(Xc[torch.randperm(Xc.shape[0], generator=g, device=DEV)[:4096]]), "features_rotated": unitr(dimc(F[torch.randperm(F.shape[0], generator=g, device=DEV)[:4096]] @ gr))}
    out = {}
    for tn, T in targets.items():
        row = {}
        for kind, Dct in (("native", Au), ("rotated", Ar)):
            sel, _, _ = omp(T, Dct, max(KS), batch=1024, record_err=False); fv = {}
            for k in KS:
                cof, _ = refit(T, Dct, sel[:, :k]); fv[k] = float((T - torch.einsum("nk,nkd->nd", cof, Dct[sel[:, :k]])).pow(2).sum(1).median())
            row[kind] = dict(fvu=fv, top_mlp=float((typ[sel[:, 0]] == T_MLP).float().mean()), max_cos_median=float(torch.cat([(T[s:s + 1024] @ Dct.T).abs().max(1).values for s in range(0, T.shape[0], 1024)]).median()))
        out[tn] = row
        log(f"{name} block {L + 1} input, {tn}: native unexplained k1/4/16 {row['native']['fvu'][1]:.2f}/{row['native']['fvu'][4]:.2f}/{row['native']['fvu'][16]:.2f} (rotated {row['rotated']['fvu'][1]:.2f}/{row['rotated']['fvu'][4]:.2f}/{row['rotated']['fvu'][16]:.2f}); top word MLP {row['native']['top_mlp']:.2f} (rotated dict {row['rotated']['top_mlp']:.2f}); max cos {row['native']['max_cos_median']:.2f} / {row['rotated']['max_cos_median']:.2f}")
    res["by_layer"][L] = out; del Au, Ar; torch.cuda.empty_cache()
o6 = res["by_layer"][6]; f_ = lambda t, k: o6[t]["native"]["fvu"][k]
res["checks"] = dict(features_sparser_than_random_0_15=f_("random_directions", 16) - f_("features", 16) >= 0.15, features_like_covariance_within_0_05=abs(f_("covariance_directions", 16) - f_("features", 16)) <= 0.05, states_sparser_than_features=f_("states", 16) < f_("features", 16),
                     features_more_mlp_than_random=o6["features"]["native"]["top_mlp"] > o6["random_directions"]["native"]["top_mlp"] + 0.2)
summ = (f"{name}: unexplained fraction by 1 / 4 / 16 native words (rotated), and share of top words that are MLP rows, by target family | " + " || ".join(f"block {L + 1} input: " + "; ".join(f"{tn} {v['native']['fvu'][1]:.2f}/{v['native']['fvu'][4]:.2f}/{v['native']['fvu'][16]:.2f} ({v['rotated']['fvu'][1]:.2f}/{v['rotated']['fvu'][4]:.2f}/{v['rotated']['fvu'][16]:.2f}), MLP top {v['native']['top_mlp']:.2f}" for tn, v in out_.items()) for L, out_ in res["by_layer"].items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e497_specificity_{name}", res, summ)
