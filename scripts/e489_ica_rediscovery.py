"""e489: are the write rows the independent components of the states? (Comon 1994; Hyvarinen and Oja 1997.) WDD takes
the dictionary from the weights and never fits it. If the rows are the directions along which the states are least
Gaussian (e488's question), then an unsupervised method that looks for exactly those directions, without the weights,
should land on them. FastICA on the centred middle-depth states (whitened to 256 principal dimensions, 64 components,
tanh contrast, symmetric decorrelation) gives 64 mixing directions; the question is how well they align with the
model's own MLP rows, against rotated rows, random directions and the principal directions.
Setup: middle depth, 16 x 512 evaluation tokens, typical positions. For each independent component, the largest
|cosine| with any native MLP row of the blocks up to the middle, with the same rows rotated, with 4096 random unit
directions, and with token embeddings; the same for the top 64 principal directions and for 64 random directions
inside the whitened subspace (the null for "any direction there"). Also the reverse: for the 256 most used native
rows, the largest |cosine| with any component; and the excess kurtosis of the components' projections against the
rows'.
Models (argument): the five.
Pre-registered (honest guesses):
- the components align with native rows far above rotated rows (median max-cosine at least twice) in all five (0.6);
- they align with native rows better than the principal directions do (0.7);
- at least a quarter of the components have a native row at cosine above 0.5 (0.4)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16; D = arch.D; C = 64; R = min(256, D)
E = eval_ids(name)[:16, :512].to(DEV)
X = block_states(model, arch, E, [L], chunk=4)[L].reshape(-1, D); keep = ~sinkmask(X); Xc = X[keep] - X[keep].mean(0); N = Xc.shape[0]
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); typ = lab["type"].to(DEV); del A
mlp = torch.nonzero(typ == T_MLP)[:, 0]; tk = torch.nonzero(typ == T_TOK)[:, 0]
ev, U = torch.linalg.eigh(torch.cov(Xc.T.double(), correction=0)); ev, U = ev.flip(0)[:R].float(), U.flip(1)[:, :R].float()
Xw = (Xc @ U) / ev.sqrt()                                                          # whitened, unit covariance
g = torch.Generator(device=DEV).manual_seed(0); W = torch.linalg.qr(torch.randn(R, C, generator=g, device=DEV))[0].T   # [C, R]
def sym_orth(M): e_, V_ = torch.linalg.eigh(M @ M.T); return V_ @ torch.diag(e_.clamp_min(1e-9).rsqrt()) @ V_.T @ M
for it in range(300):
    S = W @ Xw.T; gS = torch.tanh(S); W1 = (gS @ Xw) / N - (1 - gS.pow(2)).mean(1, keepdim=True) * W; W1 = sym_orth(W1)
    delta = float((1 - (W1 * W).sum(1).abs()).max()); W = W1
    if delta < 1e-5: break
mixing = unitr((U * ev.sqrt()[None]) @ W.T).T if False else unitr((W @ torch.diag(ev.sqrt()) @ U.T))           # [C, D]: the direction each source moves the data along
pcs_d = U[:, :C].T; rand_w = unitr(torch.randn(C, R, generator=g, device=DEV)) @ U.T; rand_w = unitr(rand_w); rand_d = unitr(torch.randn(4096, D, generator=g, device=DEV))
def maxcos(Q, Dct):
    out = []
    for s in range(0, Q.shape[0], 64): out.append((Q[s:s + 64] @ Dct.T).abs().max(1).values)
    return torch.cat(out)
fams = {"native_mlp": Au[mlp], "rotated_mlp": Ar[mlp], "random": rand_d, "token_embeddings": Au[tk]}
align = {qn: {fn: maxcos(Q, Dct) for fn, Dct in fams.items()} for qn, Q in (("ica", mixing), ("pca64", pcs_d), ("random_in_subspace", rand_w))}
sel, _, _ = omp(Xc, Au, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=Au.shape[0]); top_used = usage.topk(256).indices
rev = maxcos(Au[top_used], mixing); rev_r = maxcos(Ar[top_used], mixing)
def kurt(Dirs): z = Xc @ Dirs.T; z = z - z.mean(0); return z.pow(4).mean(0) / z.pow(2).mean(0).pow(2).clamp_min(1e-12) - 3
res = dict(model=name, level=L, n=N, components=C, whitened_dims=R, iterations=it + 1, converged=delta < 1e-5,
           alignment={qn: {fn: dict(median=float(v.median()), mean=float(v.mean()), share_over_0_5=float((v > 0.5).float().mean())) for fn, v in d.items()} for qn, d in align.items()},
           reverse=dict(most_used_native_rows_max_cos_with_ica=float(rev.median()), share_over_0_5=float((rev > 0.5).float().mean()), rotated_rows_max_cos_with_ica=float(rev_r.median())),
           kurtosis=dict(ica_median=float(kurt(mixing).median()), native_used_median=float(kurt(Au[top_used]).median()), pca64_median=float(kurt(pcs_d).median())))
Al = res["alignment"]
res["checks"] = dict(native_twice_rotated=Al["ica"]["native_mlp"]["median"] >= 2 * Al["ica"]["rotated_mlp"]["median"], ica_better_than_pca=Al["ica"]["native_mlp"]["median"] > Al["pca64"]["native_mlp"]["median"], quarter_over_0_5=Al["ica"]["native_mlp"]["share_over_0_5"] >= 0.25)
summ = (f"{name} L{L}, {N} positions, {C} components ({it + 1} iterations{'' if res['converged'] else ', not converged'}): largest |cos| of a component with any native MLP row, median {Al['ica']['native_mlp']['median']:.2f} (share above 0.5: {Al['ica']['native_mlp']['share_over_0_5']:.2f}), rotated rows {Al['ica']['rotated_mlp']['median']:.2f}, random directions {Al['ica']['random']['median']:.2f}, token embeddings {Al['ica']['token_embeddings']['median']:.2f}; "
        f"for the top-64 principal directions: native {Al['pca64']['native_mlp']['median']:.2f}, rotated {Al['pca64']['rotated_mlp']['median']:.2f}; random directions in the whitened subspace: native {Al['random_in_subspace']['native_mlp']['median']:.2f}, rotated {Al['random_in_subspace']['rotated_mlp']['median']:.2f} | "
        f"reverse: the 256 most used native rows have a component at median |cos| {res['reverse']['most_used_native_rows_max_cos_with_ica']:.2f} (share above 0.5: {res['reverse']['share_over_0_5']:.2f}; rotated rows {res['reverse']['rotated_rows_max_cos_with_ica']:.2f}) | excess kurtosis: components {res['kurtosis']['ica_median']:.1f}, used native rows {res['kurtosis']['native_used_median']:.1f}, PCs {res['kurtosis']['pca64_median']:.1f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e489_ica_{name}", res, summ)
