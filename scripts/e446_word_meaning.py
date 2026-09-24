"""e446: do native words mean something? (vision: every model ships with its dictionary)
The 400+ experiments measured what native words do for function, never whether a word has a stable meaning. A world
that reads states "as sixteen named words" needs words that are interpretable units, at least as much as the neurons
themselves.
Units compared at the middle depth (64 sequences; explanations from the first 32, tests on the other 32; typical
positions):
 - native words: the 200 MLP write rows most used in 16-word OMP descriptions; activation = coefficient;
 - the same 200 neurons: their actual activations (the standard unit of neuron interpretability);
 - rotated words: the 200 most used rows of the rotated dictionary; activation = coefficient;
 - random directions and the top-200 principal directions: activation = projection.
Judge-free proxies of meaning (no LLM judge):
 - token explanation: the tokens (current, next or previous) at a unit's 50 strongest positions in the first half
   predict its top 1% positions in the second half (AUC, best of the three);
 - coherence: mean cosine between the model's own embeddings of the tokens at the 50 strongest positions, minus
   that of 50 random positions (current tokens with the input embedding, next tokens with the unembedding);
 - self-consistency (WDD-specific): where a unit is strongly used, does its own direct logit effect (unembedding of its
   direction) rank the actual next token high? (mean percentile, 0.5 = chance). It asks whether usage agrees with
   what the word says.
Pre-registered (honest guesses):
 Mn1 (0.7) native words beat rotated words on all three;
 Mn2 (0.45) native words beat their own neurons' activations on token explanation;
 Mn3 (0.6) among native words, the most used are the least explainable (function words);
 Mn4 (0.5) self-consistency is above chance for native words and higher than for their neurons."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; ids = torch.cat([EA["eval_ids"][:32], EA["cen_ids"][:32]]).to(DEV); del EA
Bn, T = ids.shape
x = block_states(model, arch, ids, [L], chunk=4)[L][:, :-1]                      # positions 1..T-2 (a next token exists)
cur, nxt, prv = ids[:, 1:-1], ids[:, 2:], ids[:, :-2]
keep = ~sinkmask(x); half = torch.zeros(Bn, T - 2, dtype=torch.bool, device=DEV); half[:32] = True
X = x[keep]; mu = X.mean(0); Xc = X - mu; H1 = half[keep]; C, Nx, Pv = cur[keep], nxt[keep], prv[keep]; N = X.shape[0]
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk, idx = (lab[k].to(DEV) for k in ("type", "block", "index")); Ar = rotate(A, seed=7)
def code_acts(Dct, words):
    sel, cof, _ = omp(Xc, Dct, 16, batch=256, record_err=False); cnt = torch.bincount(sel.flatten(), minlength=Dct.shape[0])
    if words is None: words = torch.nonzero(typ == T_MLP)[:, 0][cnt[typ == T_MLP].topk(200).indices]
    lut = torch.full((Dct.shape[0],), -1, dtype=torch.long, device=DEV); lut[words] = torch.arange(words.numel(), device=DEV); col = lut[sel]; ok = col >= 0
    act = torch.zeros(N, words.numel(), device=DEV); act[torch.arange(N, device=DEV)[:, None].expand_as(sel)[ok], col[ok]] = cof[ok]; return words, act, cnt[words]
w_own, act_own, use_own = code_acts(A, None)
w_rot, act_rot, use_rot = code_acts(Ar, torch.topk(torch.bincount(omp(Xc, Ar, 16, batch=256, record_err=False)[0].flatten(), minlength=Ar.shape[0]), 200).indices)
# the same neurons' actual activations
nb, ni = blk[w_own], idx[w_own]; acts_n = torch.zeros(Bn, T, 200, device=DEV)
def mk(b):
    cols = torch.nonzero(nb == b)[:, 0]; neur = ni[cols]
    def pre(m, a): acts_n[s0:s0 + 4, :, cols] = a[0][..., neur].float()
    return pre
for s0 in range(0, Bn, 4):
    hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in torch.unique(nb).tolist()]
    try:
        with torch.no_grad(): model(ids[s0:s0 + 4])
    finally: [h.remove() for h in hs]
act_neu = acts_n[:, 1:-1][keep]
g = torch.Generator().manual_seed(0); Rd = unitr(torch.randn(200, D, generator=g).to(DEV)); act_rand = Xc @ Rd.T
evs, U = torch.linalg.eigh(torch.cov(Xc[H1].T.double(), correction=0)); Upc = U.flip(-1)[:, :200].float().T.contiguous(); act_pca = Xc @ Upc.T
dirs = dict(native=A[w_own], neuron=A[w_own], rotated=Ar[w_rot], random=Rd, pca=Upc)
acts = dict(native=act_own, neuron=act_neu, rotated=act_rot, random=act_rand, pca=act_pca)
E_in = unitr(arch.emb[0].detach().float()); WU = model.get_output_embeddings().weight.detach().float(); E_out = unitr(WU)
fl = model.transformer.ln_f if fam == "gpt2" else (model.gpt_neox.final_layer_norm if fam == "neox" else model.model.norm)
gf = getattr(fl, "weight", None); gf = gf.detach().float() if gf is not None else torch.ones(D, device=DEV)
def auc(score, label):
    """P(score of a positive > score of a negative) with ties counted half"""
    r = score.argsort().argsort().float(); pos = label.sum(); neg = label.numel() - pos
    if pos == 0 or neg == 0: return float("nan")
    s_sorted, inv = torch.unique(score, return_inverse=True); avg = torch.zeros_like(s_sorted).index_reduce_(0, inv, r, "mean", include_self=False)
    ra = avg[inv] + 1; return ((ra[label].sum() - pos * (pos + 1) / 2) / (pos * neg)).item()
def coherence(E, toks):
    v = E[toks]; m = v @ v.T; n = v.shape[0]; return ((m.sum() - m.diag().sum()) / (n * (n - 1))).item()
g2 = torch.Generator(device=DEV).manual_seed(1); i1, i2 = torch.nonzero(H1)[:, 0], torch.nonzero(~H1)[:, 0]
base_cur = sum(coherence(E_in, C[i1[torch.randint(0, i1.numel(), (50,), device=DEV, generator=g2)]]) for _ in range(20)) / 20
base_nxt = sum(coherence(E_out, Nx[i1[torch.randint(0, i1.numel(), (50,), device=DEV, generator=g2)]]) for _ in range(20)) / 20
res = dict(model=name, level=L, n_positions=N, units={})
for un, Aall in acts.items():
    tok_auc, coh_c, coh_n, sc = [], [], [], []
    lens = (dirs[un] * gf[None]) @ WU.T                                           # direct logit effect of each unit's direction [200, V]
    for u in range(Aall.shape[1]):
        a = Aall[:, u]; a1, a2 = a[i1], a[i2]
        top = a1.abs().topk(50).indices; lab2 = (a2.abs() >= torch.quantile(a2.abs(), 0.99)) & (a2.abs() > 0)   # top 1% (or every use, for words used less often)
        best = 0.0
        for toks in (C, Nx, Pv):
            vals, cnts = torch.unique(toks[i1][top], return_counts=True); t2 = toks[i2]; j = torch.searchsorted(vals, t2).clamp_max(vals.numel() - 1)
            sc_ = torch.where(vals[j] == t2, cnts[j], torch.zeros_like(cnts[j])).float()        # how often the token appears among the 50 strongest
            best = max(best, auc(sc_, lab2))
        tok_auc.append(best); coh_c.append(coherence(E_in, C[i1][top]) - base_cur); coh_n.append(coherence(E_out, Nx[i1][top]) - base_nxt)
        sgn = torch.sign(a1[top]); ranks = (lens[u][None, :] * sgn[:, None] > (lens[u][Nx[i1][top]] * sgn)[:, None]).float().mean(-1)   # share of tokens ranked above the actual next token
        sc.append((1 - ranks).mean().item())
    t = torch.tensor(tok_auc); res["units"][un] = dict(token_auc_median=t.nanmedian().item(), coh_cur_median=torch.tensor(coh_c).median().item(), coh_next_median=torch.tensor(coh_n).median().item(),
                                                        selfcons_median=torch.tensor(sc).median().item(), token_auc=tok_auc, selfcons=sc)
    log(f"{name} {un}: token-explanation AUC {res['units'][un]['token_auc_median']:.3f}, coherence cur {res['units'][un]['coh_cur_median']:+.3f} next {res['units'][un]['coh_next_median']:+.3f}, self-consistency {res['units'][un]['selfcons_median']:.3f}")
rk = lambda v: v.argsort().argsort().float()
ua = torch.tensor(res["units"]["native"]["token_auc"]); uu = use_own.float().cpu(); okk = ~torch.isnan(ua)
res["native_rho_usage_vs_auc"] = torch.corrcoef(torch.stack([rk(uu[okk]), rk(ua[okk])]))[0, 1].item()
U_ = res["units"]
res["checks"] = dict(Mn1=all(U_["native"][k] > U_["rotated"][k] for k in ("token_auc_median", "coh_cur_median", "coh_next_median", "selfcons_median")), Mn2=U_["native"]["token_auc_median"] > U_["neuron"]["token_auc_median"],
                     Mn3=res["native_rho_usage_vs_auc"] < 0, Mn4=U_["native"]["selfcons_median"] > 0.5 and U_["native"]["selfcons_median"] > U_["neuron"]["selfcons_median"])
summ = (f"{name} L{L}: medians over 200 units (token-explanation AUC / coherence current / coherence next / self-consistency): " + " ; ".join(f"{un} {v['token_auc_median']:.3f}/{v['coh_cur_median']:+.3f}/{v['coh_next_median']:+.3f}/{v['selfcons_median']:.3f}" for un, v in U_.items())
        + f" | native words: rank correlation of usage with explainability {res['native_rho_usage_vs_auc']:+.2f} | checks {json.dumps(res['checks'])}")
log(summ); record(f"e446_meaning_{name}", res, summ)
