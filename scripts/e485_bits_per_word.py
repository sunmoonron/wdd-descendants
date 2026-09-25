"""e485: bits per word (Shannon 1948; Fano 1961). How much does the first native word of a description say about the
next token, in bits, and how much about the current token? Held-out predictive information: a conditional next-token
model is fitted on 24 sequences from each symbol (add-one smoothing toward the marginal) and scored on 8 held-out
sequences; the information of a symbol is the cross-entropy of the marginal minus the cross-entropy given the symbol,
in bits. Symbols: the first native word, the second, the first rotated word, the current token, the previous token,
the sign-and-index of the largest principal component (32 symbols), and a random label (control, about 0).
Setup: middle depth, 32 x 512 evaluation tokens, typical positions, dictionary up to the middle block; the next-token
alphabet is the 1000 most frequent next tokens of the fitting set plus "other".
Models (argument): the five.
Pre-registered (honest guesses):
- the first native word carries more bits about the next token than the first rotated word in all five (0.7);
- it carries fewer than the current token does (0.7);
- it carries more bits about the current token than about the next one (the words are partly lexical) (0.6)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16; D = arch.D
E = eval_ids(name)[:32, :512].to(DEV); B_, T_ = E.shape
X = block_states(model, arch, E, [L], chunk=4)[L]                                   # [B, T-1, D], positions 1..T-1
X = X[:, :-1]; flat = X.reshape(-1, D)                                               # keep positions 1..T-2, whose next token exists
keep = ~sinkmask(flat); seq = torch.arange(B_, device=DEV)[:, None].expand(B_, T_ - 2).reshape(-1)[keep]
cur = E[:, 1:T_ - 1].reshape(-1)[keep]; prev = E[:, 0:T_ - 2].reshape(-1)[keep]; nxt = E[:, 2:T_].reshape(-1)[keep]
Xc = flat[keep] - flat[keep].mean(0); N = Xc.shape[0]
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
sel_n, _, _ = omp(Xc, Au, 2, batch=1024, record_err=False); sel_r, _, _ = omp(Xc, Ar, 1, batch=1024, record_err=False)
P16, _ = pcs(Xc[seq < 24], 16); Z = Xc @ P16; pc_sym = Z.abs().argmax(1) * 2 + (Z.gather(1, Z.abs().argmax(1)[:, None])[:, 0] > 0).long()
g = torch.Generator(device=DEV).manual_seed(0)
symbols = dict(native_word_1=sel_n[:, 0], native_word_2=sel_n[:, 1], rotated_word_1=sel_r[:, 0], current_token=cur, previous_token=prev, pca_component=pc_sym, random_label=torch.randint(0, 4096, (N,), generator=g, device=DEV))
train = seq < 20; test = seq >= 24
def information(sym, target, S=1000, V=1000):
    """held-out bits with a tuned smoothing (v2): symbols and targets restricted to their 1000 most frequent values on
    the training split (the rest pooled), the smoothing strength chosen on a validation split; returns bits, the
    marginal and conditional cross-entropies, the chosen strength and the share of test symbols that are not pooled"""
    tr = seq < 20; va = (seq >= 20) & (seq < 24); te = seq >= 24
    def compact(v, n):
        cnt = torch.bincount(v[tr], minlength=int(v.max()) + 1); top = cnt.topk(min(n, int((cnt > 0).sum()))).indices
        lut = torch.full((int(v.max()) + 1,), top.numel(), device=DEV, dtype=torch.long); lut[top] = torch.arange(top.numel(), device=DEV); return lut[v], top.numel()
    s, S_ = compact(sym, S); t, V_ = compact(target, V)
    marg = torch.bincount(t[tr], minlength=V_ + 1).float() + 1.0; marg = marg / marg.sum()
    joint = torch.zeros(S_ + 1, V_ + 1, device=DEV); joint.index_put_((s[tr], t[tr]), torch.ones(int(tr.sum()), device=DEV), accumulate=True); n_s = joint.sum(1, keepdim=True)
    def ce_a(alpha, mask): cond = (joint + alpha * marg[None]) / (n_s + alpha); return float(-torch.log2(cond[s[mask], t[mask]]).mean())
    best = min([0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256], key=lambda a: ce_a(a, va))
    ce_m = float(-torch.log2(marg[t[te]]).mean()); ce_c = ce_a(best, te); return ce_m - ce_c, ce_m, ce_c, best, float((s[te] < S_).float().mean())
res = dict(model=name, level=L, n=N, n_train=int(train.sum()), n_test=int(test.sum()), about_next={}, about_current={}, estimator="v2: top-1000 symbols and targets, smoothing tuned on sequences 20-23, tested on 24-31")
for sn, sv in symbols.items():
    b, cm, cc, al, cov = information(sv, nxt); res["about_next"][sn] = dict(bits=b, ce_marginal=cm, ce_conditional=cc, alpha=al, coverage=cov)
    if sn not in ("current_token",):
        b2, _, _, al2, cov2 = information(sv, cur); res["about_current"][sn] = dict(bits=b2, alpha=al2, coverage=cov2)
log(f"{name}: bits about the next token: " + ", ".join(f"{k} {v['bits']:+.3f}" for k, v in res["about_next"].items()) + " | about the current token: " + ", ".join(f"{k} {v['bits']:+.3f}" for k, v in res["about_current"].items()))
An, Ac = res["about_next"], res["about_current"]
res["checks"] = dict(native_over_rotated=An["native_word_1"]["bits"] > An["rotated_word_1"]["bits"], native_under_current_token=An["native_word_1"]["bits"] < An["current_token"]["bits"], more_about_current_than_next=Ac["native_word_1"]["bits"] > An["native_word_1"]["bits"])
summ = (f"{name} L{L}, {N} positions ({int(test.sum())} held out), marginal next-token cross-entropy {An['native_word_1']['ce_marginal']:.2f} bits: information about the next token: native word 1 {An['native_word_1']['bits']:.3f}, word 2 {An['native_word_2']['bits']:.3f}, rotated word 1 {An['rotated_word_1']['bits']:.3f}, current token {An['current_token']['bits']:.3f}, previous token {An['previous_token']['bits']:.3f}, PCA component {An['pca_component']['bits']:.3f}, random label {An['random_label']['bits']:+.3f} | "
        f"about the current token: native word 1 {Ac['native_word_1']['bits']:.3f}, rotated word 1 {Ac['rotated_word_1']['bits']:.3f}, PCA {Ac['pca_component']['bits']:.3f}, previous token {Ac['previous_token']['bits']:.3f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e485_bits_{name}", res, summ)
