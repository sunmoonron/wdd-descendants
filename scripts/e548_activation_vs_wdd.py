"""e548: does WDD add anything to what the activation side already says? e547b found an entrant's own neuron
selectively active at its future extreme contexts eight intervals before entry. If ordinary neuron statistics,
computed without the projection, predict future entry as well as S does, WDD is a detector of selective neurons with
aligned output rows and nothing more; if S adds information conditional on them, the output-side geometry carries
something the activation side does not. At every checkpoint the candidates are the non-words that enter at t+1, t+2,
t+4, t+6 or t+8 together with 4,000 random non-words that do not; for each, features at t: on the WDD side S and the
count of positions over three quarters of the floor; on the activation side, without the projection, the mean positive
activation, the excess kurtosis of the activation over positions, the share of positive activation in the top five
percent of positions, the coherence of the states at the neuron's sixteen most active positions (their mean pairwise
cosine), the write row's norm and its growth over the previous interval; on the output-geometry side the row's cosine
with the mean centred state at those sixteen positions (positions chosen by the activation, not by the projection);
and the coalition's coherence at the neuron's four most active positions (e546's within-row similarity, the alignment
weighting included). Recorded: each feature's AUC for entry at every horizon; logistic regressions on within-checkpoint
percentile ranks, trained on even origin checkpoints and tested on odd and the reverse, for the WDD set, the
activation set, the activation set with the output geometry, with the coalition, and all together, with the
increments in both directions; and, at step 8000, whether family pairs (words whose writer vectors have cosine above
0.3) share their input selectivity, the cosine of their activation vectors over positions and the overlap of their
sixteen most active positions, against random word pairs.
Pre-registered (honest guesses):
- the activation side alone predicts entry at four intervals at AUC 0.85 or more through the output-geometry
  feature, the purely input-side features at 0.6-0.75 (0.6);
- S adds information: the logistic AUC rises by 0.05 or more when the WDD features join the activation set with its
  output geometry and coalition, and the reverse increment is under 0.03 (0.5);
- at eight intervals the input-side features do not lead S (0.5);
- family pairs share input selectivity: activation-vector cosine 0.3 or more against under 0.05 for random pairs (0.6).
Arguments: name."""
"""(capture and statistics as in e546: e545 found each word's extremes written by a persistent, word-specific
ensemble of thousands of writers. Three temporal readings: the ensemble forms first and the projection follows (the
ensemble causes entry); the projection becomes extreme first and the decomposition merely names the writers afterward;
or a common context raises both together. At all sixteen checkpoints, block 12, the state at every kept position is
split into its writes (53,248 MLP writers, thirteen attention outputs, the embedding) and the rows taken from the same
model. For every clean entrant (a non-word at t-2 and t-1, a word at t and t+1) and an S-matched non-entrant (the
near miss: the same S at t-1, a non-word through t+1), at the row's top-4 positions at t and at k = -4..+2: the row's S;
the MLP part of the projection; the coherence of the contributions, the squared sum over the sum of squares less one
(how far the sum exceeds an incoherent sum); the within-row similarity of the writer vectors with the contemporaneous
alignment and with the alignment fixed at entry; the persistence to entry of the activation pattern, of the
entry-aligned writer vector and of the contemporaneous one; the attention outputs' share; the own neuron's share. The
rise of the MLP projection from k=-4 to 0 is split into the activations' change, the alignment's change from the row's
rotation, the alignment's change from the writers' motion, and the remainder. Also, at step 8000: the coherence for
persistent words, their random positions and random rows; the ensemble-by-row specificity matrix (the aggregate write
of a word's top-500 writers at its extreme positions, projected on every word); and a selection-matched null for the
ensembles' persistence from step 8000 to 16000 (the same positions weighted by a random native row's alignment).
Pre-registered (honest guesses):
- the context is there first: at k=-4 the activation pattern at an entrant's entry-time positions has cosine 0.5 or
  more with its entry-time pattern, and so does the near miss's (stability is generic) (0.6);
- the rise is the alignment's, not the activations': from k=-4 to 0 the alignment's change accounts for more than half
  of the MLP projection's rise, the row's rotation for more of it than the writers' motion, the activations' change for
  less than a quarter (0.5);
- coherence rises across the entry, at least doubling from k=-4 to 0, and is at least twice the near miss's at k=0 (0.5);
- the ensembles' persistence is mostly the contexts': the selection-matched null persists at 0.7 or more against the
  real 0.88 (0.5);
- the ensembles are private: a word's aggregate ensemble write has a cosine with its own row at least three times its
  mean cosine with other words (0.7).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
from lr_common import eval_ids
name = sys.argv[1]; B = 12; K = 16; KTOP = 4; steps = list(range(1000, 16001, 1000)); T = len(steps); CDIR = f"/workspace/wdd/cache/e524_{name}"; NTOP_E = 500
idsB = eval_ids(name)[:8, :256].to(DEV); keepc = torch.stack([torch.load(f"{CDIR}/step{n}.pt", map_location="cpu")["blocks"][B]["keep"] for n in steps]).all(0); N = int(keepc.sum())
class Stop(Exception): pass
def capture(model, arch, fam):
    cap = {"act": {}, "attn": {}, "out": None, "emb": None}; layers = arch.layers; attn_mod = lambda bb: layers[bb].attention if fam == "neox" else (layers[bb].attn if fam == "gpt2" else layers[bb].self_attn); emb_mod = model.gpt_neox.embed_in if fam == "neox" else (model.transformer.wte if fam == "gpt2" else model.model.embed_tokens)
    def h_attn(bb):
        def f(m, i, o): cap["attn"][bb] = (o[0] if isinstance(o, tuple) else o).detach().float()
        return f
    def h_act(bb):
        def f(m, a): cap["act"][bb] = a[0].detach().float(); return None
        return f
    def h_out(m, i, o): cap["out"] = (o[0] if isinstance(o, tuple) else o).detach().float(); raise Stop
    def h_emb(m, i, o): cap["emb"] = o.detach().float()
    hs = [attn_mod(bb).register_forward_hook(h_attn(bb)) for bb in range(B + 1)] + [arch.mlp_lin(bb).register_forward_pre_hook(h_act(bb)) for bb in range(B + 1)] + [emb_mod.register_forward_hook(h_emb), layers[B].register_forward_hook(h_out)]
    try: model(idsB)
    except Stop: pass
    finally: [h.remove() for h in hs]
    return cap
def flat(t): return t.reshape(8, 256, -1)[:, 1:].reshape(2040, -1)[keepc]
ACT, ATT, XS, RU, NORM, WORDS, SS, RATIO = {}, {}, {}, {}, {}, {}, {}, {}
for i, n in enumerate(steps):
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF
    for p_ in model.parameters(): p_.requires_grad_(False)
    cap = capture(model, arch, fam); Wv = torch.cat([arch.wdir(bb).float() for bb in range(B + 1)]); NORM[i] = Wv.norm(dim=1).cpu(); Ru = unitr(Wv); RU[i] = Ru.half().cpu(); m = Ru.shape[0]
    X = flat(cap["out"]); XS[i] = X.cpu(); ACT[i] = torch.cat([flat(cap["act"][bb]) for bb in range(B + 1)], 1).half().cpu(); ATT[i] = torch.stack([flat(cap["attn"][bb]) for bb in range(B + 1)], 1).sum(1).cpu()
    U = unitr(X - X.mean(0)); st = stats(U, Ru, K); RATIO[i] = st["ratio"]; SS[i] = st["S"]; WORDS[i] = wordset(st["usage"]); del st, model, cap, Wv, Ru, X, U; torch.cuda.empty_cache(); log(f"{name} step{n}: captured ({int(WORDS[i].sum())} words)")
def top_positions(i, r_): return RATIO[i][:, r_].float().topk(KTOP).indices
def gvec(i, r_, j=None):
    """alignment of every writer with row r_: the writers' norms and directions from checkpoint i, the row's direction from checkpoint j (default i)."""
    j = i if j is None else j; Ru = RU[i].float().to(DEV); return (NORM[i].to(DEV) * (Ru @ RU[j][r_].float().to(DEV)))
def cosr(a, b): return ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-9))
def within(Cm):
    Cn = Cm / Cm.norm(dim=1, keepdim=True).clamp_min(1e-9); S_ = Cn @ Cn.T; iu = torch.triu_indices(Cm.shape[0], Cm.shape[0], 1, device=Cm.device); return float(S_[iu[0], iu[1]].mean())
def coherence(Cm): return float(((Cm.sum(1) ** 2) / (Cm ** 2).sum(1).clamp_min(1e-12) - 1).median())
# ---------- candidates and features ----------
HS = [1, 2, 4, 6, 8]; words = [WORDS[i] for i in range(T)]; NSAMP = 4000; gs = torch.Generator().manual_seed(11)
FEATS = ["S", "cnt75", "act_mean_pos", "act_kurtosis", "act_top5_share", "context_coherence", "row_norm", "norm_growth", "read_align", "coalition_coherence"]
rows_all = []  # per origin t: dict of feature tensors and labels per horizon
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
for t in range(T - 1):
    nonw = ~words[t]; labels = {}; cand = torch.zeros(m, dtype=torch.bool)
    for H in HS:
        if t + H < T: labels[H] = nonw & words[t + H]; cand |= labels[H]
    rest = torch.nonzero(nonw & ~cand)[:, 0]; samp = rest[torch.randperm(rest.numel(), generator=gs)[:NSAMP]]; cand[samp] = True; C = torch.nonzero(cand)[:, 0]; nC = C.numel()
    A = ACT[t][:, C].float().to(DEV); U = XS[t].to(DEV); Uc = U - U.mean(0); Un = unitr(Uc); Ru = RU[t].float().to(DEV); Ru_c = Ru[C.to(DEV)]; nrm = NORM[t].to(DEV)
    S_ = SS[t][C].to(DEV); c75 = (RATIO[t][:, C].float().to(DEV) > 0.75).sum(0).float()
    pos = A.clamp_min(0); mean_pos = pos.mean(0); mu = A.mean(0); var = A.var(0); kurt = ((A - mu[None]) ** 4).mean(0) / var.clamp_min(1e-12) ** 2 - 3
    k5 = max(N // 20, 1); top5 = pos.topk(k5, dim=0).values.sum(0) / pos.sum(0).clamp_min(1e-9); topk16 = A.topk(16, dim=0).indices  # [16, nC]
    ctx = torch.zeros(nC, device=DEV); ralign = torch.zeros(nC, device=DEV); coal = torch.zeros(nC, device=DEV); iu = torch.triu_indices(16, 16, 1, device=DEV); iu4 = torch.triu_indices(4, 4, 1, device=DEV); ACTt = ACT[t]
    for j in range(nC):
        P = topk16[:, j]; Sm = Un[P] @ Un[P].T; ctx[j] = Sm[iu[0], iu[1]].mean(); ms = Uc[P].mean(0); ralign[j] = (ms @ Ru_c[j]) / ms.norm().clamp_min(1e-9)
        r_ = int(C[j]); g = nrm * (Ru @ Ru[r_]); a4 = ACTt[P[:4].cpu()].float().to(DEV); cv = a4 * g[None]; cn = cv / cv.norm(dim=1, keepdim=True).clamp_min(1e-9); Sc = cn @ cn.T; coal[j] = Sc[iu4[0], iu4[1]].mean()
    growth = (NORM[t][C] / NORM[t - 1][C].clamp_min(1e-9)) if t > 0 else torch.ones(nC)
    F = dict(S=S_.cpu(), cnt75=c75.cpu(), act_mean_pos=mean_pos.cpu(), act_kurtosis=kurt.cpu(), act_top5_share=top5.cpu(), context_coherence=ctx.cpu(), row_norm=NORM[t][C], norm_growth=growth, read_align=ralign.cpu(), coalition_coherence=coal.cpu())
    rows_all.append(dict(t=t, C=C, F=F, labels={H: v[C] for H, v in labels.items()})); del A, U, Uc, Un, Ru, Ru_c; torch.cuda.empty_cache(); log(f"{name} origin step{steps[t]}: {nC} candidates, entrants at H=4 {int(labels[4].sum()) if 4 in labels else 0}")
# ---------- single-feature AUCs by horizon ----------
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; res = dict(model=name, block=B, features=FEATS, auc_by_horizon={}, logistic={}, families={})
for H in HS:
    out = {}
    for f in FEATS:
        pos, neg = [], []
        for R_ in rows_all:
            if H not in R_["labels"]: continue
            y = R_["labels"][H]; x = R_["F"][f]; pos.append(x[y]); neg.append(x[~y])
        out[f] = auc(torch.cat(pos), torch.cat(neg)) if pos else None
    res["auc_by_horizon"][str(H)] = out; log(f"{name} AUC for entry at H={H}: " + ", ".join(f"{f} {f2(v)}" for f, v in out.items()))
# ---------- logistic regressions on within-checkpoint percentile ranks, H=4 ----------
def ranks(x): return (x.argsort().argsort().double() / max(x.numel() - 1, 1)).float()
SETS = {"wdd": ["S", "cnt75"], "activation": ["act_mean_pos", "act_kurtosis", "act_top5_share", "context_coherence", "row_norm", "norm_growth"], "activation+geometry": ["act_mean_pos", "act_kurtosis", "act_top5_share", "context_coherence", "row_norm", "norm_growth", "read_align"], "activation+geometry+coalition": ["act_mean_pos", "act_kurtosis", "act_top5_share", "context_coherence", "row_norm", "norm_growth", "read_align", "coalition_coherence"], "all": FEATS, "wdd+coalition": ["S", "cnt75", "coalition_coherence"]}
def fit_eval(train, test, fs, H=4):
    def mat(Rs):
        X, Y = [], []
        for R_ in Rs:
            if H not in R_["labels"]: continue
            X.append(torch.stack([ranks(R_["F"][f]) for f in fs], 1)); Y.append(R_["labels"][H].float())
        return torch.cat(X).to(DEV), torch.cat(Y).to(DEV)
    Xtr, Ytr = mat(train); Xte, Yte = mat(test); w = torch.zeros(len(fs), device=DEV, requires_grad=True); b_ = torch.zeros(1, device=DEV, requires_grad=True); opt = torch.optim.LBFGS([w, b_], lr=0.5, max_iter=200)
    pw = float((Ytr == 0).sum() / Ytr.sum().clamp_min(1))
    def closure():
        opt.zero_grad(); z = Xtr @ w + b_; loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Ytr, pos_weight=torch.tensor(pw, device=DEV)) + 1e-3 * (w ** 2).sum(); loss.backward(); return loss
    with torch.enable_grad(): opt.step(closure)
    with torch.no_grad(): sc = Xte @ w + b_; return auc(sc[Yte == 1], sc[Yte == 0]), {f: round(float(v), 3) for f, v in zip(fs, w)}
even = [R_ for R_ in rows_all if R_["t"] % 2 == 0]; odd = [R_ for R_ in rows_all if R_["t"] % 2 == 1]
for sname, fs in SETS.items():
    a1, w1 = fit_eval(even, odd, fs); a2, w2 = fit_eval(odd, even, fs); res["logistic"][sname] = dict(auc_even_to_odd=a1, auc_odd_to_even=a2, auc_mean=(a1 + a2) / 2 if (a1 is not None and a2 is not None) else None, weights_even_to_odd=w1)
    log(f"{name} logistic on {sname} ({len(fs)} features), entry at H=4, out of sample: AUC {f2(a1)} (even to odd), {f2(a2)} (odd to even); weights {w1}")
L = res["logistic"]; res["increments"] = dict(wdd_adds_to_activation_geometry_coalition=L["all"]["auc_mean"] - L["activation+geometry+coalition"]["auc_mean"], activation_side_adds_to_wdd=L["all"]["auc_mean"] - L["wdd"]["auc_mean"], geometry_adds_to_activation=L["activation+geometry"]["auc_mean"] - L["activation"]["auc_mean"], coalition_adds=L["activation+geometry+coalition"]["auc_mean"] - L["activation+geometry"]["auc_mean"])
inc = res["increments"]; log(f"{name} increments (mean out-of-sample AUC at H=4): WDD added to activation+geometry+coalition {inc['wdd_adds_to_activation_geometry_coalition']:+.3f}; the activation side added to WDD {inc['activation_side_adds_to_wdd']:+.3f}; output geometry added to the activation set {inc['geometry_adds_to_activation']:+.3f}; the coalition added {inc['coalition_adds']:+.3f}")
# ---------- families' input selectivity at step 8000 ----------
i8 = steps.index(8000); wl = torch.nonzero(WORDS[i8])[:, 0].tolist(); Ru8 = RU[i8].float().to(DEV); nrm8 = NORM[i8].to(DEV); Mrows = []
for r_ in wl:
    P = RATIO[i8][:, r_].float().topk(KTOP).indices; act = ACT[i8][P].float().to(DEV); Mrows.append((act * (nrm8 * (Ru8 @ Ru8[r_]))[None]).mean(0))
Mm = torch.stack(Mrows); Mn = Mm / Mm.norm(dim=1, keepdim=True).clamp_min(1e-9); Cw = (Mn @ Mn.T).cpu(); iu = torch.triu_indices(len(wl), len(wl), 1); fam = [(int(i_), int(j_)) for i_, j_ in zip(iu[0].tolist(), iu[1].tolist()) if Cw[i_, j_] > 0.3]
Aw = ACT[i8][:, wl].float().to(DEV); An = (Aw - Aw.mean(0)) / Aw.std(0).clamp_min(1e-6); topA = Aw.topk(16, dim=0).indices.cpu()
def pairstats(pairs):
    cs, ov = [], []
    for i_, j_ in pairs: cs.append(float((An[:, i_] * An[:, j_]).mean())); ov.append(len(set(topA[:, i_].tolist()) & set(topA[:, j_].tolist())) / 16)
    return med(cs), med(ov)
rp = torch.randperm(len(iu[0]), generator=torch.Generator().manual_seed(12))[:max(len(fam), 1)].tolist(); rpairs = [(int(iu[0][x]), int(iu[1][x])) for x in rp]
fc, fo = pairstats(fam); rc, ro = pairstats(rpairs); res["families"] = dict(n_words=len(wl), n_family_pairs=len(fam), family_activation_correlation=fc, family_top16_overlap=fo, random_activation_correlation=rc, random_top16_overlap=ro)
log(f"{name} families at step 8000 ({len(fam)} pairs of {len(wl)} words): activation correlation over positions {f2(fc)} (random word pairs {f2(rc)}), overlap of the sixteen most active positions {f2(fo)} ({f2(ro)})")
A4 = res["auc_by_horizon"]["4"]; A8 = res["auc_by_horizon"].get("8", {}); g_ = lambda x: -9 if x is None else x
res["checks"] = dict(activation_side_predicts=g_(A4["read_align"]) >= 0.85 and all(0.6 <= g_(A4[f]) <= 0.75 for f in ("act_kurtosis", "act_top5_share", "context_coherence")), wdd_adds=inc["wdd_adds_to_activation_geometry_coalition"] >= 0.05 and inc["activation_side_adds_to_wdd"] < 0.03, input_side_does_not_lead_at_8=all(g_(A8.get(f)) <= g_(A8.get("S")) for f in ("act_mean_pos", "act_kurtosis", "act_top5_share", "context_coherence")) if A8 else None, families_share_selectivity=g_(fc) >= 0.3 and g_(rc) < 0.05)
summ = f"{name} block {B}: AUC for entry at H=4: " + ", ".join(f"{f} {f2(v)}" for f, v in A4.items()) + "; at H=8: " + ", ".join(f"{f} {f2(v)}" for f, v in A8.items()) + " | logistic out of sample at H=4: " + ", ".join(f"{k} {f2(v['auc_mean'])}" for k, v in L.items()) + f" | WDD adds {inc['wdd_adds_to_activation_geometry_coalition']:+.3f}, activation side adds {inc['activation_side_adds_to_wdd']:+.3f} | families: activation correlation {f2(fc)} against {f2(rc)}, top-16 overlap {f2(fo)} against {f2(ro)} | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e548_activation_vs_wdd_{name}", res, summ)
