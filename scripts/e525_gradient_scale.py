"""e525: does a row's drift follow its gradient, and is the drift a matter of the gradient's consistency? e524b found
that the rows near the floor drift (consecutive thousand-step increments at cosine 0.24-0.31) while the typical row
jitters, and that the entrants' drift is the more directed. e522b could not test whether a row's motion follows its
own loss gradient because 65k tokens estimate a single row's gradient direction at split-half reliability 0.02-0.05.
The row's gradient is the adjoint transport of the loss to its block at the positions where its neuron fires, summed
with the activations: the WDD chain read backwards. Here it is estimated on 3M tokens of the training distribution
(NeelNanda/pile-10k) in two halves of 1.5M, at an origin checkpoint, for every row of blocks 0-12, together with the
neuron's activation statistics on the same text. Per row: the gradient's consistency (the cosine of the two halves'
estimates), its size relative to the row, the cosine of the row's actual accumulated update over the next one, two,
three and four thousand steps (e524's records) with the negative gradient and with its sign (the Adam-like direction),
the negative gradient's cosine with the state at the row's best position and its fraction in the cloud's top-32
subspace, the increment autocorrelation of e524b, and the neuron's rate of firing and mean activation. Groups: the
rows that enter cleanly within the next three thousand steps, S-matched non-entrants, all non-words, the words.
Pre-registered (honest guesses), block 12:
- the thousand-step update follows the gradient: its cosine with the negative gradient is positive for at least 70%
  of non-word rows (0.6);
- the entrants' gradients are more consistent across halves than the matched rows' (AUC 0.6 or above) (0.5);
- the gradient's consistency explains the drift: its Spearman with the increment autocorrelation across all non-words
  is 0.3 or above (0.5);
- the entrants' negative gradient points toward the state at their best position more than the matched rows' (AUC
  0.6 or above) (0.5);
- within deciles of S the gradient's consistency reads entry within three thousand steps at 0.6 or above (0.4).
Arguments: name step (the origin checkpoint, thousands)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; t0 = int(sys.argv[2]) * 1000; CDIR = f"/workspace/wdd/cache/e524_{name}"; BL = [12, 6]; LB = 12; NWORD = 256; NTOK = 1_500_000; SEQ = 512; CH = 8
ck = {k: torch.load(f"{CDIR}/step{t0 + 1000 * k}.pt") for k in range(0, 5)}
model, tok, fam = load_model(name, revision=f"step{t0}"); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF; R = (LB + 1) * DFF
for p in model.parameters(): p.requires_grad_(False)
lins = [arch.mlp_lin(bb) for bb in range(LB + 1)]
for m in lins: m.weight.requires_grad_(True)
def gradient_text():
    from datasets import load_dataset
    need = 2 * NTOK + 1; chunks, n = [], 0; ds = load_dataset("NeelNanda/pile-10k", split="train")
    for t in ds["text"]:
        if not t.strip(): continue
        ii = tok(t).input_ids; chunks.append(torch.tensor(ii)); n += len(ii)
        if n >= need: break
    ids = torch.cat(chunks)[:need]; assert ids.numel() >= need, ids.numel(); nseq = (2 * NTOK) // SEQ
    return ids[:nseq * SEQ].view(nseq, SEQ)
idsG = gradient_text(); half = idsG.shape[0] // 2; H = {1: idsG[:half], 2: idsG[half:]}; log(f"{name} step{t0}: gradient text {idsG.numel()} tokens in two halves of {half} x {SEQ}")
act_pos = torch.zeros(R, dtype=torch.float64, device=DEV); act_sum = torch.zeros(R, dtype=torch.float64, device=DEV); ntok_seen = 0
def mk(bb):
    def f(m, a):
        x = a[0].detach().float().reshape(-1, DFF); act_pos[bb * DFF:(bb + 1) * DFF] += (x > 0).sum(0).double(); act_sum[bb * DFF:(bb + 1) * DFF] += x.sum(0).double()
    return f
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def spearman(a, b):
    if a.numel() < 5: return None
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double(); ra = ra - ra.mean(); rb = rb - rb.mean(); return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))
def strat_auc(q, s, y, nbins=10):
    edges = s.quantile(torch.linspace(0, 1, nbins + 1)); tot, wsum = 0.0, 0
    for i in range(nbins):
        m = (s >= edges[i]) & (s <= edges[i + 1]) if i == nbins - 1 else (s >= edges[i]) & (s < edges[i + 1])
        v = auc(q[m & y], q[m & ~y]); n = int((m & y).sum())
        if v is not None: tot += v * n; wsum += n
    return (tot / wsum) if wsum else None
Gw = {}; lossH = {}
for h in (1, 2):
    for m in lins: m.weight.grad = None
    hs = [lins[bb].register_forward_pre_hook(mk(bb)) for bb in range(LB + 1)]; ids = H[h].to(DEV); ntok = ids.shape[0] * (ids.shape[1] - 1); tot = 0.0
    try:
        for s in range(0, ids.shape[0], CH):
            c = ids[s:s + CH]; wgt = c.shape[0] * (c.shape[1] - 1) / ntok
            with torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16): l = model(c, labels=c, use_cache=False).loss
                ls = l.float() * wgt                                                                                   # grad mode is off globally (wdd_common); the scaling must be recorded
            ls.backward(); tot += float(l) * wgt
    finally: [hh.remove() for hh in hs]
    ntok_seen += ids.numel(); lossH[h] = tot
    Gw[h] = torch.cat([(lins[bb].weight.grad.float() if fam == "gpt2" else lins[bb].weight.grad.float().T) for bb in range(LB + 1)]).clone()
    log(f"{name} step{t0}: half {h} done, loss {tot:.4f}")
for m in lins: m.weight.grad = None; m.weight.requires_grad_(False)
G = (Gw[1] + Gw[2]) / 2; cons = (Gw[1] * Gw[2]).sum(1) / (Gw[1].norm(dim=1) * Gw[2].norm(dim=1)).clamp_min(1e-12); act_rate = (act_pos / ntok_seen).float(); act_mean = (act_sum / ntok_seen).float()
del Gw, model; torch.cuda.empty_cache()
Wv = {k: (ck[k]["rows"].float() * ck[k]["norms"][:, None]).to(DEV) for k in ck}
res = dict(model=name, origin=t0, gradient_tokens=int(idsG.numel()), loss_halves=lossH, by_block={})
for b in BL:
    Rb = (b + 1) * DFF; Bk = {k: ck[k]["blocks"][b] for k in ck}
    words = {}
    for k in ck:
        u = Bk[k]["usage"]; w = torch.zeros(Rb, dtype=torch.bool); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; words[k] = w
    Gb = G[:Rb]; ng = -Gb; ngu = unitr(ng); w0 = Wv[0][:Rb]; wu0 = unitr(w0)
    U0 = Bk[0]["U"].float().to(DEV); a0 = Bk[0]["a"].to(DEV); V0 = Bk[0]["V"].to(DEV); S0 = Bk[0]["S"].float()
    Q = dict(consistency=cons[:Rb].cpu(), gradient_size=(Gb.norm(dim=1) / w0.norm(dim=1).clamp_min(1e-9)).cpu(), toward_best_state=(ngu * U0[a0]).sum(1).cpu(), in_top_32=(ngu @ V0).pow(2).sum(1).cpu(), radial=(ngu * wu0).sum(1).cpu(), act_rate=act_rate[:Rb].cpu(), act_mean=act_mean[:Rb].cpu(), S=S0)
    for k in (1, 2, 3, 4):
        dw = Wv[k][:Rb] - w0; Q[f"follow_{k}"] = ((dw * ng).sum(1) / (dw.norm(dim=1) * ng.norm(dim=1)).clamp_min(1e-12)).cpu()
        if k == 1: Q["follow_sign_1"] = ((dw * torch.sign(ng)).sum(1) / (dw.norm(dim=1) * torch.sign(ng).norm(dim=1)).clamp_min(1e-12)).cpu(); Q["step_rel_1"] = (dw.norm(dim=1) / w0.norm(dim=1).clamp_min(1e-9)).cpu()
    d1 = unitr(Wv[1][:Rb] - w0); d2 = unitr(Wv[2][:Rb] - Wv[1][:Rb]); Q["increment_autocorrelation"] = (d1 * d2).sum(1).cpu()
    # groups: clean entrants within three thousand steps, matched, non-words, words
    nonw0 = ~words[0]; ent = torch.zeros(Rb, dtype=torch.bool); stay = nonw0.clone()
    for k in (1, 2, 3):
        e = stay & ~words[k - 1] & words[k] & words[k + 1]; ent |= e; stay &= ~words[k]
    never = nonw0 & ~words[1] & ~words[2] & ~words[3] & ~words[4]; pool = torch.nonzero(never)[:, 0]; taken = torch.zeros(Rb, dtype=torch.bool); matched = []
    for i in torch.nonzero(ent)[:, 0].tolist():
        c = pool[~taken[pool]]; j = c[(S0[c] - S0[i]).abs().argmin()]; taken[j] = True; matched.append(int(j))
    mm = torch.zeros(Rb, dtype=torch.bool); mm[torch.tensor(matched, dtype=torch.long)] = True
    groups = {"entrants": ent, "matched": mm, "non_words": nonw0, "words": words[0]}
    out = dict(n_entrants=int(ent.sum()), n_non_words=int(nonw0.sum()), n_words=int(words[0].sum()), medians={g: {q: float(v[m].nanmedian()) for q, v in Q.items()} for g, m in groups.items()}, positive_fraction={g: {q: float((Q[q][m] > 0).float().mean()) for q in ("follow_1", "follow_2", "follow_4", "follow_sign_1", "toward_best_state")} for g, m in groups.items()},
               auc_entrants_vs_matched={q: auc(Q[q][ent].nan_to_num(0), Q[q][mm].nan_to_num(0)) for q in ("consistency", "gradient_size", "toward_best_state", "in_top_32", "act_rate", "act_mean", "follow_1", "increment_autocorrelation")},
               within_S_deciles={q: strat_auc(Q[q][nonw0].nan_to_num(0), S0[nonw0], ent[nonw0]) for q in ("consistency", "gradient_size", "toward_best_state", "in_top_32", "act_rate", "act_mean", "increment_autocorrelation")},
               spearman_non_words={f"{x}_vs_{y}": spearman(Q[x][nonw0].nan_to_num(0), Q[y][nonw0].nan_to_num(0)) for x, y in (("consistency", "increment_autocorrelation"), ("consistency", "act_rate"), ("consistency", "S"), ("follow_1", "consistency"), ("follow_1", "act_rate"), ("follow_1", "increment_autocorrelation"), ("gradient_size", "step_rel_1"), ("toward_best_state", "S"), ("act_rate", "S"))})
    res["by_block"][b] = out; e_, m_, a_, w_ = (out["medians"][g] for g in ("entrants", "matched", "non_words", "words")); pf = out["positive_fraction"]; A = out["auc_entrants_vs_matched"]; W_ = out["within_S_deciles"]; sp = out["spearman_non_words"]; fm = lambda x: "n/a" if x is None else f"{x:.2f}"
    log(f"{name} step{t0} block {b} ({out['n_entrants']} entrants within three thousand steps, {out['n_non_words']} non-words): the gradient's split-half consistency, entrants/matched/non-words/words {e_['consistency']:.3f}/{m_['consistency']:.3f}/{a_['consistency']:.3f}/{w_['consistency']:.3f} (AUC entrants vs matched {fm(A['consistency'])}; within S deciles {fm(W_['consistency'])}); the thousand-step update's cosine with the negative gradient {e_['follow_1']:.3f}/{m_['follow_1']:.3f}/{a_['follow_1']:.3f}/{w_['follow_1']:.3f}, positive for {pf['entrants']['follow_1']:.2f}/{pf['matched']['follow_1']:.2f}/{pf['non_words']['follow_1']:.2f}/{pf['words']['follow_1']:.2f}; with its sign {a_['follow_sign_1']:.3f} (positive {pf['non_words']['follow_sign_1']:.2f}); over two/three/four thousand steps {a_['follow_2']:.3f}/{a_['follow_3']:.3f}/{a_['follow_4']:.3f} (entrants {e_['follow_2']:.3f}/{e_['follow_3']:.3f}/{e_['follow_4']:.3f})")
    log(f"{name} step{t0} block {b}: the negative gradient toward the state at the best position {e_['toward_best_state']:.3f}/{m_['toward_best_state']:.3f}/{a_['toward_best_state']:.3f}/{w_['toward_best_state']:.3f} (AUC {fm(A['toward_best_state'])}, within S {fm(W_['toward_best_state'])}), in the top-32 subspace {e_['in_top_32']:.3f}/{m_['in_top_32']:.3f}/{a_['in_top_32']:.3f}/{w_['in_top_32']:.3f} (isotropic {32 / D:.3f}; AUC {fm(A['in_top_32'])}), radial {e_['radial']:.3f}/{m_['radial']:.3f}/{a_['radial']:.3f}; gradient size over row norm {e_['gradient_size']:.2e}/{m_['gradient_size']:.2e}/{a_['gradient_size']:.2e}/{w_['gradient_size']:.2e} (AUC {fm(A['gradient_size'])}); firing rate {e_['act_rate']:.3f}/{m_['act_rate']:.3f}/{a_['act_rate']:.3f}/{w_['act_rate']:.3f} (AUC {fm(A['act_rate'])}, within S {fm(W_['act_rate'])}); increment autocorrelation {e_['increment_autocorrelation']:.2f}/{m_['increment_autocorrelation']:.2f}/{a_['increment_autocorrelation']:.2f}/{w_['increment_autocorrelation']:.2f}")
    log(f"{name} step{t0} block {b}: Spearman over non-words: consistency vs increment autocorrelation {fm(sp['consistency_vs_increment_autocorrelation'])}, consistency vs firing rate {fm(sp['consistency_vs_act_rate'])}, consistency vs S {fm(sp['consistency_vs_S'])}, following vs consistency {fm(sp['follow_1_vs_consistency'])}, following vs firing rate {fm(sp['follow_1_vs_act_rate'])}, following vs autocorrelation {fm(sp['follow_1_vs_increment_autocorrelation'])}, gradient size vs step size {fm(sp['gradient_size_vs_step_rel_1'])}, toward best state vs S {fm(sp['toward_best_state_vs_S'])}, firing rate vs S {fm(sp['act_rate_vs_S'])}")
    del U0, V0, Gb, ng, ngu; torch.cuda.empty_cache()
B = res["by_block"][12]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(update_follows_gradient_70=B["positive_fraction"]["non_words"]["follow_1"] >= 0.7, entrants_more_consistent=g_(B["auc_entrants_vs_matched"]["consistency"]) >= 0.6, consistency_explains_drift=g_(B["spearman_non_words"]["consistency_vs_increment_autocorrelation"]) >= 0.3, gradient_toward_best_state=g_(B["auc_entrants_vs_matched"]["toward_best_state"]) >= 0.6, consistency_within_S=g_(B["within_S_deciles"]["consistency"]) >= 0.6)
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name} step{t0} ({idsG.numel()} gradient tokens): " + " | ".join(f"block {b}: consistency entrants/matched/non-words {o['medians']['entrants']['consistency']:.3f}/{o['medians']['matched']['consistency']:.3f}/{o['medians']['non_words']['consistency']:.3f} (AUC {fm(o['auc_entrants_vs_matched']['consistency'])}, within S {fm(o['within_S_deciles']['consistency'])}); update follows -gradient: cosine {o['medians']['non_words']['follow_1']:.3f}, positive {o['positive_fraction']['non_words']['follow_1']:.2f} (entrants {o['medians']['entrants']['follow_1']:.3f}); -gradient toward best state {o['medians']['entrants']['toward_best_state']:.3f}/{o['medians']['matched']['toward_best_state']:.3f}/{o['medians']['non_words']['toward_best_state']:.3f} (AUC {fm(o['auc_entrants_vs_matched']['toward_best_state'])}); firing rate {o['medians']['entrants']['act_rate']:.3f}/{o['medians']['matched']['act_rate']:.3f}/{o['medians']['non_words']['act_rate']:.3f} (AUC {fm(o['auc_entrants_vs_matched']['act_rate'])}); Spearman consistency vs autocorrelation {fm(o['spearman_non_words']['consistency_vs_increment_autocorrelation'])}, consistency vs firing {fm(o['spearman_non_words']['consistency_vs_act_rate'])}" for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e525_gradient_scale_{name}_step{t0}", res, summ)
