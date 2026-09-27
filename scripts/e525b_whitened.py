"""e525b: is the drift the whitened gradient? e525 found that a row's mean gradient at a checkpoint, estimated on
1.5M tokens per half, agrees with itself between halves at cosine 0.08-0.18 and that the row's actual thousand-step
update follows it at cosine 0.01. Adam does not move a parameter along its mean gradient: it moves it along the mean
divided, coordinate by coordinate, by the root of the running mean square, so that coordinates with small but
consistent gradients move as much as coordinates with large noisy ones. If the rows' drift is that whitened
direction, the raw gradient would look uninformative while the update is fully determined. This run accumulates, over
the same 3M tokens in two halves, the per-micro-batch gradients' mean and mean square for every row of blocks 0-12,
and forms the whitened direction (the mean over the per-coordinate standard deviation). Per row: the split-half
consistency of the raw and of the whitened direction, the cosine of the actual one-, two-, three- and four-thousand-step
updates with each, the whitened direction's cosine with the state at the row's best position and its fraction in the
cloud's top-32 subspace, the median per-coordinate signal-to-noise ratio; entrants within three thousand steps
against S-matched non-entrants, all non-words and the words.
Pre-registered (honest guesses), block 12:
- the thousand-step update follows the whitened direction at a median cosine above 0.05 (0.5), at least three times
  the raw gradient's (0.5);
- the whitened direction is more consistent across halves than the raw gradient (median cosine 0.3 or above) (0.4);
- the entrants' updates follow the whitened direction more than the matched rows' (AUC 0.6 or above) (0.4);
- the entrants' whitened direction points toward the state at their best position more than the matched rows' (AUC
  0.6 or above) (0.4).
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
Gw = {}; Wh = {}; lossH = {}; EPS = 1e-12
grad_of = lambda: torch.cat([(lins[bb].weight.grad.float() if fam == "gpt2" else lins[bb].weight.grad.float().T) for bb in range(LB + 1)])
for h in (1, 2):
    S1 = torch.zeros(R, D, device=DEV, dtype=torch.float64); S2 = torch.zeros(R, D, device=DEV, dtype=torch.float64); nb = 0
    hs = [lins[bb].register_forward_pre_hook(mk(bb)) for bb in range(LB + 1)]; ids = H[h].to(DEV); ntok = ids.shape[0] * (ids.shape[1] - 1); tot = 0.0
    try:
        for s in range(0, ids.shape[0], CH):
            c = ids[s:s + CH]; wgt = c.shape[0] * (c.shape[1] - 1) / ntok
            for m in lins: m.weight.grad = None
            with torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16): l = model(c, labels=c, use_cache=False).loss
                ls = l.float()                                                                                          # the micro-batch's own mean loss; grad mode is off globally (wdd_common)
            ls.backward(); tot += float(l) * wgt; g = grad_of().double(); S1 += g; S2 += g * g; nb += 1
    finally: [hh.remove() for hh in hs]
    ntok_seen += ids.numel(); lossH[h] = tot; mu = S1 / nb; var = (S2 / nb - mu * mu).clamp_min(0); sd = var.sqrt()
    Gw[h] = mu.float(); Wh[h] = (mu / (sd + EPS)).float(); del S1, S2, mu, var, sd
    log(f"{name} step{t0}: half {h} done ({nb} micro-batches), loss {tot:.4f}")
for m in lins: m.weight.grad = None; m.weight.requires_grad_(False)
cosr = lambda A, B: (A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1)).clamp_min(1e-12)
G = (Gw[1] + Gw[2]) / 2; Gwh = (Wh[1] + Wh[2]) / 2; cons = cosr(Gw[1], Gw[2]); cons_white = cosr(Wh[1], Wh[2]); snr = Gwh.abs().median(1).values                                                                                       # median per-coordinate |mean| / sd
act_rate = (act_pos / ntok_seen).float(); act_mean = (act_sum / ntok_seen).float()
del Gw, Wh, model; torch.cuda.empty_cache()
Wv = {k: (ck[k]["rows"].float() * ck[k]["norms"][:, None]).to(DEV) for k in ck}
res = dict(model=name, origin=t0, gradient_tokens=int(idsG.numel()), loss_halves=lossH, by_block={})
for b in BL:
    Rb = (b + 1) * DFF; Bk = {k: ck[k]["blocks"][b] for k in ck}
    words = {}
    for k in ck:
        u = Bk[k]["usage"]; w = torch.zeros(Rb, dtype=torch.bool); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; words[k] = w
    Gb = G[:Rb]; ng = -Gb; ngu = unitr(ng); w0 = Wv[0][:Rb]; wu0 = unitr(w0); nw_ = -Gwh[:Rb]; nwu = unitr(nw_)
    U0 = Bk[0]["U"].float().to(DEV); a0 = Bk[0]["a"].to(DEV); V0 = Bk[0]["V"].to(DEV); S0 = Bk[0]["S"].float()
    Q = dict(consistency=cons[:Rb].cpu(), consistency_white=cons_white[:Rb].cpu(), snr=snr[:Rb].cpu(), gradient_size=(Gb.norm(dim=1) / w0.norm(dim=1).clamp_min(1e-9)).cpu(), toward_best_state=(ngu * U0[a0]).sum(1).cpu(), toward_best_state_white=(nwu * U0[a0]).sum(1).cpu(), in_top_32=(ngu @ V0).pow(2).sum(1).cpu(), in_top_32_white=(nwu @ V0).pow(2).sum(1).cpu(), radial=(ngu * wu0).sum(1).cpu(), radial_white=(nwu * wu0).sum(1).cpu(), act_rate=act_rate[:Rb].cpu(), act_mean=act_mean[:Rb].cpu(), S=S0)
    for k in (1, 2, 3, 4):
        dw = Wv[k][:Rb] - w0; Q[f"follow_{k}"] = ((dw * ng).sum(1) / (dw.norm(dim=1) * ng.norm(dim=1)).clamp_min(1e-12)).cpu(); Q[f"follow_white_{k}"] = ((dw * nw_).sum(1) / (dw.norm(dim=1) * nw_.norm(dim=1)).clamp_min(1e-12)).cpu()
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
    out = dict(n_entrants=int(ent.sum()), n_non_words=int(nonw0.sum()), n_words=int(words[0].sum()), medians={g: {q: float(v[m].nanmedian()) for q, v in Q.items()} for g, m in groups.items()}, positive_fraction={g: {q: float((Q[q][m] > 0).float().mean()) for q in ("follow_1", "follow_2", "follow_4", "follow_sign_1", "toward_best_state", "follow_white_1", "follow_white_2", "follow_white_4", "toward_best_state_white")} for g, m in groups.items()},
               auc_entrants_vs_matched={q: auc(Q[q][ent].nan_to_num(0), Q[q][mm].nan_to_num(0)) for q in ("consistency", "consistency_white", "snr", "gradient_size", "toward_best_state", "toward_best_state_white", "in_top_32", "in_top_32_white", "act_rate", "act_mean", "follow_1", "follow_white_1", "increment_autocorrelation")},
               within_S_deciles={q: strat_auc(Q[q][nonw0].nan_to_num(0), S0[nonw0], ent[nonw0]) for q in ("consistency", "consistency_white", "snr", "gradient_size", "toward_best_state", "toward_best_state_white", "in_top_32", "in_top_32_white", "act_rate", "act_mean", "increment_autocorrelation")},
               spearman_non_words={f"{x}_vs_{y}": spearman(Q[x][nonw0].nan_to_num(0), Q[y][nonw0].nan_to_num(0)) for x, y in (("consistency", "increment_autocorrelation"), ("consistency_white", "increment_autocorrelation"), ("consistency", "act_rate"), ("consistency_white", "act_rate"), ("consistency", "S"), ("follow_1", "consistency"), ("follow_white_1", "consistency_white"), ("follow_white_1", "increment_autocorrelation"), ("follow_1", "act_rate"), ("follow_1", "increment_autocorrelation"), ("gradient_size", "step_rel_1"), ("toward_best_state", "S"), ("act_rate", "S"), ("snr", "consistency_white"))})
    res["by_block"][b] = out; e_, m_, a_, w_ = (out["medians"][g] for g in ("entrants", "matched", "non_words", "words")); pf = out["positive_fraction"]; A = out["auc_entrants_vs_matched"]; W_ = out["within_S_deciles"]; sp = out["spearman_non_words"]; fm = lambda x: "n/a" if x is None else f"{x:.2f}"
    log(f"{name} step{t0} block {b} ({out['n_entrants']} entrants within three thousand steps, {out['n_non_words']} non-words): split-half consistency raw / whitened, entrants/matched/non-words/words {e_['consistency']:.3f}/{m_['consistency']:.3f}/{a_['consistency']:.3f}/{w_['consistency']:.3f} / {e_['consistency_white']:.3f}/{m_['consistency_white']:.3f}/{a_['consistency_white']:.3f}/{w_['consistency_white']:.3f} (AUC entrants vs matched {fm(A['consistency'])} / {fm(A['consistency_white'])}); median per-coordinate signal-to-noise {a_['snr']:.3f} (entrants {e_['snr']:.3f}); the thousand-step update's cosine with the raw / whitened direction, non-words {a_['follow_1']:.3f} / {a_['follow_white_1']:.3f} (positive {pf['non_words']['follow_1']:.2f} / {pf['non_words']['follow_white_1']:.2f}; entrants {e_['follow_1']:.3f} / {e_['follow_white_1']:.3f}, matched {m_['follow_1']:.3f} / {m_['follow_white_1']:.3f}, words {w_['follow_1']:.3f} / {w_['follow_white_1']:.3f}; AUC entrants vs matched {fm(A['follow_1'])} / {fm(A['follow_white_1'])}); over two/three/four thousand steps raw {a_['follow_2']:.3f}/{a_['follow_3']:.3f}/{a_['follow_4']:.3f}, whitened {a_['follow_white_2']:.3f}/{a_['follow_white_3']:.3f}/{a_['follow_white_4']:.3f}")
    log(f"{name} step{t0} block {b}: toward the state at the best position raw / whitened, entrants/matched/non-words {e_['toward_best_state']:.3f}/{m_['toward_best_state']:.3f}/{a_['toward_best_state']:.3f} / {e_['toward_best_state_white']:.3f}/{m_['toward_best_state_white']:.3f}/{a_['toward_best_state_white']:.3f} (AUC {fm(A['toward_best_state'])} / {fm(A['toward_best_state_white'])}; within S {fm(W_['toward_best_state'])} / {fm(W_['toward_best_state_white'])}); in the top-32 subspace raw / whitened {a_['in_top_32']:.3f} / {a_['in_top_32_white']:.3f} (entrants {e_['in_top_32']:.3f} / {e_['in_top_32_white']:.3f}; isotropic {32 / D:.3f}); radial raw / whitened {a_['radial']:.3f} / {a_['radial_white']:.3f}; firing rate {e_['act_rate']:.3f}/{m_['act_rate']:.3f}/{a_['act_rate']:.3f} (AUC {fm(A['act_rate'])}); increment autocorrelation {e_['increment_autocorrelation']:.2f}/{m_['increment_autocorrelation']:.2f}/{a_['increment_autocorrelation']:.2f}")
    log(f"{name} step{t0} block {b}: Spearman over non-words: consistency raw / whitened vs increment autocorrelation {fm(sp['consistency_vs_increment_autocorrelation'])} / {fm(sp['consistency_white_vs_increment_autocorrelation'])}, vs firing rate {fm(sp['consistency_vs_act_rate'])} / {fm(sp['consistency_white_vs_act_rate'])}; following raw vs consistency {fm(sp['follow_1_vs_consistency'])}, whitened vs its consistency {fm(sp['follow_white_1_vs_consistency_white'])}, whitened following vs autocorrelation {fm(sp['follow_white_1_vs_increment_autocorrelation'])}; signal-to-noise vs whitened consistency {fm(sp['snr_vs_consistency_white'])}; gradient size vs step size {fm(sp['gradient_size_vs_step_rel_1'])}")
    del U0, V0, Gb, ng, ngu; torch.cuda.empty_cache()
B = res["by_block"][12]; g_ = lambda x: -9 if x is None else x; a_ = B["medians"]["non_words"]
res["checks"] = dict(whitened_followed_over_0_05=a_["follow_white_1"] > 0.05, whitened_three_times_raw=a_["follow_white_1"] >= 3 * max(a_["follow_1"], 1e-6), whitened_consistency_over_0_3=a_["consistency_white"] >= 0.3, entrants_follow_more=g_(B["auc_entrants_vs_matched"]["follow_white_1"]) >= 0.6, whitened_toward_best_state=g_(B["auc_entrants_vs_matched"]["toward_best_state_white"]) >= 0.6)
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name} step{t0} ({idsG.numel()} gradient tokens, whitened): " + " | ".join(f"block {b}: split-half consistency raw / whitened, non-words {o['medians']['non_words']['consistency']:.3f} / {o['medians']['non_words']['consistency_white']:.3f} (entrants {o['medians']['entrants']['consistency']:.3f} / {o['medians']['entrants']['consistency_white']:.3f}); the thousand-step update follows the raw / whitened direction at {o['medians']['non_words']['follow_1']:.3f} / {o['medians']['non_words']['follow_white_1']:.3f} (positive {o['positive_fraction']['non_words']['follow_1']:.2f} / {o['positive_fraction']['non_words']['follow_white_1']:.2f}; entrants {o['medians']['entrants']['follow_white_1']:.3f}, matched {o['medians']['matched']['follow_white_1']:.3f}, AUC {fm(o['auc_entrants_vs_matched']['follow_white_1'])}); whitened direction toward the best state entrants/matched/non-words {o['medians']['entrants']['toward_best_state_white']:.3f}/{o['medians']['matched']['toward_best_state_white']:.3f}/{o['medians']['non_words']['toward_best_state_white']:.3f} (AUC {fm(o['auc_entrants_vs_matched']['toward_best_state_white'])}); signal-to-noise per coordinate {o['medians']['non_words']['snr']:.3f}" for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e525b_whitened_{name}_step{t0}", res, summ)
