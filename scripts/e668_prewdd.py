"""e668 (session 150; Gemini 4's "Pre-WDD"): can the upstream rows that feed a known attribute neuron be named from the
weights alone, with no data? Qwen2.5-0.5B, the top row of the direct-contribution scan for each attribute (gender: block 4,
row 3019, e620's row; number: block 7, row 870). Data-free: the target's read vector in residual space (its up_proj row,
and as a variant its gate_proj row, times the input RMSNorm gain) against every upstream MLP write column (all blocks below
the target): by |cosine| (Gemini's one-line baseline), by the raw dot product (the virtual weight, norm included) and by
OMP (k = 32); and against the token embeddings. Data-driven reference: each upstream row's activation change between the
poles times its virtual weight, signed toward the target's own marked-pole drive (the first-order direct path). Tests:
(1) are the data-free nominees attribute-selective (|activation change| over its standard deviation on discovery items,
percentile among all upstream rows)? (2) negating the top 8 or 32 nominees with the target untouched: the target's
activation on held-out marked items, the held-out flips, the wikitext KL, against the data-driven feeders, random rows of
the same blocks and the target itself; (3) do the target's embedding scores separate the marked items' word tokens from
the unmarked ones? Uses wdd_demo.py's harness (items, scan, edits, wikitext). Arguments: [--smoke]. Pre-registered in
e668_prereg.json."""
import sys, time, json, random, collections
SMOKE = "--smoke" in sys.argv; sys.argv = [sys.argv[0], "--auto", "--no-color"]
import numpy as np, torch
import wdd_demo as wd
torch.set_grad_enabled(False); t0 = time.time(); DEV = wd.DEV
def log(s): print(f"[e668] {s} ({time.time() - t0:.0f}s)", flush=True)
lm = wd.LM("Qwen/Qwen2.5-0.5B"); IT = {a: lm.items(a) for a in wd.POLES}; SC = {a: lm.scan(IT[a]) for a in wd.POLES}; lm.wiki_setup(n=4 if SMOKE else 12)
KS = [8] if SMOKE else [8, 32]; NRAND = 1 if SMOKE else 3; E = lm.model.get_input_embeddings().weight.detach().float()
def omp1(x, A, k):
    r = x.clone(); S = []
    for _ in range(k):
        c = (A @ r).abs(); c[S] = -1; S.append(int(c.argmax())); As = A[S]; coef = torch.linalg.lstsq(As.T, x[:, None]).solution[:, 0]; r = x - As.T @ coef
    return S
def up_pre(its, b, j, which="up"):
    """the target's up_proj (or gate_proj) pre-activation at each item's word position"""
    groups = collections.defaultdict(list); out = [None] * len(its)
    for n, it in enumerate(its): groups[len(it["ids"])].append(n)
    for idx in groups.values():
        for s0 in range(0, len(idx), lm.BS):
            ch = idx[s0:s0 + lm.BS]; cap = {}; h = getattr(lm.layers[b].mlp, which + "_proj").register_forward_hook(lambda m, a, o: cap.__setitem__("u", o[..., j].float()))
            try: lm.model(torch.tensor([its[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1)
            finally: h.remove()
            for k, n in enumerate(ch): out[n] = float(cap["u"][k, its[n]["p"]])
    return torch.tensor(out)
def target_act(its, b, j):
    lm.run(its, capture=True); return torch.tensor([float(it["A"][b, j]) for it in its])
RES = dict(model=lm.hf, smoke=SMOKE, targets={})
for a in wd.POLES:
    b, j = SC[a]["order"][0]; lay = lm.layers[b]; g = lay.post_attention_layernorm.weight.detach().float()
    r_up = g * lay.mlp.up_proj.weight[j].detach().float(); r_gate = g * lay.mlp.gate_proj.weight[j].detach().float(); ru, rg_ = r_up / r_up.norm(), r_gate / r_gate.norm()
    Wup = torch.cat([lm.W(bb).detach().float().T for bb in range(b)]); nrm = Wup.norm(dim=1).clamp_min(1e-12); Uu = Wup / nrm[:, None]; M = Wup.shape[0]
    rows = lambda idx: [divmod(int(i), lm.DFF) for i in idx]
    disc = [it for it in IT[a] if it["disc"]]; A_ = torch.stack([it["A"] for it in disc])[:, :b].reshape(len(disc), -1); s_ = torch.tensor([it["side"] for it in disc])
    da = A_[s_ == 1].mean(0) - A_[s_ == 0].mean(0); sel = da.abs() / A_.std(0).clamp_min(1e-6); selpct = sel.argsort().argsort().float() / (M - 1)
    du = up_pre(disc, b, j); sgn = float(torch.sign(du[s_ == 1].mean() - du[s_ == 0].mean())) or 1.0; dg = up_pre(disc, b, j, "gate"); sgn_g = float(torch.sign(dg[s_ == 1].mean() - dg[s_ == 0].mean())) or 1.0
    vw = (Wup @ r_up).cpu(); dd = sgn * da * vw   # each upstream row's first-order push on the target's up pre-activation toward its marked-pole value
    vwg = (Wup @ r_gate).cpu(); ddg = sgn_g * da * vwg   # the same through the gate
    sc_cos = (Uu @ ru).cpu(); sc_raw = vw; sc_gate = (Uu @ rg_).cpu()
    NOM = {"cos": sc_cos.abs().argsort(descending=True)[:max(KS)].tolist(), "raw": sc_raw.abs().argsort(descending=True)[:max(KS)].tolist(), "gate_cos": sc_gate.abs().argsort(descending=True)[:max(KS)].tolist(),
           "omp": omp1(ru, Uu, max(KS)), "data_driven": dd.argsort(descending=True)[:max(KS)].tolist(), "data_driven_gate": ddg.argsort(descending=True)[:max(KS)].tolist()}
    rg = random.Random(668); NOM.update({f"random{q}": rg.sample(range(M), max(KS)) for q in range(NRAND)})
    sp = lambda x, y: float(np.corrcoef(np.argsort(np.argsort(x.numpy())), np.argsort(np.argsort(y.numpy())))[0, 1])
    t1 = {k: dict(sel_pct_median=float(selpct[v[:32]].median()), sel_pct_mean=float(selpct[v[:32]].mean()), jaccard_dd=len(set(v[:32]) & set(NOM["data_driven"][:32])) / len(set(v[:32]) | set(NOM["data_driven"][:32])), blocks=collections.Counter(int(i) // lm.DFF for i in v[:32])) for k, v in NOM.items()}
    corr = dict(spearman_abscos_sel=sp(sc_cos.abs(), sel), spearman_absraw_sel=sp(sc_raw.abs(), sel), spearman_absgate_sel=sp(sc_gate.abs(), sel), spearman_abscos_absdd=sp(sc_cos.abs(), dd.abs()),
                jaccard_gate8=len(set(NOM["gate_cos"][:8]) & set(NOM["data_driven_gate"][:8])) / len(set(NOM["gate_cos"][:8]) | set(NOM["data_driven_gate"][:8])), jaccard_up8=len(set(NOM["cos"][:8]) & set(NOM["data_driven"][:8])) / len(set(NOM["cos"][:8]) | set(NOM["data_driven"][:8])))
    log(f"{a}: target (block {b}, row {j}); marked-pole sign of its up drive {sgn:+.0f}; {M} upstream rows; selectivity percentile of the top 32 (median): " + ", ".join(f"{k} {v['sel_pct_median']:.2f}" for k, v in t1.items()) + f"; Spearman with selectivity over all upstream rows: |cos| {corr['spearman_abscos_sel']:.3f}, |raw| {corr['spearman_absraw_sel']:.3f}, |gate cos| {corr['spearman_absgate_sel']:.3f}; top-8 Jaccard with data: up {corr['jaccard_up8']:.2f}, gate {corr['jaccard_gate8']:.2f}")
    # (2) surgery with the target untouched
    held = [it for it in IT[a] if not it["disc"]]; hm = [it for it in held if it["side"] == 1]; act0 = target_act(hm, b, j); base = lm.flips(IT[a]); t2 = {}
    for k in KS:
        for name, v in list(NOM.items()) + [("target", None)]:
            if name == "target" and k != KS[0]: continue
            ed = [(b, j)] if name == "target" else rows(v[:k])
            with wd.Negate(lm, ed):
                f = lm.flips(IT[a]); kl = lm.cost()[1]; act1 = target_act(hm, b, j)
            t2[f"{name}_{k}" if name != "target" else "target"] = dict(flip_marked=f[1], flip_unmarked=f[0], kl=kl, target_act_ratio=float(act1.abs().mean() / act0.abs().mean().clamp_min(1e-9)), target_act_mean=float(act1.mean()), target_act0_mean=float(act0.mean()))
        log(f"{a} k={k}: flips marked->unmarked " + ", ".join(f"{n} {t2[n + '_' + str(k)]['flip_marked']:.2f} (target |a| x{t2[n + '_' + str(k)]['target_act_ratio']:.2f}, KL {t2[n + '_' + str(k)]['kl']:.4f})" for n in NOM) + f"; the target itself {t2['target']['flip_marked']:.2f}")
    # (3) the embedding readout
    wt = lambda side: sorted({it["ids"][it["p"]] for it in IT[a] if it["side"] == side}); m1, m0 = wt(1), wt(0); only1, only0 = sorted(set(m1) - set(m0)), sorted(set(m0) - set(m1))
    es = (E @ ru).cpu(); auc = lambda pos, neg, sc=None: float(np.mean([[float((es if sc is None else sc)[p_] > (es if sc is None else sc)[n_]) + 0.5 * float((es if sc is None else sc)[p_] == (es if sc is None else sc)[n_]) for n_ in neg] for p_ in pos]))
    esu = ((E / E.norm(dim=1, keepdim=True).clamp_min(1e-12)) @ ru).cpu(); esg = ((E / E.norm(dim=1, keepdim=True).clamp_min(1e-12)) @ rg_).cpu()
    en1 = sorted({it["cand"][1] for it in IT[a]}); en0 = sorted({it["cand"][0] for it in IT[a]}); en1, en0 = sorted(set(en1) - set(en0)), sorted(set(en0) - set(en1))
    topv = es.abs().argsort(descending=True)[:20].tolist(); toks = [lm.tok.decode([t]) for t in topv]
    t3 = dict(auc_english_unit_up=auc(en1, en0, esu), auc_english_unit_gate=auc(en1, en0, esg), auc_english_raw_up=auc(en1, en0), n_english=(len(en1), len(en0)), english_tokens=([lm.tok.decode([t]) for t in en1], [lm.tok.decode([t]) for t in en0]),
              auc_marked_vs_unmarked=auc(only1, only0), auc_word_unit_up=auc(only1, only0, esu), n_marked=len(only1), n_unmarked=len(only0), top_tokens=toks, top_signs=[float(torch.sign(es[t])) for t in topv], marked_tokens=[lm.tok.decode([t]) for t in only1][:20], unmarked_tokens=[lm.tok.decode([t]) for t in only0][:20])
    log(f"{a}: embedding AUC, English answer tokens (unit embeddings) up {t3['auc_english_unit_up']:.2f}, gate {t3['auc_english_unit_gate']:.2f} ({len(en1)} vs {len(en0)}); word-position tokens {t3['auc_marked_vs_unmarked']:.2f} raw, {t3['auc_word_unit_up']:.2f} unit; top tokens by |raw score|: {toks[:12]}")
    RES["targets"][a] = dict(target=[b, j], sign=sgn, n_upstream=M, t1=t1, corr=corr, t2=t2, t3=t3, base_flips=base, nominees={k: rows(v[:32]) for k, v in NOM.items()},
                             dd_top_vw=[float(vw[i]) for i in NOM["data_driven"][:8]], cos_top_values=[float(sc_cos[i]) for i in NOM["cos"][:8]])
RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S"); RES["seconds"] = time.time() - t0
json.dump(RES, open(f"results/e668_prewdd{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else (dict(o) if isinstance(o, collections.Counter) else str(o))); log("done"); print("done", flush=True)
