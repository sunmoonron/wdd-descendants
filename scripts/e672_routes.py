"""e672 (session 152; Cammarata 2026, "Neural Networks Are Already Clean": recursive generalization, tested where the essay
could not, on public training checkpoints). The essay groups a neuron's top-100 firings by the upstream neurons that fed it
("route share") and predicts that apparent polysemanticity is mostly a main job with local offshoots that (1) are much less
causally important than the main job and (2) grow out from it over training "in a particular order". Pythia-160m, block 6
(3,072 GELU neurons, parallel residual), 256 windows of 256 tokens from wikitext-2 train, the final checkpoint (143000):
for 300 sampled neurons with a substantial top-100, each top firing's pre-activation is split exactly (through the block's
LayerNorm at that position) into the contributions of every upstream MLP neuron of blocks 0-5, the six attention outputs and
the embedding. Two groupings of the firings into a core route and an offshoot: "cluster" (primary), cosine k-means (k = 2)
on the contributions of the 256 largest units, a neuron multi-route when the split's silhouette beats the 95th percentile of
20 unimodal nulls with the same mean and covariance (the firings mixed by a random rotation, as in SigClust) and the minor route
holds ten firings or more, the core the larger route; "top upstream" (the essay's figure), each firing by the
upstream MLP neuron that adds the most, the core the most common one and the offshoot the rest. Causal: the neuron zeroed in
every window that holds its firings, at each firing the KL divergence of the next-token distribution (primary) and the drop
in the next token's log probability, core against offshoot, raw and per unit of activation; reference, the strongest firings
against the rest. Developmental: at 11 checkpoints (128 to 143000) each firing's percentile among all positions for that
neuron; a firing's early score is its mean percentile before the final checkpoint; the lead is the core's mean early score
minus the offshoot's after regressing out final activation (primary), against 200 random splits of the same sizes; secondary,
each route's onset (the first checkpoint from which half of it stays in the neuron's top 1%) and the strength split. Saves per-neuron damage
for e673. Arguments: [--smoke]. Pre-registered in e672_prereg.json."""
import sys, os, time, json, math, random, collections
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False); DEV = "cuda:0" if torch.cuda.is_available() else "cpu"; t0 = time.time()
SMOKE = "--smoke" in sys.argv; HF = "EleutherAI/pythia-160m"; B = 6; T = 256
NWIN = 48 if SMOKE else 256; NNEU = 40 if SMOKE else 300; TOPN = 100; KUNITS = 256; NSPLIT = 200; NNULL = 20; MINOFF = 10
CKPTS = [512, 1000, 4000, 16000, 143000] if SMOKE else [128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
def log(s): print(f"[e672] {s} ({time.time() - t0:.0f}s)", flush=True)
tok = AutoTokenizer.from_pretrained(HF)
def load(step): return AutoModelForCausalLM.from_pretrained(HF, revision=("main" if step == 143000 else f"step{step}"), dtype=torch.float32).to(DEV).eval()
import datasets
text = "\n".join(t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
IDS = tok(text, return_tensors="pt").input_ids[0]; starts = np.linspace(0, len(IDS) - T - 1, NWIN).astype(int); Wc = torch.stack([IDS[s:s + T] for s in starts]); W = Wc.to(DEV)
P = T - 2; N = NWIN * P; TOPPCT = max(TOPN, N // 100)   # positions 1..T-2 of each window (no sink, a next token always exists); a neuron's top 1% of them
win_of = lambda n: n // P; pos_of = lambda n: n % P + 1
class Stop(Exception): pass
def capture(model, full):
    """block-6 MLP activations [N, DFF]; with full, the upstream MLP activations of blocks 0-5, the attention outputs, the embedding and the block-6 input"""
    L = model.gpt_neox.layers; out = collections.defaultdict(list)
    for s0 in range(0, NWIN, 8):
        cap = {}; hs = [L[B].mlp.dense_4h_to_h.register_forward_pre_hook(lambda m, a: cap.__setitem__("a6", a[0][:, 1:T - 1].detach().half()))]
        def stop(m, a, o): raise Stop
        hs.append(L[B].register_forward_hook(stop))
        if full:
            for b in range(B):
                hs.append(L[b].mlp.dense_4h_to_h.register_forward_pre_hook(lambda m, a, b=b: cap.__setitem__(("up", b), a[0][:, 1:T - 1].detach().half())))
                hs.append(L[b].attention.register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(("att", b), (o[0] if isinstance(o, tuple) else o)[:, 1:T - 1].detach().half())))
            hs.append(model.gpt_neox.embed_in.register_forward_hook(lambda m, a, o: cap.__setitem__("emb", o[:, 1:T - 1].detach().half())))
            hs.append(L[B].register_forward_pre_hook(lambda m, a, k: cap.__setitem__("xin", (a[0] if a else k["hidden_states"])[:, 1:T - 1].detach().float()), with_kwargs=True))
        try: model(W[s0:s0 + 8])
        except Stop: pass
        finally: [h.remove() for h in hs]
        out["a6"].append(cap["a6"].reshape(-1, cap["a6"].shape[-1]))
        if full:
            out["up"].append(torch.cat([cap[("up", b)] for b in range(B)], -1).reshape(-1, B * cap["a6"].shape[-1]).cpu())
            out["att"].append(torch.stack([cap[("att", b)] for b in range(B)], 2).reshape(-1, B, cap["emb"].shape[-1]).cpu()); out["emb"].append(cap["emb"].reshape(-1, cap["emb"].shape[-1]).cpu()); out["xin"].append(cap["xin"].reshape(-1, cap["xin"].shape[-1]))
    return {k: torch.cat(v) for k, v in out.items()}
def binom_p(k, n):
    """one-sided P(X >= k) for X ~ Binomial(n, 1/2)"""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else None
# ---------------- the final checkpoint: routes
model = load(143000); L = model.gpt_neox.layers; cfg = model.config; D, DFF = cfg.hidden_size, cfg.intermediate_size
C = capture(model, True); A6 = C["a6"]; log(f"captured {N} positions")
ln = L[B].post_attention_layernorm; gam, bet = ln.weight.detach().float(), ln.bias.detach().float(); xin = C["xin"]; sig = (xin.var(-1, unbiased=False) + ln.eps).sqrt()
WIN_ = L[B].mlp.dense_h_to_4h.weight.detach().float(); BIN = L[B].mlp.dense_h_to_4h.bias.detach().float()
WUP = torch.cat([L[b].mlp.dense_4h_to_h.weight.detach().float().T for b in range(B)])   # [6*DFF, D] upstream write rows
BIAS = sum(L[b].mlp.dense_4h_to_h.bias.detach().float() for b in range(B))
mx = A6.float().max(0).values; a100 = A6.float().topk(TOPN, dim=0).values[-1]
cand = torch.nonzero((mx > 1.0) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(672).sample(cand, min(NNEU, len(cand))))
log(f"{len(cand)} of {DFF} block-6 neurons have a substantial top-{TOPN}; {len(NEU)} sampled")
def unit_name(u):
    if u < B * DFF: return f"L{u // DFF}:{u % DFF}"
    return f"attn{u - B * DFF}" if u < B * DFF + B else "embedding"
def kmeans2(X, iters=30, inits=10, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed); best = None
    for _ in range(inits):
        Cc = X[torch.randperm(X.shape[0], generator=g)[:2]].clone()
        for _ in range(iters):
            lab = (X @ Cc.T).argmax(1); Cc = torch.stack([X[lab == k].mean(0) if (lab == k).any() else Cc[k] for k in range(2)]); Cc = Cc / Cc.norm(dim=1, keepdim=True).clamp_min(1e-9)
        lab = (X @ Cc.T).argmax(1); obj = float((X @ Cc.T).max(1).values.sum())
        if best is None or obj > best[0]: best = (obj, lab.clone())
    return best[1]
def silhouette(X, lab):
    Dm = 1 - X @ X.T; M = torch.nn.functional.one_hot(lab, 2).float(); S = Dm @ M; n = M.sum(0)
    if (n < 2).any(): return 0.0
    own = S.gather(1, lab[:, None])[:, 0] / (n[lab] - 1); oth = S.gather(1, (1 - lab)[:, None])[:, 0] / n[1 - lab]
    return float(((oth - own) / torch.maximum(own, oth).clamp_min(1e-9)).mean())
rownorm = lambda X: X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
def null_silhouettes(Xraw, seed):
    """the same pipeline on unimodal data with the firings' mean and covariance: the centred firings mixed by a random rotation"""
    g = torch.Generator().manual_seed(seed); mu = Xraw.mean(0, keepdim=True); Xc = Xraw - mu; out = []
    for _ in range(NNULL):
        Q, R = torch.linalg.qr(torch.randn(Xraw.shape[0], Xraw.shape[0], generator=g)); Q = Q * torch.sign(torch.diagonal(R))[None]
        Xn = rownorm(mu + Q @ Xc); out.append(silhouette(Xn, kmeans2(Xn, inits=5)))
    return out
def context(n, k=8):
    w, p = win_of(int(n)), pos_of(int(n))
    return tok.decode(Wc[w, max(0, p - k):p].tolist()) + " [[" + tok.decode([int(Wc[w, p])]) + "]] " + tok.decode([int(Wc[w, p + 1])])
def route_tokens(idx, mask):
    sel = [int(n) for n in idx[mask]]
    return dict(current=collections.Counter(tok.decode([int(Wc[win_of(n), pos_of(n)])]) for n in sel).most_common(3), examples=[context(n) for n in sel[:3]])
NR = {}; recon = []
for j in NEU:
    tv, ti = A6[:, j].float().topk(TOPN); idx = ti.cpu(); r = WIN_[j] * gam; r = r - r.mean()
    Vj = WUP @ r; s_ = sig[ti]
    cm = C["up"][idx].to(DEV).float() * Vj[None] / s_[:, None]; ca = (C["att"][idx].to(DEV).float() @ r) / s_[:, None]; ce = (C["emb"][idx].to(DEV).float() @ r) / s_
    Cj = torch.cat([cm, ca, ce[:, None]], 1)
    h_true = (xin[ti] - xin[ti].mean(-1, keepdim=True)) / s_[:, None] @ (WIN_[j] * gam) + WIN_[j] @ bet + BIN[j]
    h_rec = Cj.sum(1) + (BIAS @ r) / s_ + WIN_[j] @ bet + BIN[j]; recon.append(float((h_rec - h_true).abs().max() / h_true.abs().max().clamp_min(1e-9)))
    units = Cj.abs().sum(0).topk(KUNITS).indices; Xraw = Cj[:, units].cpu(); X = rownorm(Xraw)
    lab = kmeans2(X); sil = silhouette(X, lab); n1 = int(lab.sum()); core = 1 if n1 >= TOPN - n1 else 0; is_core = (lab == core); nulls = null_silhouettes(Xraw, j)
    topup = cm.argmax(1).cpu(); cnt = collections.Counter(topup.tolist()); u0, n0 = cnt.most_common(1)[0]; is_top = topup == u0
    drivers = lambda mask: [unit_name(int(units[u])) for u in Cj[mask.to(DEV)][:, units].sum(0).argsort(descending=True)[:4].tolist()]
    NR[j] = dict(idx=idx, act=tv.cpu(), is_core=is_core, is_top=is_top, silhouette=sil, null_sil95=float(np.percentile(nulls, 95)), null_sil_mean=float(np.mean(nulls)), n_core=int(is_core.sum()), n_off=int((~is_core).sum()),
                 drivers=dict(core=drivers(is_core), offshoot=drivers(~is_core)), top_unit=unit_name(u0), top_share=n0 / TOPN, n_distinct_top=len(cnt),
                 drive=dict(pre_activation=float(h_true.mean()), mlp=float(cm.sum(1).mean()), attention=float(ca.sum(1).mean()), embedding=float(ce.mean())),
                 tokens=dict(core=route_tokens(idx, is_core), offshoot=route_tokens(idx, ~is_core)), cur=torch.tensor([int(Wc[win_of(int(n)), pos_of(int(n))]) for n in idx]))
SPLITS = dict(cluster={j: NR[j]["is_core"] for j in NEU}, top_upstream={j: NR[j]["is_top"] for j in NEU})
ELIG = dict(cluster=[j for j in NEU if NR[j]["silhouette"] > NR[j]["null_sil95"] and NR[j]["n_off"] >= MINOFF], top_upstream=[j for j in NEU if MINOFF <= int(NR[j]["is_top"].sum()) <= TOPN - MINOFF])
med = lambda xs: float(np.median(xs)) if len(xs) else None
log(f"cluster split: {len(ELIG['cluster'])} of {len(NEU)} neurons multi-route (silhouette median {med([NR[j]['silhouette'] for j in NEU]):.2f}, unimodal-null mean {med([NR[j]['null_sil_mean'] for j in NEU]):.2f} and 95th percentile {med([NR[j]['null_sil95'] for j in NEU]):.2f}; core share median {med([NR[j]['n_core'] / TOPN for j in ELIG['cluster']])}); "
    f"top-upstream split: the most common top upstream neuron carries a median {med([NR[j]['top_share'] for j in NEU]):.2f} of the top {TOPN} ({len(ELIG['top_upstream'])} neurons testable); net drive at the firings (median), pre-activation "
    f"{med([NR[j]['drive']['pre_activation'] for j in NEU]):.2f} = upstream MLP {med([NR[j]['drive']['mlp'] for j in NEU]):.2f} + attention {med([NR[j]['drive']['attention'] for j in NEU]):.2f} + embedding {med([NR[j]['drive']['embedding'] for j in NEU]):.2f} + constants; max relative error of the exact split {max(recon):.1e}")
# ---------------- causal: the neuron removed, the next token's log probability at its firings
def ablate(model, j, idx):
    """per firing: the drop in the next token's log probability and the KL divergence of the next-token distribution, neuron j zeroed in the window (model as is and ablated in one batch)"""
    byw = collections.defaultdict(list)
    for i, n in enumerate(idx.tolist()): byw[win_of(n)].append((i, pos_of(n)))
    wins = sorted(byw); dlp = torch.zeros(TOPN); kl = torch.zeros(TOPN)
    for s0 in range(0, len(wins), 8):
        ws = wins[s0:s0 + 8]; x = W[ws]; nb = len(ws)
        def hook(m, a):
            h = a[0].clone(); h[nb:, :, j] = 0.0; return (h,)
        hd = model.gpt_neox.layers[B].mlp.dense_4h_to_h.register_forward_pre_hook(hook)
        try: lg = model(torch.cat([x, x])).logits
        finally: hd.remove()
        rows = [(k, i, p) for k, w in enumerate(ws) for i, p in byw[w]]; kk = torch.tensor([r[0] for r in rows], device=DEV); ii = torch.tensor([r[1] for r in rows]); pp = torch.tensor([r[2] for r in rows], device=DEV)
        lb = lg[kk, pp].float().log_softmax(-1); la = lg[kk + nb, pp].float().log_softmax(-1); tg = x[kk, pp + 1][:, None]; del lg
        dlp[ii] = (lb.gather(-1, tg) - la.gather(-1, tg))[:, 0].cpu(); kl[ii] = (lb.exp() * (lb - la)).sum(-1).clamp_min(0).cpu()
    return dlp, kl
DMG, KLD = {}, {}
for j in NEU: DMG[j], KLD[j] = ablate(model, j, NR[j]["idx"])
log(f"ablations done; per firing, median over neurons: KL {med([float(KLD[j].mean()) for j in NEU]):.2e} nats, log-probability drop {med([float(DMG[j].mean()) for j in NEU]):.2e} nats")
def causal(split, elig):
    rows = {}
    for j in elig:
        d, k, a, c = DMG[j], KLD[j], NR[j]["act"], split[j]; st = torch.zeros(TOPN, dtype=torch.bool); st[:int(c.sum())] = True   # act is sorted, so the strongest firings come first
        g = k.sqrt() / a   # KL grows with the square of a small perturbation: the root per unit of activation
        rows[j] = dict(kl_core=float(k[c].mean()), kl_off=float(k[~c].mean()), dlp_core=float(d[c].mean()), dlp_off=float(d[~c].mean()), gain_core=float(g[c].mean()), gain_off=float(g[~c].mean()),
                       gain_strong=float(g[st].mean()), gain_weak=float(g[~st].mean()), act_core=float(a[c].mean()), act_off=float(a[~c].mean()), overlap_strong=float((c & st).sum() / max(int(c.sum()), 1)))
    n = len(rows); cnt = lambda f: sum(bool(f(v)) for v in rows.values())
    k_kl, k_dlp, k_g, k_st = cnt(lambda v: v["kl_off"] < v["kl_core"]), cnt(lambda v: v["dlp_off"] < v["dlp_core"]), cnt(lambda v: v["gain_off"] < v["gain_core"]), cnt(lambda v: v["gain_weak"] < v["gain_strong"])
    sh = lambda k: k / n if n else None; vals = lambda key: med([v[key] for v in rows.values()])
    return rows, dict(n=n, off_less_kl=k_kl, off_less_kl_share=sh(k_kl), p_off_less_kl=binom_p(k_kl, n), off_less_dlp_share=sh(k_dlp), p_off_less_dlp=binom_p(k_dlp, n), off_less_gain_share=sh(k_g), p_off_less_gain=binom_p(k_g, n),
                      weak_less_gain_share=sh(k_st), p_weak_less_gain=binom_p(k_st, n), median_kl_core=vals("kl_core"), median_kl_off=vals("kl_off"), median_kl_ratio=med([v["kl_off"] / max(v["kl_core"], 1e-12) for v in rows.values()]),
                      median_dlp_core=vals("dlp_core"), median_dlp_off=vals("dlp_off"), median_act_core=vals("act_core"), median_act_off=vals("act_off"), median_overlap_strong=vals("overlap_strong"))
CAUSAL = {}
for nm in SPLITS:
    for scope, el in (("multi", ELIG[nm]), ("all", NEU)):
        rows, sm = causal(SPLITS[nm], el); CAUSAL[f"{nm}/{scope}"] = dict(summary=sm, rows=rows)
        log(f"causal {nm}/{scope} ({sm['n']} neurons): removing the neuron moves the next-token distribution less at offshoot firings than at core firings in {sm['off_less_kl_share']} (p {sm['p_off_less_kl']}; median KL core {sm['median_kl_core']} / offshoot {sm['median_kl_off']}, ratio {sm['median_kl_ratio']}); "
            f"log-probability drop {sm['off_less_dlp_share']} (p {sm['p_off_less_dlp']}); per unit of activation {sm['off_less_gain_share']} (p {sm['p_off_less_gain']}); strongest-vs-rest reference {sm['weak_less_gain_share']}; activation core {sm['median_act_core']} / offshoot {sm['median_act_off']}")
del model, L; torch.cuda.empty_cache()
# ---------------- developmental: each route over training
NEUt = torch.tensor(NEU, device=DEV); IDXM = torch.stack([NR[j]["idx"] for j in NEU], 1).to(DEV); SEL, ACT, PCT = {}, {}, {}
for step in CKPTS:
    m_ = load(step); a6 = capture(m_, False)["a6"]; del m_
    sub = a6[:, NEUt].float(); thr = sub.topk(TOPPCT, dim=0).values[-1]; vals = sub.gather(0, IDXM)
    PCT[step] = (torch.searchsorted(sub.T.contiguous().sort(1).values, vals.T.contiguous()).float() / N).cpu()   # [neurons, TOPN]: share of positions below each firing
    SEL[step] = (vals >= thr[None]).T.cpu(); ACT[step] = vals.T.cpu(); del a6, sub; torch.cuda.empty_cache()
    log(f"checkpoint {step}: the final top-{TOPN} firings sit at a median percentile {float(PCT[step].median()):.3f}; share inside the neuron's top 1%, median over neurons {float(SEL[step].float().mean(1).median()):.2f}")
KI = {j: k for k, j in enumerate(NEU)}
def curves(j, mask): return [float(SEL[s][KI[j]][mask].float().mean()) for s in CKPTS], [float(ACT[s][KI[j]][mask].mean()) for s in CKPTS]
def onset_sel(cv):
    for i in range(len(cv)):
        if all(x >= 0.5 for x in cv[i:]): return i
    return None
def onset_act(cv):
    if cv[-1] <= 0: return None
    for i in range(len(cv)):
        if all(x >= 0.5 * cv[-1] for x in cv[i:]): return i
    return None
cmp_ = lambda a, b: (None if a is None or b is None else (-1 if a < b else (1 if a > b else 0)))
EARLY = torch.stack([PCT[s] for s in CKPTS[:-1]]).mean(0)   # [neurons, TOPN]: mean percentile before the final checkpoint
def develop(nm, split, elig):
    rows = {}; pool = collections.Counter()
    for j in elig:
        c = split[j]; nc = int(c.sum()); st = torch.zeros(TOPN, dtype=torch.bool); st[:nc] = True
        if nc in (0, TOPN): continue
        sc, ac = curves(j, c); so, ao = curves(j, ~c); ss, _ = curves(j, st); sw, _ = curves(j, ~st)
        e = EARLY[KI[j]].double(); a = NR[j]["act"].double(); Xd = torch.stack([torch.ones_like(a), a], 1); res = e - Xd @ torch.linalg.lstsq(Xd, e[:, None]).solution[:, 0]
        lead = float(e[c].mean() - e[~c].mean()); lead_res = float(res[c].mean() - res[~c].mean())
        cur = NR[j]["cur"].tolist(); tc = collections.Counter(cur); typ = torch.tensor([tc[t] / TOPN for t in cur], dtype=torch.float64)   # how common the firing's own token is among the neuron's top firings
        X3 = torch.stack([torch.ones_like(a), a, typ], 1); res3 = e - X3 @ torch.linalg.lstsq(X3, e[:, None]).solution[:, 0]; lead_res3 = float(res3[c].mean() - res3[~c].mean())
        conc = lambda m: max(collections.Counter(np.array(cur)[m.numpy()].tolist()).values()) / max(int(m.sum()), 1)
        rng = random.Random((6720 if nm == "cluster" else 16720) + j); nl, nr_ = [], []
        for q in range(NSPLIT):
            mk = torch.zeros(TOPN, dtype=torch.bool); mk[rng.sample(range(TOPN), nc)] = True; nl.append(float(e[mk].mean() - e[~mk].mean())); nr_.append(float(res[mk].mean() - res[~mk].mean()))
            if q < 20: x = cmp_(onset_sel(curves(j, mk)[0]), onset_sel(curves(j, ~mk)[0])); pool[x] += 1
        pct_c = [float(PCT[s][KI[j]][c].mean()) for s in CKPTS]; pct_o = [float(PCT[s][KI[j]][~c].mean()) for s in CKPTS]
        rows[j] = dict(lead=lead, lead_res=lead_res, lead_res_tok=lead_res3, tokconc_core=conc(c), tokconc_off=conc(~c), z=(lead - np.mean(nl)) / max(np.std(nl), 1e-12), z_res=(lead_res - np.mean(nr_)) / max(np.std(nr_), 1e-12), lead_strength=float(e[st].mean() - e[~st].mean()),
                       onset_core=onset_sel(sc), onset_off=onset_sel(so), cmp=cmp_(onset_sel(sc), onset_sel(so)), cmp_act=cmp_(onset_act(ac), onset_act(ao)), cmp_strength=cmp_(onset_sel(ss), onset_sel(sw)), pct_core=pct_c, pct_off=pct_o)
    def tally(key):
        v = [r[key] for r in rows.values() if r[key] is not None]; e, l = sum(x == -1 for x in v), sum(x == 1 for x in v)
        return dict(n=len(v), core_earlier=e, same=len(v) - e - l, off_earlier=l, core_earlier_share_of_differing=e / (e + l) if e + l else None, p_core_earlier=binom_p(e, e + l))
    nd = pool[-1] + pool[1]; n = len(rows); kr = sum(r["lead_res"] > 0 for r in rows.values()); kl_ = sum(r["lead"] > 0 for r in rows.values()); ks = sum(r["lead_strength"] > 0 for r in rows.values())
    kt = sum(r["lead_res_tok"] > 0 for r in rows.values())
    lead_sum = dict(n=n, core_leads_res=kr, core_leads_res_tok_share=kt / n if n else None, p_core_leads_res_tok=binom_p(kt, n), median_lead_res_tok=med([r["lead_res_tok"] for r in rows.values()]),
                    median_tokconc_core=med([r["tokconc_core"] for r in rows.values()]), median_tokconc_off=med([r["tokconc_off"] for r in rows.values()]), core_leads_res_share=kr / n if n else None, p_core_leads_res=binom_p(kr, n), core_leads_raw_share=kl_ / n if n else None, p_core_leads_raw=binom_p(kl_, n),
                    median_lead_res=med([r["lead_res"] for r in rows.values()]), median_z_res=med([r["z_res"] for r in rows.values()]), median_lead_raw=med([r["lead"] for r in rows.values()]), median_z_raw=med([r["z"] for r in rows.values()]),
                    strong_leads_share=ks / n if n else None, median_lead_strength=med([r["lead_strength"] for r in rows.values()]),
                    median_pct_core_by_ckpt=[med([r["pct_core"][i] for r in rows.values()]) for i in range(len(CKPTS))], median_pct_off_by_ckpt=[med([r["pct_off"][i] for r in rows.values()]) for i in range(len(CKPTS))])
    return rows, dict(lead=lead_sum, selection=tally("cmp"), mean_activation=tally("cmp_act"), strength=tally("cmp_strength"), null_first_earlier_share_of_differing=pool[-1] / nd if nd else None, null_differing_share=nd / max(sum(pool.values()), 1),
                      median_onset_core=med([CKPTS[r["onset_core"]] for r in rows.values() if r["onset_core"] is not None]), median_onset_off=med([CKPTS[r["onset_off"]] for r in rows.values() if r["onset_off"] is not None]))
DEVEL = {}
for nm in SPLITS:
    for scope, el in (("multi", ELIG[nm]), ("all", NEU)):
        rows, sm = develop(nm, SPLITS[nm], el); DEVEL[f"{nm}/{scope}"] = dict(summary=sm, rows=rows); s1 = sm["selection"]; ld = sm["lead"]
        log(f"developmental {nm}/{scope} ({ld['n']} neurons): the core's firings were top firings earlier than the offshoot's (final activation regressed out) in {ld['core_leads_res_share']} (p {ld['p_core_leads_res']}; median lead {ld['median_lead_res']}, median z against random splits {ld['median_z_res']}); "
            f"raw {ld['core_leads_raw_share']} (p {ld['p_core_leads_raw']}, median z {ld['median_z_raw']}); with the firing's token typicality also regressed out {ld['core_leads_res_tok_share']} (p {ld['p_core_leads_res_tok']}); "
            f"most common token's share core {ld['median_tokconc_core']} / offshoot {ld['median_tokconc_off']}; strongest firings lead in {ld['strong_leads_share']}; median percentile by checkpoint core {ld['median_pct_core_by_ckpt']} / offshoot {ld['median_pct_off_by_ckpt']}")
        log(f"developmental onsets {nm}/{scope} ({s1['n']} neurons): core first {s1['core_earlier']}, same checkpoint {s1['same']}, offshoot first {s1['off_earlier']} (core share of differing {s1['core_earlier_share_of_differing']}, p {s1['p_core_earlier']}); "
            f"random splits differ in {sm['null_differing_share']:.2f}, first group earlier in {sm['null_first_earlier_share_of_differing']} of those; by mean activation {sm['mean_activation']['core_earlier_share_of_differing']}; strongest firings first {sm['strength']['core_earlier_share_of_differing']}; median onsets core {sm['median_onset_core']} / offshoot {sm['median_onset_off']}")
jl = lambda x: x.tolist() if torch.is_tensor(x) else x
RES = dict(model=HF, block=B, windows=NWIN, positions=N, top_pct_rank=TOPPCT, checkpoints=CKPTS, n_candidates=len(cand), n_sampled=len(NEU), n_null=NNULL, n_random_splits=NSPLIT, min_offshoot=MINOFF,
           n_multi=dict((k, len(v)) for k, v in ELIG.items()), silhouette_median=med([NR[j]["silhouette"] for j in NEU]), null_sil_mean_median=med([NR[j]["null_sil_mean"] for j in NEU]), null_sil95_median=med([NR[j]["null_sil95"] for j in NEU]),
           top_share_median=med([NR[j]["top_share"] for j in NEU]), max_split_error=max(recon), drive_median={k: med([NR[j]["drive"][k] for j in NEU]) for k in ("pre_activation", "mlp", "attention", "embedding")},
           causal={k: v["summary"] for k, v in CAUSAL.items()}, developmental={k: v["summary"] for k, v in DEVEL.items()},
           selection_share_by_checkpoint={str(s): float(SEL[s].float().mean(1).median()) for s in CKPTS}, percentile_by_checkpoint={str(s): float(PCT[s].median()) for s in CKPTS},
           neurons={str(j): dict(silhouette=NR[j]["silhouette"], null_sil95=NR[j]["null_sil95"], n_core=NR[j]["n_core"], n_off=NR[j]["n_off"], top_unit=NR[j]["top_unit"], top_share=NR[j]["top_share"], n_distinct_top=NR[j]["n_distinct_top"], drive=NR[j]["drive"],
                                 drivers=NR[j]["drivers"], tokens=NR[j]["tokens"], multi=j in ELIG["cluster"], damage_mean=float(DMG[j].mean()), kl_mean=float(KLD[j].mean()), act_mean=float(NR[j]["act"].mean()),
                                 causal=CAUSAL["cluster/all"]["rows"].get(j), dev=DEVEL["cluster/all"]["rows"].get(j)) for j in NEU},
           _time=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - t0)
os.makedirs("results", exist_ok=True); sfx = "_smoke" if SMOKE else ""
json.dump(RES, open(f"results/e672_routes{sfx}.json", "w"), indent=1, default=jl)
torch.save(dict(neurons=NEU, idx={j: NR[j]["idx"] for j in NEU}, act={j: NR[j]["act"] for j in NEU}, damage={j: DMG[j] for j in NEU}, kl={j: KLD[j] for j in NEU}, is_core={j: NR[j]["is_core"] for j in NEU}, multi=ELIG["cluster"], starts=starts, nwin=NWIN, T=T), f"results/e672_for_e673{sfx}.pt")
log("done"); print("done", flush=True)
