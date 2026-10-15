"""e674 (session 153; the pre-registered replication of e672, Gemini 4's design after Cammarata 2026). e672 found at
Pythia-160m block 6 that a neuron's distinct routes form in no order and its offshoot firings cost as much as its core
firings, and, after the fact, that inside a single route the larger part of the top firings became top firings before the
rest. Here the same measurement on other blocks and models, each a separate run: Pythia-160m blocks 3 and 9, Pythia-410m
blocks 6, 12 and 18 (GELU, parallel residual), OLMo-1B-0724 blocks 4, 8 and 12 (SwiGLU, sequential residual, a LayerNorm
without parameters). Per run: 256 windows of 256 tokens from wikitext-2 train; 300 sampled neurons with a substantial
top 100; each top firing's MLP input split exactly into every upstream MLP neuron, attention output and the embedding
(through the LayerNorm at that position; for SwiGLU the gate and value pre-activations each split exactly and combined
into the first-order attribution of the activation, silu'(g) u dg + silu(g) du); k-means (k = 2) on the attributions of
the 256 largest units, two-route when the silhouette beats the 95th percentile of 20 unimodal nulls with the firings'
mean and covariance; the neuron zeroed in each window holding its firings, the KL divergence of the next-token
distribution at each firing; each firing's percentile among all positions at 11 checkpoints, the lead of the larger
group over the rest with final activation (and the firing's token typicality) regressed out. For SwiGLU also Gemini's
leak filter: the share of gate-closed firings (gate < 0, value < 0) in the top 100, in each route, and the whole analysis
again on the top 100 gate-open firings. Arguments: --model pythia160|pythia410|olmo1b --block B [--smoke].
Pre-registered in e674_prereg.json."""
import sys, os, time, json, math, random, collections, argparse
import numpy as np, torch
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args()
torch.set_grad_enabled(False); DEV = "cuda:0"; t0 = time.time(); SMOKE = A_.smoke; B = A_.block
torch.set_num_threads(int(os.environ.get("E674_THREADS", "4")))   # tiny CPU matrices: 128 threads only add overhead, and nine runs share the box
PY_CK = [128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
if A_.model == "olmo1b":
    R_ = json.load(open("/workspace/wdd/olmo_refs.json")); OREV = {int(k): v for k, v in R_["olmo_revisions"].items()}; FINAL = 10 ** 9
    SPEC = dict(hf="allenai/OLMo-1B-0724-hf", fam="olmo", ckpts=sorted(OREV) + [FINAL], rev=lambda s: "main" if s == FINAL else OREV[s])
else:
    SPEC = dict(hf={"pythia160": "EleutherAI/pythia-160m", "pythia410": "EleutherAI/pythia-410m"}[A_.model], fam="neox", ckpts=PY_CK, rev=lambda s: "main" if s == 143000 else f"step{s}")
FAM = SPEC["fam"]; CKPTS = SPEC["ckpts"] if not SMOKE else [SPEC["ckpts"][2], SPEC["ckpts"][5], SPEC["ckpts"][-1]]
T = 256; NWIN = 48 if SMOKE else 256; NNEU = 30 if SMOKE else 300; TOPN = 100; KUNITS = 256; NSPLIT = 200; NNULL = 20; MINOFF = 10
TAG = f"{A_.model}_b{B}" + ("_smoke" if SMOKE else "")
def log(s): print(f"[e674 {TAG}] {s} ({time.time() - t0:.0f}s)", flush=True)
tok = AutoTokenizer.from_pretrained(SPEC["hf"])
def load(step): return AutoModelForCausalLM.from_pretrained(SPEC["hf"], revision=SPEC["rev"](step), dtype=torch.float32).to(DEV).eval()
import datasets
text = "\n".join(t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
IDS = tok(text, return_tensors="pt").input_ids[0]; starts = np.linspace(0, len(IDS) - T - 1, NWIN).astype(int); Wc = torch.stack([IDS[s:s + T] for s in starts]); W = Wc.to(DEV)
P = T - 2; N = NWIN * P; TOPPCT = max(TOPN, N // 100)
win_of = lambda n: n // P; pos_of = lambda n: n % P + 1
def binom_p(k, n): return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else None
med = lambda xs: float(np.median(xs)) if len(xs) else None
# ---------------- the family's parts
def layers(m): return m.gpt_neox.layers if FAM == "neox" else m.model.layers
def act_mod(m, b): return layers(m)[b].mlp.dense_4h_to_h if FAM == "neox" else layers(m)[b].mlp.down_proj
def attn_mod(m, l): return layers(m)[l].attention if FAM == "neox" else layers(m)[l].self_attn
def emb_mod(m): return m.gpt_neox.embed_in if FAM == "neox" else m.model.embed_tokens
NATT = B if FAM == "neox" else B + 1   # parallel residual: block b's own attention does not feed its MLP; sequential: it does
class Stop(Exception): pass
def capture_block(model, gate=False):
    """block-B activations [N, DFF] (and the SwiGLU gate pre-activation) at every kept position"""
    L = layers(model); out = collections.defaultdict(list)
    for s0 in range(0, NWIN, 8):
        cap = {}; hs = [act_mod(model, B).register_forward_pre_hook(lambda m, a: cap.__setitem__("a", a[0][:, 1:T - 1].detach().half()))]
        if gate: hs.append(L[B].mlp.gate_proj.register_forward_hook(lambda m, a, o: cap.__setitem__("g", o[:, 1:T - 1].detach().half())))
        def stop(m, a, o): raise Stop
        hs.append(L[B].register_forward_hook(stop))
        try: model(W[s0:s0 + 8])
        except Stop: pass
        finally: [h.remove() for h in hs]
        for k, v in cap.items(): out[k].append(v.reshape(-1, v.shape[-1]))
    return {k: torch.cat(v) for k, v in out.items()}
def capture_upstream(model, rows_of):
    """at the needed positions only: every upstream MLP activation, the attention outputs feeding block B's MLP, the embedding, the MLP's LayerNorm input"""
    L = layers(model); DFF = model.config.intermediate_size; D = model.config.hidden_size; n = len(rows_of)
    UP = torch.zeros(n, B * DFF, dtype=torch.half); AT = torch.zeros(n, NATT, D, dtype=torch.half); EM = torch.zeros(n, D, dtype=torch.half); XM = torch.zeros(n, D)
    byw = collections.defaultdict(list)
    for gpos, r in rows_of.items(): byw[win_of(gpos)].append((pos_of(gpos), r))
    for s0 in range(0, NWIN, 8):
        sel = [(w - s0, p, r) for w in range(s0, min(s0 + 8, NWIN)) for p, r in byw.get(w, [])]
        if not sel: continue
        bi = torch.tensor([x[0] for x in sel], device=DEV); pi = torch.tensor([x[1] for x in sel], device=DEV); rr = torch.tensor([x[2] for x in sel])
        cap = {}; hs = []
        for l in range(B): hs.append(act_mod(model, l).register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(("up", l), a[0][bi, pi].detach().half().cpu())))
        for l in range(NATT): hs.append(attn_mod(model, l).register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("att", l), (o[0] if isinstance(o, tuple) else o)[bi, pi].detach().half().cpu())))
        hs.append(emb_mod(model).register_forward_hook(lambda m, a, o: cap.__setitem__("emb", o[bi, pi].detach().half().cpu())))
        hs.append(L[B].post_attention_layernorm.register_forward_pre_hook(lambda m, a: cap.__setitem__("xm", a[0][bi, pi].detach().float().cpu())))
        def stop(m, a, o): raise Stop
        hs.append(L[B].register_forward_hook(stop))
        try: model(W[s0:s0 + 8])
        except Stop: pass
        finally: [h.remove() for h in hs]
        if B: UP[rr] = torch.cat([cap[("up", l)] for l in range(B)], 1)
        AT[rr] = torch.stack([cap[("att", l)] for l in range(NATT)], 1); EM[rr] = cap["emb"]; XM[rr] = cap["xm"]
    return UP, AT, EM, XM
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
    g = torch.Generator().manual_seed(seed); mu = Xraw.mean(0, keepdim=True); Xc = Xraw - mu; out = []
    for _ in range(NNULL):
        Q, R = torch.linalg.qr(torch.randn(Xraw.shape[0], Xraw.shape[0], generator=g)); Q = Q * torch.sign(torch.diagonal(R))[None]
        Xn = rownorm(mu + Q @ Xc); out.append(silhouette(Xn, kmeans2(Xn, inits=5)))
    return out
# ---------------- the final checkpoint
model = load(CKPTS[-1]); L = layers(model); cfg = model.config; D, DFF = cfg.hidden_size, cfg.intermediate_size
C1 = capture_block(model, gate=(FAM == "olmo")); A6 = C1["a"]; G6 = C1.get("g"); log(f"captured {N} positions, block {B} of {len(L)}")
mx = A6.float().max(0).values; a100 = A6.float().topk(TOPN, dim=0).values[-1]
floor_ = 1.0 if FAM == "neox" else float(mx.quantile(0.10))
cand = torch.nonzero((mx > floor_) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(674 + B).sample(cand, min(NNEU, len(cand))))
log(f"{len(cand)} of {DFF} neurons have a substantial top-{TOPN} (maximum above {floor_:.2f}); {len(NEU)} sampled")
SETS = {"all": {}}; LEAK = {}
for j in NEU:
    tv, ti = A6[:, j].float().topk(TOPN); SETS["all"][j] = (ti.cpu(), tv.cpu())
    if FAM == "olmo":
        LEAK[j] = (G6[ti, j].float() < 0).cpu()
        open_ = A6[:, j].float().masked_fill(G6[:, j].float() <= 0, -1e9); tv2, ti2 = open_.topk(TOPN); SETS.setdefault("gate_open", {})[j] = (ti2.cpu(), tv2.cpu())
need = sorted({int(n) for S in SETS.values() for (ti, _) in S.values() for n in ti}); rows_of = {n: k for k, n in enumerate(need)}
UP, AT, EM, XM = capture_upstream(model, rows_of); log(f"upstream captured at {len(need)} positions ({B * DFF} MLP units, {NATT} attention outputs)")
ln = L[B].post_attention_layernorm; eps = getattr(ln, "eps", getattr(ln, "variance_epsilon", 1e-5))
gam = ln.weight.detach().float() if getattr(ln, "weight", None) is not None else torch.ones(D, device=DEV); bet = ln.bias.detach().float() if getattr(ln, "bias", None) is not None else torch.zeros(D, device=DEV)
if FAM == "neox":
    WIN_ = {"h": L[B].mlp.dense_h_to_4h.weight.detach().float()}; BIN = {"h": L[B].mlp.dense_h_to_4h.bias.detach().float()}
    WUP = torch.cat([L[l].mlp.dense_4h_to_h.weight.detach().float().T for l in range(B)]) if B else torch.zeros(0, D, device=DEV); BIAS = sum((L[l].mlp.dense_4h_to_h.bias.detach().float() for l in range(B)), torch.zeros(D, device=DEV))
else:
    WIN_ = {"g": L[B].mlp.gate_proj.weight.detach().float(), "u": L[B].mlp.up_proj.weight.detach().float()}; BIN = {"g": torch.zeros(DFF, device=DEV), "u": torch.zeros(DFF, device=DEV)}
    WUP = torch.cat([L[l].mlp.down_proj.weight.detach().float().T for l in range(B)]) if B else torch.zeros(0, D, device=DEV); BIAS = torch.zeros(D, device=DEV)
silu = torch.nn.functional.silu
def dsilu(x): s = torch.sigmoid(x); return s * (1 + x * (1 - s))
def unit_name(u):
    if u < B * DFF: return f"L{u // DFF}:{u % DFF}"
    return f"attn{u - B * DFF}" if u < B * DFF + NATT else "embedding"
def attribution(j, ti):
    """per firing, the exact split of each pre-activation into upstream units; returns the attribution matrix and the largest relative reconstruction error"""
    rr = torch.tensor([rows_of[int(n)] for n in ti]); xm = XM[rr].to(DEV); sig = (xm.var(-1, unbiased=False) + eps).sqrt(); up = UP[rr].to(DEV).float(); at = AT[rr].to(DEV).float(); em = EM[rr].to(DEV).float(); C, err = {}, 0.0
    for k, Wk in WIN_.items():
        r = Wk[j] * gam; r = r - r.mean(); Ck = torch.cat([up * (WUP @ r)[None] / sig[:, None], (at @ r) / sig[:, None], ((em @ r) / sig)[:, None]], 1)
        h_true = (xm - xm.mean(-1, keepdim=True)) / sig[:, None] @ (Wk[j] * gam) + Wk[j] @ bet + BIN[k][j]
        h_rec = Ck.sum(1) + (BIAS @ r) / sig + Wk[j] @ bet + BIN[k][j]; err = max(err, float((h_rec - h_true).abs().max() / h_true.abs().max().clamp_min(1e-9))); C[k] = (Ck, h_true)
    if FAM == "neox": return C["h"][0], err
    (Cg, g), (Cu, u) = C["g"], C["u"]; return Cg * (dsilu(g) * u)[:, None] + Cu * silu(g)[:, None], err
def context(n, k=8):
    w, p = win_of(int(n)), pos_of(int(n))
    return tok.decode(Wc[w, max(0, p - k):p].tolist()) + " [[" + tok.decode([int(Wc[w, p])]) + "]] " + tok.decode([int(Wc[w, p + 1])])
def routes(j, ti, tv):
    Cj, err = attribution(j, ti); units = Cj.abs().sum(0).topk(min(KUNITS, Cj.shape[1])).indices; Xraw = Cj[:, units].cpu(); X = rownorm(Xraw)
    lab = kmeans2(X); sil = silhouette(X, lab); n1 = int(lab.sum()); core = 1 if n1 >= TOPN - n1 else 0; is_core = lab == core; nulls = null_silhouettes(Xraw, j)
    cur = torch.tensor([int(Wc[win_of(int(n)), pos_of(int(n))]) for n in ti])
    drivers = lambda m: [unit_name(int(units[u])) for u in Cj[m.to(DEV)][:, units].sum(0).argsort(descending=True)[:4].tolist()]
    return dict(idx=ti, act=tv, is_core=is_core, silhouette=sil, null_sil95=float(np.percentile(nulls, 95)), null_sil_mean=float(np.mean(nulls)), n_core=int(is_core.sum()), n_off=int((~is_core).sum()), err=err, cur=cur,
                multi=bool(sil > float(np.percentile(nulls, 95)) and int((~is_core).sum()) >= MINOFF), drivers=dict(core=drivers(is_core), offshoot=drivers(~is_core)),
                examples=dict(core=[context(n) for n in ti[is_core][:2]], offshoot=[context(n) for n in ti[~is_core][:2]]))
NR = {s: {} for s in SETS}
for s, S in SETS.items():
    for j, (ti, tv) in S.items(): NR[s][j] = routes(j, ti, tv)
    log(f"[{s}] {sum(v['multi'] for v in NR[s].values())} of {len(NEU)} neurons two-route (silhouette median {med([v['silhouette'] for v in NR[s].values()]):.2f}, null mean {med([v['null_sil_mean'] for v in NR[s].values()]):.2f}, 95th {med([v['null_sil95'] for v in NR[s].values()]):.2f}); split error at most {max(v['err'] for v in NR[s].values()):.1e}")
del UP, AT, EM, XM
# ---------------- causal: the neuron zeroed in each window holding its firings
def ablate(j, idx):
    byw = collections.defaultdict(list)
    for i, n in enumerate(idx.tolist()): byw[win_of(n)].append((i, pos_of(n)))
    wins = sorted(byw); kl = torch.zeros(TOPN)
    for s0 in range(0, len(wins), 8):
        ws = wins[s0:s0 + 8]; x = W[ws]; nb = len(ws)
        def hook(m, a):
            h = a[0].clone(); h[nb:, :, j] = 0.0; return (h,)
        hd = act_mod(model, B).register_forward_pre_hook(hook)
        try: lg = model(torch.cat([x, x])).logits
        finally: hd.remove()
        rows = [(k, i, p) for k, w in enumerate(ws) for i, p in byw[w]]; kk = torch.tensor([r[0] for r in rows], device=DEV); ii = torch.tensor([r[1] for r in rows]); pp = torch.tensor([r[2] for r in rows], device=DEV)
        lb = lg[kk, pp].float().log_softmax(-1); la = lg[kk + nb, pp].float().log_softmax(-1); del lg
        kl[ii] = (lb.exp() * (lb - la)).sum(-1).clamp_min(0).cpu()
    return kl
ABL = {s: sorted([j for j in NEU if NR[s][j]["multi"]] + random.Random(6741 + B).sample([j for j in NEU if not NR[s][j]["multi"]], min(60, sum(not NR[s][j]["multi"] for j in NEU)))) for s in SETS}
TF32 = FAM == "olmo"   # OLMo-1B's ablations in TF32 (about four times faster); the smoke compares a few neurons against fp32
if TF32 and SMOKE:
    j0 = ABL["all"][0]; k32 = ablate(j0, NR["all"][j0]["idx"]); torch.backends.cuda.matmul.allow_tf32 = True; ktf = ablate(j0, NR["all"][j0]["idx"]); torch.backends.cuda.matmul.allow_tf32 = False
    log(f"TF32 check on neuron {j0}: KL fp32 median {float(k32.median()):.2e}, TF32 {float(ktf.median()):.2e}, max abs difference {float((k32 - ktf).abs().max()):.2e}, Spearman {float(np.corrcoef(k32.argsort().argsort(), ktf.argsort().argsort())[0, 1]):.4f}")
torch.backends.cuda.matmul.allow_tf32 = TF32
KLD = {s: {j: ablate(j, NR[s][j]["idx"]) for j in ABL[s]} for s in SETS}; torch.backends.cuda.matmul.allow_tf32 = False; log(f"ablations done ({sum(len(v) for v in ABL.values())} neurons, TF32 {TF32})")
def causal(s, js):
    rows = {}
    for j in js:
        k, a, c = KLD[s][j], NR[s][j]["act"], NR[s][j]["is_core"]; g = k.sqrt() / a.clamp_min(1e-6)
        rows[j] = dict(kl_core=float(k[c].mean()), kl_off=float(k[~c].mean()), gain_core=float(g[c].mean()), gain_off=float(g[~c].mean()), act_core=float(a[c].mean()), act_off=float(a[~c].mean()))
    n = len(rows); kk = sum(v["kl_off"] < v["kl_core"] for v in rows.values()); kg = sum(v["gain_off"] < v["gain_core"] for v in rows.values())
    return rows, dict(n=n, off_less_kl=kk, off_less_kl_share=kk / n if n else None, p_off_less_kl=binom_p(kk, n), p_off_more_kl=binom_p(n - kk, n), off_less_gain_share=kg / n if n else None, p_off_less_gain=binom_p(kg, n),
                      median_kl_core=med([v["kl_core"] for v in rows.values()]), median_kl_off=med([v["kl_off"] for v in rows.values()]), median_act_core=med([v["act_core"] for v in rows.values()]), median_act_off=med([v["act_off"] for v in rows.values()]))
CAUSAL = {}
for s in SETS:
    for scope in ("two_route", "one_route"):
        js = [j for j in ABL[s] if NR[s][j]["multi"] == (scope == "two_route")]; rows, sm = causal(s, js); CAUSAL[f"{s}/{scope}"] = dict(summary=sm, rows=rows)
        log(f"causal [{s}] {scope} ({sm['n']}): offshoot firings cost less than core in {sm['off_less_kl_share']} (p {sm['p_off_less_kl']}; more: p {sm['p_off_more_kl']}); median KL core {sm['median_kl_core']} / offshoot {sm['median_kl_off']}")
del model; torch.cuda.empty_cache()
# ---------------- developmental
NEUt = torch.tensor(NEU, device=DEV); IDXM = {s: torch.stack([NR[s][j]["idx"] for j in NEU], 1).to(DEV) for s in SETS}; PCT = {s: {} for s in SETS}; SEL = {s: {} for s in SETS}
for step in CKPTS:
    m_ = load(step); a6 = capture_block(m_)["a"]; del m_
    sub = a6[:, NEUt].float(); srt = sub.T.contiguous().sort(1).values; thr = sub.topk(TOPPCT, dim=0).values[-1]
    for s in SETS:
        vals = sub.gather(0, IDXM[s]); PCT[s][step] = (torch.searchsorted(srt, vals.T.contiguous()).float() / N).cpu(); SEL[s][step] = (vals >= thr[None]).T.cpu()
    del a6, sub, srt; torch.cuda.empty_cache(); log(f"checkpoint {step}: final top firings at median percentile {float(PCT['all'][step].median()):.3f}")
KI = {j: k for k, j in enumerate(NEU)}
def onset_sel(cv):
    for i in range(len(cv)):
        if all(x >= 0.5 for x in cv[i:]): return i
    return None
cmp_ = lambda a, b: (None if a is None or b is None else (-1 if a < b else (1 if a > b else 0)))
def develop(s, js, seed0):
    EARLY = torch.stack([PCT[s][c] for c in CKPTS[:-1]]).mean(0); rows = {}
    for j in js:
        c = NR[s][j]["is_core"]; nc = int(c.sum())
        if nc in (0, TOPN): continue
        e = EARLY[KI[j]].double(); a = NR[s][j]["act"].double(); one = torch.ones_like(a)
        X2 = torch.stack([one, a], 1); res = e - X2 @ torch.linalg.lstsq(X2, e[:, None]).solution[:, 0]
        cur = NR[s][j]["cur"].tolist(); tc = collections.Counter(cur); typ = torch.tensor([tc[t] / TOPN for t in cur], dtype=torch.float64)
        X3 = torch.stack([one, a, typ], 1); res3 = e - X3 @ torch.linalg.lstsq(X3, e[:, None]).solution[:, 0]
        st = torch.zeros(TOPN, dtype=torch.bool); st[:nc] = True; rng = random.Random(seed0 + j); nl = []
        for q in range(NSPLIT):
            mk = torch.zeros(TOPN, dtype=torch.bool); mk[rng.sample(range(TOPN), nc)] = True; nl.append(float(res[mk].mean() - res[~mk].mean()))
        sc = [float(SEL[s][x][KI[j]][c].float().mean()) for x in CKPTS]; so = [float(SEL[s][x][KI[j]][~c].float().mean()) for x in CKPTS]
        rows[j] = dict(lead_res=float(res[c].mean() - res[~c].mean()), lead_res_tok=float(res3[c].mean() - res3[~c].mean()), lead_raw=float(e[c].mean() - e[~c].mean()), lead_strength=float(e[st].mean() - e[~st].mean()),
                       z_res=(float(res[c].mean() - res[~c].mean()) - np.mean(nl)) / max(np.std(nl), 1e-12), cmp=cmp_(onset_sel(sc), onset_sel(so)),
                       pct_core=[float(PCT[s][x][KI[j]][c].mean()) for x in CKPTS], pct_off=[float(PCT[s][x][KI[j]][~c].mean()) for x in CKPTS])
    n = len(rows); kr = sum(r["lead_res"] > 0 for r in rows.values()); kt = sum(r["lead_res_tok"] > 0 for r in rows.values()); ks = sum(r["lead_strength"] > 0 for r in rows.values())
    v = [r["cmp"] for r in rows.values() if r["cmp"] is not None]; e_, l_ = sum(x == -1 for x in v), sum(x == 1 for x in v)
    return rows, dict(n=n, leads_res=kr, leads_res_share=kr / n if n else None, p_leads_res=binom_p(kr, n), leads_res_tok=kt, leads_res_tok_share=kt / n if n else None, p_leads_res_tok=binom_p(kt, n),
                      median_lead_res=med([r["lead_res"] for r in rows.values()]), median_z_res=med([r["z_res"] for r in rows.values()]), strong_leads_share=ks / n if n else None,
                      onset_core_first=e_, onset_same=len(v) - e_ - l_, onset_off_first=l_)
DEVEL = {}
for s in SETS:
    for scope in ("one_route", "two_route"):
        js = [j for j in NEU if NR[s][j]["multi"] == (scope == "two_route")]; rows, sm = develop(s, js, 67400 + (0 if s == "all" else 50000)); DEVEL[f"{s}/{scope}"] = dict(summary=sm, rows=rows)
        log(f"developmental [{s}] {scope} ({sm['n']}): the larger group's firings were top firings earlier (activation regressed out) in {sm['leads_res_share']} (p {sm['p_leads_res']}); with token typicality {sm['leads_res_tok_share']} (p {sm['p_leads_res_tok']}); onsets core first {sm['onset_core_first']}, same {sm['onset_same']}, offshoot first {sm['onset_off_first']}")
LK = {}
if FAM == "olmo":
    two = [j for j in NEU if NR["all"][j]["multi"]]; withleak = [j for j in two if LEAK[j].any()]
    enr = [(float(LEAK[j][~NR["all"][j]["is_core"]].float().mean()), float(LEAK[j][NR["all"][j]["is_core"]].float().mean())) for j in withleak]; ke = sum(o > c for o, c in enr); kt_ = sum(o != c for o, c in enr)
    collapsed = [j for j in two if not NR["gate_open"][j]["multi"]]
    LK = dict(leak_share_top100_median=med([float(LEAK[j].float().mean()) for j in NEU]), neurons_with_any_leak=sum(bool(LEAK[j].any()) for j in NEU), leak_dominated=sum(float(LEAK[j].float().mean()) > 0.5 for j in NEU),
              two_route=len(two), two_route_with_leak=len(withleak), offshoot_more_leak=ke, differing=kt_, p_offshoot_more_leak=binom_p(ke, kt_), median_leak_core=med([c for o, c in enr]), median_leak_off=med([o for o, c in enr]),
              collapsed_after_filter=len(collapsed), collapsed_share=len(collapsed) / len(two) if two else None, two_route_after_filter=sum(NR["gate_open"][j]["multi"] for j in NEU))
    log(f"leak: median gate-closed share of the top 100 {LK['leak_share_top100_median']}; {LK['neurons_with_any_leak']} neurons with any, {LK['leak_dominated']} leak-dominated; offshoots more gate-closed than cores in {ke} of {kt_} differing two-route neurons (p {LK['p_offshoot_more_leak']}); "
        f"{len(collapsed)} of {len(two)} two-route neurons become one-route on gate-open firings; {LK['two_route_after_filter']} two-route after filtering")
jl = lambda x: x.tolist() if torch.is_tensor(x) else x
RES = dict(model=SPEC["hf"], key=A_.model, block=B, tf32_ablations=TF32, n_ablated={s: len(v) for s, v in ABL.items()}, n_layers=len(L), fam=FAM, windows=NWIN, positions=N, checkpoints=CKPTS, n_candidates=len(cand), n_sampled=len(NEU), floor=floor_,
           n_two_route={s: sum(NR[s][j]["multi"] for j in NEU) for s in SETS}, silhouette_median={s: med([NR[s][j]["silhouette"] for j in NEU]) for s in SETS}, max_split_error={s: max(NR[s][j]["err"] for j in NEU) for s in SETS},
           causal={k: v["summary"] for k, v in CAUSAL.items()}, developmental={k: v["summary"] for k, v in DEVEL.items()}, leak=LK, percentile_by_checkpoint={str(c): float(PCT["all"][c].median()) for c in CKPTS},
           neurons={s: {str(j): dict(silhouette=NR[s][j]["silhouette"], null_sil95=NR[s][j]["null_sil95"], multi=NR[s][j]["multi"], n_core=NR[s][j]["n_core"], drivers=NR[s][j]["drivers"], examples=NR[s][j]["examples"],
                                     leak_share=(float(LEAK[j].float().mean()) if FAM == "olmo" and s == "all" else None), kl_mean=(float(KLD[s][j].mean()) if j in KLD[s] else None),
                                     dev=next((DEVEL[f"{s}/{sc}"]["rows"].get(j) for sc in ("one_route", "two_route") if j in DEVEL[f"{s}/{sc}"]["rows"]), None),
                                     causal=next((CAUSAL[f"{s}/{sc}"]["rows"].get(j) for sc in ("one_route", "two_route") if j in CAUSAL[f"{s}/{sc}"]["rows"]), None)) for j in NEU} for s in SETS},
           _time=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - t0)
os.makedirs("/workspace/wdd/results", exist_ok=True); json.dump(RES, open(f"/workspace/wdd/results/e674_{TAG}.json", "w"), indent=1, default=jl); log("done"); print("done", flush=True)
