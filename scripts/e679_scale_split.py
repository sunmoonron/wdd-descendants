"""e679 (session 154; Gemini 4's fusion of the essay's scale figures: a smaller vision model learns coarser two-sided
neurons that a larger model trained on the same data splits into narrower ones). Pythia-160m and Pythia-410m saw the same
data in the same order. For e674's and e672's sampled Pythia-160m neurons (blocks 3, 6, 9; the same routes, recomputed
with the same seeds), every Pythia-410m neuron of the five blocks around the matching relative depth is a candidate; a
candidate covers a set of firings by the share of them inside its own top 1% of the same 65,024 wikitext positions. A
two-route neuron splits when its core's best-covering 410m neuron differs from its offshoot's and each covers its own
route by at least 0.25 more than the other; against 20 random splits of the same firings into groups of the same sizes,
and against one-route neurons. Arguments: --block B [--seed S] [--window W] [--smoke]. Pre-registered in e679_prereg.json."""
import sys, os, time, json, math, random, collections, argparse
import numpy as np, torch
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default="/workspace/wdd/results"); ap.add_argument("--window", type=int, default=2); ap.add_argument("--seed", type=int, default=None); A_ = ap.parse_args()
torch.set_grad_enabled(False); DEV = "cuda:0"; t0 = time.time(); SMOKE = A_.smoke; B = A_.block
torch.set_num_threads(int(os.environ.get("E674_THREADS", "4")))   # tiny CPU matrices: 128 threads only add overhead, and nine runs share the box
PY_CK = [128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
A_.model = "pythia160"
SPEC = dict(hf="EleutherAI/pythia-160m", fam="neox", ckpts=PY_CK, rev=lambda s: "main" if s == 143000 else f"step{s}")
FAM = SPEC["fam"]; CKPTS = SPEC["ckpts"] if not SMOKE else [SPEC["ckpts"][2], SPEC["ckpts"][5], SPEC["ckpts"][-1]]
T = 256; NWIN = 48 if SMOKE else 256; NNEU = 30 if SMOKE else 300; TOPN = 100; KUNITS = 256; NSPLIT = 200; NNULL = 20; MINOFF = 10
TAG = f"{A_.model}_b{B}" + ("_smoke" if SMOKE else "")
def log(s): print(f"[e679 {TAG}] {s} ({time.time() - t0:.0f}s)", flush=True)
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
cand = torch.nonzero((mx > floor_) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(A_.seed if A_.seed is not None else 674 + B).sample(cand, min(NNEU, len(cand))))   # e674's neurons (e672's with --seed 672 at Pythia-160m block 6)
log(f"{len(cand)} of {DFF} neurons have a substantial top-{TOPN} (maximum above {floor_:.2f}); {len(NEU)} sampled")
SETS = {"all": {}}; LEAK = {}
for j in NEU:
    tv, ti = A6[:, j].float().topk(TOPN); SETS["all"][j] = (ti.cpu(), tv.cpu())
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
    cen = X.mean(0); cen = cen / cen.norm().clamp_min(1e-9); cos_c = X @ cen   # each firing's cosine with the centroid of all its top firings' attributions
    cur = torch.tensor([int(Wc[win_of(int(n)), pos_of(int(n))]) for n in ti])
    drivers = lambda m: [unit_name(int(units[u])) for u in Cj[m.to(DEV)][:, units].sum(0).argsort(descending=True)[:4].tolist()]
    return dict(idx=ti, act=tv, is_core=is_core, cos_c=cos_c, silhouette=sil, null_sil95=float(np.percentile(nulls, 95)), null_sil_mean=float(np.mean(nulls)), n_core=int(is_core.sum()), n_off=int((~is_core).sum()), err=err, cur=cur,
                multi=bool(sil > float(np.percentile(nulls, 95)) and int((~is_core).sum()) >= MINOFF), drivers=dict(core=drivers(is_core), offshoot=drivers(~is_core)),
                examples=dict(core=[context(n) for n in ti[is_core][:2]], offshoot=[context(n) for n in ti[~is_core][:2]]))
NR = {s: {} for s in SETS}
for s, S in SETS.items():
    for j, (ti, tv) in S.items(): NR[s][j] = routes(j, ti, tv)
    log(f"[{s}] {sum(v['multi'] for v in NR[s].values())} of {len(NEU)} neurons two-route (silhouette median {med([v['silhouette'] for v in NR[s].values()]):.2f}, null mean {med([v['null_sil_mean'] for v in NR[s].values()]):.2f}, 95th {med([v['null_sil95'] for v in NR[s].values()]):.2f}); split error at most {max(v['err'] for v in NR[s].values()):.1e}")
del UP, AT, EM, XM
# ---------------- Pythia-410m, the same windows: candidate neurons in the blocks around the matching relative depth
del model; torch.cuda.empty_cache()
B4 = 2 * B; CB = [b for b in range(B4 - A_.window, B4 + A_.window + 1) if 0 <= b < 24]
m4 = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", dtype=torch.float32).to(DEV).eval(); L4 = m4.gpt_neox.layers; out4 = collections.defaultdict(list)
for s0 in range(0, NWIN, 8):
    cap = {}; hs = [L4[b].mlp.dense_4h_to_h.register_forward_pre_hook(lambda m, a, b=b: cap.__setitem__(b, a[0][:, 1:T - 1].detach().half())) for b in CB]
    def stop(m, a, o): raise Stop
    hs.append(L4[max(CB)].register_forward_hook(stop))
    try: m4(W[s0:s0 + 8])
    except Stop: pass
    finally: [h.remove() for h in hs]
    for b in CB: out4[b].append(cap[b].reshape(-1, cap[b].shape[-1]).cpu())   # kept on the CPU: the GPU is shared
A4 = torch.cat([torch.cat(out4[b]) for b in CB], 1); del out4, m4; torch.cuda.empty_cache()   # [N, len(CB)*4096]
K1 = max(TOPN, N // 100); TOP1 = torch.zeros(N, A4.shape[1], dtype=torch.bool, device=DEV); DEAD = torch.zeros(A4.shape[1], dtype=torch.bool, device=DEV)
for c0 in range(0, A4.shape[1], 4096):   # membership by rank, exactly K1 positions per candidate; a candidate whose top K1 are all tied (dead or constant) is dropped
    tv, ti = A4[:, c0:c0 + 4096].to(DEV).float().topk(K1, dim=0); TOP1[:, c0:c0 + 4096].scatter_(0, ti, True); DEAD[c0:c0 + 4096] = (tv[0] - tv[-1]) <= 1e-6
TOP1[:, DEAD] = False; del A4; torch.cuda.empty_cache()
cname = lambda k: f"L{CB[k // 4096]}:{k % 4096}"
log(f"Pythia-410m blocks {CB}: {TOP1.shape[1]} candidate neurons ({int(DEAD.sum())} dropped as dead or constant), each one's top {K1} of {N} positions by rank")
def cover(idx): return TOP1[idx.to(DEV)].float().mean(0)   # [candidates]: the share of the given firings inside each candidate's top 1%
SPLIT_MARGIN = 0.25
def split_test(idx, c):
    cc, co = cover(idx[c]), cover(idx[~c]); bc, bo = int(cc.argmax()), int(co.argmax())
    ok = bc != bo and float(cc[bc] - co[bc]) >= SPLIT_MARGIN and float(co[bo] - cc[bo]) >= SPLIT_MARGIN
    return ok, dict(best_core=cname(bc), best_off=cname(bo), cov_core_by_core_best=float(cc[bc]), cov_off_by_core_best=float(co[bc]), cov_off_by_off_best=float(co[bo]), cov_core_by_off_best=float(cc[bo]))
ROWS = {}
for j in NEU:
    r = NR["all"][j]; idx, c = r["idx"], r["is_core"]; ca = cover(idx); ball = int(ca.argmax()); rng = random.Random(6790 + j); nc = int(c.sum())
    ok, det = split_test(idx, c); null = []
    for q in range(20):
        mk = torch.zeros(TOPN, dtype=torch.bool); mk[rng.sample(range(TOPN), nc)] = True; null.append(split_test(idx, mk)[0])
    ROWS[j] = dict(multi=r["multi"], n_core=nc, best_all=cname(ball), cov_all=float(ca[ball]), split=ok, null_split=float(np.mean(null)), **det)
def tally(js):
    n = len(js); k = sum(ROWS[j]["split"] for j in js); nul = float(np.mean([ROWS[j]["null_split"] for j in js])) if js else None
    return dict(n=n, split=k, split_share=k / n if n else None, null_split_share=nul, median_cov_all=med([ROWS[j]["cov_all"] for j in js]))
two = [j for j in NEU if ROWS[j]["multi"]]; one = [j for j in NEU if not ROWS[j]["multi"]]
SUM = dict(two_route=tally(two), one_route=tally(one))
log(f"two-route ({len(two)}): their two routes find different Pythia-410m specialists in {SUM['two_route']['split_share']} (random splits of the same firings {SUM['two_route']['null_split_share']}); one-route ({len(one)}): {SUM['one_route']['split_share']} (random {SUM['one_route']['null_split_share']}); "
    f"best single 410m neuron covers a median {SUM['two_route']['median_cov_all']} of a two-route neuron's top 100 against {SUM['one_route']['median_cov_all']} of a one-route neuron's")
RES = dict(block160=B, blocks410=CB, n_dead_candidates=int(DEAD.sum()), seed=(A_.seed if A_.seed is not None else 674 + B), positions=N, n_sampled=len(NEU), n_two_route=len(two), margin=SPLIT_MARGIN, summary=SUM, neurons={str(j): ROWS[j] for j in NEU},
           examples=[dict(neuron=j, **{k: v for k, v in ROWS[j].items() if k in ("best_core", "best_off", "cov_core_by_core_best", "cov_off_by_off_best", "split")}, tokens_core=NR["all"][j]["examples"]["core"], tokens_off=NR["all"][j]["examples"]["offshoot"]) for j in two[:12]],
           _time=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - t0)
os.makedirs(A_.out, exist_ok=True); json.dump(RES, open(f"{A_.out}/e679_{TAG}.json", "w"), indent=1); log("done"); print("done", flush=True)
