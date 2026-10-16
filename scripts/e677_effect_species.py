"""e677 (session 154; Gemini 4's fusion of the essay's 'when does the answer take shape' figure, whose vision panel marks
each neuron by whether removing it changed the next block's readout or the final readout beyond a matched null, with its
Poe neuron: removing it drops 'dreary' from 87% to 50% while removing only its direct contribution at the output leaves
85%). Per MLP neuron at the last position of e568's 280 prompts, three effects of removing it, each a KL divergence:
local (the block's own readout: a least-squares affine lens from the block's residual to the final residual, fitted on
wikitext, then the final norm and the unembedding), direct (its write subtracted from the final residual, nothing
downstream recomputed) and total (the neuron zeroed, the prefix cached, every later block recomputed). Each against a
null of random directions with the write's norm (the 95th percentile, interpolated over a grid of norms per prompt).
Species: a direct speaker is beyond the total null with direct at least half of total; an indirect router is beyond
the total null with direct under half; a washed-out writer is beyond the local null and not the total null. Pythia-410m
blocks 2, 6, 12, 17, 22; Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct blocks 3, 12, 21 (does fine-tuning change a neuron's
species?). Arguments: --model pythia410|qwen05|qwen05i [--smoke]. Pre-registered in e677_prereg.json."""
import sys, os, time, json, math, collections, argparse
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--blocks", default=""); A_ = ap.parse_args(); SMOKE = A_.smoke
torch.set_grad_enabled(False); torch.set_num_threads(8); DEV = "cuda:0"; t0 = time.time()
SPEC = {"pythia410": ("EleutherAI/pythia-410m", "neox", [2, 6, 12, 17, 22]), "qwen05": ("Qwen/Qwen2.5-0.5B", "qwen", [3, 12, 21]), "qwen05i": ("Qwen/Qwen2.5-0.5B-Instruct", "qwen", [3, 12, 21])}[A_.model]
HF, FAM, BLOCKS = SPEC; BLOCKS = [int(x) for x in A_.blocks.split(",")] if A_.blocks else (BLOCKS[1:2] if SMOKE else BLOCKS)
TAG = A_.model + (f"_b{A_.blocks.replace(chr(44), chr(45))}" if A_.blocks else "") + ("_smoke" if SMOKE else ""); MAXSEQ = 1024; NDIR = 16; NORMS = 8
def log(s): print(f"[e677 {TAG}] {s} ({time.time() - t0:.0f}s)", flush=True)
PRM = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e673_prompts.json")))
tok = AutoTokenizer.from_pretrained(HF); model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32).to(DEV).eval()
for p_ in model.parameters(): p_.requires_grad_(False)
base_ = model.gpt_neox if FAM == "neox" else model.model; L = base_.layers; NL = len(L); D = model.config.hidden_size; DFF = model.config.intermediate_size
FNORM = base_.final_layer_norm if FAM == "neox" else base_.norm; UNEMB = model.get_output_embeddings()
act_mod = lambda b: L[b].mlp.dense_4h_to_h if FAM == "neox" else L[b].mlp.down_proj
WOUT = lambda b: act_mod(b).weight.detach().float()   # [D, DFF]
PROMPTS = [("The capital of %s is" % c) for c, _ in PRM["caps"]] + [("The capital of %s is" % c) for c in PRM["fake_countries"]] + [("The birthplace of %s is" % p) for p in PRM["real_people"]] + [("The birthplace of %s is" % p) for p in PRM["fake_people"]]
if SMOKE: PROMPTS = PROMPTS[::7]
NP = len(PROMPTS); IDS = [tok(p, add_special_tokens=False)["input_ids"] for p in PROMPTS]; GRP = collections.defaultdict(list)
for i, x in enumerate(IDS): GRP[len(x)].append(i)
NEU = list(range(512)) if SMOKE else list(range(DFF)); NEUt = torch.tensor(NEU, device=DEV)
log(f"{NP} prompts, blocks {BLOCKS} of {NL}, {len(NEU)} neurons per block")
# ---------------- the clean pass: block outputs, the final residual, activations at the last position; the lens fit on wikitext
XB = {b: torch.zeros(NP, D, device=DEV) for b in BLOCKS}; XL = torch.zeros(NP, D, device=DEV); A = {b: torch.zeros(NP, DFF, device=DEV) for b in BLOCKS}
for ln_, idx in GRP.items():
    x = torch.tensor([IDS[i] for i in idx], device=DEV); cap = {}; hs = []
    for b in BLOCKS:
        hs.append(L[b].register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(("x", b), (o[0] if isinstance(o, tuple) else o)[:, -1])))
        hs.append(act_mod(b).register_forward_pre_hook(lambda m, a, b=b: cap.__setitem__(("a", b), a[0][:, -1])))
    hs.append(FNORM.register_forward_pre_hook(lambda m, a: cap.__setitem__("xl", a[0][:, -1])))
    try: base_(input_ids=x, use_cache=False)
    finally: [h.remove() for h in hs]
    it = torch.tensor(idx, device=DEV); XL[it] = cap["xl"].float()
    for b in BLOCKS: XB[b][it] = cap[("x", b)].float(); A[b][it] = cap[("a", b)].float()
def readout(X): return UNEMB(FNORM(X)).float().log_softmax(-1)
LP0 = readout(XL); TOP = LP0.argmax(-1)
import datasets
wt = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip()]
wid = tok("\n".join(wt[:4000]), return_tensors="pt", add_special_tokens=False).input_ids[0]; NW = 16 if SMOKE else 64; st = np.linspace(0, len(wid) - 257, NW).astype(int); WW = torch.stack([wid[s:s + 256] for s in st]).to(DEV)
LENS = {}
for b in BLOCKS:
    xs, xl = [], []
    for s0 in range(0, NW, 8):
        cap = {}; h1 = L[b].register_forward_hook(lambda m, a, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[:, 1:])); h2 = FNORM.register_forward_pre_hook(lambda m, a: cap.__setitem__("xl", a[0][:, 1:]))
        try: base_(input_ids=WW[s0:s0 + 8], use_cache=False)
        finally: h1.remove(); h2.remove()
        xs.append(cap["x"].reshape(-1, D).float()); xl.append(cap["xl"].reshape(-1, D).float())
    Xs, Xl = torch.cat(xs), torch.cat(xl); Xa = torch.cat([Xs, torch.ones(Xs.shape[0], 1, device=DEV)], 1)
    M = torch.linalg.solve(Xa.T @ Xa + 1e-3 * torch.eye(D + 1, device=DEV) * float((Xa * Xa).mean()), Xa.T @ Xl)   # ridge, the affine translator of the tuned lens
    r2 = 1 - float(((Xa @ M - Xl) ** 2).sum() / ((Xl - Xl.mean(0)) ** 2).sum()); LENS[b] = (M, r2)
lens = lambda b, X: readout(torch.cat([X, torch.ones(X.shape[0], 1, device=DEV)], 1) @ LENS[b][0])
log("lens fit, R^2 of the final residual from the block's: " + ", ".join(f"block {b} {LENS[b][1]:.3f}" for b in BLOCKS))
kl = lambda P, Q: (P.exp() * (P - Q)).sum(-1).clamp_min(0)   # broadcasts: P [.., V] against Q [.., V]
VOC = LP0.shape[-1]; ROWS = max(64, int(1.5e9 / (VOC * 4)))   # rows of vocabulary-sized tensors per chunk
# ---------------- direct and local effects, exact, in chunks of neurons
def exact_effects(b):
    W = WOUT(b); Ab = A[b]; Ldir = torch.zeros(len(NEU), NP, device=DEV); Lloc = torch.zeros(len(NEU), NP, device=DEV); LPl0 = lens(b, XB[b]); CH = max(1, ROWS // NP)
    for c0 in range(0, len(NEU), CH):
        js = NEUt[c0:c0 + CH]; dlt = Ab[:, js].T[:, :, None] * W[:, js].T[:, None, :]   # [ch, NP, D]: each neuron's write at each prompt
        Ldir[c0:c0 + len(js)] = kl(LP0[None], readout((XL[None] - dlt).reshape(-1, D)).reshape(len(js), NP, -1))
        Lloc[c0:c0 + len(js)] = kl(LPl0[None], lens(b, (XB[b][None] - dlt).reshape(-1, D)).reshape(len(js), NP, -1))
    return Ldir, Lloc
# ---------------- total effect: the neuron zeroed at the last position, the prefix cached
def total_effects(b, inject=None):
    """with inject = [NP, K, D] vectors, the null instead: each vector subtracted from block b's output at the last position"""
    K = len(NEU) if inject is None else inject.shape[1]; out = torch.zeros(K, NP, device=DEV)
    for ln_, idx in GRP.items():
        it = torch.tensor(idx, device=DEV); n = len(idx); NB = max(1, min(256, MAXSEQ // n)); x = torch.tensor([IDS[i] for i in idx], device=DEV)
        o = base_(input_ids=x[:, :-1], use_cache=True); cache = o.past_key_values; cache.batch_repeat_interleave(NB); last = x[:, -1:].repeat_interleave(NB, 0)
        rows = torch.arange(n * NB, device=DEV); slot = rows % NB; lp0 = None
        for c0 in range(-NB, K, NB):   # the first pass changes nothing: the baseline on the same cached path
            ks = torch.arange(c0, c0 + NB, device=DEV); live = (ks >= 0) & (ks < K); kk = ks.clamp(0, K - 1)[slot]
            if inject is None:
                neur = NEUt[kk]
                def hook(m, a):
                    h = a[0].clone(); keep = live[slot]; h[rows[keep], -1, neur[keep]] = 0.0; return (h,)
                hd = act_mod(b).register_forward_pre_hook(hook)
            else:
                vec = inject[it.repeat_interleave(NB), kk]   # [n*NB, D]
                def hook(m, a, o):
                    h = (o[0] if isinstance(o, tuple) else o).clone(); keep = live[slot]; h[rows[keep], -1] -= vec[keep]; return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
                hd = L[b].register_forward_hook(hook)
            try: hl = base_(input_ids=last, past_key_values=cache, use_cache=True).last_hidden_state[:, -1]
            finally: hd.remove(); cache.crop(-1)
            lp = UNEMB(hl).float().log_softmax(-1)
            if c0 < 0: lp0 = lp; continue
            v = kl(lp0, lp); m = live[slot]; out[ks.clamp(0, K - 1)[slot][m], it.repeat_interleave(NB)[m]] = v[m]
        del cache
    return out
def nulls(b):
    """per prompt, KL of random directions at a grid of norms, for each of the three effects; returns the 95th percentile per (prompt, norm)"""
    nrm = (A[b].abs() * WOUT(b).norm(dim=0)[None]); lo, hi = float(nrm[nrm > 0].quantile(0.01)), float(nrm.max())
    grid = torch.logspace(math.log10(max(lo, 1e-4)), math.log10(hi), NORMS, device=DEV); g = torch.Generator(device=DEV).manual_seed(677 + b)
    U = torch.randn(NP, NDIR, D, device=DEV, generator=g); U = U / U.norm(dim=-1, keepdim=True)
    V = (U[:, None] * grid[None, :, None, None]).reshape(NP, NORMS * NDIR, D)   # [NP, NORMS*NDIR, D]
    tot = total_effects(b, inject=V).T.reshape(NP, NORMS, NDIR); LPl0 = lens(b, XB[b]); dr = torch.zeros(NP, NORMS * NDIR, device=DEV); lo_ = torch.zeros(NP, NORMS * NDIR, device=DEV); CP = max(1, ROWS // (NORMS * NDIR))
    for p0 in range(0, NP, CP):
        sl = slice(p0, p0 + CP); n_ = V[sl].shape[0]
        dr[sl] = kl(LP0[sl, None], readout((XL[sl, None] - V[sl]).reshape(-1, D)).reshape(n_, NORMS * NDIR, -1)); lo_[sl] = kl(LPl0[sl, None], lens(b, (XB[b][sl, None] - V[sl]).reshape(-1, D)).reshape(n_, NORMS * NDIR, -1))
    dr = dr.reshape(NP, NORMS, NDIR); lo_ = lo_.reshape(NP, NORMS, NDIR)
    q = lambda T: T.quantile(0.95, dim=-1)
    return grid, {"total": q(tot), "direct": q(dr), "local": q(lo_)}
def interp(grid, Q, norms):
    """the null's 95th percentile at each (neuron, prompt) norm, log-log interpolation over the grid; [NEU, NP]"""
    lg = torch.log(grid); x = torch.log(norms.clamp_min(float(grid[0])))
    k = torch.searchsorted(lg, x.contiguous()).clamp(1, len(grid) - 1); x0, x1 = lg[k - 1], lg[k]; w = ((x - x0) / (x1 - x0)).clamp(0, 1)
    q = torch.log(Q.clamp_min(1e-12)).T   # [NORMS, NP]
    i_ = torch.arange(norms.shape[1], device=DEV)[None].expand_as(k); return torch.exp(q[k - 1, i_] * (1 - w) + q[k, i_] * w)
RES = dict(model=HF, blocks={}, n_prompts=NP, n_neurons=len(NEU))
for b in BLOCKS:
    Ldir, Lloc = exact_effects(b); Ltot = total_effects(b); grid, Q = nulls(b)
    norms = (A[b][:, NEUt].abs() * WOUT(b)[:, NEUt].norm(dim=0)[None]).T   # [NEU, NP]
    Nt, Nd, Nl = interp(grid, Q["total"], norms), interp(grid, Q["direct"], norms), interp(grid, Q["local"], norms)
    T_, D_, Lo_ = Ltot.sum(1), Ldir.sum(1), Lloc.sum(1); bt, bd, bl = T_ > Nt.sum(1), D_ > Nd.sum(1), Lo_ > Nl.sum(1)
    direct_sp = bt & (D_ >= 0.5 * T_); router = bt & (D_ < 0.5 * T_); washed = bl & ~bt; quiet = ~bt & ~bl
    sp = torch.full((len(NEU),), 3, dtype=torch.long, device=DEV); sp[direct_sp] = 0; sp[router] = 1; sp[washed] = 2   # 0 direct, 1 router, 2 washed-out, 3 quiet
    rk = lambda v: v.argsort().argsort().float(); sr = lambda a, c: float(torch.corrcoef(torch.stack([rk(a), rk(c)]))[0, 1])
    top = T_.argsort(descending=True)[:100]
    R = dict(n=len(NEU), beyond_total=int(bt.sum()), beyond_local=int(bl.sum()), beyond_direct=int(bd.sum()), direct_speakers=int(direct_sp.sum()), routers=int(router.sum()), washed_out=int(washed.sum()), quiet=int(quiet.sum()),
             router_share_of_beyond_total=float(router.sum() / bt.sum().clamp_min(1)), washed_share_of_beyond_local=float(washed.sum() / bl.sum().clamp_min(1)),
             median_direct_over_total_top100=float((D_[top] / T_[top].clamp_min(1e-12)).median()), spearman_direct_total=sr(D_, T_), spearman_local_total=sr(Lo_, T_), lens_r2=LENS[b][1],
             total_null_ratio_median=float((T_ / Nt.sum(1).clamp_min(1e-12)).median()), species=sp.cpu().tolist(), top10_total=[[NEU[int(j)], float(T_[j] / NP), float(D_[j] / NP), float(Lo_[j] / NP)] for j in top[:10]])
    RES["blocks"][str(b)] = R
    log(f"block {b}: beyond the null on total {R['beyond_total']}, local {R['beyond_local']}, direct {R['beyond_direct']} of {len(NEU)}; direct speakers {R['direct_speakers']}, indirect routers {R['routers']} (share of beyond-total {R['router_share_of_beyond_total']:.2f}), washed-out {R['washed_out']} (share of beyond-local {R['washed_share_of_beyond_local']:.2f}), quiet {R['quiet']}; "
        f"top-100 by total: median direct/total {R['median_direct_over_total_top100']:.2f}; Spearman direct~total {R['spearman_direct_total']:.2f}, local~total {R['spearman_local_total']:.2f}; lens R^2 {LENS[b][1]:.3f}")
RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S"); RES["seconds"] = time.time() - t0
os.makedirs("/workspace/wdd/results", exist_ok=True); json.dump(RES, open(f"/workspace/wdd/results/e677_{TAG}.json", "w"), indent=1); log("done"); print("done", flush=True)
