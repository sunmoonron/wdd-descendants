"""e696 (session 157): the bypass. e693 found the weak activations' summed write per block low-rank in data space:
sixteen principal directions of it recover most of the zeroing cost. Here the question is whether that envelope is a
linear function of the block's input, so that the weak neurons need not run at all: on held-out text a ridge regression
from the normalised block input to the weak write is solved in closed form (the full map, and its projection on the weak
write's 16 and 64 principal directions); at test time all but the strongest quarter of each neuron's activations are
zeroed at every block at once and the predicted write is added back instead, against zero, the per-neuron mean and the
oracle projections of the true weak write (e693's pca16 and pca64). Scored by next-token KL and loss on wikitext-2 test
and by ARC-Easy accuracy (four-way, continuation log-likelihood) under the same hooks. Arguments: --model M [--smoke].
Pre-registered in e696_e701_prereg.json."""
import argparse
from c11 import *
BF16 = {"qwen7b", "qwen14b", "qwen32b"}
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e696_{name}" + ("_smoke" if SMOKE else ""); M = load(name, dtype=torch.bfloat16 if name in BF16 else torch.float32); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWC = 16 if SMOKE else 128; NWE = 8 if SMOKE else 48; NWH = 16 if SMOKE else 96; NQ = 40 if SMOKE else 400; Q = 0.25; RS_ = [16, 64]; LAM = 1e-2
CONDS = ["zero", "nmean", "pca16", "pca64", "bypass16", "bypass64", "bypassD"]
ids_all = wikitext_ids(M, "train"); half = len(ids_all) // 2
cal_ids = windows_from(ids_all[:half], NWC, T).to(DEV); ho_ids = windows_from(ids_all[half:], NWH, T).to(DEV); ev_ids = windows_from(wikitext_ids(M, "test"), NWE, T).to(DEV)
def write_mat(b): w = write_mod(M, b).weight; return w.T if M["fam"] == "gpt2" else w
XIN = {}   # the normalised input of each block's MLP, caught at its first read module
def xin_hook(b):
    def h(mod, a): XIN[b] = a[0]
    return h
HX = [list(read_mods(M, b).values())[0].register_forward_pre_hook(xin_hook(b)) for b in range(NB)]
# ---------------- calibration
NPOS = NWC * T; SIGNED = {b: [] for b in range(NB)}
with Capture(M, list(range(NB)), ["act"]) as C:
    for s0 in range(0, NWC, 8):
        run_base(M, cal_ids[s0:s0 + 8])
        for b in range(NB): SIGNED[b].append(C.cap[("act", b)].reshape(-1, DFF).half().cpu())
TAU = torch.zeros(NB, DFF, device=DEV); MEANW = torch.zeros(NB, DFF, device=DEV)
for b in range(NB):
    X_ = torch.cat(SIGNED[b]); SIGNED[b] = None
    for c0 in range(0, DFF, 512):
        xs = X_[:, c0:c0 + 512].to(DEV).float(); t = xs.abs().quantile(1 - Q, dim=0); wk = xs.abs() < t[None]; TAU[b, c0:c0 + 512] = t; MEANW[b, c0:c0 + 512] = (xs * wk).sum(0) / wk.sum(0).clamp_min(1)
    del X_
# ---------------- held-out pass: per block the weak write's second moment, and the moments for the regression from the normalised input to the weak write
COV = [torch.zeros(D, D, device=DEV) for _ in range(NB)]; SXX = [torch.zeros(D, D, device=DEV) for _ in range(NB)]; SXD = [torch.zeros(D, D, device=DEV) for _ in range(NB)]; SX = [torch.zeros(D, device=DEV) for _ in range(NB)]; SD = [torch.zeros(D, device=DEV) for _ in range(NB)]; n = 0
with Capture(M, list(range(NB)), ["act"]) as C:
    for s0 in range(0, NWH, 8):
        run_base(M, ho_ids[s0:s0 + 8]); n += 8 * T
        for b in range(NB):
            a = C.cap[("act", b)].reshape(-1, DFF); wk = a.abs() < TAU[b].to(a.dtype)[None]; dx = ((a * wk).to(write_mat(b).dtype) @ write_mat(b).T).float(); x = XIN[b].reshape(-1, D).float()
            COV[b] += dx.T @ dx; SXX[b] += x.T @ x; SXD[b] += x.T @ dx; SX[b] += x.sum(0); SD[b] += dx.sum(0)
PROJ = {}; WMAP = {}; XBAR = {}; DBAR = {}; R2 = []
for b in range(NB):
    ev, evec = torch.linalg.eigh(COV[b])
    for r in RS_: PROJ[(f"pca{r}", b)] = evec[:, -r:].contiguous()
    xb, db = SX[b] / n, SD[b] / n; Cxx = SXX[b] / n - torch.outer(xb, xb); Cxd = SXD[b] / n - torch.outer(xb, db); lam = LAM * Cxx.diagonal().mean()
    W = torch.linalg.solve(Cxx + lam * torch.eye(D, device=DEV), Cxd); WMAP[b] = W; XBAR[b] = xb; DBAR[b] = db   # d_hat = db + (x - xb) @ W
    Cdd = COV[b] / n - torch.outer(db, db); expl = (Cxd.T @ W).diagonal().sum(); R2.append(float(expl / Cdd.diagonal().sum().clamp_min(1e-12)))   # the weak write's variance the linear map explains (on the fitting data)
L_(f"regressions solved on {n} held-out positions: the linear map explains {med(R2):.2f} of the weak write's variance (median over blocks; min {min(R2):.2f}, max {max(R2):.2f})")
# ---------------- hooks: at every block, all but the strongest quarter zeroed; the replacement write added to the block's output
COND = {}; HOLD = {}; STAT = collections.defaultdict(float)
def pre_hook(b):
    def h(mod, a):
        if not COND: return None
        x = a[0]; wk = x.abs() < TAU[b].to(x.dtype)[None, None]; kind = COND["kind"]; STAT["removed"] += float((x.float() ** 2)[wk].sum()); STAT["all"] += float((x.float() ** 2).sum())
        if kind == "nmean": return (torch.where(wk, MEANW[b].to(x.dtype)[None, None].expand_as(x), x),)
        if kind.startswith("pca"): HOLD[b] = x * wk
        return (torch.where(wk, torch.zeros_like(x), x),)
    return h
def fwd_hook(b):
    def h(mod, a, out):
        if not COND or COND["kind"] in ("zero", "nmean"): return None
        kind = COND["kind"]
        if kind.startswith("pca"): P = PROJ[(kind, b)]; dx = (HOLD.pop(b).to(write_mat(b).dtype) @ write_mat(b).T).float(); add = (dx @ P) @ P.T
        else:
            x = XIN[b].float(); dh = DBAR[b][None, None] + (x - XBAR[b]) @ WMAP[b]
            if kind == "bypassD": add = dh
            else: P = PROJ[("pca" + kind[6:], b)]; add = (dh @ P) @ P.T
        return out + add.to(out.dtype)
    return h
HS = [write_mod(M, b).register_forward_pre_hook(pre_hook(b)) for b in range(NB)] + [write_mod(M, b).register_forward_hook(fwd_hook(b)) for b in range(NB)]
def evaluate():
    STAT.clear(); LPs = []
    for s0 in range(0, NWE, 8): o = M["base"](input_ids=ev_ids[s0:s0 + 8], use_cache=False); LPs.append(logprobs_at(M, o.last_hidden_state[:, :-1]))
    return torch.cat(LPs), (STAT["removed"] / max(STAT["all"], 1e-12) if STAT else 0.0)
LP0, _ = evaluate(); TGT = ev_ids[:, 1:]; PRED0 = LP0.argmax(-1); LOSS0 = float(-LP0.gather(-1, TGT[..., None]).float().mean())
def compare(LP1):
    ks, fl = [], []
    for w in range(NWE): ks.append(kl(LP0[w].float(), LP1[w].float())); fl.append(LP1[w].argmax(-1) != PRED0[w])
    return torch.stack(ks), torch.stack(fl), float(-LP1.gather(-1, TGT[..., None]).float().mean())
# ---------------- ARC-Easy, four-way by the continuation's log-likelihood, under the same hooks
import datasets
arc = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"); items = []
for ex in arc:
    if len(ex["choices"]["text"]) != 4 or ex["answerKey"] not in ex["choices"]["label"]: continue
    items.append((ex["question"], ex["choices"]["text"], ex["choices"]["label"].index(ex["answerKey"])))
    if len(items) >= NQ: break
tok = M["tok"]; SEQS = []; META = []
for qi, (q, ch, ans) in enumerate(items):
    pre = tok(f"Question: {q}\nAnswer:", add_special_tokens=False).input_ids
    for ci, c in enumerate(ch): full = pre + tok(" " + c, add_special_tokens=False).input_ids; SEQS.append(full); META.append((qi, ci, len(pre)))
pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
def arc_accuracy():
    scores = torch.zeros(len(items), 4, device=DEV); BS = 32
    for s in range(0, len(SEQS), BS):
        batch = SEQS[s:s + BS]; L = max(len(x) for x in batch); x = torch.full((len(batch), L), pad, device=DEV, dtype=torch.long)
        for i, sq in enumerate(batch): x[i, :len(sq)] = torch.tensor(sq, device=DEV)
        o = M["base"](input_ids=x, use_cache=False); lp = logprobs_at(M, o.last_hidden_state[:, :-1]).reshape(len(batch), L - 1, -1)
        for i, sq in enumerate(batch):
            qi, ci, lp_ = META[s + i]; t = torch.tensor(sq[lp_:], device=DEV); scores[qi, ci] = lp[i, lp_ - 1:len(sq) - 1].gather(-1, t[:, None]).sum()
    return float((scores.argmax(1) == torch.tensor([a for _, _, a in items], device=DEV)).float().mean())
ACC0 = arc_accuracy()
res = dict(model=M["hf"], n_blocks=NB, DFF=DFF, eval_positions=int(TGT.numel()), base_loss=LOSS0, arc_items=len(items), arc_clean=ACC0, keep_quantile=Q, ridge=LAM, linear_r2_per_block=R2, linear_r2_median=med(R2), conditions={})
L_(f"clean: loss {LOSS0:.3f}, ARC-Easy {ACC0:.3f} on {len(items)} items")
for kind in CONDS:
    COND.update(kind=kind); LP1, remf = evaluate(); k_, flip, loss1 = compare(LP1); del LP1; acc = arc_accuracy()
    res["conditions"][kind] = dict(removed_energy_fraction=remf, kl_mean=float(k_.mean()), kl_median=float(k_.median()), loss_delta=loss1 - LOSS0, flip_rate=float(flip.float().mean()), arc=acc)
    r = res["conditions"][kind]; L_(f"{kind}: KL mean {r['kl_mean']:.4f}, loss {r['loss_delta']:+.4f}, flips {r['flip_rate']:.4f}, ARC-Easy {acc:.3f}")
COND.clear(); [h.remove() for h in HS + HX]
c = {k: v["kl_mean"] for k, v in res["conditions"].items()}; acc = {k: v["arc"] for k, v in res["conditions"].items()}; rec = lambda k: 1 - c[k] / c["zero"]
V = dict(L1_bypass16_matches_pca16=bool(rec("bypass16") >= 0.8 * rec("pca16")), L2_bypass16_halves_zero=bool(c["bypass16"] < 0.5 * c["zero"]), L3_bypassD_matches_pca16=bool(rec("bypassD") >= 0.9 * rec("pca16")), L4_arc_bypass16_within_2pts=bool(acc["bypass16"] >= ACC0 - 0.02), L5_arc_zero_drops_5pts=bool(acc["zero"] <= ACC0 - 0.05), L6_linear_r2_above_half=bool(med(R2) > 0.5))
res["verdicts"] = V
summ = (f"[{name}] all but the strongest quarter of each neuron's activations zeroed at every block, KL in nats on {TGT.numel()} tokens and ARC-Easy accuracy on {len(items)} items (clean {ACC0:.3f}): " + ", ".join(f"{k} {c[k]:.3f} ({acc[k]:.3f})" for k in CONDS) + f"; recovered by the oracle 16 directions {rec('pca16'):.2f}, by the predicted 16 {rec('bypass16'):.2f}, by the full linear prediction {rec('bypassD'):.2f}; the linear map explains {med(R2):.2f} of the weak write's variance; verdicts {V}")
save(TAG, res, summ)
