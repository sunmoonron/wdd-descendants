"""e693 (session 156): the weak-activation ladder. e681 found that zeroing all but the strongest quarter of each neuron's
activations costs 0.24 to 8.7 nats, and that the per-neuron mean of the removed activations recovers most of that in the
GELU models and a fraction in the gated ones. Here the removed activations are replaced, without training, by summaries
of rising resolution built on held-out text: zero; the per-neuron mean; the per-token mean (a lookup table of the neuron's
weak activation given the current token); the per-bigram mean with backoff; a random held-out occurrence of the same
token (the resample control: does the variation within a token matter?); and the removed activations' summed write
projected onto r principal directions of that write (data, r = 16 and 64) or onto the block's r dominant write directions
(the weights' singular vectors, r = 16 and 64). Measured as e681: next-token KL to the unedited model on wikitext-2 test,
the loss change and the flips. Arguments: --model M [--smoke] [--lite]. Pre-registered in e692_e694_prereg.json."""
import argparse
from c11 import *
BF16 = {"qwen7b", "qwen14b", "qwen32b"}
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--lite", action="store_true"); A_ = ap.parse_args(); SMOKE, LITE = A_.smoke, A_.lite
name = A_.model; TAG = f"e693_{name}" + ("_smoke" if SMOKE else ""); M = load(name, dtype=torch.bfloat16 if name in BF16 else torch.float32); NB, D, DFF, VOC = M["NB"], M["D"], M["DFF"], M["V"]; T = 256; L_ = lambda s: log(TAG, s)
NWC = 16 if SMOKE else (48 if LITE else 128); NWE = 8 if SMOKE else (24 if LITE else 48); NWH = 16 if SMOKE else (48 if LITE else 96); Q = 0.25; PCA_R = [1, 4, 16, 64]; SVD_R = [16, 64]; SPCA_R = [16, 64]
CONDS = ["zero", "nmean", "token", "bigram", "resample"] + [f"pca{r}" for r in PCA_R] + [f"svd{r}" for r in SVD_R] + [f"spca{r}" for r in SPCA_R]
ids_all = wikitext_ids(M, "train"); half = len(ids_all) // 2
cal_ids = windows_from(ids_all[:half], NWC, T).to(DEV); ho_ids = windows_from(ids_all[half:], NWH, T).to(DEV); ev_ids = windows_from(wikitext_ids(M, "test"), NWE, T).to(DEV)
def write_mat(b): w = write_mod(M, b).weight; return w.T if M["fam"] == "gpt2" else w   # [D, DFF] view in the model's dtype
# ---------------- calibration: per neuron, the threshold under which an activation is weak (the strongest quarter kept by count) and the mean of the weak activations
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
L_(f"calibrated on {NPOS} positions, {NB} blocks of {DFF}")
# ---------------- evaluation tokens and bigrams (compact ids), held-out mapping, resample rows
uniq, inv = torch.unique(ev_ids.reshape(-1), return_inverse=True); VE = uniq.numel(); TOKIDX = inv
prev_e = torch.cat([torch.full((NWE, 1), -1, device=DEV, dtype=torch.long), ev_ids[:, :-1]], 1); bk_e = ((prev_e + 1) * VOC + ev_ids).reshape(-1); uniq2, inv2 = torch.unique(bk_e, return_inverse=True); VB = uniq2.numel(); BIIDX = inv2
tok_of_bi = torch.zeros(VB, dtype=torch.long, device=DEV).scatter_(0, inv2, inv)
ht = ho_ids.reshape(-1); hpos = torch.searchsorted(uniq, ht).clamp(max=VE - 1); hval = uniq[hpos] == ht; hidx = torch.where(hval, hpos, torch.full_like(hpos, VE))
prev_h = torch.cat([torch.full((NWH, 1), -1, device=DEV, dtype=torch.long), ho_ids[:, :-1]], 1); bk_h = ((prev_h + 1) * VOC + ho_ids).reshape(-1); hpos2 = torch.searchsorted(uniq2, bk_h).clamp(max=VB - 1); hval2 = uniq2[hpos2] == bk_h; hidx2 = torch.where(hval2, hpos2, torch.full_like(hpos2, VB))
cnt_tok = torch.bincount(hidx, minlength=VE + 1)[:VE]; cnt_bi = torch.bincount(hidx2, minlength=VB + 1)[:VB]
rows_h = torch.nonzero(hval)[:, 0]; o_h = torch.argsort(hidx[rows_h]); rows_sorted = rows_h[o_h]; starts = torch.searchsorted(hidx[rows_h][o_h], torch.arange(VE + 1, device=DEV))
gen = torch.Generator(device=DEV).manual_seed(693); m_e = cnt_tok[inv]; r_e = (torch.rand(inv.numel(), device=DEV, generator=gen) * m_e).long().clamp(max=(m_e - 1).clamp_min(0))
RSROW = torch.where(m_e > 0, rows_sorted[(starts[inv] + r_e).clamp(max=max(rows_sorted.numel() - 1, 0))], torch.full_like(inv, -1)) if rows_sorted.numel() else torch.full_like(inv, -1)
COVER = dict(token=float((cnt_tok[inv] >= 2).float().mean()), bigram=float((cnt_bi[inv2] >= 2).float().mean()), resample=float((m_e > 0).float().mean()))
# ---------------- held-out pass: per-token and per-bigram sums of the weak activations, the held-out activations (for the resample) and the covariance of the weak write
hi_c, hi2_c = hidx.cpu(), hidx2.cpu(); SUMT = [torch.zeros(VE + 1, DFF) for _ in range(NB)]; CNTT = [torch.zeros(VE + 1, DFF) for _ in range(NB)]; SUMB = [torch.zeros(VB + 1, DFF) for _ in range(NB)]; CNTB = [torch.zeros(VB + 1, DFF) for _ in range(NB)]
HACT = [torch.zeros(NWH * T, DFF, dtype=torch.half) for _ in range(NB)]; COV = [torch.zeros(D, D, device=DEV) for _ in range(NB)]; COVX = [torch.zeros(D, D, device=DEV) for _ in range(NB)]
with Capture(M, list(range(NB)), ["act", "xin"]) as C:
    for s0 in range(0, NWH, 8):
        run_base(M, ho_ids[s0:s0 + 8]); sl = slice(s0 * T, (s0 + 8) * T); hi, hi2 = hi_c[sl], hi2_c[sl]
        for b in range(NB):
            a = C.cap[("act", b)].reshape(-1, DFF); wk = a.abs() < TAU[b].to(a.dtype)[None]; aw = (a * wk); awc = aw.float().cpu(); wkc = wk.float().cpu()
            SUMT[b].index_add_(0, hi, awc); CNTT[b].index_add_(0, hi, wkc); SUMB[b].index_add_(0, hi2, awc); CNTB[b].index_add_(0, hi2, wkc); HACT[b][sl] = a.half().cpu()
            dx = (aw.to(write_mat(b).dtype) @ write_mat(b).T).float(); COV[b] += dx.T @ dx; xin = C.cap[("xin", b)].reshape(-1, D).float(); COVX[b] += xin.T @ xin
TABT = [torch.where(CNTT[b][:VE] >= 2, SUMT[b][:VE] / CNTT[b][:VE].clamp_min(1), MEANW[b].cpu()[None].expand(VE, DFF)) for b in range(NB)]; del SUMT, CNTT
TABB = [torch.where(CNTB[b][:VB] >= 2, SUMB[b][:VB] / CNTB[b][:VB].clamp_min(1), TABT[b][tok_of_bi.cpu()]) for b in range(NB)]; del SUMB, CNTB
PROJ = {}; ENERGY16 = []
for b in range(NB):
    ev, evec = torch.linalg.eigh(COV[b]); W = write_mat(b).float(); ew, ewec = torch.linalg.eigh(W @ W.T); ex, exec_ = torch.linalg.eigh(COVX[b]); ENERGY16.append(float(ev[-16:].sum() / ev.sum().clamp_min(1e-12)))
    for r in PCA_R: PROJ[(f"pca{r}", b)] = evec[:, -r:].contiguous()
    for r in SVD_R: PROJ[(f"svd{r}", b)] = ewec[:, -r:].contiguous()
    for r in SPCA_R: PROJ[(f"spca{r}", b)] = exec_[:, -r:].contiguous()
    del W
DIR = dict(native_max_cos=[], mean_dir_cos=[], stream_overlap=[], cross_block=[], energy_top16=ENERGY16)   # what the weak write's principal directions are
for b in range(NB):
    Pw = PROJ[("pca16", b)]; W = write_mat(b).float(); Wn = W / W.norm(dim=0, keepdim=True).clamp_min(1e-6); DIR["native_max_cos"].append(float((Pw.T @ Wn).abs().max(1).values.median()))
    m = W @ MEANW[b]; m = m / m.norm().clamp_min(1e-9); DIR["mean_dir_cos"].append(float((Pw[:, -1] @ m).abs())); DIR["stream_overlap"].append(float((PROJ[("spca16", b)].T @ Pw).pow(2).sum() / 16))
    if b + 1 < NB: DIR["cross_block"].append(float((PROJ[("pca16", b + 1)].T @ Pw).pow(2).sum() / 16))
    del W
L_(f"tables from {NWH * T} held-out positions: coverage {COVER}; {VE} tokens, {VB} bigrams in the evaluation set")
# ---------------- conditions as hooks on every block's write module
COND = {}; HOLD = {}; STAT = collections.defaultdict(float)
def pre_hook(b):
    def h(mod, a):
        if not COND: return None
        x = a[0]; wk = x.abs() < TAU[b].to(x.dtype)[None, None]; kind = COND["kind"]; rows = COND["rows"]
        STAT["removed"] += float((x.float() ** 2)[wk].sum()); STAT["all"] += float((x.float() ** 2).sum()); STAT["npos"] += float(wk.float().sum()); STAT["nall"] += float(wk.numel())
        if kind == "nmean": rep = MEANW[b].to(x.dtype)[None, None].expand_as(x)
        elif kind == "token": rep = TABT[b][TOKIDX[rows].cpu()].to(DEV).to(x.dtype).reshape(x.shape)
        elif kind == "bigram": rep = TABB[b][BIIDX[rows].cpu()].to(DEV).to(x.dtype).reshape(x.shape)
        elif kind == "resample":
            rr = RSROW[rows]; v = HACT[b][rr.clamp(min=0).cpu()].to(DEV).float(); tab = TABT[b][TOKIDX[rows].cpu()].to(DEV); wkh = v.abs() < TAU[b][None]
            rep = torch.where((rr >= 0)[:, None] & wkh, v, tab).to(x.dtype).reshape(x.shape)
        else: rep = torch.zeros_like(x)
        if kind.startswith(("pca", "svd", "spca")): HOLD[b] = x * wk
        return (torch.where(wk, rep, x),)
    return h
def fwd_hook(b):
    def h(mod, a, out):
        if not COND or not COND["kind"].startswith(("pca", "svd", "spca")): return None
        P = PROJ[(COND["kind"], b)]; xw = HOLD.pop(b); dx = (xw.to(write_mat(b).dtype) @ write_mat(b).T).float(); return out + ((dx @ P) @ P.T).to(out.dtype)
    return h
HS = [write_mod(M, b).register_forward_pre_hook(pre_hook(b)) for b in range(NB)] + [write_mod(M, b).register_forward_hook(fwd_hook(b)) for b in range(NB)]
def evaluate():
    STAT.clear(); LPs = []
    for s0 in range(0, NWE, 8):
        if COND: COND["rows"] = torch.arange(s0 * T, (s0 + 8) * T, device=DEV)
        o = M["base"](input_ids=ev_ids[s0:s0 + 8], use_cache=False); LPs.append(logprobs_at(M, o.last_hidden_state[:, :-1]))
    return torch.cat(LPs), ((STAT["removed"] / max(STAT["all"], 1e-12), STAT["npos"] / max(STAT["nall"], 1e-12)) if STAT else (0.0, 0.0))
LP0, _ = evaluate(); TGT = ev_ids[:, 1:]; PRED0 = LP0.argmax(-1); LOSS0 = float(-LP0.gather(-1, TGT[..., None]).float().mean())
def compare(LP1):
    ks, fl = [], []
    for w in range(NWE): ks.append(kl(LP0[w].float(), LP1[w].float())); fl.append(LP1[w].argmax(-1) != PRED0[w])
    return torch.stack(ks), torch.stack(fl), float(-LP1.gather(-1, TGT[..., None]).float().mean())
res = dict(model=M["hf"], n_blocks=NB, DFF=DFF, calib_positions=NPOS, heldout_positions=NWH * T, eval_positions=int(TGT.numel()), base_loss=LOSS0, keep_quantile=Q, coverage=COVER, directions={k: med(v) for k, v in DIR.items()}, directions_per_block=DIR, conditions={})
for kind in CONDS:
    COND.update(kind=kind); LP1, (remf, posf) = evaluate(); k_, flip, loss1 = compare(LP1); del LP1
    res["conditions"][kind] = dict(removed_energy_fraction=remf, removed_position_fraction=posf, kl_mean=float(k_.mean()), kl_median=float(k_.median()), loss_delta=loss1 - LOSS0, flip_rate=float(flip.float().mean()))
    r = res["conditions"][kind]; L_(f"{kind}: KL mean {r['kl_mean']:.4f}, loss {r['loss_delta']:+.4f}, flips {r['flip_rate']:.4f} (removed energy {remf:.3f}, positions {posf:.3f})")
COND.clear(); [h.remove() for h in HS]
c = {k: v["kl_mean"] for k, v in res["conditions"].items()}
V = dict(V1_token_halves_zero=bool(c["token"] < 0.5 * c["zero"]), V1b_token_below_005=bool(c["token"] < 0.05), V1c_token_beats_nmean=bool(c["token"] < 0.8 * c["nmean"]), V2_bigram_beats_token=bool(c["bigram"] < 0.8 * c["token"]),
         V3_resample_above_token=bool(c["resample"] > 1.5 * c["token"]), V4_pca16_halves_zero=bool(c["pca16"] < 0.5 * c["zero"]), V5_svd16_halves_zero=bool(c["svd16"] < 0.5 * c["zero"]), V6_pca64_below_02=bool(c["pca64"] < 0.2), V7_stream16_within_1p2_of_pca16=bool(c["spca16"] < 1.2 * c["pca16"]), V8_pcs_native_rows=bool(res["directions"]["native_max_cos"] > 0.5), V9_pcs_shared_across_blocks=bool(res["directions"]["cross_block"] > 0.5), V10_weak_write_energy_top16_above_08=bool(res["directions"]["energy_top16"] > 0.8))
res["verdicts"] = V
summ = (f"[{name}] the strongest quarter of each neuron's activations kept, the rest replaced, KL in nats on {TGT.numel()} tokens: " + ", ".join(f"{k} {c[k]:.3f}" for k in CONDS) + f"; recovered by the per-token table {1 - c['token'] / max(c['zero'], 1e-9):.2f} of the zeroing cost (per-neuron mean {1 - c['nmean'] / max(c['zero'], 1e-9):.2f}); "
        f"the weak write's 16 principal directions: energy share {res['directions']['energy_top16']:.2f}, max cosine with a native write row {res['directions']['native_max_cos']:.2f}, first direction's cosine with the mean weak write {res['directions']['mean_dir_cos']:.2f}, overlap with the stream's 16 {res['directions']['stream_overlap']:.2f}, with the next block's 16 {res['directions']['cross_block']:.2f}; coverage {COVER}; verdicts {V}")
save(TAG, res, summ)
