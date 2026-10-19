"""e693b (session 156): what the weak write's directions are. e693 found the removed weak activations' summed write,
projected on sixteen principal directions of its own, recovering most of the zeroing cost, and those directions neither
native rows nor the stream's dominant subspace. Here they are named by weights alone: the removed write (all but the
strongest quarter of each neuron's activations) projected on the right singular directions of the next block's read
matrix (what the next block's neurons read most, gamma folded), of all later blocks' read matrices stacked, and of the
unembedding; and the overlap of the weak write's own principal directions with each of those subspaces, with the stream's
mean direction and with its massive coordinates. Measured as e693 (KL on wikitext-2 test). Arguments: --model M [--smoke].
Pre-registered in e693b_prereg.json."""
import argparse
from c11 import *
BF16 = {"qwen7b", "qwen14b", "qwen32b"}
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e693b_{name}" + ("_smoke" if SMOKE else ""); M = load(name, dtype=torch.bfloat16 if name in BF16 else torch.float32); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWC = 16 if SMOKE else 128; NWE = 8 if SMOKE else 48; NWH = 16 if SMOKE else 96; Q = 0.25; RS_ = [16, 64]
SUBS = ["readsvd", "allread", "unemb", "strong"]; CONDS = ["zero", "pca16"] + [f"{s}{r}" for s in SUBS for r in RS_] + ["full16"]
ids_all = wikitext_ids(M, "train"); half = len(ids_all) // 2
cal_ids = windows_from(ids_all[:half], NWC, T).to(DEV); ho_ids = windows_from(ids_all[half:], NWH, T).to(DEV); ev_ids = windows_from(wikitext_ids(M, "test"), NWE, T).to(DEV)
def write_mat(b): w = write_mod(M, b).weight; return w.T if M["fam"] == "gpt2" else w
# ---------------- calibration: the weak threshold per neuron and the mean weak activation
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
# ---------------- held-out pass: the weak write's second moment and the stream's mean per block
COV = [torch.zeros(D, D, device=DEV) for _ in range(NB)]; COVS = [torch.zeros(D, D, device=DEV) for _ in range(NB)]; COVF = [torch.zeros(D, D, device=DEV) for _ in range(NB)]; MU = [torch.zeros(D, device=DEV) for _ in range(NB)]; nrows = 0
with Capture(M, list(range(NB)), ["act", "xin"]) as C:
    for s0 in range(0, NWH, 8):
        run_base(M, ho_ids[s0:s0 + 8]); nrows += 8 * T
        for b in range(NB):
            a = C.cap[("act", b)].reshape(-1, DFF); wk = a.abs() < TAU[b].to(a.dtype)[None]; dx = ((a * wk).to(write_mat(b).dtype) @ write_mat(b).T).float(); COV[b] += dx.T @ dx; MU[b] += C.cap[("xin", b)].reshape(-1, D).float().sum(0)
            ds = ((a * ~wk).to(write_mat(b).dtype) @ write_mat(b).T).float(); COVS[b] += ds.T @ ds; df = dx + ds; COVF[b] += df.T @ df   # the strong write (the kept quarter) and the whole MLP write
MU = [m / nrows for m in MU]
# ---------------- the subspaces, from weights alone
def reads(b):
    Rd = read_w(M, b); return torch.cat([read_tilde(M, b, Rd[k]) for k in Rd])   # [n, D] gamma-folded read rows in residual space
head = M["model"].get_output_embeddings().weight.detach().float(); g = getattr(M["fnorm"], "weight", None); WU = head * g.detach().float()[None] if g is not None else head
def topdirs(G, r): ev, evec = torch.linalg.eigh(G); return evec[:, -r:].contiguous()
GU = WU.T @ WU; GR = [None] * NB
for b in range(NB): R_ = reads(b); GR[b] = R_.T @ R_; del R_
PROJ = {}; DIRS = {k: [] for k in ("overlap_readsvd16", "overlap_allread16", "overlap_unemb16", "overlap_strong16", "overlap_full16", "pc1_mean_dir", "massive_share_pc1", "massive_share_16", "energy_top16")}
for b in range(NB):
    ev, evec = torch.linalg.eigh(COV[b]); Pw = evec[:, -16:].contiguous(); PROJ[("pca16", b)] = Pw
    Gn = GR[b + 1] if b + 1 < NB else GU; Ga = sum(GR[b + 1:], torch.zeros(D, D, device=DEV)) + (GU if b + 1 >= NB else 0)
    for r in RS_: PROJ[(f"readsvd{r}", b)] = topdirs(Gn, r); PROJ[(f"allread{r}", b)] = topdirs(Ga, r); PROJ[(f"unemb{r}", b)] = topdirs(GU, r); PROJ[(f"strong{r}", b)] = topdirs(COVS[b], r)
    PROJ[("full16", b)] = topdirs(COVF[b], 16); DIRS["overlap_strong16"].append(float((PROJ[("strong16", b)].T @ Pw).pow(2).sum() / 16)); DIRS["overlap_full16"].append(float((PROJ[("full16", b)].T @ Pw).pow(2).sum() / 16))
    ov = lambda Pa, Pb: float((Pa.T @ Pb).pow(2).sum() / Pa.shape[1]); DIRS["overlap_readsvd16"].append(ov(Pw, PROJ[("readsvd16", b)])); DIRS["overlap_allread16"].append(ov(Pw, PROJ[("allread16", b)])); DIRS["overlap_unemb16"].append(ov(Pw, PROJ[("unemb16", b)]))
    mu = MU[b] / MU[b].norm().clamp_min(1e-9); DIRS["pc1_mean_dir"].append(float((Pw[:, -1] @ mu).abs())); big = MU[b].abs().topk(4).indices
    DIRS["massive_share_pc1"].append(float(Pw[big, -1].pow(2).sum())); DIRS["massive_share_16"].append(float(Pw[big].pow(2).sum() / 16)); DIRS["energy_top16"].append(float(ev[-16:].sum() / ev.sum().clamp_min(1e-12)))
L_(f"subspaces built; overlap of the weak write's 16 directions with the next block's 16 read directions median {med(DIRS['overlap_readsvd16']):.2f}, with all later reads {med(DIRS['overlap_allread16']):.2f}, with the unembedding {med(DIRS['overlap_unemb16']):.2f}")
# ---------------- conditions: the weak activations zeroed, their write added back projected on the subspace
COND = {}; HOLD = {}; STAT = collections.defaultdict(float)
def pre_hook(b):
    def h(mod, a):
        if not COND: return None
        x = a[0]; wk = x.abs() < TAU[b].to(x.dtype)[None, None]; STAT["removed"] += float((x.float() ** 2)[wk].sum()); STAT["all"] += float((x.float() ** 2).sum())
        if COND["kind"] != "zero": HOLD[b] = x * wk
        return (torch.where(wk, torch.zeros_like(x), x),)
    return h
def fwd_hook(b):
    def h(mod, a, out):
        if not COND or COND["kind"] == "zero": return None
        P = PROJ[(COND["kind"], b)]; xw = HOLD.pop(b); dx = (xw.to(write_mat(b).dtype) @ write_mat(b).T).float(); return out + ((dx @ P) @ P.T).to(out.dtype)
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
res = dict(model=M["hf"], n_blocks=NB, DFF=DFF, eval_positions=int(TGT.numel()), base_loss=LOSS0, keep_quantile=Q, directions={k: med(v) for k, v in DIRS.items()}, directions_per_block=DIRS, conditions={})
for kind in CONDS:
    COND.update(kind=kind); LP1, remf = evaluate(); k_, flip, loss1 = compare(LP1); del LP1
    res["conditions"][kind] = dict(removed_energy_fraction=remf, kl_mean=float(k_.mean()), kl_median=float(k_.median()), loss_delta=loss1 - LOSS0, flip_rate=float(flip.float().mean()))
    r = res["conditions"][kind]; L_(f"{kind}: KL mean {r['kl_mean']:.4f}, loss {r['loss_delta']:+.4f}, flips {r['flip_rate']:.4f}")
COND.clear(); [h.remove() for h in HS]
c = {k: v["kl_mean"] for k, v in res["conditions"].items()}; dr = res["directions"]
V = dict(D1_readsvd16_halves_zero=bool(c["readsvd16"] < 0.5 * c["zero"]), D1b_readsvd64_halves_zero=bool(c["readsvd64"] < 0.5 * c["zero"]), D2_allread16_halves_zero=bool(c["allread16"] < 0.5 * c["zero"]), D2b_allread64_halves_zero=bool(c["allread64"] < 0.5 * c["zero"]),
         D3_overlap_readsvd16_above_03=bool(dr["overlap_readsvd16"] > 0.3), D3b_overlap_allread16_above_03=bool(dr["overlap_allread16"] > 0.3), D4_unemb16_halves_zero=bool(c["unemb16"] < 0.5 * c["zero"]), D5_pc1_massive=bool(dr["massive_share_pc1"] > 0.3), D6_weights_match_pca=bool(min(c["readsvd16"], c["allread16"]) < 1.2 * c["pca16"]), D7_strong16_matches_pca16=bool(c["strong16"] < 1.2 * c["pca16"]), D8_overlap_strong16_above_05=bool(dr["overlap_strong16"] > 0.5))
res["verdicts"] = V
summ = (f"[{name}] weak activations (all but the strongest quarter) zeroed and their write added back on a subspace, KL in nats on {TGT.numel()} tokens: " + ", ".join(f"{k} {c[k]:.3f}" for k in CONDS) +
        f"; the weak write's 16 principal directions: overlap with the next block's 16 read directions {dr['overlap_readsvd16']:.2f}, with all later reads' 16 {dr['overlap_allread16']:.2f}, with the unembedding's 16 {dr['overlap_unemb16']:.2f}, with the strong write's 16 {dr['overlap_strong16']:.2f} and the whole MLP write's 16 {dr['overlap_full16']:.2f}; first direction's cosine with the stream mean {dr['pc1_mean_dir']:.2f}, its share in the 4 massive coordinates {dr['massive_share_pc1']:.2f} (the 16: {dr['massive_share_16']:.2f}); energy in the 16 {dr['energy_top16']:.2f}; verdicts {V}")
save(TAG, res, summ)
