"""e702 (session 158): native field atoms. e693 found the weak activations' summed write per block living in sixteen
data-derived principal directions, e693b that neither the write rows, the read rows nor the unembedding name them, and
e696 that a fitted linear map of the block's normalised input predicts the write. Here the directions are sought in the
block's own linearisation: the Jacobian of the MLP at the mean normalised input, J = W_out diag(phi'(W_in xbar + b)) W_in
(the SwiGLU product rule for gated blocks), and its weak-weighted form with each neuron's slope scaled by the share of
positions at which it is weak. Rungs, all but the strongest quarter of each neuron's activations zeroed at every block:
the true weak write on J's and on weak-J's 16 and 64 left singular directions (as e693b scored subspaces); the first-order
Taylor prediction of the weak write from the weights, the mean state and the weak shares alone, in full and on weak-J's
16 directions; against zero, the oracle sixteen (e693) and the fitted bypass (e696). Then the fitted bypass and the Taylor
prediction, frozen, scored on four Pile domains (GitHub, ArXiv, EuroParl, StackExchange). Arguments: --model M [--smoke].
Pre-registered in e701_e704_prereg.json."""
import argparse
from c11 import *
BF16 = {"qwen7b", "qwen14b", "qwen32b"}
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e702_{name}" + ("_smoke" if SMOKE else ""); M = load(name, dtype=torch.bfloat16 if name in BF16 else torch.float32); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWC = 16 if SMOKE else 128; NWE = 8 if SMOKE else 48; NWH = 16 if SMOKE else 96; NWO = 4 if SMOKE else 16; Q = 0.25; LAM = 1e-2; GATED = M["fam"] == "gated"
CONDS = ["zero", "pca16", "bypass16", "jac16", "jac64", "wjac16", "wjac64", "taylor", "taylor16"]; OOD = ["Github", "ArXiv", "EuroParl", "StackExchange"]; OCONDS = ["zero", "pca16", "bypass16", "taylor"]
ids_all = wikitext_ids(M, "train"); half = len(ids_all) // 2
cal_ids = windows_from(ids_all[:half], NWC, T).to(DEV); ho_ids = windows_from(ids_all[half:], NWH, T).to(DEV); ev_ids = windows_from(wikitext_ids(M, "test"), NWE, T).to(DEV)
def write_mat(b): w = write_mod(M, b).weight; return w.T if M["fam"] == "gpt2" else w
XIN = {}
def xin_hook(b):
    def h(mod, a): XIN[b] = a[0]
    return h
HX = [list(read_mods(M, b).values())[0].register_forward_pre_hook(xin_hook(b)) for b in range(NB)]
# ---------------- calibration: thresholds, the weak share per neuron
NPOS = NWC * T; SIGNED = {b: [] for b in range(NB)}
with Capture(M, list(range(NB)), ["act"]) as C:
    for s0 in range(0, NWC, 8):
        run_base(M, cal_ids[s0:s0 + 8])
        for b in range(NB): SIGNED[b].append(C.cap[("act", b)].reshape(-1, DFF).half().cpu())
TAU = torch.zeros(NB, DFF, device=DEV); PW = torch.zeros(NB, DFF, device=DEV)
for b in range(NB):
    X_ = torch.cat(SIGNED[b]); SIGNED[b] = None
    for c0 in range(0, DFF, 512):
        xs = X_[:, c0:c0 + 512].to(DEV).float(); t = xs.abs().quantile(1 - Q, dim=0); TAU[b, c0:c0 + 512] = t; PW[b, c0:c0 + 512] = (xs.abs() < t[None]).float().mean(0)
    del X_
# ---------------- held-out pass: the weak write's second moment, the regression moments, the mean normalised input
COV = [torch.zeros(D, D, device=DEV) for _ in range(NB)]; SXX = [torch.zeros(D, D, device=DEV) for _ in range(NB)]; SXD = [torch.zeros(D, D, device=DEV) for _ in range(NB)]; SX = [torch.zeros(D, device=DEV) for _ in range(NB)]; SD = [torch.zeros(D, device=DEV) for _ in range(NB)]; n = 0
with Capture(M, list(range(NB)), ["act"]) as C:
    for s0 in range(0, NWH, 8):
        run_base(M, ho_ids[s0:s0 + 8]); n += 8 * T
        for b in range(NB):
            a = C.cap[("act", b)].reshape(-1, DFF); wk = a.abs() < TAU[b].to(a.dtype)[None]; dx = ((a * wk).to(write_mat(b).dtype) @ write_mat(b).T).float(); x = XIN[b].reshape(-1, D).float()
            COV[b] += dx.T @ dx; SXX[b] += x.T @ x; SXD[b] += x.T @ dx; SX[b] += x.sum(0); SD[b] += dx.sum(0)
PROJ = {}; WMAP = {}; XBAR = {}; DBAR = {}; JAC = {}; WK0 = {}; SLOPE = {}; OV = {k: [] for k in ("jac16", "wjac16")}
def act_and_slope(b, x):
    """the block's activations at input x [D] and their derivative with respect to the pre-activations (gated: the two partials)"""
    Rd = read_w(M, b); RB = read_bias(M, b)
    if not GATED:
        z = Rd["h"] @ x + RB["h"]; h = 1e-3; a = M["act"](z); sl = (M["act"](z + h) - M["act"](z - h)) / (2 * h); return a, sl
    g = Rd["g"] @ x + RB["g"]; u = Rd["u"] @ x + RB["u"]; sg = torch.sigmoid(g); silu = g * sg; dsilu = sg * (1 + g * (1 - sg)); return silu * u, (silu, u * dsilu)
for b in range(NB):
    ev, evec = torch.linalg.eigh(COV[b]); PROJ[("pca16", b)] = evec[:, -16:].contiguous()
    xb, db = SX[b] / n, SD[b] / n; Cxx = SXX[b] / n - torch.outer(xb, xb); Cxd = SXD[b] / n - torch.outer(xb, db); lam = LAM * Cxx.diagonal().mean(); WMAP[b] = torch.linalg.solve(Cxx + lam * torch.eye(D, device=DEV), Cxd); XBAR[b] = xb; DBAR[b] = db
    Wo = write_mat(b).float(); Rd = read_w(M, b); a0, sl = act_and_slope(b, xb)
    if not GATED: J = (Wo * sl[None]) @ Rd["h"]; Jw = (Wo * (sl * PW[b])[None]) @ Rd["h"]
    else: J = (Wo * sl[0][None]) @ Rd["u"] + (Wo * sl[1][None]) @ Rd["g"]; Jw = (Wo * (sl[0] * PW[b])[None]) @ Rd["u"] + (Wo * (sl[1] * PW[b])[None]) @ Rd["g"]
    for nm, Jm in (("jac", J), ("wjac", Jw)):
        Uj, Sj, _ = torch.linalg.svd(Jm, full_matrices=False)
        for r in (16, 64): PROJ[(f"{nm}{r}", b)] = Uj[:, :r].contiguous()
    JAC[b] = Jw; WK0[b] = (Wo @ (PW[b] * a0)); SLOPE[b] = sl   # the Taylor prediction: weak(x) ~ W_out diag(p)[a(xbar) + slope (W_in (x - xbar))]
    ov = lambda Pa, Pb: float((Pa.T @ Pb).pow(2).sum() / Pa.shape[1]); OV["jac16"].append(ov(PROJ[("pca16", b)], PROJ[("jac16", b)])); OV["wjac16"].append(ov(PROJ[("pca16", b)], PROJ[("wjac16", b)]))
    del Wo, J, Jw
L_(f"subspaces built on {n} held-out positions: overlap of the weak write's 16 principal directions with the Jacobian's 16 {med(OV['jac16']):.2f}, with the weak-weighted Jacobian's 16 {med(OV['wjac16']):.2f}")
# ---------------- hooks
COND = {}; HOLD = {}; STAT = collections.defaultdict(float)
def pre_hook(b):
    def h(mod, a):
        if not COND: return None
        x = a[0]; wk = x.abs() < TAU[b].to(x.dtype)[None, None]; STAT["removed"] += float((x.float() ** 2)[wk].sum()); STAT["all"] += float((x.float() ** 2).sum())
        if COND["kind"] in ("pca16", "jac16", "jac64", "wjac16", "wjac64"): HOLD[b] = x * wk
        return (torch.where(wk, torch.zeros_like(x), x),)
    return h
def fwd_hook(b):
    def h(mod, a, out):
        if not COND or COND["kind"] == "zero": return None
        kind = COND["kind"]
        if kind in ("pca16", "jac16", "jac64", "wjac16", "wjac64"): P = PROJ[(kind, b)]; dx = (HOLD.pop(b).to(write_mat(b).dtype) @ write_mat(b).T).float(); add = (dx @ P) @ P.T
        elif kind == "bypass16": x = XIN[b].float(); dh = DBAR[b][None, None] + (x - XBAR[b]) @ WMAP[b]; P = PROJ[("pca16", b)]; add = (dh @ P) @ P.T
        else:
            x = XIN[b].float(); dh = WK0[b][None, None] + (x - XBAR[b]) @ JAC[b].T
            if kind == "taylor16": P = PROJ[("wjac16", b)]; dh = (dh @ P) @ P.T
            add = dh
        return out + add.to(out.dtype)
    return h
HS = [write_mod(M, b).register_forward_pre_hook(pre_hook(b)) for b in range(NB)] + [write_mod(M, b).register_forward_hook(fwd_hook(b)) for b in range(NB)]
def evaluate(wins):
    STAT.clear(); LPs = []
    for s0 in range(0, wins.shape[0], 8): o = M["base"](input_ids=wins[s0:s0 + 8], use_cache=False); LPs.append(logprobs_at(M, o.last_hidden_state[:, :-1]))
    return torch.cat(LPs)
def score(wins, conds):
    COND.clear(); LP0 = evaluate(wins); TGT = wins[:, 1:]; LOSS0 = float(-LP0.gather(-1, TGT[..., None]).float().mean()); out = {}
    for kind in conds:
        COND.update(kind=kind); LP1 = evaluate(wins); ks = torch.stack([kl(LP0[w].float(), LP1[w].float()) for w in range(wins.shape[0])]); out[kind] = dict(kl_mean=float(ks.mean()), loss_delta=float(-LP1.gather(-1, TGT[..., None]).float().mean()) - LOSS0); del LP1
    COND.clear(); return out, LOSS0
res = dict(model=M["hf"], n_blocks=NB, DFF=DFF, keep_quantile=Q, overlap_pca16_jac16=med(OV["jac16"]), overlap_pca16_wjac16=med(OV["wjac16"]), overlaps_per_block=OV)
res["conditions"], res["base_loss"] = score(ev_ids, CONDS); c = {k: v["kl_mean"] for k, v in res["conditions"].items()}; rec = lambda k: 1 - c[k] / c["zero"]
L_("wikitext: " + ", ".join(f"{k} {c[k]:.3f}" for k in CONDS))
import datasets
pile = datasets.load_dataset("NeelNanda/pile-10k", split="train"); res["ood"] = {}
for dom in OOD:
    docs = [ex["text"] for ex in pile if ex["meta"]["pile_set_name"] == dom][:60]; toks = M["tok"]("\n".join(docs), return_tensors="pt", add_special_tokens=False).input_ids[0]
    wins = windows_from(toks, NWO, T).to(DEV); sc, l0 = score(wins, OCONDS); res["ood"][dom] = dict(conditions=sc, base_loss=l0, n_tokens=int(toks.numel()))
    L_(f"{dom}: " + ", ".join(f"{k} {sc[k]['kl_mean']:.3f}" for k in OCONDS))
COND.clear(); [h.remove() for h in HS + HX]
orec = lambda dom, k: 1 - res["ood"][dom]["conditions"][k]["kl_mean"] / res["ood"][dom]["conditions"]["zero"]["kl_mean"]
V = dict(J1_wjac16_overlaps_pca16_above_03=bool(res["overlap_pca16_wjac16"] > 0.3), J2_wjac64_matches_pca16=bool(rec("wjac64") >= 0.8 * rec("pca16")), J2b_wjac16_halves_zero=bool(c["wjac16"] < 0.5 * c["zero"]), J3_taylor_matches_bypass16=bool(rec("taylor") >= 0.8 * rec("bypass16")), J3b_taylor_halves_zero=bool(c["taylor"] < 0.5 * c["zero"]),
         J4_bypass16_holds_on_code=bool(orec("Github", "bypass16") >= 0.7 * rec("bypass16")), J4b_bypass16_holds_on_europarl=bool(orec("EuroParl", "bypass16") >= 0.7 * rec("bypass16")), J5_taylor_half_of_bypass_ood=bool(med([orec(d, "taylor") for d in OOD]) >= 0.5 * med([orec(d, "bypass16") for d in OOD])))
res["verdicts"] = V
summ = (f"[{name}] all but the strongest quarter zeroed at every block, KL in nats on wikitext-2 test: " + ", ".join(f"{k} {c[k]:.3f}" for k in CONDS) + f"; recovered by the oracle 16 {rec('pca16'):.2f}, the fitted bypass {rec('bypass16'):.2f}, the weak-Jacobian 16/64 {rec('wjac16'):.2f}/{rec('wjac64'):.2f}, the Taylor prediction {rec('taylor'):.2f} (on 16 {rec('taylor16'):.2f}); "
        f"overlap of the weak write's 16 directions with the Jacobian's 16 {res['overlap_pca16_jac16']:.2f}, with the weak-weighted Jacobian's {res['overlap_pca16_wjac16']:.2f}; out of domain, recovered by the fitted bypass / the Taylor prediction: " + ", ".join(f"{d} {orec(d, 'bypass16'):.2f}/{orec(d, 'taylor'):.2f}" for d in OOD) + f"; verdicts {V}")
save(TAG, res, summ)
