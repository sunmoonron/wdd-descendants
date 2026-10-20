"""e710 (session 160): branches, not polynomials. e705 found the gated models' remainder after the linear bypass not to be a
random quadratic of the block's input. Here the remainder is read as control flow: positions are clustered by the block's
open set (the sign pattern of its gate pre-activations, or of its pre-activations in a GELU model, through a fixed random
projection), one closed-form linear bypass is fitted per branch, and the mixture is scored like e705's single map and
oracle on wikitext-2 test and the four held-out Pile domains; a mixture over clusters of the normalised input itself,
with the same number of branches, is the control that separates a branch from a local linearisation. Arguments:
--model M [--smoke]. Pre-registered in e707_e712_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e710_{name}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWC = 16 if SMOKE else 128; NWE = 8 if SMOKE else 48; NWH = 16 if SMOKE else 768; NWO = 4 if SMOKE else 16; NKM = 4000 if SMOKE else 24000; Q = 0.25; LAM = 1e-2; GATED = M["fam"] == "gated"; PK = "g" if GATED else "h"
MIX = {"mix4_open": ("open", 4), "mix16_open": ("open", 16), "mix16_state": ("state", 16)}; CONDS = ["zero", "pca16", "lin"] + list(MIX); OOD = ["Github", "ArXiv", "EuroParl", "StackExchange"]
ids_all = wikitext_ids(M, "train"); half = len(ids_all) // 2
cal_ids = windows_from(ids_all[:half], NWC, T).to(DEV); ho_ids = windows_from(ids_all[half:], NWH, T).to(DEV); ev_ids = windows_from(wikitext_ids(M, "test"), NWE, T).to(DEV)
def write_mat(b): w = write_mod(M, b).weight; return w.T if M["fam"] == "gpt2" else w
XIN = {}
def xin_hook(b):
    def h(mod, a): XIN[b] = a[0]
    return h
HX = [list(read_mods(M, b).values())[0].register_forward_pre_hook(xin_hook(b)) for b in range(NB)]
gen = torch.Generator(device=DEV).manual_seed(710); RP = torch.randn(NB, DFF, 64, device=DEV, generator=gen) / DFF ** 0.5; RPX = torch.randn(D, 64, device=DEV, generator=gen) / D ** 0.5
# ---------------- calibration: thresholds
SIGNED = {b: [] for b in range(NB)}
with Capture(M, list(range(NB)), ["act"]) as C:
    for s0 in range(0, NWC, 8):
        run_base(M, cal_ids[s0:s0 + 8])
        for b in range(NB): SIGNED[b].append(C.cap[("act", b)].reshape(-1, DFF).half().cpu())
TAU = torch.zeros(NB, DFF, device=DEV)
for b in range(NB):
    X_ = torch.cat(SIGNED[b]); SIGNED[b] = None
    for c0 in range(0, DFF, 512): TAU[b, c0:c0 + 512] = X_[:, c0:c0 + 512].to(DEV).float().abs().quantile(1 - Q, dim=0)
    del X_
# ---------------- pass 1 on the fitting windows: the branch features (open-set bits and the state, each through a random projection), then k-means per block
def feats_of(C, b): return dict(open=((C.cap[(f"pre_{PK}", b)].reshape(-1, DFF) > 0).float() @ RP[b]), state=(XIN[b].reshape(-1, D).float() @ RPX))
Z = {("open", b): [] for b in range(NB)}; Z.update({("state", b): [] for b in range(NB)})
with Capture(M, list(range(NB)), ["pre"]) as C:
    for s0 in range(0, NWH, 8):
        run_base(M, ho_ids[s0:s0 + 8])
        for b in range(NB):
            f = feats_of(C, b)
            for kind in ("open", "state"): Z[(kind, b)].append(f[kind].half())
Z = {k: torch.cat(v) for k, v in Z.items()}; NPH = Z[("open", 0)].shape[0]
def kmeans_e(X, K, iters=25, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed); Cc = X[torch.randperm(X.shape[0], generator=g, device=DEV)[:K]].clone()
    for _ in range(iters):
        lab = torch.cdist(X, Cc).argmin(1); Cc = torch.stack([X[lab == c].mean(0) if (lab == c).any() else Cc[c] for c in range(K)])
    return Cc
CENT = {}; sub = torch.randperm(NPH, generator=torch.Generator(device=DEV).manual_seed(1), device=DEV)[:NKM]
for b in range(NB):
    for cond, (kind, K) in MIX.items(): CENT[(cond, b)] = kmeans_e(Z[(kind, b)][sub].float(), K, seed=b)
assign = lambda z, Cc: torch.cdist(z.float(), Cc).argmin(1)
L_(f"branches: k-means on {NKM} of {NPH} fitting positions per block, {', '.join(f'{c} K={K}' for c, (_, K) in MIX.items())}")
# ---------------- pass 2: moments of the weak write and of the normalised input, globally and per branch
def zero_m(K): return dict(SFF=torch.zeros(K, D, D, device=DEV), SFD=torch.zeros(K, D, D, device=DEV), SF=torch.zeros(K, D, device=DEV), SD=torch.zeros(K, D, device=DEV), n=torch.zeros(K, device=DEV))
MO = {("lin", b): zero_m(1) for b in range(NB)}; MO.update({(c, b): zero_m(K) for c, (_, K) in MIX.items() for b in range(NB)}); COV = [torch.zeros(D, D, device=DEV) for _ in range(NB)]
def accumulate(Mo, f, dx, lab):
    for c in range(Mo["n"].numel()):
        m = lab == c
        if m.any(): fc, dc = f[m], dx[m]; Mo["SFF"][c] += fc.T @ fc; Mo["SFD"][c] += fc.T @ dc; Mo["SF"][c] += fc.sum(0); Mo["SD"][c] += dc.sum(0); Mo["n"][c] += m.sum()
with Capture(M, list(range(NB)), ["act", "pre"]) as C:
    for s0 in range(0, NWH, 8):
        run_base(M, ho_ids[s0:s0 + 8])
        for b in range(NB):
            a = C.cap[("act", b)].reshape(-1, DFF); wk = a.abs() < TAU[b].to(a.dtype)[None]; dx = ((a * wk).to(write_mat(b).dtype) @ write_mat(b).T).float(); f = XIN[b].reshape(-1, D).float(); fz = feats_of(C, b); COV[b] += dx.T @ dx
            accumulate(MO[("lin", b)], f, dx, torch.zeros(f.shape[0], dtype=torch.long, device=DEV))
            for cond, (kind, K) in MIX.items(): accumulate(MO[(cond, b)], f, dx, assign(fz[kind], CENT[(cond, b)]))
PROJ = {b: torch.linalg.eigh(COV[b])[1][:, -16:].contiguous() for b in range(NB)}; MAPS = {}
for b in range(NB):
    for cond in ["lin"] + list(MIX):
        Mo = MO[(cond, b)]; K = Mo["n"].numel(); Ws, fbs, dbs = [], [], []
        for c in range(K):
            n = float(Mo["n"][c]); fb, db = Mo["SF"][c] / max(n, 1), Mo["SD"][c] / max(n, 1); Cff = Mo["SFF"][c] / max(n, 1) - torch.outer(fb, fb); Cfd = Mo["SFD"][c] / max(n, 1) - torch.outer(fb, db)
            lam = LAM * Cff.diagonal().mean() * (1 + D / max(n, 1)); Ws.append(torch.linalg.solve(Cff + lam * torch.eye(D, device=DEV), Cfd)); fbs.append(fb); dbs.append(db)
        MAPS[(cond, b)] = (torch.stack(Ws), torch.stack(fbs), torch.stack(dbs)); MO[(cond, b)] = None   # the moments freed as each map is solved
del MO; torch.cuda.empty_cache(); L_("maps solved")
# ---------------- evaluation hooks
COND = {}; HOLD = {}; CAPE = {}
def pre_hook(b):
    def h(mod, a):
        if not COND: return None
        x = a[0]; wk = x.abs() < TAU[b].to(x.dtype)[None, None]
        if COND["kind"] == "pca16": HOLD[b] = x * wk
        return (torch.where(wk, torch.zeros_like(x), x),)
    return h
def fwd_hook(b):
    def h(mod, a, out):
        if not COND or COND["kind"] == "zero": return None
        kind = COND["kind"]; P = PROJ[b]
        if kind == "pca16": dx = (HOLD.pop(b).to(write_mat(b).dtype) @ write_mat(b).T).float(); add = (dx @ P) @ P.T; return out + add.to(out.dtype)
        Ws, fbs, dbs = MAPS[(kind, b)]; f = XIN[b].float(); sh = f.shape; f = f.reshape(-1, D)
        if kind == "lin": lab = torch.zeros(f.shape[0], dtype=torch.long, device=DEV)
        else: fz = feats_of(CAPE["C"], b); lab = assign(fz[MIX[kind][0]], CENT[(kind, b)])
        dh = torch.zeros_like(f)
        for c in range(Ws.shape[0]):
            m = lab == c
            if m.any(): dh[m] = dbs[c] + (f[m] - fbs[c]) @ Ws[c]
        add = ((dh @ P) @ P.T).reshape(sh); return out + add.to(out.dtype)
    return h
HS = [write_mod(M, b).register_forward_pre_hook(pre_hook(b)) for b in range(NB)] + [write_mod(M, b).register_forward_hook(fwd_hook(b)) for b in range(NB)]
def evaluate(wins):
    LPs = []
    with Capture(M, list(range(NB)), ["pre"]) as C:
        CAPE["C"] = C
        for s0 in range(0, wins.shape[0], 8): o = M["base"](input_ids=wins[s0:s0 + 8], use_cache=False); LPs.append(logprobs_at(M, o.last_hidden_state[:, :-1]))
    return torch.cat(LPs)
def score(wins, conds):
    COND.clear(); LP0 = evaluate(wins); TGT = wins[:, 1:]; LOSS0 = float(-LP0.gather(-1, TGT[..., None]).float().mean()); out = {}
    for kind in conds:
        COND.update(kind=kind); LP1 = evaluate(wins); ks = torch.stack([kl(LP0[w].float(), LP1[w].float()) for w in range(wins.shape[0])]); out[kind] = dict(kl_mean=float(ks.mean()), loss_delta=float(-LP1.gather(-1, TGT[..., None]).float().mean()) - LOSS0); del LP1
    COND.clear(); return out, LOSS0
res = dict(model=M["hf"], n_blocks=NB, DFF=DFF, keep_quantile=Q, gated=GATED, branches={c: K for c, (_, K) in MIX.items()}, fitting_positions=NPH)
res["conditions"], res["base_loss"] = score(ev_ids, CONDS); c = {k: v["kl_mean"] for k, v in res["conditions"].items()}; rec = lambda k: 1 - c[k] / c["zero"]
L_("wikitext: " + ", ".join(f"{k} {c[k]:.3f}" for k in CONDS)); res["ood"] = {}
import datasets
pile = datasets.load_dataset("NeelNanda/pile-10k", split="train")
for dom in OOD:
    docs = [ex["text"] for ex in pile if ex["meta"]["pile_set_name"] == dom][:60]; toks = M["tok"]("\n".join(docs), return_tensors="pt", add_special_tokens=False).input_ids[0]
    wins = windows_from(toks, NWO, T).to(DEV); sc, l0 = score(wins, CONDS); res["ood"][dom] = dict(conditions=sc, base_loss=l0, n_tokens=int(toks.numel())); L_(f"{dom}: " + ", ".join(f"{k} {sc[k]['kl_mean']:.3f}" for k in CONDS))
COND.clear(); [h.remove() for h in HS + HX]
orec = lambda dom, k: 1 - res["ood"][dom]["conditions"][k]["kl_mean"] / res["ood"][dom]["conditions"]["zero"]["kl_mean"]; mo = lambda k: med([orec(d, k) for d in OOD])
V = dict(B1_mix16_closes_half_gap=bool(rec("mix16_open") >= rec("lin") + 0.5 * (rec("pca16") - rec("lin"))), B2_mix16_adds_under_005=bool(rec("mix16_open") < rec("lin") + 0.05), B3_open_beats_state_by_005=bool(rec("mix16_open") >= rec("mix16_state") + 0.05),
         B4_mix16_transfers=bool(mo("mix16_open") >= 0.7 * rec("mix16_open")), B5_mix16_over_mix4_by_003=bool(rec("mix16_open") >= rec("mix4_open") + 0.03))
res["verdicts"] = V; res["recovered"] = {k: rec(k) for k in CONDS if k != "zero"}; res["recovered_ood"] = {k: mo(k) for k in CONDS if k != "zero"}
summ = (f"[{name}] all but the strongest quarter zeroed at every block, KL in nats on wikitext-2 test: " + ", ".join(f"{k} {c[k]:.3f}" for k in CONDS) + "; recovered: " + ", ".join(f"{k} {rec(k):.2f}" for k in CONDS[1:]) + "; out of domain (median over four Pile domains): " + ", ".join(f"{k} {mo(k):.2f}" for k in CONDS[1:]) + f"; verdicts {V}")
save(TAG, res, summ)
