"""e705 (session 159): the gated remainder. e702 found the fitted linear bypass transferring across domains at 0.65 to 0.97 of
its recovery in the GELU models and 0.20 to 0.64 in the gated ones, whose weak regime is bilinear at the origin. Here two
closed-form predictors of the weak write's sixteen coordinates are compared with the linear one: a quadratic predictor on
fixed random bilinear features (256 products of two random projections of the normalised input, ridge-solved like the
linear map), and the same two predictors fitted on a mixed Pile stream instead of wikitext; all four are frozen and scored
on wikitext-2 test and on the four held-out Pile domains (GitHub, ArXiv, EuroParl, StackExchange). The field itself is
checked for domain stability: the overlap of the weak write's sixteen directions found on wikitext with those found on the
Pile. Arguments: --model M [--smoke]. Pre-registered in e705_e706_prereg.json."""
import argparse
from c11 import *
BF16 = {"qwen7b", "qwen14b", "qwen32b"}
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e705_{name}" + ("_smoke" if SMOKE else ""); M = load(name, dtype=torch.bfloat16 if name in BF16 else torch.float32); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWC = 16 if SMOKE else 128; NWE = 8 if SMOKE else 48; NWH = 16 if SMOKE else 96; NWO = 4 if SMOKE else 16; Q = 0.25; LAM = 1e-2; GATED = M["fam"] == "gated"
CONDS = ["zero", "pca16", "lin_wiki", "quad_wiki", "lin_pile", "quad_pile"]; OOD = ["Github", "ArXiv", "EuroParl", "StackExchange"]; OCONDS = CONDS; NQF = 256
ids_all = wikitext_ids(M, "train"); half = len(ids_all) // 2
cal_ids = windows_from(ids_all[:half], NWC, T).to(DEV); ho_ids = windows_from(ids_all[half:], NWH, T).to(DEV); ev_ids = windows_from(wikitext_ids(M, "test"), NWE, T).to(DEV)
def write_mat(b): w = write_mod(M, b).weight; return w.T if M["fam"] == "gpt2" else w
XIN = {}
def xin_hook(b):
    def h(mod, a): XIN[b] = a[0]
    return h
HX = [list(read_mods(M, b).values())[0].register_forward_pre_hook(xin_hook(b)) for b in range(NB)]
# ---------------- calibration: thresholds
NPOS = NWC * T; SIGNED = {b: [] for b in range(NB)}
with Capture(M, list(range(NB)), ["act"]) as C:
    for s0 in range(0, NWC, 8):
        run_base(M, cal_ids[s0:s0 + 8])
        for b in range(NB): SIGNED[b].append(C.cap[("act", b)].reshape(-1, DFF).half().cpu())
TAU = torch.zeros(NB, DFF, device=DEV)
for b in range(NB):
    X_ = torch.cat(SIGNED[b]); SIGNED[b] = None
    for c0 in range(0, DFF, 512): TAU[b, c0:c0 + 512] = X_[:, c0:c0 + 512].to(DEV).float().abs().quantile(1 - Q, dim=0)
    del X_
# ---------------- a mixed Pile fitting stream (the four test domains excluded)
import datasets
pile = datasets.load_dataset("NeelNanda/pile-10k", split="train"); docs = [ex["text"] for ex in pile if ex["meta"]["pile_set_name"] not in OOD][:400]
pile_ids_ = M["tok"]("\n".join(docs), return_tensors="pt", add_special_tokens=False).input_ids[0]; pi_ids = windows_from(pile_ids_, NWH, T).to(DEV)
# ---------------- held-out passes: moments of the weak write and of the linear and quadratic features of the normalised input, on wikitext and on the Pile
gen = torch.Generator(device=DEV).manual_seed(705); R1 = torch.randn(D, NQF, device=DEV, generator=gen) / D ** 0.5; R2 = torch.randn(D, NQF, device=DEV, generator=gen) / D ** 0.5; DF = D + NQF
def feats(x): return torch.cat([x, (x @ R1) * (x @ R2)], -1)   # the linear features and 256 random bilinear ones
def moments(wins, nwin):
    COV = [torch.zeros(D, D, device=DEV) for _ in range(NB)]; SFF = [torch.zeros(DF, DF, device=DEV) for _ in range(NB)]; SFD = [torch.zeros(DF, D, device=DEV) for _ in range(NB)]; SF = [torch.zeros(DF, device=DEV) for _ in range(NB)]; SD = [torch.zeros(D, device=DEV) for _ in range(NB)]; n = 0
    with Capture(M, list(range(NB)), ["act"]) as C:
        for s0 in range(0, nwin, 8):
            run_base(M, wins[s0:s0 + 8]); n += 8 * T
            for b in range(NB):
                a = C.cap[("act", b)].reshape(-1, DFF); wk = a.abs() < TAU[b].to(a.dtype)[None]; dx = ((a * wk).to(write_mat(b).dtype) @ write_mat(b).T).float(); f = feats(XIN[b].reshape(-1, D).float())
                COV[b] += dx.T @ dx; SFF[b] += f.T @ f; SFD[b] += f.T @ dx; SF[b] += f.sum(0); SD[b] += dx.sum(0)
    return dict(COV=COV, SFF=SFF, SFD=SFD, SF=SF, SD=SD, n=n)
MW = moments(ho_ids, NWH); MP = moments(pi_ids, NWH); L_(f"moments on {MW['n']} wikitext and {MP['n']} Pile positions")
PROJ = {}; MAPS = {}; OVF = []
for b in range(NB):
    ev, evec = torch.linalg.eigh(MW["COV"][b]); PROJ[("pca16", b)] = evec[:, -16:].contiguous(); evp, evecp = torch.linalg.eigh(MP["COV"][b]); OVF.append(float((evecp[:, -16:].T @ PROJ[("pca16", b)]).pow(2).sum() / 16))
    for nm, Mo in (("wiki", MW), ("pile", MP)):
        n = Mo["n"]; fb, db = Mo["SF"][b] / n, Mo["SD"][b] / n; Cff = Mo["SFF"][b] / n - torch.outer(fb, fb); Cfd = Mo["SFD"][b] / n - torch.outer(fb, db)
        for kind, k in (("lin", D), ("quad", DF)):
            lam = LAM * Cff[:k, :k].diagonal().mean(); W = torch.linalg.solve(Cff[:k, :k] + lam * torch.eye(k, device=DEV), Cfd[:k]); MAPS[(f"{kind}_{nm}", b)] = (W, fb[:k], db)
L_(f"maps solved; the field's sixteen directions on wikitext and on the Pile overlap by {med(OVF):.2f} (median over blocks)")
# ---------------- hooks
COND = {}; HOLD = {}; STAT = collections.defaultdict(float)
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
        kind = COND["kind"]; P = PROJ[("pca16", b)]
        if kind == "pca16": dx = (HOLD.pop(b).to(write_mat(b).dtype) @ write_mat(b).T).float(); add = (dx @ P) @ P.T
        else:
            W, fb, db = MAPS[(kind, b)]; f = feats(XIN[b].float()); f = f[..., :W.shape[0]]; dh = db[None, None] + (f - fb) @ W; add = (dh @ P) @ P.T
        return out + add.to(out.dtype)
    return h
HS = [write_mod(M, b).register_forward_pre_hook(pre_hook(b)) for b in range(NB)] + [write_mod(M, b).register_forward_hook(fwd_hook(b)) for b in range(NB)]
def evaluate(wins):
    LPs = []
    for s0 in range(0, wins.shape[0], 8): o = M["base"](input_ids=wins[s0:s0 + 8], use_cache=False); LPs.append(logprobs_at(M, o.last_hidden_state[:, :-1]))
    return torch.cat(LPs)
def score(wins, conds):
    COND.clear(); LP0 = evaluate(wins); TGT = wins[:, 1:]; LOSS0 = float(-LP0.gather(-1, TGT[..., None]).float().mean()); out = {}
    for kind in conds:
        COND.update(kind=kind); LP1 = evaluate(wins); ks = torch.stack([kl(LP0[w].float(), LP1[w].float()) for w in range(wins.shape[0])]); out[kind] = dict(kl_mean=float(ks.mean()), loss_delta=float(-LP1.gather(-1, TGT[..., None]).float().mean()) - LOSS0); del LP1
    COND.clear(); return out, LOSS0
res = dict(model=M["hf"], n_blocks=NB, DFF=DFF, keep_quantile=Q, gated=GATED, field_overlap_wiki_pile=med(OVF), field_overlap_per_block=OVF, quad_features=NQF)
res["conditions"], res["base_loss"] = score(ev_ids, CONDS); c = {k: v["kl_mean"] for k, v in res["conditions"].items()}; rec = lambda k: 1 - c[k] / c["zero"]
L_("wikitext: " + ", ".join(f"{k} {c[k]:.3f}" for k in CONDS)); res["ood"] = {}
for dom in OOD:
    docs = [ex["text"] for ex in pile if ex["meta"]["pile_set_name"] == dom][:60]; toks = M["tok"]("\n".join(docs), return_tensors="pt", add_special_tokens=False).input_ids[0]
    wins = windows_from(toks, NWO, T).to(DEV); sc, l0 = score(wins, OCONDS); res["ood"][dom] = dict(conditions=sc, base_loss=l0, n_tokens=int(toks.numel())); L_(f"{dom}: " + ", ".join(f"{k} {sc[k]['kl_mean']:.3f}" for k in OCONDS))
COND.clear(); [h.remove() for h in HS + HX]
orec = lambda dom, k: 1 - res["ood"][dom]["conditions"][k]["kl_mean"] / res["ood"][dom]["conditions"]["zero"]["kl_mean"]; mo = lambda k: med([orec(d, k) for d in OOD])
V = dict(G1_quad_beats_linear_by_01=bool(rec("quad_wiki") >= rec("lin_wiki") + 0.1), G1b_quad_matches_oracle=bool(rec("quad_wiki") >= 0.9 * rec("pca16")), G2_pile_linear_transfers=bool(mo("lin_pile") >= 0.7 * rec("lin_pile")), G2b_wiki_linear_transfers=bool(mo("lin_wiki") >= 0.7 * rec("lin_wiki")),
         G3_pile_quad_ood_above_06=bool(mo("quad_pile") >= 0.6), G4_field_domain_stable=bool(med(OVF) > 0.7), G5_quad_ood_beats_linear_ood_by_01=bool(mo("quad_wiki") >= mo("lin_wiki") + 0.1))
res["verdicts"] = V
summ = (f"[{name}] all but the strongest quarter zeroed at every block, KL in nats on wikitext-2 test: " + ", ".join(f"{k} {c[k]:.3f}" for k in CONDS) + f"; recovered by the oracle 16 {rec('pca16'):.2f}, the linear bypass {rec('lin_wiki'):.2f}, the quadratic bypass {rec('quad_wiki'):.2f}, the Pile-fitted linear {rec('lin_pile'):.2f} and quadratic {rec('quad_pile'):.2f}; "
        f"out of domain (median over GitHub, ArXiv, EuroParl, StackExchange) the linear bypass recovers {mo('lin_wiki'):.2f}, the quadratic {mo('quad_wiki'):.2f}, the Pile-fitted linear {mo('lin_pile'):.2f} and quadratic {mo('quad_pile'):.2f}; the field's sixteen directions on wikitext and the Pile overlap by {med(OVF):.2f}; verdicts {V}")
save(TAG, res, summ)
