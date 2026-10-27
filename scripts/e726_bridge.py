"""e726 (session 166): the bridge to the sparsified MLP. Shabalin and Piotrowski (2026) replace a late MLP by its top-k neurons per token
and find that a trained d-by-d linear map on the pruned output restores the model's next-token distribution (KL) while leaving the
reconstruction poor; they call for an attribution of the non-neuron structure that map carries. The record's candidate is the brake's
carrier: the weak write is low-rank and mostly linear in the block's normalised input (e693, e696). Here, at one block of Qwen3-4B, the
MLP output is replaced by its top-64 neurons per token (by magnitude, with their signs) plus one of: nothing; the closed-form rank-16
map of the weak write from the normalised block input, fitted on held-out text by reduced-rank regression (the field map); the same map
restricted to a random rank-16 output subspace; the true weak write projected on the field map's subspace (the subspace's ceiling); a
trained rank-16 map (same shape, trained on the KL); and a trained d-by-d map on the pruned output (their correction). Scored by the
downstream KL from the unedited model over all positions, with the corrected readout, and by the fraction of the MLP output's variance
left unexplained. Arguments: --model qwen3_4b --block B [--smoke]. Pre-registered in e726_e727_prereg.json. The log-prob targets (evaluation and training windows) are kept in CPU memory and moved
to the GPU per chunk: on the GPU they need 37 GB at the full window counts (the first full run ran out of memory); the measurement is unchanged."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen3_4b"); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e726_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name, dtype=torch.bfloat16); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else 48; NWH = 16 if SMOKE else 96; NTR = 16 if SMOKE else 192; STEPS = 10 if SMOKE else 200; K = 64; R = 16; CH = 4
ids_all = wikitext_ids(M, "train"); n3 = len(ids_all) // 3
ids = windows_from(ids_all[:n3], NWIN, T).to(DEV); ids_h = windows_from(ids_all[n3:2 * n3], NWH, T).to(DEV); ids_t = windows_from(ids_all[2 * n3:], NTR, T, seed=726).to(DEV); N = NWIN * (T - 1)
mlpB = mlp(M, B); lnB = ln_mlp(M, B); dp = write_mod(M, B); WD = write_w(M, B)   # [D, DFF]
CAP = {}
h1 = lnB.register_forward_hook(lambda m, a, o: CAP.__setitem__("xt", o)); h2 = dp.register_forward_pre_hook(lambda m, a: CAP.__setitem__("a", a[0]))
def topk_out(a):
    """the MLP output from the k largest activations by magnitude per token, signs kept"""
    idx = a.abs().topk(K, dim=-1).indices; ak = torch.zeros_like(a).scatter_(-1, idx, a.gather(-1, idx)); return ak @ WD.T.to(a.dtype)
def capture(ids_, need_lp):
    XT, A, Y, LP = [], [], [], []
    for s0 in range(0, ids_.shape[0], CH):
        o = M["base"](input_ids=ids_[s0:s0 + CH], use_cache=False)
        if need_lp: LP.append(logprobs_at(M, o.last_hidden_state[:, :-1]).float().cpu())
        XT.append(CAP["xt"][:, :-1].float().reshape(-1, D)); A.append(CAP["a"][:, :-1].reshape(-1, DFF)); Y.append((CAP["a"][:, :-1].reshape(-1, DFF).float() @ WD.T))
    return dict(XT=torch.cat(XT), A=torch.cat(A), Y=torch.cat(Y), LP=torch.cat(LP) if need_lp else None)
E = capture(ids, True); H = capture(ids_h, False); L_(f"block {B} of {NB}: {N} evaluation positions, {H['XT'].shape[0]} held-out, {DFF} neurons, top-{K} kept")
# ---------------- the maps, fitted on held-out text
YKh = topk_out(H["A"]).float(); Wh = H["Y"] - YKh; Xh = H["XT"]
G = Xh.T @ Xh + 1e-3 * Xh.shape[0] * torch.eye(D, device=DEV); WF = torch.linalg.solve(G, Xh.T @ Wh)   # [D, D] full-rank least squares: weak write from the normalised input
U16 = torch.linalg.eigh((Xh @ WF).T @ (Xh @ WF))[1][:, -R:]; WFR = WF @ U16 @ U16.T   # reduced-rank regression: the fitted weak write's top-R output directions
gen = torch.Generator(device=DEV).manual_seed(726 + B); UR = torch.linalg.qr(torch.randn(D, R, device=DEV, generator=gen))[0]; WRR = WF @ UR @ UR.T
fvu = lambda yhat, y: float((y - yhat).pow(2).sum() / (y - y.mean(0)).pow(2).sum())
L_(f"held-out: weak write is {float(Wh.pow(2).sum() / H['Y'].pow(2).sum()):.3f} of the MLP output's energy; the full map explains {1 - fvu(Xh @ WF, Wh):.3f} of its variance, rank-{R} {1 - fvu(Xh @ WFR, Wh):.3f}, random rank-{R} {1 - fvu(Xh @ WRR, Wh):.3f}, the projection on its subspace {1 - fvu(Wh @ U16 @ U16.T, Wh):.3f}")
del H
# ---------------- the trained maps (on the KL, through the rest of the model)
LPT = capture(ids_t, True)["LP"].reshape(NTR, T - 1, -1)
PA = torch.zeros(D, R, device=DEV, requires_grad=True); PB = torch.zeros(R, D, device=DEV, requires_grad=True); torch.nn.init.normal_(PA, std=0.02); torch.nn.init.normal_(PB, std=0.02)
WREC = torch.eye(D, device=DEV, requires_grad=True)
MODE = {"name": "full"}
def mlp_hook(m, a, o):
    nm = MODE["name"]
    if nm == "full": return None
    a_ = CAP["a"]; xt = CAP["xt"].float(); yk = topk_out(a_).float()
    if nm == "topk": y = yk
    elif nm == "field": y = yk + xt @ WFR
    elif nm == "random": y = yk + xt @ WRR
    elif nm == "proj": y = yk + (o.float() - yk) @ U16 @ U16.T
    elif nm == "train16": y = yk + (xt @ PA) @ PB
    elif nm == "trainfull": y = yk @ WREC
    elif nm == "zero": y = torch.zeros_like(yk)
    elif nm == "mean": y = MODE["ymean"].expand_as(yk)
    return y.to(o.dtype)
h3 = mlpB.register_forward_hook(mlp_hook); MODE["ymean"] = E["Y"].mean(0)
def train(params, nm, lr):
    opt = torch.optim.Adam(params, lr=lr); MODE["name"] = nm; hist = []
    for step in range(STEPS):
        w = (step * CH) % NTR; x = ids_t[w:w + CH]
        with torch.enable_grad():
            o = M["base"](input_ids=x, use_cache=False); lp = logprobs_at(M, o.last_hidden_state[:, :-1]).float(); loss = kl(LPT[w:w + CH].to(DEV), lp).mean()
            if nm == "trainfull": loss = loss + 0.1 * (CAP["a"][:, :-1].float().reshape(-1, DFF) @ WD.T - (topk_out(CAP["a"][:, :-1].reshape(-1, DFF)).float() @ WREC)).pow(2).mean() / E["Y"].var(0).mean()
            opt.zero_grad(); loss.backward(); opt.step(); hist.append(float(loss))
        if step % max(1, STEPS // 5) == 0: L_(f"{nm} step {step}: KL {float(loss):.4f}")
    MODE["name"] = "full"; return hist
H16 = train([PA, PB], "train16", 1e-3); HF = train([WREC], "trainfull", 2e-4)
# ---------------- the scores
def score(nm):
    MODE["name"] = nm; ks, fv = [], []
    for s0 in range(0, NWIN, CH):
        o = M["base"](input_ids=ids[s0:s0 + CH], use_cache=False); lp = logprobs_at(M, o.last_hidden_state[:, :-1]).float(); ks.append(kl(E["LP"].reshape(NWIN, T - 1, -1)[s0:s0 + CH].to(DEV), lp).reshape(-1))
    MODE["name"] = "full"; k = torch.cat(ks); return k
CONDS = ["topk", "field", "random", "proj", "train16", "trainfull", "zero", "mean"]; KLs = {}
for nm in CONDS: k = score(nm); KLs[nm] = k; L_(f"{nm}: KL {float(k.mean()):.4f} (median {float(k.median()):.4f})")
# the variance each reconstruction leaves of the MLP output on the evaluation positions (no forward needed)
with torch.no_grad():
    YK = topk_out(E["A"]).float(); XT = E["XT"]; REC = {"topk": YK, "field": YK + XT @ WFR, "random": YK + XT @ WRR, "proj": YK + (E["Y"] - YK) @ U16 @ U16.T, "train16": YK + (XT @ PA) @ PB, "trainfull": YK @ WREC, "zero": torch.zeros_like(YK), "mean": MODE["ymean"].expand_as(YK)}
    FVU = {nm: fvu(REC[nm], E["Y"]) for nm in CONDS}
KLm = {nm: float(KLs[nm].mean()) for nm in CONDS}; rec = {nm: 1 - KLm[nm] / max(KLm["topk"], 1e-9) for nm in CONDS}
boot = {}
rng = np.random.default_rng(726)
for nm in ("field", "train16", "trainfull", "random", "proj"):
    kt, kn = KLs["topk"].cpu().numpy().reshape(NWIN, -1).mean(1), KLs[nm].cpu().numpy().reshape(NWIN, -1).mean(1); idx = rng.integers(0, NWIN, (2000, NWIN)); bs = 1 - kn[idx].mean(1) / kt[idx].mean(1); boot[nm] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
S_ = dict(k=K, rank=R, positions=N, kl=KLm, recovered=rec, recovered_ci=boot, fvu=FVU, weak_energy_share_heldout=float(Wh.pow(2).sum() / (Wh + YKh).pow(2).sum()), train_curves=dict(train16=H16, trainfull=HF))
V = dict(B1_topk_costs_half_nat=bool(KLm["topk"] > 0.5), B2_field_map_recovers_half=bool(rec["field"] >= 0.5), B3_trained_full_beats_field_map=bool(rec["trainfull"] > rec["field"]), B4_trained_rank16_within_01_of_field=bool(abs(rec["train16"] - rec["field"]) <= 0.1), B5_random_rank16_under_half_of_field=bool(rec["random"] < 0.5 * rec["field"]), B6_kl_fvu_orderings_differ=bool((rec["trainfull"] > rec["field"]) != (FVU["trainfull"] < FVU["field"])))
res = dict(model=M["hf"], block=B, n_blocks=NB, summary=S_, verdicts=V)
summ = (f"[{name} block {B}] top-{K} neurons per token leave {KLm['topk']:.3f} nats of KL (mean ablation {KLm['mean']:.3f}, zero {KLm['zero']:.3f}); recovered of that: closed-form rank-{R} field map {rec['field']:.3f} [{boot['field'][0]:.2f}, {boot['field'][1]:.2f}], random rank-{R} {rec['random']:.3f}, projection on the field subspace {rec['proj']:.3f}, trained rank-{R} {rec['train16']:.3f}, trained d-by-d map {rec['trainfull']:.3f} [{boot['trainfull'][0]:.2f}, {boot['trainfull'][1]:.2f}]; FVU of the MLP output: top-k {FVU['topk']:.3f}, field {FVU['field']:.3f}, random {FVU['random']:.3f}, trained rank-{R} {FVU['train16']:.3f}, trained full {FVU['trainfull']:.3f}; verdicts {V}")
for h in (h1, h2, h3): h.remove()
save(TAG, res, summ)
