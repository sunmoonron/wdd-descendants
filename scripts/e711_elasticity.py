"""e711 (session 160): the elasticity test. A unit is real to the model if training defends it, whatever a human can read
off it. A trained model's block-B write rows are perturbed and training is continued on Pile text (the training distribution) with AdamW whose moments
are first primed at zero learning rate (so the restart does not shock the weights), then a low rate with warmup: sixteen strong neurons' write rows rotated by forty-five degrees (rot), sixteen alive but weak
neurons' rows likewise (rotw), every strong row of the block at once, up to 256 (rotall), sixteen strong rows halved (scale), the block's weak-write field swapped with a random
sixteen-dimensional subspace by an orthogonal map on every write row (field), sixteen pairs of neurons swapped whole
(perm, an exact symmetry, the loop's own drift) and nothing (none). Logged along the way: each perturbed row's cosine to
its original and to its perturbed self (the return fraction), the held-out loss recovery, for the field the writers'
restoration against the readers' following, and with --freeze the perturbed rows held fixed so that only migration can
recover the loss. Arguments: --model M --block B --cond C [--freeze] [--steps S] [--lr LR] [--smoke]. Pre-registered in
e707_e712_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="pythia160"); ap.add_argument("--block", type=int, default=6); ap.add_argument("--cond", required=True, choices=["none", "perm", "rot", "rotw", "rotall", "scale", "field"]); ap.add_argument("--freeze", action="store_true")
ap.add_argument("--steps", type=int, default=800); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke; FREEZE = A_.freeze; COND = A_.cond
name = A_.model; B = A_.block; TAG = f"e711_{name}_b{B}_{COND}" + ("_frozen" if FREEZE else "") + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; L_ = lambda s: log(TAG, s)
STEPS = 12 if SMOKE else A_.steps; LR = A_.lr; BS = 32; TL = 256 if SMOKE else 1024; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; NCAL = 32; NEV = 4 if SMOKE else 16; WARM = 50; PRIME = 4 if SMOKE else 30; Q = 0.25; NPERT = 16; THETA = math.pi / 4
LOGS = sorted({s for s in (0, 10, 25, 50, 100, 200, 400, 800, 1600) if s <= STEPS} | {STEPS})
def wt103_ids(M):
    f = f"{CACHED}/wt103_{M['hf'].replace('/', '_')}.pt"
    if os.path.exists(f): return torch.load(f)
    import datasets; ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]; out = []
    for i in range(0, len(ds), 20000): out += M["tok"]("".join(ds[i:i + 20000]), add_special_tokens=False)["input_ids"]
    ids = torch.tensor(out); os.makedirs(CACHED, exist_ok=True); tmp = f + f".{os.getpid()}.tmp"; torch.save(ids, tmp); os.replace(tmp, f); return ids
stream = pile_ids(M, ndocs=10000); ev_ids = windows_from(wikitext_ids(M, "test"), NEV, TL).to(DEV); cal_ids = windows_from(wikitext_ids(M, "train"), NCAL, 256).to(DEV); rng = np.random.default_rng(711); gen = torch.Generator(device=DEV).manual_seed(711 + B)
def evaluate():
    tot = 0.0
    for s0 in range(0, NEV, 4): x = ev_ids[s0:s0 + 4]; tot += float(M["model"](input_ids=x, labels=x).loss) * x.shape[0]
    return tot / NEV
def weak_write_cov():
    cov = torch.zeros(D, D, device=DEV); W = write_w(M, B)
    with Capture(M, [B], ["act"]) as C:
        for s0 in range(0, NCAL, 8):
            run_base(M, cal_ids[s0:s0 + 8]); a = C.cap[("act", B)].reshape(-1, DFF).float(); dw = (a * (a.abs() < TAU)) @ W.T; cov += dw.T @ dw
    return cov
# ---------------- calibration on the unperturbed model
AB = []
with Capture(M, [B], ["act"]) as C:
    for s0 in range(0, NCAL, 8): run_base(M, cal_ids[s0:s0 + 8]); AB.append(C.cap[("act", B)].reshape(-1, DFF).float())
ab = torch.cat(AB); TAU = torch.cat([ab[:, c0:c0 + 512].abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)]); mxa = ab.abs().max(0).values; mx = ab.max(0).values; a100 = ab.topk(100, dim=0).values[-1]
GATED = M["fam"] == "gated"; floor_act = 1.0 if not GATED else float(mx.quantile(0.1)); cand = torch.nonzero((mx > floor_act) & (a100 >= 0.25 * mx))[:, 0].tolist(); STRONG = sorted(random.Random(711 + B).sample(cand, NPERT))
lo, hi = mxa.quantile(0.1), mxa.quantile(0.3); wcand = [j for j in torch.nonzero((mxa > lo) & (mxa < hi))[:, 0].tolist() if j not in STRONG]; WEAKN = sorted(random.Random(712 + B).sample(wcand, NPERT))
F = torch.linalg.eigh(weak_write_cov())[1][:, -16:].contiguous(); G0 = torch.randn(D, 16, device=DEV, generator=gen); G0 -= F @ (F.T @ G0); G = torch.linalg.qr(G0)[0]
loss_base = evaluate(); W0 = write_w(M, B).clone(); L_(f"base loss {loss_base:.4f}; strong {STRONG[:4]}..., weak {WEAKN[:4]}...; field found")
# ---------------- the perturbation
Wp = W0.clone(); PERT = []
ALLS = sorted(random.Random(713 + B).sample(cand, min(256, len(cand))))   # every strong candidate (capped at 256): the collective rotation
if COND in ("rot", "rotw", "rotall"):
    PERT = STRONG if COND == "rot" else (WEAKN if COND == "rotw" else ALLS)
    for n in PERT: w = W0[:, n]; u = torch.randn(D, device=DEV, generator=gen); u -= (u @ w) * w / w.pow(2).sum(); u /= u.norm(); Wp[:, n] = math.cos(THETA) * w + math.sin(THETA) * w.norm() * u
elif COND == "scale": PERT = STRONG; Wp[:, PERT] *= 0.5
elif COND == "field": PERT = list(range(DFF)); Rm = torch.eye(D, device=DEV) - F @ F.T - G @ G.T + G @ F.T + F @ G.T; Wp = Rm @ W0
elif COND == "perm":
    PERT = STRONG; pairs = [(STRONG[i], STRONG[i + NPERT // 2]) for i in range(NPERT // 2)]
    for i, j in pairs:
        Wp[:, [i, j]] = W0[:, [j, i]]
        for mod in read_mods(M, B).values():
            wt = mod.weight.data
            if M["fam"] == "gpt2": wt[:, [i, j]] = wt[:, [j, i]].clone()
            else: wt[[i, j]] = wt[[j, i]].clone()
            if getattr(mod, "bias", None) is not None: mod.bias.data[[i, j]] = mod.bias.data[[j, i]].clone()
wm = write_mod(M, B).weight
with torch.no_grad(): wm.data.copy_(Wp.T if M["fam"] == "gpt2" else Wp)
loss_pert = evaluate(); COSP = torch.nn.functional.cosine_similarity(Wp[:, PERT], W0[:, PERT], dim=0) if PERT else torch.zeros(0); L_(f"{COND}: loss {loss_base:.4f} -> {loss_pert:.4f}; perturbed columns {len(PERT)}")
# ---------------- readers of the field: every read matrix after block B and the unembedding
def readers():
    out = []
    for b in range(B + 1, NB):
        out += [w for w in read_w(M, b).values()]; am = attn_mod(M, b)
        for nm in ("query_key_value", "q_proj", "k_proj", "v_proj"):
            if hasattr(am, nm): out.append(getattr(am, nm).weight.detach().float())
        if hasattr(am, "c_attn"): out.append(am.c_attn.weight.detach().float().T)
    out.append(M["model"].get_output_embeddings().weight.detach().float()); return out
def reader_energy(): R = readers(); return float(sum((r @ F).pow(2).sum() for r in R)), float(sum((r @ G).pow(2).sum() for r in R))
EF0, EG0 = reader_energy()
# ---------------- training
MASK_ON = [True]; params = [p for p in M["model"].parameters()]
for p in params: p.requires_grad_(True)
if FREEZE and PERT:
    mask = torch.ones_like(wm)
    if M["fam"] == "gpt2": mask[PERT] = 0
    else: mask[:, PERT] = 0
    wm.register_hook(lambda g: g * mask if MASK_ON[0] else g)
opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0); UNP = torch.tensor(random.Random(5).sample([j for j in range(DFF) if j not in PERT] or list(range(DFF)), 64), device=DEV)
def snapshot(step, loss_t):
    W = write_w(M, B); r = dict(step=step, loss=loss_t, recovery=(loss_pert - loss_t) / (loss_pert - loss_base) if abs(loss_pert - loss_base) > 0.01 else None, loss_change=loss_t - loss_base, drift_cos=float(torch.nn.functional.cosine_similarity(W[:, UNP], W0[:, UNP], dim=0).mean()))
    if PERT and COND != "field":
        co = torch.nn.functional.cosine_similarity(W[:, PERT], W0[:, PERT], dim=0); cp = torch.nn.functional.cosine_similarity(W[:, PERT], Wp[:, PERT], dim=0); nr = W[:, PERT].norm(dim=0) / W0[:, PERT].norm(dim=0)
        r.update(cos_orig=float(co.mean()), cos_pert=float(cp.mean()), norm_ratio=float(nr.mean()))
        if COND in ("rot", "rotw", "rotall"): r["return_fraction"] = float(((co - COSP) / (1 - COSP)).mean())
        if COND == "scale": r["norm_recovery"] = float(((nr - 0.5) / 0.5).mean())
    if COND == "field":
        CF, CF0, CFp = F.T @ W, F.T @ W0, F.T @ Wp; cs = lambda A_, B_: float((A_ * B_).sum() / (A_.norm() * B_.norm()).clamp_min(1e-12)); c0, cp = cs(CF, CF0), cs(CFp, CF0); r.update(writer_cos_orig=c0, writer_cos_pert=cs(CF, CFp), writer_restore=(c0 - cp) / (1 - cp))
        EF, EG = reader_energy(); r.update(reader_energy_F=EF, reader_energy_G=EG, readers_follow=(EG - EG0) / (EF0 - EG0) if EF0 != EG0 else None)
        Fn = torch.linalg.eigh(weak_write_cov())[1][:, -16:]; r.update(field_overlap_orig=float((Fn.T @ F).pow(2).sum() / 16), field_overlap_swapped=float((Fn.T @ G).pow(2).sum() / 16))
    return r
NG = 2 if SMOKE else 24; grng = np.random.default_rng(7110)
def restoring_gradient():
    """the expected gradient of the loss at the current weights over NG batches (no step), read on block B's write rows: for each perturbed row the cosine of the descent direction with the way back to the original row, its gradient norm against the unperturbed reference rows', and for the field the same on the whole matrix and on the swapped plane"""
    MASK_ON[0] = False; opt.zero_grad(set_to_none=True); Gs = torch.zeros_like(wm); MB = 8 if D <= 768 else 4
    for i in range(NG):
        starts = grng.integers(0, len(stream) - TL - 1, BS); x = torch.stack([stream[s:s + TL] for s in starts]).to(DEV)
        with torch.enable_grad():
            for m0 in range(0, BS, MB): l_ = M["model"](input_ids=x[m0:m0 + MB], labels=x[m0:m0 + MB]).loss * (MB / BS); l_.backward()
        Gs += wm.grad.detach(); opt.zero_grad(set_to_none=True)
    MASK_ON[0] = True; G = (Gs / NG); G = G.T if M["fam"] == "gpt2" else G   # [D, DFF], column n = the gradient on neuron n's write row
    W = write_w(M, B); cs = lambda A_, B_: float((A_ * B_).sum() / (A_.norm() * B_.norm()).clamp_min(1e-12)); out = dict(ref_grad_norm=float(G[:, UNP].norm(dim=0).median()))
    if PERT and COND != "field":
        back = W0[:, PERT] - W[:, PERT]; gn = -G[:, PERT]; cos_back = torch.nn.functional.cosine_similarity(gn, back, dim=0) if COND != "none" else torch.zeros(len(PERT), device=DEV)
        out.update(restoring_cos=float(cos_back.mean()), restoring_cos_median=float(cos_back.median()), pert_grad_norm=float(G[:, PERT].norm(dim=0).median()), force_ratio=float(G[:, PERT].norm(dim=0).median() / max(out["ref_grad_norm"], 1e-12)))
        if COND == "scale": out["norm_cos"] = float(torch.nn.functional.cosine_similarity(gn, W0[:, PERT], dim=0).mean())   # the descent direction along the original row: positive grows the norm back
    if COND == "field":
        back = W0 - W; out.update(restoring_cos=cs(-G, back), plane_cos=cs(-torch.cat([F.T @ G, G_.T @ G]), torch.cat([F.T @ back, G_.T @ back])), grad_energy_F=float((F.T @ G).pow(2).sum()), grad_energy_G=float((G_.T @ G).pow(2).sum()), grad_energy_total=float(G.pow(2).sum()))
    return out
G_ = G; GRAD0 = restoring_gradient(); L_(f"restoring gradient at the perturbed point: {GRAD0}")
TRAJ = [snapshot(0, loss_pert)]; L_(f"step 0: {TRAJ[0]}")
for step in range(-PRIME + 1, STEPS + 1):   # steps up to 0 prime the moments at zero learning rate
    for g_ in opt.param_groups: g_["lr"] = 0.0 if step <= 0 else LR * min(1.0, step / WARM)
    starts = rng.integers(0, len(stream) - TL - 1, BS); x = torch.stack([stream[s:s + TL] for s in starts]).to(DEV); MB = 8 if D <= 768 else 4; loss = 0.0
    with torch.enable_grad():
        for m0 in range(0, BS, MB): l_ = M["model"](input_ids=x[m0:m0 + MB], labels=x[m0:m0 + MB]).loss * (MB / BS); l_.backward(); loss += float(l_)   # gradient accumulation over micro-batches
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if step in LOGS and step > 0: TRAJ.append(snapshot(step, evaluate())); L_(f"step {step}: train loss {loss:.4f}; " + ", ".join(f"{k} {v:.3f}" for k, v in TRAJ[-1].items() if isinstance(v, float) and k != "step"))
GRAD1 = restoring_gradient(); L_(f"restoring gradient at the end: {GRAD1}")
last = TRAJ[-1]
res = dict(model=M["hf"], block=B, cond=COND, frozen=FREEZE, steps=STEPS, lr=LR, warmup=WARM, primed_steps=PRIME, batch_tokens=BS * TL, loss_base=loss_base, loss_pert=loss_pert, perturbed=PERT if COND != "field" else "all", strong=STRONG, weak=WEAKN, cos_pert_orig=float(COSP.mean()) if PERT and COND != "field" else None, trajectory=TRAJ, final=last, grad_start=GRAD0, grad_end=GRAD1, corpus="pile-10k, ten thousand documents")
summ = (f"[{name} block {B} {COND}{' frozen' if FREEZE else ''}] loss {loss_base:.4f} -> {loss_pert:.4f} -> {last['loss']:.4f} after {STEPS} steps (recovery {last.get('recovery')}); restoring cosine at the start {GRAD0.get('restoring_cos')}, force ratio {GRAD0.get('force_ratio')}, at the end {GRAD1.get('restoring_cos')}; " + ", ".join(f"{k} {v:.3f}" for k, v in last.items() if isinstance(v, float) and k not in ("loss", "step")))
save(TAG, res, summ)
