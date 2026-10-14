"""e639 (2026-10-12): rotate the writers. The ring's writers (the rows the ledger attribution names for 25, 50, 75 and
90 percent of the MLP ring pattern at the best block) have their write columns' in-plane components rotated by one class
step forward (and, for the 90 percent set, backward) in the ring's plane; the quantile-matched low-attribution rows and
random rows of the same count are rotated the same way as controls. Measured: the class means' angular shift on the fixed
plane at the best block and at the last block (in units of the class step), the fixed-plane radius, the forward task's
answer offsets (0, +1, +2, -1, other), and the general loss on the templates and on neutral text. A rotation preserves
every column's norm, so general damage should be small; if the readout reads the angle on this plane and the writers set
it, the answers move from +1 to +2 with the dose. Standalone. Arguments: model [--smoke]."""
import sys, os, json, math, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = {"pythia160": "EleutherAI/pythia-160m", "pythia410": "EleutherAI/pythia-410m", "pythia14b": "EleutherAI/pythia-1.4b", "gpt2": "gpt2", "gpt2m": "gpt2-medium", "gpt2l": "gpt2-large", "qwen05": "Qwen/Qwen2.5-0.5B", "qwen05i": "Qwen/Qwen2.5-0.5B-Instruct", "olmo1b": "allenai/OLMo-1B-0724-hf", "smollm135": "HuggingFaceTB/SmolLM2-135M", "smollm360": "HuggingFaceTB/SmolLM2-360M", "opt125": "facebook/opt-125m", "opt350": "facebook/opt-350m", "opt13b": "facebook/opt-1.3b"}
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); dev = "cuda:0" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
T_DAY = ["The meeting is scheduled for {X}, so please be on time.", "She told me she would arrive on {X} afternoon.", "We usually go to the market on {X} mornings.", "My favourite day of the week is {X} because of the weather.", "The report is due on {X}, according to the manager.", "They play football every {X} at the park.", "On {X} the shop closes early.", "I will call you back on {X} with the details.", "The concert takes place on {X} evening.", "He was born on a {X} in the spring.", "Classes resume on {X} after the break.", "The flight departs on {X} at noon.", "We celebrated the win last {X} with friends.", "Every {X} she visits her grandmother.", "The deadline was moved to {X} by the client.", "On {X} nights the restaurant is full.", "The doctor's appointment is on {X} at ten.", "Our team meets on {X} to plan the week.", "The package should arrive by {X}, the courier said.", "He forgot that {X} was a holiday this year.", "The library is open late on {X}.", "The train on {X} was delayed by an hour.", "Rehearsals are held on {X} in the hall.", "Payday falls on {X} this month.", "The game was postponed from {X} to the following week.", "She always runs on {X} before work.", "Dinner with the neighbours is on {X}.", "The exhibition opens on {X} and runs for a month.", "I have a dentist visit on {X}, unfortunately.", "The bus schedule changes on {X} for the holiday."]
T_MONTH = ["The conference will be held in {X} next year.", "She moved to the city in {X} after graduating.", "The harvest usually begins in {X} in this region.", "In {X} the days grow noticeably longer.", "The invoice is dated {X} and remains unpaid.", "They were married in {X} under a clear sky.", "Enrolment for the course opens in {X}.", "The river tends to flood in {X} when the snow melts.", "His birthday is in {X}, near the end of the month.", "The festival takes place every {X} downtown.", "Prices typically rise in {X} before the holidays.", "In {X} the committee publishes its annual report.", "The museum reopened in {X} after renovations.", "We planted the garden in {X} this year.", "The exam results are released in {X}.", "The ship set sail in {X} for the southern ports.", "In {X} the company announced record profits.", "The lease expires in {X}, so we must decide soon.", "Migration peaks in {X} along this coast.", "The elections are scheduled for {X}.", "The new season of the show premieres in {X}.", "Tax returns are due in {X} in that country.", "In {X} the lake freezes over completely.", "The trial is expected to begin in {X}.", "Construction should finish by {X}, the builder promised.", "The first frost usually arrives in {X}.", "Their anniversary falls in {X} every year.", "The orchestra tours Europe in {X}.", "In {X} the valley fills with wildflowers.", "The grant application window closes in {X}."]
TASK = {"days": ["Today is {X}. Tomorrow is", "If today is {X}, then tomorrow is", "The day after {X} is", "Yesterday was {X}, so today is"], "months": ["This month is {X}. Next month is", "The month after {X} is", "If this month is {X}, then next month is", "Last month was {X}, so this month is"], "days_rev": ["Today is {X}. Yesterday was", "The day before {X} is", "If today is {X}, then yesterday was"], "months_rev": ["This month is {X}. Last month was", "The month before {X} is", "If this month is {X}, then last month was"], "days_two": ["Today is {X}. The day after tomorrow is", "Two days after {X} is"], "months_two": ["This month is {X}. Two months from now it will be", "Two months after {X} is"]}
NRAND = 2 if SMOKE else 5; FRACS = [0.25, 0.5, 0.75, 0.9]
if SMOKE: T_DAY, T_MONTH = T_DAY[:8], T_MONTH[:8]
tok = AutoTokenizer.from_pretrained(MODELS[name]); model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.float32).to(dev).eval()
cfg = model.config; fam = "neox" if hasattr(model, "gpt_neox") else ("gpt2" if hasattr(model, "transformer") else ("opt" if hasattr(model, "model") and hasattr(model.model, "decoder") else "llama"))
layers = model.gpt_neox.layers if fam == "neox" else (model.transformer.h if fam == "gpt2" else (model.model.decoder.layers if fam == "opt" else model.model.layers)); NB = len(layers); D = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.n_embd
def mlp_out(l): return l.mlp.dense_4h_to_h if fam == "neox" else (l.mlp.c_proj if fam == "gpt2" else (l.fc2 if fam == "opt" else l.mlp.down_proj))
def wrows(b):
    w = mlp_out(layers[b]).weight.detach().float(); return w if fam == "gpt2" else w.T
DFF = wrows(0).shape[0]; W_NP = {}
def wnp(b):
    if b not in W_NP: W_NP[b] = wrows(b).cpu().numpy()
    return W_NP[b]
def unembed_reading(direction, topn=20):
    """tokens promoted by a write direction, read through the final norm's scale and the unembedding"""
    lm = model.get_output_embeddings().weight.detach().float(); d = torch.as_tensor(direction, device=dev, dtype=torch.float32)
    fn = model.gpt_neox.final_layer_norm if fam == "neox" else (model.transformer.ln_f if fam == "gpt2" else (model.model.decoder.final_layer_norm if fam == "opt" else model.model.norm))
    g = fn.weight.detach().float() if getattr(fn, "weight", None) is not None else torch.ones(D, device=dev); logits = lm @ (d * g); top = logits.topk(topn).indices.tolist(); return [tok.decode([t]) for t in top]
def prompts_for(names, templates):
    items = []
    for ci, X in enumerate(names):
        for ti, tpl in enumerate(templates):
            pre, post = tpl.split("{X}"); text = pre + X + post; ids = tok(text, return_tensors="pt").input_ids[0]; ids_px = tok(pre + X, return_tensors="pt").input_ids[0]; items.append(dict(cls=ci, tpl=ti, ids=ids, pos=ids_px.numel() - 1))
    return items
def run(items):
    S, A = [], []; msum = torch.zeros(NB + 1, D, device=dev); mcnt = 0; cap = {}
    hs = [mlp_out(layers[b]).register_forward_pre_hook((lambda b_: lambda m, a: cap.__setitem__(b_, a[0].detach().reshape(-1, a[0].shape[-1])))(b)) for b in range(NB)]
    try:
        for it in items:
            out = model(it["ids"][None].to(dev), output_hidden_states=True); H = torch.stack(out.hidden_states)[:, 0]; msum += H[:, 1:].sum(1); mcnt += H.shape[1] - 1; S.append(H[:, it["pos"]].cpu()); A.append(torch.stack([cap[b][it["pos"]] for b in range(NB)]).cpu())
    finally: [h.remove() for h in hs]
    return torch.stack(S), torch.stack(A), (msum / mcnt).cpu()
def adjacency(angles, k):
    order = np.argsort(angles); pos = np.empty(k, dtype=int); pos[order] = np.arange(k); return int(sum(1 for i in range(k) if (pos[i] - pos[(i + 1) % k]) % k in (1, k - 1)))
def plane_of(means):
    M = means - means.mean(0, keepdims=True); U, s, Vt = np.linalg.svd(M, full_matrices=False); return Vt[0], Vt[1], float((s[:2] ** 2).sum() / (s ** 2).sum())
def means_of(Z, cls, k): return np.stack([Z[cls == c].mean(0) for c in range(k)])
def ring_at(X, cls, k, L, mu):
    Z = (X[:, L] - mu[L]).numpy(); m = means_of(Z, cls, k); u, v, share = plane_of(m); ang = np.arctan2(m @ v, m @ u); return dict(adjacency=adjacency(ang, k), share=share, radius=float(np.sqrt(((m - m.mean(0)) @ np.stack([u, v]).T) ** 2).sum(1).mean()), u=u, v=v, angles=ang, means=m, Z=Z)
def phase_stats(Aall, cls, k, phases_all, ang, ks, rng, sel_mask=None):
    """alignment of the top-k class-selective rows' write phases with the class angle, for several k; selectivity computed on sel_mask prompts (all if None)"""
    m_ = sel_mask if sel_mask is not None else np.ones(len(cls), bool); sd = Aall[m_].std(0) + 1e-6; out = {kk: [] for kk in ks}; rnd = []
    for c in range(k):
        inc = m_ & (cls == c); exc = m_ & (cls != c); sel = (Aall[inc].mean(0) - Aall[exc].mean(0)) / sd; order = np.argsort(-sel); sgn_all = np.sign(Aall[inc].mean(0) + 1e-12)
        for kk in ks:
            top = order[:kk]; ph = phases_all[top] + np.where(sgn_all[top] < 0, np.pi, 0.0); out[kk].append(float(np.cos(ph - ang[c]).mean()))
        r = rng.choice(Aall.shape[1], 20, replace=False); rnd.append(float(np.cos(phases_all[r] - ang[c]).mean()))
    return {kk: float(np.mean(v)) for kk, v in out.items()}, float(np.mean(rnd))
def task_eval(kind, names):
    """accuracy of the next-class prediction among the class tokens, and unrestricted top-1"""
    cand = [tok(" " + X, add_special_tokens=False).input_ids[0] for X in names]; k = len(names); hit, hit_all, margin = [], [], []
    for ci, X in enumerate(names):
        for tpl in TASK[kind]:
            ids = tok(tpl.replace("{X}", X), return_tensors="pt").input_ids.to(dev); lg = model(ids).logits[0, -1].float(); sub = lg[cand]; pred = int(sub.argmax()); tgt = (ci + 1) % k
            hit.append(float(pred == tgt)); hit_all.append(float(int(lg.argmax()) == cand[tgt])); srt = sub.sort(descending=True).values; margin.append(float(sub[tgt] - (srt[1] if pred == tgt else srt[0])))
    return dict(acc_among_classes=float(np.mean(hit)), acc_top1=float(np.mean(hit_all)), margin=float(np.mean(margin)), n=len(hit))
SAVED = {}
def zero_rows(rows, Lb):
    """zero the write columns of the rows (indexed over blocks 0..Lb-1); remembers originals"""
    for r in rows:
        b, j = int(r) // DFF, int(r) % DFF; w = mlp_out(layers[b]).weight
        if (b, j) in SAVED: continue
        if fam == "gpt2": SAVED[(b, j)] = w[j].detach().clone(); w[j] = 0
        else: SAVED[(b, j)] = w[:, j].detach().clone(); w[:, j] = 0
def restore():
    for (b, j), v in SAVED.items():
        w = mlp_out(layers[b]).weight
        if fam == "gpt2": w[j] = v
        else: w[:, j] = v
    SAVED.clear()


NEUTRAL = ["The committee reviewed the proposal and asked for a revised budget before the end of the quarter.", "A narrow path led down to the river, where the water ran fast over smooth grey stones.", "The new library opened with a collection of maps, manuscripts and recordings from the region.", "She adjusted the telescope and waited for the clouds to clear over the eastern hills.", "Prices for copper rose sharply after the mine announced a temporary halt in production.", "The recipe calls for two eggs, a cup of flour and a pinch of salt, mixed slowly.", "Engineers traced the fault to a cracked seal in the pump's secondary housing.", "The orchestra rehearsed the final movement twice before the conductor was satisfied.", "He parked the truck behind the warehouse and carried the crates in through the side door.", "The village council met to discuss the repair of the bridge damaged by the storm."]
def mean_loss(texts):
    tot, cnt = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(dev); lg = model(ids).logits[0, :-1].float(); tot += float(torch.nn.functional.cross_entropy(lg, ids[0, 1:], reduction="sum")); cnt += ids.shape[1] - 1
    return tot / max(cnt, 1)

def task_eval2(kind, names, shift):
    cand = [tok(" " + X, add_special_tokens=False).input_ids[0] for X in names]; k = len(names); hit = []
    for ci, X in enumerate(names):
        for tpl in TASK[kind]:
            ids = tok(tpl.replace("{X}", X), return_tensors="pt").input_ids.to(dev); lg = model(ids).logits[0, -1].float(); sub = lg[cand]; hit.append(float(int(sub.argmax()) == (ci + shift) % k))
    return float(np.mean(hit))
rng = np.random.default_rng(0); R = dict(model=name, NB=NB, D=D, DFF=DFF, fam=fam); PREP = {}
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def rotate_rows(rows, u, v, theta):
    """rotate the in-plane component of each row's write column by theta in the (u, v) plane; remembers originals"""
    ut = torch.as_tensor(u, device=dev, dtype=torch.float32); vt = torch.as_tensor(v, device=dev, dtype=torch.float32); c, s_ = math.cos(theta), math.sin(theta)
    for r in rows:
        b, j = int(r) // DFF, int(r) % DFF; w = mlp_out(layers[b]).weight
        if (b, j) in SAVED: continue
        col = w[j] if fam == "gpt2" else w[:, j]; SAVED[(b, j)] = col.detach().clone(); a_, b_ = float(col @ ut), float(col @ vt)
        new = col + (a_ * c - b_ * s_ - a_) * ut + (a_ * s_ + b_ * c - b_) * vt
        if fam == "gpt2": w[j] = new
        else: w[:, j] = new
def offsets(kind, names):
    """distribution of the forward task's answers among the classes, as offsets from the cue class"""
    cand = [tok(" " + X, add_special_tokens=False).input_ids[0] for X in names]; k = len(names); cnt = {"0": 0, "+1": 0, "+2": 0, "-1": 0, "other": 0}; n = 0
    for ci, X in enumerate(names):
        for tpl in TASK[kind]:
            ids = tok(tpl.replace("{X}", X), return_tensors="pt").input_ids.to(dev); lg = model(ids).logits[0, -1].float(); pred = int(lg[cand].argmax()); off = (pred - ci) % k; n += 1
            cnt["0" if off == 0 else "+1" if off == 1 else "+2" if off == 2 else "-1" if off == k - 1 else "other"] += 1
    return {key: val / n for key, val in cnt.items()}
for kind, names, templates in (("days", DAYS, T_DAY), ("months", MONTHS, T_MONTH)):
    k = len(names); items = prompts_for(names, templates); X, A, mu = run(items); cls = np.array([it["cls"] for it in items]); texts = [tpl.replace("{X}", X_) for X_ in names for tpl in templates]
    per = {L: ring_at(X, cls, k, L, mu) for L in range(NB + 1)}; Lb = max(range(1, NB), key=lambda L: (per[L]["adjacency"], per[L]["share"])); rb = per[Lb]; u, v = rb["u"], rb["v"]; Lf = NB - 1; rf = per[Lf]
    Au = np.concatenate([A[:, b].numpy() * (wnp(b) @ u)[None] for b in range(Lb)], 1); Av = np.concatenate([A[:, b].numpy() * (wnp(b) @ v)[None] for b in range(Lb)], 1); Cu = means_of(Au, cls, k); Cv = means_of(Av, cls, k); Cu -= Cu.mean(0); Cv -= Cv.mean(0); Tu, Tv = Cu.sum(1), Cv.sum(1)
    attr = (Cu * Tu[:, None] + Cv * Tv[:, None]).sum(0) / max(float((Tu ** 2 + Tv ** 2).sum()), 1e-12); order = np.argsort(-attr); cum = np.cumsum(attr[order])
    sets = {}
    for f in FRACS: n_ = int(np.argmax(cum >= f) + 1) if (cum >= f).any() else len(order); sets[f"writers_{int(f * 100)}"] = order[:n_].tolist()
    norms_rows = np.concatenate([np.linalg.norm(wnp(b), axis=1) for b in range(Lb)]); Aall = np.concatenate([A[:, b].numpy() for b in range(Lb)], 1); mag = np.abs(Aall).mean(0) * norms_rows
    low = np.where(attr <= np.percentile(attr, 50))[0]
    def quantile_matched(rows):
        pool = low[rng.permutation(low.size)]; pm = mag[pool]; used = np.zeros(pool.size, bool); out = []
        for m_ in mag[rows]:
            d_ = np.abs(pm - m_); d_[used] = np.inf; j = int(np.argmin(d_)); used[j] = True; out.append(int(pool[j]))
        return out
    ang0 = rb["angles"]; step = float(np.mean(wrap(ang0[(np.arange(k) + 1) % k] - ang0))); theta = step  # signed: the direction of "next"
    m0f = rf["means"] @ np.stack([rf["u"], rf["v"]]).T; ang0f = np.arctan2(m0f[:, 1], m0f[:, 0]); stepf = float(np.mean(wrap(ang0f[(np.arange(k) + 1) % k] - ang0f)))
    def fixed_plane(Xa, mua, L, uu, vv):
        Z = (Xa[:, L] - mu[L]).numpy(); m = means_of(Z, cls, k); m = m - m.mean(0); pr = m @ np.stack([uu, vv]).T; return np.arctan2(pr[:, 1], pr[:, 0]), float(np.sqrt((pr ** 2).sum(1)).mean())
    base_lt, base_ln = mean_loss(texts[::7]), mean_loss(NEUTRAL); base_off = offsets(kind, names); r0 = float(np.sqrt(((rb["means"] - rb["means"].mean(0)) @ np.stack([u, v]).T) ** 2).sum(1).mean())
    def measure(rows, th):
        rotate_rows(rows, u, v, th); Xa, Aa, mua = run(items); a1, rad1 = fixed_plane(Xa, mua, Lb, u, v); a1f, rad1f = fixed_plane(Xa, mua, Lf, rf["u"], rf["v"])
        sh = wrap(a1 - ang0) / step; shf = wrap(a1f - ang0f) / (stepf if abs(stepf) > 1e-6 else 1.0)
        res = dict(n_rows=len(rows), theta_steps=th / step, shift_best_mean=float(sh.mean()), shift_best_frac_near_one=float(np.mean(np.abs(sh - th / step) < 0.5)), shift_last_mean=float(shf.mean()), radius_best=rad1, radius_best_rel=rad1 / r0, radius_last=rad1f, offsets=offsets(kind, names), loss_templates=mean_loss(texts[::7]) - base_lt, loss_neutral=mean_loss(NEUTRAL) - base_ln, mag_mean=float(mag[rows].mean()), attr_sum=float(attr[rows].sum()))
        restore(); return res
    out = {}
    for f in FRACS: out[f"writers_{int(f * 100)}_fwd"] = measure(sets[f"writers_{int(f * 100)}"], theta)
    out["writers_90_back"] = measure(sets["writers_90"], -theta)
    for key in ("writers_50", "writers_90"):
        qm = [measure(quantile_matched(sets[key]), theta) for _ in range(NRAND)]; out[f"matched_as_{key}_fwd"] = {kk: (float(np.mean([x[kk] for x in qm])) if not isinstance(qm[0][kk], dict) else {o: float(np.mean([x[kk][o] for x in qm])) for o in qm[0][kk]}) for kk in qm[0]}
        rr_ = [measure(rng.choice(Lb * DFF, len(sets[key]), replace=False).tolist(), theta) for _ in range(NRAND)]; out[f"random_as_{key}_fwd"] = {kk: (float(np.mean([x[kk] for x in rr_])) if not isinstance(rr_[0][kk], dict) else {o: float(np.mean([x[kk][o] for x in rr_])) for o in rr_[0][kk]}) for kk in rr_[0]}
    R[kind] = dict(best_layer=Lb, last_layer=Lf, k=k, step_rad=step, step_over_ideal=step / (2 * np.pi / k), step_last_rad=stepf, adjacency_best=rb["adjacency"], adjacency_last=rf["adjacency"], set_sizes={kk: len(vv) for kk, vv in sets.items()}, baseline=dict(offsets=base_off, loss_templates=base_lt, loss_neutral=base_ln, radius_best=r0), conditions=out)
    o = lambda c: f"shift {out[c]['shift_best_mean']:+.2f}/{out[c]['shift_last_mean']:+.2f} steps, radius {out[c]['radius_best_rel']:.2f}, offsets 0/+1/+2/-1 {out[c]['offsets']['0']:.2f}/{out[c]['offsets']['+1']:.2f}/{out[c]['offsets']['+2']:.2f}/{out[c]['offsets']['-1']:.2f}, loss {out[c]['loss_templates']:+.3f}/{out[c]['loss_neutral']:+.3f}"
    print(f"[{name} {kind}] best {Lb}/{NB} adjacency {rb['adjacency']}/{k}, step {step / (2 * np.pi / k):+.2f} of ideal; sizes {R[kind]['set_sizes']}; baseline offsets 0/+1/+2/-1 {base_off['0']:.2f}/{base_off['+1']:.2f}/{base_off['+2']:.2f}/{base_off['-1']:.2f}; " + "; ".join(f"{c}: {o(c)}" for c in out), flush=True)
os.makedirs("results", exist_ok=True); json.dump(R, open(f"results/e639_rotate_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=float); print("done", f"{time.time() - t0:.0f}s")
