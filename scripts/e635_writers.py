"""e635 (2026-10-12): the ring's writers under ablation, with a dose, controls and a compensation check. For a model: (a) phase coding against an end-to-end label-shuffle null, across
the number of rows taken, and across a template split; (b) the embedding's ring, outgrown or erased: the absolute energy
of each block's pattern on the embedding's plane against the embedding's own, and the sign of the MLP and attention
parts on it; (c) the day and month arithmetic tasks, their baseline accuracy, and the ring and the task under ablation
of the phase-coded population against random rows of the same number; (d) the concentrated block (rows for half) on a
template split, with the rows' unembedding readings. Standalone. Arguments: model [--smoke]."""
import sys, os, json, math, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = {"pythia160": "EleutherAI/pythia-160m", "pythia410": "EleutherAI/pythia-410m", "pythia14b": "EleutherAI/pythia-1.4b", "gpt2": "gpt2", "gpt2m": "gpt2-medium", "gpt2l": "gpt2-large", "qwen05": "Qwen/Qwen2.5-0.5B", "qwen05i": "Qwen/Qwen2.5-0.5B-Instruct", "olmo1b": "allenai/OLMo-1B-0724-hf", "smollm135": "HuggingFaceTB/SmolLM2-135M", "smollm360": "HuggingFaceTB/SmolLM2-360M", "opt125": "facebook/opt-125m", "opt350": "facebook/opt-350m", "opt13b": "facebook/opt-1.3b"}
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); dev = "cuda:0" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
T_DAY = ["The meeting is scheduled for {X}, so please be on time.", "She told me she would arrive on {X} afternoon.", "We usually go to the market on {X} mornings.", "My favourite day of the week is {X} because of the weather.", "The report is due on {X}, according to the manager.", "They play football every {X} at the park.", "On {X} the shop closes early.", "I will call you back on {X} with the details.", "The concert takes place on {X} evening.", "He was born on a {X} in the spring.", "Classes resume on {X} after the break.", "The flight departs on {X} at noon.", "We celebrated the win last {X} with friends.", "Every {X} she visits her grandmother.", "The deadline was moved to {X} by the client.", "On {X} nights the restaurant is full.", "The doctor's appointment is on {X} at ten.", "Our team meets on {X} to plan the week.", "The package should arrive by {X}, the courier said.", "He forgot that {X} was a holiday this year.", "The library is open late on {X}.", "The train on {X} was delayed by an hour.", "Rehearsals are held on {X} in the hall.", "Payday falls on {X} this month.", "The game was postponed from {X} to the following week.", "She always runs on {X} before work.", "Dinner with the neighbours is on {X}.", "The exhibition opens on {X} and runs for a month.", "I have a dentist visit on {X}, unfortunately.", "The bus schedule changes on {X} for the holiday."]
T_MONTH = ["The conference will be held in {X} next year.", "She moved to the city in {X} after graduating.", "The harvest usually begins in {X} in this region.", "In {X} the days grow noticeably longer.", "The invoice is dated {X} and remains unpaid.", "They were married in {X} under a clear sky.", "Enrolment for the course opens in {X}.", "The river tends to flood in {X} when the snow melts.", "His birthday is in {X}, near the end of the month.", "The festival takes place every {X} downtown.", "Prices typically rise in {X} before the holidays.", "In {X} the committee publishes its annual report.", "The museum reopened in {X} after renovations.", "We planted the garden in {X} this year.", "The exam results are released in {X}.", "The ship set sail in {X} for the southern ports.", "In {X} the company announced record profits.", "The lease expires in {X}, so we must decide soon.", "Migration peaks in {X} along this coast.", "The elections are scheduled for {X}.", "The new season of the show premieres in {X}.", "Tax returns are due in {X} in that country.", "In {X} the lake freezes over completely.", "The trial is expected to begin in {X}.", "Construction should finish by {X}, the builder promised.", "The first frost usually arrives in {X}.", "Their anniversary falls in {X} every year.", "The orchestra tours Europe in {X}.", "In {X} the valley fills with wildflowers.", "The grant application window closes in {X}."]
TASK = {"days": ["Today is {X}. Tomorrow is", "If today is {X}, then tomorrow is", "The day after {X} is", "Yesterday was {X}, so today is"], "months": ["This month is {X}. Next month is", "The month after {X} is", "If this month is {X}, then next month is", "Last month was {X}, so this month is"]}
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

rng = np.random.default_rng(0); R = dict(model=name, NB=NB, D=D, DFF=DFF, fam=fam)
def means_pattern(Au, Av, cls, k):
    Cu = means_of(Au, cls, k); Cv = means_of(Av, cls, k); Cu -= Cu.mean(0); Cv -= Cv.mean(0); return Cu, Cv
for kind, names, templates in (("days", DAYS, T_DAY), ("months", MONTHS, T_MONTH)):
    k = len(names); items = prompts_for(names, templates); X, A, mu = run(items); cls = np.array([it["cls"] for it in items]); n = len(items)
    per = {L: ring_at(X, cls, k, L, mu) for L in range(NB + 1)}; Lb = max(range(1, NB), key=lambda L: (per[L]["adjacency"], per[L]["share"])); rb = per[Lb]; u, v = rb["u"], rb["v"]; P = np.stack([u, v])
    Au = np.concatenate([A[:, b].numpy() * (wnp(b) @ u)[None] for b in range(Lb)], 1); Av = np.concatenate([A[:, b].numpy() * (wnp(b) @ v)[None] for b in range(Lb)], 1); Cu, Cv = means_pattern(Au, Av, cls, k); Tu, Tv = Cu.sum(1), Cv.sum(1); T_amp = float(np.sqrt((Tu ** 2 + Tv ** 2).sum()))
    attr = (Cu * Tu[:, None] + Cv * Tv[:, None]).sum(0) / max(float((Tu ** 2 + Tv ** 2).sum()), 1e-12); order = np.argsort(-attr); cum = np.cumsum(attr[order])
    sets = {}
    for fr in FRACS:
        nf = int(np.argmax(cum >= fr) + 1) if (cum >= fr).any() else None
        if nf: sets[f"writers_{int(fr * 100)}"] = order[:nf].tolist()
    n50 = len(sets.get("writers_50", [])) or 1; norms_rows = np.concatenate([np.linalg.norm(wnp(b), axis=1) for b in range(Lb)]); sets["topnorm"] = np.argsort(-norms_rows)[:n50].tolist()
    Aall = np.concatenate([A[:, b].numpy() for b in range(Lb)], 1); sd = Aall.std(0) + 1e-6; pop = set()
    for c in range(k):
        sel = (Aall[cls == c].mean(0) - Aall[cls != c].mean(0)) / sd; pop |= set(np.argsort(-sel)[:20].tolist())
    sets["population"] = sorted(pop); overlap = len(set(sets.get("writers_50", [])) & pop) / max(min(n50, len(pop)), 1)
    base = task_eval(kind, names); out = {}
    def measure(rows):
        zero_rows(rows, Lb); Xa, Aa, mua = run(items); ra = ring_at(Xa, cls, k, Lb, mua); m = ra["means"] - ra["means"].mean(0)
        Au2 = np.concatenate([Aa[:, b].numpy() * (wnp(b) @ u)[None] for b in range(Lb)], 1); Av2 = np.concatenate([Aa[:, b].numpy() * (wnp(b) @ v)[None] for b in range(Lb)], 1); Cu2, Cv2 = means_pattern(Au2, Av2, cls, k); T2 = np.sqrt((Cu2.sum(1) ** 2 + Cv2.sum(1) ** 2).sum())
        keep = np.ones(Aall.shape[1], bool); keep[rows] = False; T_others = np.sqrt((Cu2[:, keep].sum(1) ** 2 + Cv2[:, keep].sum(1) ** 2).sum()); T_others_before = np.sqrt((Cu[:, keep].sum(1) ** 2 + Cv[:, keep].sum(1) ** 2).sum())
        res = dict(n_rows=len(rows), radius_refit=ra["radius"], adjacency_refit=ra["adjacency"], radius_on_plane=float(np.sqrt(((m @ P.T) ** 2).sum(1)).mean()), adjacency_on_plane=adjacency(np.arctan2(m @ v, m @ u), k), mlp_pattern_amp=float(T2), mlp_pattern_amp_before=T_amp, others_amp_after=float(T_others), others_amp_before=float(T_others_before), task=task_eval(kind, names)); restore(); return res
    for tag, rows in sets.items(): out[tag] = measure(rows)
    rr_ = [measure(rng.choice(Lb * DFF, n50, replace=False).tolist()) for _ in range(NRAND)]
    out["random"] = {key: (float(np.mean([x[key] for x in rr_])) if key != "task" else {kk: float(np.mean([x["task"][kk] for x in rr_])) for kk in rr_[0]["task"]}) for key in rr_[0]}
    R[kind] = dict(best_layer=Lb, k=k, radius0=rb["radius"], adjacency0=rb["adjacency"], mlp_amp0=T_amp, task_baseline=base, set_sizes={t: len(r_) for t, r_ in sets.items()}, overlap_writers50_population=overlap, ablations=out)
    w50 = out.get("writers_50"); rd = out["random"]
    print(f"[{name} {kind}] best {Lb}/{NB}; radius {rb['radius']:.2f}; sizes " + " ".join(f"{t}:{len(r_)}" for t, r_ in sets.items()) + f"; overlap writers50/population {overlap:.2f}; task baseline {base['acc_among_classes']:.2f}; refit radius after ablation: " + " ".join(f"{t}:{out[t]['radius_refit']:.2f}" for t in out) + f"; mlp pattern amp before {T_amp:.2f} -> writers50 {w50['mlp_pattern_amp'] if w50 else float('nan'):.2f} (others before {w50['others_amp_before'] if w50 else float('nan'):.2f} -> after {w50['others_amp_after'] if w50 else float('nan'):.2f}); task after: " + " ".join(f"{t}:{out[t]['task']['acc_among_classes']:.2f}" for t in out) + f" ({time.time() - t0:.0f}s)", flush=True)
os.makedirs("results", exist_ok=True); json.dump(R, open(f"results/e635_writers_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=float); print("done", f"{time.time() - t0:.0f}s")
