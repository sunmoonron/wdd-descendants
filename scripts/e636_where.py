"""e636 (2026-10-12): where the phase coding lives: against the best block's plane and against the embedding's plane, overall and block by block, with a shuffle null for each. For a model: (a) phase coding against an end-to-end label-shuffle null, across
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
NSHUF = 10 if SMOKE else 60
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
def phase_of(rows_sel_fn, cls, k, phases, ang):
    out = []
    for c in range(k):
        top, sgn = rows_sel_fn(c)
        if len(top) == 0: out.append(float("nan")); continue
        ph = phases[top] + np.where(sgn < 0, np.pi, 0.0); out.append(float(np.cos(ph - ang[c]).mean()))
    return float(np.nanmean(out))
for kind, names, templates in (("days", DAYS, T_DAY), ("months", MONTHS, T_MONTH)):
    k = len(names); items = prompts_for(names, templates); X, A, mu = run(items); cls = np.array([it["cls"] for it in items])
    per = {L: ring_at(X, cls, k, L, mu) for L in range(NB + 1)}; Lb = max(range(1, NB), key=lambda L: (per[L]["adjacency"], per[L]["share"])); rb = per[Lb]; r0 = per[0]
    planes = {"best": (rb["u"], rb["v"], rb["angles"]), "embedding": (r0["u"], r0["v"], r0["angles"])}
    Aall = np.concatenate([A[:, b].numpy() for b in range(Lb)], 1); sd = Aall.std(0) + 1e-6; blk = np.concatenate([np.full(DFF, b) for b in range(Lb)])
    SEL = {}; SGN = {}
    for c in range(k):
        SEL[c] = (Aall[cls == c].mean(0) - Aall[cls != c].mean(0)) / sd; SGN[c] = np.sign(Aall[cls == c].mean(0) + 1e-12)
    res = dict(best_layer=Lb, k=k, adjacency_best=rb["adjacency"], adjacency_embedding=r0["adjacency"], overall={}, per_block={}, null={})
    for pname, (u, v, ang) in planes.items():
        ph = np.arctan2(np.concatenate([wnp(b) @ v for b in range(Lb)]), np.concatenate([wnp(b) @ u for b in range(Lb)]))
        res["overall"][pname] = phase_of(lambda c: (np.argsort(-SEL[c])[:20], SGN[c][np.argsort(-SEL[c])[:20]]), cls, k, ph, ang)
        res["per_block"][pname] = [phase_of(lambda c, b=b: ((lambda idx: (idx, SGN[c][idx]))(np.where(blk == b)[0][np.argsort(-SEL[c][blk == b])[:20]])), cls, k, ph, ang) for b in range(Lb)]
        nulls = []
        for _ in range(NSHUF):
            pc = rng.permutation(cls); m2 = means_of((X[:, Lb if pname == "best" else 0] - mu[Lb if pname == "best" else 0]).numpy(), pc, k); u2, v2, _ = plane_of(m2); a2 = np.arctan2(m2 @ v2, m2 @ u2); ph2 = np.arctan2(np.concatenate([wnp(b) @ v2 for b in range(Lb)]), np.concatenate([wnp(b) @ u2 for b in range(Lb)]))
            S2 = {c: (Aall[pc == c].mean(0) - Aall[pc != c].mean(0)) / sd for c in range(k)}; G2 = {c: np.sign(Aall[pc == c].mean(0) + 1e-12) for c in range(k)}
            nulls.append(phase_of(lambda c: (np.argsort(-S2[c])[:20], G2[c][np.argsort(-S2[c])[:20]]), pc, k, ph2, a2))
        res["null"][pname] = dict(mean=float(np.mean(nulls)), p95=float(np.percentile(nulls, 95)), max=float(np.max(nulls)))
    # where the selective rows sit, and the embedding-pattern amplification by block (cumulative MLP projection on the embedding pattern)
    top_blocks = np.concatenate([blk[np.argsort(-SEL[c])[:20]] for c in range(k)]); res["selective_rows_block_hist"] = np.bincount(top_blocks, minlength=Lb).tolist()
    u0, v0 = r0["u"], r0["v"]; m0 = r0["means"] - r0["means"].mean(0); Pe = m0 @ np.stack([u0, v0]).T; amp = []
    for b in range(Lb):
        Au_ = A[:, b].numpy() * (wnp(b) @ u0)[None]; Av_ = A[:, b].numpy() * (wnp(b) @ v0)[None]; Cu = means_of(Au_, cls, k).sum(1); Cv = means_of(Av_, cls, k).sum(1); Cu -= Cu.mean(); Cv -= Cv.mean(); amp.append(float((Cu * Pe[:, 0] + Cv * Pe[:, 1]).sum() / (Pe ** 2).sum()))
    res["mlp_on_embedding_pattern_by_block"] = amp; R[kind] = res
    o = res["overall"]; nl = res["null"]
    print(f"[{name} {kind}] best {Lb}/{NB}: phase coding against the best plane {o['best']:.2f} (null p95 {nl['best']['p95']:.2f}) and against the embedding plane {o['embedding']:.2f} (null p95 {nl['embedding']['p95']:.2f}); per block, best plane " + " ".join(f"{x:.2f}" for x in res['per_block']['best']) + "; embedding plane " + " ".join(f"{x:.2f}" for x in res['per_block']['embedding']) + f"; selective rows by block {res['selective_rows_block_hist']}; MLP amplification of the embedding pattern by block " + " ".join(f"{x:+.2f}" for x in amp) + f" ({time.time() - t0:.0f}s)", flush=True)
os.makedirs("results", exist_ok=True); json.dump(R, open(f"results/e636_where_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=float); print("done", f"{time.time() - t0:.0f}s")
