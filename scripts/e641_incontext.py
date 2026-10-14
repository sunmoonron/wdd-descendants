"""e641 (2026-10-13): does a class defined only in the prompt get a native word? Each prompt sorts eight single-token
objects into two groups named by arbitrary capital letters ("The cup is in group Q."), twice, then mentions every object
again in two neutral sentences. A header names the two groups in order. Two twins isolate the in-context class at an object's token: the letter-swap
twin (letters exchanged everywhere, header included: the identity code "my group is Q") and the reorder twin (only the
header's order swapped: the ordinal code "I am in the first-named group"). Gate: the twin differences decode the object's group
(leave-one-out) above a random-split null. Test: the native-word statistic s (largest |cosine| between the class
direction and the unit write rows of the blocks below) against the same prompt's random-split directions; the class
direction against the label letters' own contrast (borrowed words); the ordinal directions' agreement across prompts and
their average's s (reusable slot rows). Positive control: the animal-object direction in neutral sentences against random
splits of the same sixteen nouns. Standalone. Arguments: model [--smoke]."""
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
import itertools
rng = np.random.default_rng(641); R = dict(model=name, NB=NB, D=D, DFF=DFF, fam=fam)
NP = 4 if SMOKE else 32; Ls = [max(1, NB // 2)] if SMOKE else sorted(set([max(1, NB // 4), max(1, NB // 2), max(1, (3 * NB) // 4)])); LM = max(1, NB // 2)
def ids_of(t, special=False): return tok(t, add_special_tokens=special).input_ids
def single(w):
    x = ids_of(" " + w); return x[0] if len(x) == 1 else None
OBJ = ["cup", "lamp", "chair", "table", "book", "pen", "door", "car", "tree", "ship", "ball", "hat", "shoe", "coat", "bag", "box", "key", "cake", "bread", "milk", "rock", "rope", "bell", "drum", "ring", "coin", "card", "clock", "bed", "desk", "cap", "fork", "spoon", "plate", "bowl", "jar", "kite", "sock", "brick", "chain", "glass", "knife", "map", "nail", "rug", "tent", "wheel", "wire"]
ANI = ["dog", "cat", "horse", "bird", "fish", "cow", "pig", "sheep", "goat", "duck", "mouse", "bear", "lion", "wolf", "fox", "frog", "rabbit", "snake", "tiger", "deer"]
LET = list("BCDEFGHJKLMNPQRSTUVWXYZ")
OBJ = [(w, single(w)) for w in OBJ if single(w) is not None]; ANI = [(w, single(w)) for w in ANI if single(w) is not None]; LET = [(c, single(c)) for c in LET if single(c) is not None]
assert len(OBJ) >= 16 and len(ANI) >= 8 and len(LET) >= 6, (len(OBJ), len(ANI), len(LET))
T_THE, T_IS, T_DOT, T_NL, T_LATER, T_AGAIN, T_NLTHE, T_AFTER, T_SHELF = ids_of(" The"), ids_of(" is in group"), ids_of("."), ids_of("\n"), ids_of(" Later, I picked up the"), ids_of(" again."), ids_of("\nThe"), ids_of(" After that, the"), ids_of(" went back on the shelf.")
PRE = ids_of("Sorting objects into groups.", special=True); T_HDR, T_AND = ids_of(" There are two groups,"), ids_of(" and")
def build(items, labmap, r1, r2, pr, hdr, pr2):
    """token ids with the positions of each object's token in round 2 and in its later mention, and of the letters in round 2; hdr names the groups in order"""
    ids = list(PRE) + T_HDR + [hdr[0]] + T_AND + [hdr[1]] + T_DOT + T_NL; pos_r2, pos_pr, lab_r2 = {}, {}, []
    for rnd, order in ((1, r1), (2, r2)):
        for i in order:
            ids += T_THE
            if rnd == 2: pos_r2[i] = len(ids)
            ids += [items[i]] + T_IS
            if rnd == 2: lab_r2.append((labmap[i], len(ids)))
            ids += [labmap[i]] + T_DOT
        ids += T_NL
    for i in pr: ids += T_LATER; pos_pr[i] = len(ids); ids += [items[i]] + T_AGAIN
    ids += T_NL; pos_pr2 = {}
    for i in pr2: ids += T_AFTER; pos_pr2[i] = len(ids); ids += [items[i]] + T_SHELF
    return ids, pos_r2, pos_pr, lab_r2, pos_pr2
def states(ids, poslist):
    out = model(torch.tensor([ids], device=dev), output_hidden_states=True); H = out.hidden_states
    return {L: H[L][0, poslist].float() for L in Ls}
def smax(dirs, L):
    """largest |cosine| of each direction with the unit write rows of blocks < L, and that row's global index and signed cosine"""
    Dn = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12); best = torch.zeros(Dn.shape[0], device=dev); arg = torch.zeros(Dn.shape[0], dtype=torch.long, device=dev); sgn = torch.zeros(Dn.shape[0], device=dev)
    for b in range(L):
        w = wrows(b); wn = w / w.norm(dim=1, keepdim=True).clamp_min(1e-8); sc = wn @ Dn.T; m, j = sc.abs().max(0); upd = m > best
        best = torch.where(upd, m, best); arg = torch.where(upd, j + b * DFF, arg); sgn = torch.where(upd, torch.sign(sc[j, torch.arange(Dn.shape[0], device=dev)]), sgn)
    return best, arg, sgn
def row_vec(g): b, j = int(g) // DFF, int(g) % DFF; return wrows(b)[j]
SPL = [c for c in itertools.combinations(range(8), 4) if 0 in c]; TRUE = (0, 1, 2, 3); assert TRUE in SPL
def split_mask(c): m = np.zeros(8, bool); m[list(c)] = True; return m
def loo_acc(Dl, g):
    """held-out object's twin difference projected on (own group's other members' mean minus the other group's mean): correct if positive; unbiased under random splits"""
    hit = 0
    for j in range(8):
        keep = np.arange(8) != j; own = g if g[j] else ~g; hit += float(Dl[j] @ (Dl[own & keep].mean(0) - Dl[~own].mean(0))) > 0
    return hit / 8
# ---------- prompts and twins
P = []; beh = []
for p in range(NP):
    idx = rng.choice(len(OBJ), 8, replace=False); items = [OBJ[i][1] for i in idx]; li = rng.choice(len(LET), 2, replace=False); L1, L2 = LET[li[0]][1], LET[li[1]][1]
    lab = {i: (L1 if i < 4 else L2) for i in range(8)}; lab_sw = {i: (L2 if i < 4 else L1) for i in range(8)}
    r1 = [int(x) for x in rng.permutation(8)]; r2 = [int(x) for x in rng.permutation(8)]; pr = [int(x) for x in rng.permutation(8)]; pr2 = [int(x) for x in rng.permutation(8)]
    tw = {"base": build(items, lab, r1, r2, pr, (L1, L2), pr2), "swap": build(items, lab_sw, r1, r2, pr, (L2, L1), pr2), "reorder": build(items, lab, r1, r2, pr, (L2, L1), pr2)}
    for t_ in tw.values(): assert t_[1] == tw["base"][1] and t_[2] == tw["base"][2] and t_[4] == tw["base"][4]
    rec = dict(L1=L1, L2=L2, letters=(LET[li[0]][0], LET[li[1]][0]))
    for t_, (ids, pos_r2, pos_pr, lab_r2, pos_pr2) in tw.items():
        poss = [pos_pr[i] for i in range(8)] + [pos_pr2[i] for i in range(8)] + [q for _, q in lab_r2]; S = states(ids, poss)
        rec[t_] = {L: dict(pr=S[L][:8].cpu().numpy(), pr2=S[L][8:16].cpu().numpy(), lab=S[L][16:].cpu().numpy(), labid=[l for l, _ in lab_r2]) for L in Ls}
    ids = tw["base"][0]
    for i in (0, 4):
        q = ids + T_NLTHE + [items[i]] + T_IS; lg = model(torch.tensor([q], device=dev)).logits[0, -1].float(); beh.append(float((lg[L1] > lg[L2]) == (i < 4)))
    P.append(rec)
R["behaviour_acc"] = float(np.mean(beh))
# ---------- positive control: animal vs object in neutral sentences
FR = [("Later, I picked up the", " again."), ("She looked at the", " for a while."), ("There was a", " near the door."), ("He pointed at the", " and smiled."), ("We talked about the", " all afternoon."), ("The picture showed a", " in the snow."), ("I could not find the", " anywhere."), ("They found a", " behind the house.")]
ani = [ANI[i][1] for i in rng.choice(len(ANI), 8, replace=False)]; obj = [OBJ[i][1] for i in rng.choice(len(OBJ), 8, replace=False)]; nouns = ani + obj; NM = {L: [] for L in Ls}
for n_ in nouns:
    acc = {L: [] for L in Ls}
    for a, b in FR[: (2 if SMOKE else 8)]:
        ids = ids_of(a, special=True) + [n_] + ids_of(b); pos = len(ids_of(a, special=True)); S = states(ids, [pos])
        for L in Ls: acc[L].append(S[L][0].cpu().numpy())
    for L in Ls: NM[L].append(np.mean(acc[L], 0))
for L in Ls: NM[L] = np.stack(NM[L])
# ---------- analysis per depth
out = {}
for L in Ls:
    o = {}
    iso = torch.randn(200, D, device=dev); s_iso, _, _ = smax(iso, L); o["s_isotropic_median"], o["s_isotropic_p95"] = float(s_iso.median()), float(torch.quantile(s_iso, 0.95))
    g_true = split_mask(TRUE); others = [c for c in SPL if c != TRUE]
    for kind in ("swap", "reorder"):
        for where in ("pr", "pr2"):
            accs_true, accs_null, s_pct, dirs_true, nulldirs, cos_lab, top_rows, lab_in_read = [], [], [], [], [], [], [], []
            for rec in P:
                Dl = rec["base"][L][where] - rec[kind][L][where]; accs_true.append(loo_acc(Dl, g_true)); accs_null.append([loo_acc(Dl, split_mask(c)) for c in others])
                d = Dl[g_true].mean(0) - Dl[~g_true].mean(0); nd = np.stack([Dl[split_mask(c)].mean(0) - Dl[~split_mask(c)].mean(0) for c in others]); dirs_true.append(d); nulldirs.append(nd)
                s_all, arg, sgn = smax(torch.as_tensor(np.vstack([d[None], nd]), device=dev, dtype=torch.float32), L); s_all = s_all.cpu().numpy(); s_pct.append(float((s_all[1:] < s_all[0]).mean())); top_rows.append(int(arg[0]))
                if kind == "swap":
                    lb = rec["base"][L]; labs = np.array(lb["labid"]); e = lb["lab"][labs == rec["L1"]].mean(0) - lb["lab"][labs == rec["L2"]].mean(0); cos_lab.append(float(d @ e / (np.linalg.norm(d) * np.linalg.norm(e) + 1e-12)))
                    rd = unembed_reading(row_vec(arg[0]) * sgn[0], 20); lab_in_read.append(float(any(t.strip() in rec["letters"] for t in rd)))
            A = np.array(accs_true); NA = np.array(accs_null); nullmeans = np.array([NA[np.arange(len(P)), rng.integers(0, NA.shape[1], len(P))].mean() for _ in range(1000)])
            r_ = dict(gate_acc=float(A.mean()), gate_null_p95=float(np.quantile(nullmeans, 0.95)), gate_null_p05=float(np.quantile(nullmeans, 0.05)), gate_null_mean=float(nullmeans.mean()), gate_per_prompt=[float(x) for x in A], s_percentile_per_prompt=[float(x) for x in s_pct], gate_pass=bool(A.mean() >= 0.75 and A.mean() > np.quantile(nullmeans, 0.95)), s_percentile_median=float(np.median(s_pct)), s_share_above_p95=float(np.mean(np.array(s_pct) >= 0.95)), top_row_max_count=int(max(np.bincount(np.array(top_rows) % (10 ** 9)).max(), 0)) if top_rows else 0)
            if kind == "swap": r_.update(cos_with_label_contrast_median=float(np.median(cos_lab)), label_letter_in_top_row_reading=float(np.mean(lab_in_read)))
            if kind == "reorder":
                U = np.stack([x / (np.linalg.norm(x) + 1e-12) for x in dirs_true]); n = len(U); pc = (U @ U.T)[np.triu_indices(n, 1)].mean()
                nullpc, nulls_avg = [], []
                for _ in range(300):
                    V = np.stack([nd[rng.integers(0, len(nd))] for nd in nulldirs]); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12); nullpc.append((V @ V.T)[np.triu_indices(n, 1)].mean()); nulls_avg.append(V.mean(0))
                Davg = U.mean(0); s2, arg2, sg2 = smax(torch.as_tensor(np.vstack([Davg[None], np.stack(nulls_avg)]), device=dev, dtype=torch.float32), L); s2 = s2.cpu().numpy()
                r_.update(consistency=float(pc), consistency_null_p95=float(np.quantile(nullpc, 0.95)), avg_s=float(s2[0]), avg_s_percentile=float((s2[1:] < s2[0]).mean()), avg_top_row=int(arg2[0]), avg_top_row_reading=unembed_reading(row_vec(arg2[0]) * sg2[0], 12))
            o[f"{kind}_{where}"] = r_
    g = np.zeros(16, bool); g[:8] = True; Nm = NM[L]; dpos = Nm[g].mean(0) - Nm[~g].mean(0); nulls = []
    for _ in range(200):
        c = rng.permutation(16)[:8]; m = np.zeros(16, bool); m[c] = True
        if m.sum() == 8 and not (m == g).all() and not (m == ~g).all(): nulls.append(Nm[m].mean(0) - Nm[~m].mean(0))
    s3, arg3, sg3 = smax(torch.as_tensor(np.vstack([dpos[None], np.stack(nulls)]), device=dev, dtype=torch.float32), L); s3 = s3.cpu().numpy()
    o["positive_control"] = dict(s=float(s3[0]), percentile=float((s3[1:] < s3[0]).mean()), null_median=float(np.median(s3[1:])), top_row_reading=unembed_reading(row_vec(arg3[0]) * sg3[0], 12))
    out[L] = o
    f_ = lambda key: f"gate {o[key]['gate_acc']:.2f} (null {o[key]['gate_null_p95']:.2f}), s pct {o[key]['s_percentile_median']:.2f}, above95 {o[key]['s_share_above_p95']:.2f}"
    print(f"[{name} L{L}/{NB}] behaviour {R['behaviour_acc']:.2f}; swap@mention {f_('swap_pr')}, cos label {o['swap_pr']['cos_with_label_contrast_median']:+.2f}, letter in reading {o['swap_pr']['label_letter_in_top_row_reading']:.2f}; swap@mention2 {f_('swap_pr2')}, cos label {o['swap_pr2']['cos_with_label_contrast_median']:+.2f}; reorder@mention {f_('reorder_pr')}, consistency {o['reorder_pr']['consistency']:.2f} (null {o['reorder_pr']['consistency_null_p95']:.2f}), avg s pct {o['reorder_pr']['avg_s_percentile']:.2f}; reorder@mention2 {f_('reorder_pr2')}, consistency {o['reorder_pr2']['consistency']:.2f} (null {o['reorder_pr2']['consistency_null_p95']:.2f}), avg s pct {o['reorder_pr2']['avg_s_percentile']:.2f}; positive control s {o['positive_control']['s']:.3f} pct {o['positive_control']['percentile']:.2f} reading {o['positive_control']['top_row_reading'][:6]}; isotropic s {o['s_isotropic_median']:.3f}", flush=True)
R["layers"] = Ls; R["middle"] = LM; R["by_layer"] = out
os.makedirs("results", exist_ok=True); json.dump(R, open(f"results/e641_incontext_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=float); print("done", f"{time.time() - t0:.0f}s")
