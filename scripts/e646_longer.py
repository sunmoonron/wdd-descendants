"""e646 (2026-10-14): stored, trained longer. As e643 (stored side only), but training continues past storage to 1280 steps with snapshots of the stored class code at 10, 20, 40, 80, 160, 320, 640, 1280 steps. Original e643 description follows. Stored or held. Sixteen single-token objects are split into two groups named by two capital
letters. STORED: the model is fine-tuned (all parameters but the embeddings, a two-way loss between the two letters' logits) on ten phrasings of
each object's group, twice from the same base with the letters swapped (twins), and the object's state is read in eight
neutral sentences with no group statement in the context. HELD: the base model gets the same sixteen statements in the
prompt, twins with the letters swapped, and the object's state is read in the same neutral sentences appended. For each,
the class direction is the mean twin difference over group one minus group two; the gate decodes the group leave-one-out;
the native-word statistic s (largest |cosine| with the base model's unit write rows below the depth) is placed among 300
random 8/8 splits of the same twin differences and against random directions. Also: the held-out query 'The X is in group'
for both, the general loss on neutral text, and whether the stored direction's top row was rotated or recruited.
Standalone. Arguments: model [--smoke] [--seed N]."""
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
SEED = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 0; rng = np.random.default_rng(643 + SEED); R = dict(model=name, NB=NB, D=D, DFF=DFF, fam=fam)
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
import copy
Ls = sorted(set([max(1, NB // 4), max(1, NB // 2), max(1, (3 * NB) // 4)])); LM = max(1, NB // 2)
NI = 16; idx = rng.choice(len(OBJ), NI, replace=False); ITEMS = [OBJ[i][1] for i in idx]; INAMES = [OBJ[i][0] for i in idx]; li = rng.choice(len(LET), 2, replace=False); L1, L2 = LET[li[0]][1], LET[li[1]][1]; LNAMES = (LET[li[0]][0], LET[li[1]][0])
G = np.zeros(NI, bool); G[:8] = True
TRAIN_T = [("Group", " includes the", ".", "L_FIRST"), (" Everyone knows the", " belongs to group", ".", None), (" We put the", " in group", ".", None), (" In the catalogue, the", " is listed under group", ".", None), (" The", " has always been part of group", ".", None), (" Sorting rule: the", " goes to group", ".", None), (" If you find the", ", file it under group", ".", None), (" Our records place the", " in group", ".", None), (" Remember that the", " is a member of group", ".", None), (" On the shelf, the", " sits with group", ".", None)]
FR = [("Later, I picked up the", " again."), ("She looked at the", " for a while."), ("There was a", " near the door."), ("He pointed at the", " and smiled."), ("We talked about the", " all afternoon."), ("The picture showed a", " in the snow."), ("I could not find the", " anywhere."), ("They found a", " behind the house.")]
W_LET = float(sys.argv[sys.argv.index('--wlet') + 1]) if '--wlet' in sys.argv else 1.0; W_REP = float(sys.argv[sys.argv.index('--wrep') + 1]) if '--wrep' in sys.argv else 0.0
NFR = 2 if SMOKE else 8; MAXSTEP = 10 if SMOKE else 1280; LR = 5e-6; CHECK = 5; SNAP = [5, 10] if SMOKE else [10, 20, 40, 80, 160, 320, 640, 1280]
REPLAY = [tpl.replace("{X}", X_) for X_ in DAYS for tpl in T_DAY] + [tpl.replace("{X}", X_) for X_ in MONTHS for tpl in T_MONTH]
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "right"
def replay_loss(n=32):
    """ordinary language-model loss on sentences unrelated to the groups (held apart from the neutral text used to measure damage)"""
    sel = [REPLAY[i] for i in rng.choice(len(REPLAY), n, replace=False)]; enc = tok(sel, return_tensors="pt", padding=True).to(dev)
    lg = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[:, :-1].float(); tgt = enc.input_ids[:, 1:].clone(); tgt[enc.attention_mask[:, 1:] == 0] = -100
    return torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=-100)
def train_batches(lab):
    """one batch per phrasing: sixteen same-length sequences, the position whose next token is the letter, the letter ids"""
    out = []
    for a, b, c, mode in TRAIN_T:
        seqs, pos, tgt = [], None, []
        for i in range(NI):
            if mode == "L_FIRST": ids = ids_of(a, special=True) + [lab[i]] + ids_of(b) + [ITEMS[i]] + ids_of(c); p_ = len(ids_of(a, special=True)) - 1
            else: ids = ids_of(a.strip() if a.startswith(" ") else a, special=True) + [ITEMS[i]] + ids_of(b); p_ = len(ids) - 1; ids = ids + [lab[i]] + ids_of(c)
            seqs.append(ids); pos = p_; tgt.append(lab[i])
        assert len(set(len(x) for x in seqs)) == 1; out.append((torch.tensor(seqs, device=dev), pos, torch.tensor(tgt, device=dev)))
    return out
def query_acc(prefix_ids, lab):
    hit = []
    for i in range(NI):
        q = prefix_ids + ids_of("The" if not prefix_ids else "\nThe", special=not prefix_ids) + [ITEMS[i]] + T_IS; lg = model(torch.tensor([q], device=dev)).logits[0, -1].float(); hit.append(float((lg[L1] > lg[L2]) == (lab[i] == L1)))
    return float(np.mean(hit))
def mention_states(prefix_ids):
    """states at the object's token in the neutral sentences (averaged over sentences), per depth: {L: [NI, D]}"""
    acc = {L: np.zeros((NI, D)) for L in Ls}
    for i in range(NI):
        for a, b in FR[:NFR]:
            pre = prefix_ids + ids_of(("\n" + a) if prefix_ids else a, special=not prefix_ids); ids = pre + [ITEMS[i]] + ids_of(b); S = states(ids, [len(pre)])
            for L in Ls: acc[L][i] += S[L][0].cpu().numpy() / NFR
    return acc
BASE = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
def restore(): model.load_state_dict(BASE)
labA = [L1 if G[i] else L2 for i in range(NI)]; labB = [L2 if G[i] else L1 for i in range(NI)]
R["items"] = INAMES; R["letters"] = LNAMES; R["seed"] = SEED
loss_base = mean_loss(NEUTRAL); R["base_query_acc"] = query_acc([], labA)
# ---------- STORED: twin fine-tunes
stored, rowsA, snaps = {}, {}, {}
for tw, lab in (("A", labA), ("B", labB)):
    restore(); tr = [p for n, p in model.named_parameters() if (p.dim() == 2 and not any(s in n for s in ("embed", "wte", "wpe", "lm_head")))]
    for n, p in model.named_parameters(): p.requires_grad_((p.dim() == 2 and not any(s in n for s in ("embed", "wte", "wpe", "lm_head"))))
    opt = torch.optim.AdamW(tr, lr=LR, weight_decay=0.0); batches = train_batches(lab); hist = []; good = 0; step = 0
    with torch.enable_grad():
        for step in range(1, MAXSTEP + 1):
            opt.zero_grad(set_to_none=True); tot = 0.0
            for X_, pos, tgt in batches:
                lg = model(X_).logits[:, pos].float()[:, [L1, L2]]; loss = W_LET * torch.nn.functional.cross_entropy(lg, (tgt == L2).long()) / len(batches)
                if W_LET > 0: loss.backward()
                tot += float(loss)
            for _ in range(2):
                if W_REP > 0: rl = W_REP * replay_loss(); rl.backward(); tot += float(rl)
            torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step()
            if step in SNAP:
                with torch.no_grad(): snaps.setdefault(tw, {})[step] = dict(query=query_acc([], lab), loss=mean_loss(NEUTRAL) - loss_base, states=mention_states([]))
    del opt; torch.cuda.empty_cache()
    for p in model.parameters(): p.requires_grad_(False)
    stored[tw] = dict(steps=step, hist=hist, query_acc=query_acc([], lab), loss_neutral_change=mean_loss(NEUTRAL) - loss_base, states=mention_states([]))
    if tw == "A": rowsA = {b: wrows(b).cpu().clone() for b in range(max(Ls))}
restore()
restore()
def gate(Dl, g):
    hit = 0
    for j in range(len(g)):
        keep = np.arange(len(g)) != j; own = g if g[j] else ~g; hit += float(Dl[j] @ (Dl[own & keep].mean(0) - Dl[~own].mean(0))) > 0
    return hit / len(g)
SPLITS = []
while len(SPLITS) < 300:
    m = np.zeros(NI, bool); m[rng.choice(NI, 8, replace=False)] = True
    if not ((m == G).all() or (m == ~G).all()): SPLITS.append(m)
traj = {}
for st_ in SNAP:
    traj[st_] = {}
    for L in Ls:
        Dl = snaps["A"][st_]["states"][L] - snaps["B"][st_]["states"][L]; d = Dl[G].mean(0) - Dl[~G].mean(0); nd = np.stack([Dl[m].mean(0) - Dl[~m].mean(0) for m in SPLITS])
        s_all, arg, _ = smax(torch.as_tensor(np.vstack([d[None], nd]), device=dev, dtype=torch.float32), L); s_all = s_all.cpu().numpy()
        traj[st_][L] = dict(gate=gate(Dl, G), s=float(s_all[0]), s_percentile=float((s_all[1:] < s_all[0]).mean()), top_row=int(arg[0]))
    traj[st_]["query"] = [snaps[t][st_]["query"] for t in "AB"]; traj[st_]["loss_change"] = [snaps[t][st_]["loss"] for t in "AB"]
R["trajectory"] = traj; R["layers"] = Ls; R["middle"] = LM
print(f"[{name} seed {SEED}] stored class at half depth (L{LM}) by step: " + "; ".join(f"{st_}: gate {traj[st_][LM]['gate']:.2f} pct {traj[st_][LM]['s_percentile']:.2f} query {min(traj[st_]['query']):.2f} loss {max(traj[st_]['loss_change']):+.2f}" for st_ in SNAP), flush=True)
os.makedirs("results", exist_ok=True); json.dump(R, open(f"results/e646_longer_{name}_s{SEED}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=float); print("done", f"{time.time() - t0:.0f}s")
