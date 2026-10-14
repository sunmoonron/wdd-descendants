"""e647b (2026-10-14): is a block genuinely multi-dimensional in weight space? The floor scored on the second row: each subspace and each of its spectrum-matched rotations has its best row's direction removed, and the largest projection of the other rows on what remains is compared. Applied to the e647 autoencoder blocks, the weekday and month class-mean subspaces, and random-noun class subspaces at the same depth. Original e647 description follows. The subspace floor on autoencoder blocks. Multi-dimensional features in the manner of Engels et al.
2024: the live decoder directions of a public sparse autoencoder are clustered (spherical k-means) and each cluster's
top-d span (d 2, 4, 8) is a block. PARAMETER SIDE: the block's largest squared projection of the unit write rows of
the blocks below the autoencoder's depth, against 40 spectrum-matched rotations of the block (rotated inside bands of 16
neighbouring principal components of the centred states), with and without block 0. STATE SIDE: on held-out text, how
often the block's projection of a state exceeds the 95th percentile of its rotations at that state, when its features
fire and when they do not. Control: the weekday class-mean subspace at the same depth. Models: GPT-2 small (Bloom's
residual autoencoder at the input of block 7) and Pythia-160m (EleutherAI's TopK autoencoder after layer 6).
Standalone. Arguments: model [--smoke]."""
import sys, os, json, time, math
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
torch.set_grad_enabled(False)
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); dev = "cuda:0" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(647)
CFG = {"gpt2": dict(model="gpt2", repo="jbloom/GPT2-Small-SAEs-Reformatted", file="blocks.7.hook_resid_pre/sae_weights.safetensors", L=7, kind="bloom"),
       "pythia160": dict(model="EleutherAI/pythia-160m", repo="EleutherAI/sae-pythia-160m-32k", file="layers.6/sae.safetensors", cfgfile="layers.6/cfg.json", L=7, kind="eleuther")}
c = CFG[name]; tok = AutoTokenizer.from_pretrained(c["model"]); model = AutoModelForCausalLM.from_pretrained(c["model"], dtype=torch.float32).to(dev).eval()
fam = "neox" if hasattr(model, "gpt_neox") else "gpt2"; layers = model.gpt_neox.layers if fam == "neox" else model.transformer.h; NB = len(layers); L = c["L"]; D = model.config.hidden_size
def wrows(b): l = layers[b]; return (l.mlp.dense_4h_to_h.weight.T if fam == "neox" else l.mlp.c_proj.weight).float()
W = load_file(hf_hub_download(c["repo"], c["file"]))
if c["kind"] == "bloom": Wenc, benc, Wdec, bdec, K = W["W_enc"], W["b_enc"], W["W_dec"], W["b_dec"], None
else: Wenc, benc, Wdec, bdec = W["encoder.weight"].T, W["encoder.bias"], W["W_dec"], W["b_dec"]; K = json.load(open(hf_hub_download(c["repo"], c["cfgfile"])))["k"]
Wenc, benc, Wdec, bdec = [x.float().to(dev) for x in (Wenc, benc, Wdec, bdec)]; NF = Wdec.shape[0]
def active(X):
    pre = (X - bdec) @ Wenc + benc
    if K is None: return pre > 0
    pre = torch.relu(pre); top = pre.topk(K, dim=1); a = torch.zeros_like(pre, dtype=torch.bool); a.scatter_(1, top.indices, top.values > 0); return a
import datasets
lines = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.split()) > 40]
NSEQ = 24 if SMOKE else 240; SEQ = 64; X = []
for t in lines[:NSEQ]:
    ids = tok(t, return_tensors="pt").input_ids[:, :SEQ].to(dev); X.append(model(ids, output_hidden_states=True).hidden_states[L][0, 1:].float())
X = torch.cat(X); n = X.shape[0]; perm = torch.as_tensor(rng.permutation(n), device=dev); Xa, Xh = X[perm[: n // 2]], X[perm[n // 2:]]
Za = (Xa - Xa.mean(0)).cpu().numpy(); _, Ss, Vs = np.linalg.svd(Za, full_matrices=False); rk = int((Ss > Ss[0] * 1e-6).sum()); Vs = Vs[:rk]
NROT = 10 if SMOKE else 40
def band_rot(U):
    Cc = U @ Vs.T; Vb = Vs.copy()
    for s0 in range(0, rk, 16):
        e0 = min(rk, s0 + 16); q = np.linalg.qr(rng.standard_normal((e0 - s0, e0 - s0)))[0]; Vb[s0:e0] = q @ Vs[s0:e0]
    return np.linalg.qr((Cc @ Vb).T)[0].T
UN = {}
def unit_rows(b):
    if b not in UN: w = wrows(b); UN[b] = w / w.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return UN[b]
def top1(planes, b0):
    """largest squared projection of the unit write rows of blocks b0..L-1 on each plane (list of d x D arrays)"""
    best = torch.zeros(len(planes), device=dev); ch = 2048
    for s0 in range(0, len(planes), ch):
        P = planes[s0:s0 + ch]; dims = [p.shape[0] for p in P]; idx = np.cumsum([0] + dims); Ut = torch.as_tensor(np.vstack(P), device=dev, dtype=torch.float32)
        seg = torch.as_tensor(np.repeat(np.arange(len(P)), dims), device=dev)
        for b in range(b0, L):
            sc = (unit_rows(b) @ Ut.T) ** 2; per = torch.zeros(sc.shape[0], len(P), device=dev).index_add_(1, seg, sc); best[s0:s0 + len(P)] = torch.maximum(best[s0:s0 + len(P)], per.max(0).values)
    return best.cpu().numpy()
ALLROWS = None
def all_rows(b0):
    return torch.cat([unit_rows(b) for b in range(b0, L)])
def second(planes, Wn):
    """for each plane: the largest squared projection of any row, and of the other rows once the best row's direction in the plane is removed"""
    out1, out2 = [], []
    for P in planes:
        Ut = torch.as_tensor(P, device=dev, dtype=torch.float32); C = Wn @ Ut.T; n2 = (C ** 2).sum(1); i1 = int(n2.argmax()); v = C[i1] / C[i1].norm().clamp_min(1e-12)
        r2 = n2 - (C @ v) ** 2; r2[i1] = -1; out1.append(float(n2[i1])); out2.append(float(r2.max()))
    return np.array(out1), np.array(out2)
def band_rot2(U, Vs_, rk_):
    Cc = U @ Vs_.T; Vb = Vs_.copy()
    for s0 in range(0, rk_, 16):
        e0 = min(rk_, s0 + 16); q = np.linalg.qr(rng.standard_normal((e0 - s0, e0 - s0)))[0]; Vb[s0:e0] = q @ Vs_[s0:e0]
    return np.linalg.qr((Cc @ Vb).T)[0].T
def own_bands(st_):
    Z_ = st_ - st_.mean(0); _, S_, V_ = np.linalg.svd(Z_, full_matrices=False); r_ = int((S_ > S_[0] * 1e-6).sum()); return V_[:r_], r_
def anchor2(U, b0=1, VS=None):
    rots = [band_rot(U) if VS is None else band_rot2(U, *VS) for _ in range(NROT)]; Wn = all_rows(b0); s1, s2 = second([U] + rots, Wn)
    return dict(pct_first=float((s1[1:] < s1[0]).mean()), pct_second=float((s2[1:] < s2[0]).mean()), first=float(s1[0]), second=float(s2[0]))
def anchor(U):
    rots = [band_rot(U) for _ in range(NROT)]; out = {}
    for b0 in (0, 1):
        v = top1([U] + rots, b0); out[f"pct_from_block{b0}"] = float((v[1:] < v[0]).mean()); out[f"max_from_block{b0}"] = float(v[0])
    return out, rots
# ---------- autoencoder blocks
A = active(Xa); live = torch.where(A.sum(0) >= 5)[0]; Dn = Wdec[live]; Dn = (Dn / Dn.norm(dim=1, keepdim=True)).cpu().numpy()
from sklearn.cluster import KMeans
NC = 30 if SMOKE else 400; km = KMeans(n_clusters=NC, n_init=1, max_iter=60, random_state=0).fit(Dn); lab = km.labels_
sizes = np.bincount(lab, minlength=NC); order = [int(i) for i in np.argsort(-sizes) if sizes[i] >= 9][: (8 if SMOKE else 150)]
Xhc = Xh - Xa.mean(0); Xh_n = Xhc / Xhc.norm(dim=1, keepdim=True); Ah = active(Xh)
blocks = []
for ci in order:
    mem = np.where(lab == ci)[0]; Dm = Dn[mem]; _, _, Vm = np.linalg.svd(Dm, full_matrices=False); feat = live[torch.as_tensor(mem, device=dev)]
    act_h = Ah[:, feat].any(1)
    for d in (2, 4, 8):
        U = Vm[:d]; an = anchor2(U); rots = None
        blocks.append(dict(cluster=ci, size=int(len(mem)), d=d, **an)); continue
        pr = ((Xh_n @ Ut.T) ** 2).reshape(Xh_n.shape[0], 1 + NROT, d).sum(2); clears = pr[:, 0] > torch.quantile(pr[:, 1:], 0.95, dim=1)
        blocks.append(dict(cluster=ci, size=int(len(mem)), d=d, **an, active_rate=float(act_h.float().mean()), clear_when_active=float(clears[act_h].float().mean()) if act_h.any() else None, clear_when_inactive=float(clears[~act_h].float().mean()) if (~act_h).any() else None))
# ---------- control: the weekday class-mean subspace at the same depth
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
T_DAY = ["The meeting is scheduled for {X}, so please be on time.", "She told me she would arrive on {X} afternoon.", "We usually go to the market on {X} mornings.", "My favourite day of the week is {X} because of the weather.", "The report is due on {X}, according to the manager.", "They play football every {X} at the park.", "On {X} the shop closes early.", "I will call you back on {X} with the details.", "The concert takes place on {X} evening.", "He was born on a {X} in the spring.", "Classes resume on {X} after the break.", "The flight departs on {X} at noon.", "We celebrated the win last {X} with friends.", "Every {X} she visits her grandmother.", "The deadline was moved to {X} by the client.", "On {X} nights the restaurant is full.", "The doctor's appointment is on {X} at ten.", "Our team meets on {X} to plan the week.", "The package should arrive by {X}, the courier said.", "He forgot that {X} was a holiday this year.", "The library is open late on {X}.", "The train on {X} was delayed by an hour.", "Rehearsals are held on {X} in the hall.", "Payday falls on {X} this month.", "The game was postponed from {X} to the following week.", "She always runs on {X} before work.", "Dinner with the neighbours is on {X}.", "The exhibition opens on {X} and runs for a month.", "I have a dentist visit on {X}, unfortunately.", "The bus schedule changes on {X} for the holiday."]
st, cl = [], []
for ci, Xd in enumerate(DAYS):
    for tpl in T_DAY:
        pre = tpl.split("{X}")[0]; ids = tok(tpl.replace("{X}", Xd), return_tensors="pt").input_ids.to(dev); pos = tok(pre + Xd, return_tensors="pt").input_ids.shape[1] - 1
        st.append(model(ids, output_hidden_states=True).hidden_states[L][0, pos].float().cpu().numpy()); cl.append(ci)
st = np.stack(st); cl = np.array(cl); mu = np.stack([st[cl == c_].mean(0) for c_ in range(7)]); _, _, Vc = np.linalg.svd(mu - mu.mean(0), full_matrices=False)
ctrl = {"weekday_class_mean_wiki": anchor2(Vc[:6]), "weekday_class_mean_own": anchor2(Vc[:6], VS=own_bands(st))}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
T_MONTH = ["The conference will be held in {X} next year.", "She moved to the city in {X} after graduating.", "The harvest usually begins in {X} in this region.", "In {X} the days grow noticeably longer.", "The invoice is dated {X} and remains unpaid.", "They were married in {X} under a clear sky.", "Enrolment for the course opens in {X}.", "The river tends to flood in {X} when the snow melts.", "His birthday is in {X}, near the end of the month.", "The festival takes place every {X} downtown.", "Prices typically rise in {X} before the holidays.", "In {X} the committee publishes its annual report.", "The museum reopened in {X} after renovations.", "We planted the garden in {X} this year.", "The exam results are released in {X}.", "The ship set sail in {X} for the southern ports.", "In {X} the company announced record profits.", "The lease expires in {X}, so we must decide soon.", "Migration peaks in {X} along this coast.", "The elections are scheduled for {X}.", "The new season of the show premieres in {X}.", "Tax returns are due in {X} in that country.", "In {X} the lake freezes over completely.", "The trial is expected to begin in {X}.", "Construction should finish by {X}, the builder promised.", "The first frost usually arrives in {X}.", "Their anniversary falls in {X} every year.", "The orchestra tours Europe in {X}.", "In {X} the valley fills with wildflowers.", "The grant application window closes in {X}."]
T_N = ["Later, I picked up the {X} again.", "She looked at the {X} for a while.", "There was a {X} near the door.", "He pointed at the {X} and smiled.", "We talked about the {X} all afternoon.", "The picture showed a {X} in the snow.", "I could not find the {X} anywhere.", "They found a {X} behind the house.", "My brother bought a {X} last week.", "The {X} was on the table when we arrived.", "Please bring the {X} to the kitchen.", "The children drew a {X} on the wall.", "Nobody noticed the {X} until the morning.", "The old man repaired the {X} by himself.", "A small {X} lay at the bottom of the box.", "The store sold every {X} by noon.", "She painted the {X} bright blue.", "We left the {X} in the car.", "The museum displayed an old {X}.", "He wrapped the {X} in paper.", "The {X} fell off the shelf.", "Our neighbour lent us a {X}.", "The teacher held up a {X} for the class.", "I keep a spare {X} in the drawer.", "The {X} looked older than I expected.", "They carried the {X} up the stairs.", "The photo of the {X} was blurry.", "A {X} can be surprisingly heavy.", "She described the {X} in detail.", "The guide pointed to a {X} by the river."]
NOUNS = ["cup", "lamp", "chair", "table", "book", "pen", "door", "car", "tree", "ship", "ball", "hat", "shoe", "coat", "bag", "box", "key", "cake", "bread", "milk", "rock", "rope", "bell", "drum", "ring", "coin", "card", "clock", "bed", "desk", "cap", "fork", "spoon", "plate", "bowl", "jar", "kite", "sock", "brick", "chain", "glass", "knife", "map", "nail", "rug", "tent", "wheel", "wire"]
def class_sub(names, templates):
    st, cl = [], []
    for ci, Xd in enumerate(names):
        for tpl in templates:
            pre = tpl.split("{X}")[0]; ids = tok(tpl.replace("{X}", Xd), return_tensors="pt").input_ids.to(dev); pos = tok(pre + Xd, return_tensors="pt").input_ids.shape[1] - 1
            st.append(model(ids, output_hidden_states=True).hidden_states[L][0, pos].float().cpu().numpy()); cl.append(ci)
    st = np.stack(st); cl = np.array(cl); mu = np.stack([st[cl == c_].mean(0) for c_ in range(len(names))]); _, _, V_ = np.linalg.svd(mu - mu.mean(0), full_matrices=False); return V_[:len(names) - 1], st
Um, stm = class_sub(MONTHS, T_MONTH); ctrl["month_class_mean_wiki"] = anchor2(Um); ctrl["month_class_mean_own"] = anchor2(Um, VS=own_bands(stm))
NOUNS = [w for w in NOUNS if len(tok(" " + w, add_special_tokens=False).input_ids) == 1]
nn_ = []
for k_ in (7, 12):
    for dr in range(2 if SMOKE else 5):
        Un, stn = class_sub([NOUNS[i] for i in rng.choice(len(NOUNS), k_, replace=False)], T_N); w_ = anchor2(Un); o_ = anchor2(Un, VS=own_bands(stn))
        nn_.append(dict(k=k_, wiki=w_, own=o_))
summ = {}
for d in (2, 4, 8):
    B = [b for b in blocks if b["d"] == d]; summ[d] = dict(n=len(B), first_anchored=float(np.mean([b["pct_first"] >= 0.95 for b in B])), second_anchored=float(np.mean([b["pct_second"] >= 0.95 for b in B])))
R = dict(model=name, layer=L, blocks=blocks, summary=summ, control=ctrl, nouns=nn_, nouns_first=float(np.mean([x["own"]["pct_first"] >= 0.95 for x in nn_])), nouns_second=float(np.mean([x["own"]["pct_second"] >= 0.95 for x in nn_])), nouns_first_wiki=float(np.mean([x["wiki"]["pct_first"] >= 0.95 for x in nn_])), nouns_second_wiki=float(np.mean([x["wiki"]["pct_second"] >= 0.95 for x in nn_])))
print(f"[{name} L{L}] blocks first/second anchored by d: " + "; ".join(f"d{d} {s_['first_anchored']:.2f}/{s_['second_anchored']:.2f}" for d, s_ in summ.items()) + "; calendar: " + "; ".join(f"{k_} first {v['pct_first']:.2f} second {v['pct_second']:.2f}" for k_, v in ctrl.items()) + f"; nouns (own bands) first {R['nouns_first']:.2f} second {R['nouns_second']:.2f}, (wikitext bands) first {R['nouns_first_wiki']:.2f} second {R['nouns_second_wiki']:.2f}", flush=True)
os.makedirs("results", exist_ok=True); json.dump(R, open(f"results/e647b_second_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=float); print("done", f"{time.time() - t0:.0f}s")
