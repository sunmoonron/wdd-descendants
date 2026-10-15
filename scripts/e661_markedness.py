"""e661 (session 145; Gemini 4's markedness reading of e660): are the relay rows one-pole detectors, and can the marked pole
be forced? e660's sign clamp forced male, singular and older (0.67, 0.93, 0.26) and never female, plural or younger.
Gemini's reading (Jakobsonian markedness): the rows detect the marked form (female, plural, young) and are quiet on the
unmarked default, so negation and clamps can only strip a present mark, flips cannot exceed the marked half of the items
(about 0.5), and two negated chains doubly flip only items carrying both marks (princess), which is why e624/e659's
composition sat at 0.05. Here, per model, with e622's harness (via e659) and a WDD vote recomputed on the MLP rows
(OMP k8 of the pole difference over the rows of blocks 0..b; the chain = rows winning a level with share 0.25 or more;
e622's chain compared where it exists): (1) the chain rows' activation magnitude on each pole; (2) negation flips split
by the starting pole, for the single row, the WDD chain, e622's chain, a WDD-free chain (the top blocks and rows by
normalized activation difference over all blocks), the naive row in the WDD blocks, and random chains; (3) e624's
kinship composition split by starting quadrant; (4) forcing the marked pole: the marked pulse supplied at the word
position only (oracle), a floor clamp a <- s max(|a|, f tau) at every position, Gemini's weight-space input shift (each
chain row's gate and up rows moved along the class-mean input direction so unmarked items get the marked pre-activation
difference), against a steering vector, with wikitext KL off target; (5) the marked forcing of gender and generation
together on the kinship items, by quadrant. Models: Qwen2.5-0.5B, -0.5B-Instruct, -1.5B, -1.5B-Instruct. Arguments: model
[--smoke]."""
import sys
src = open("e659_relay_chain.py").read().replace('MODELS = {"qwen05": "Qwen/Qwen2.5-0.5B",', 'MODELS = {"qwen15": "Qwen/Qwen2.5-1.5B", "qwen05": "Qwen/Qwen2.5-0.5B",')
exec(src[src.index("import sys, os, time"): src.index("# ---- e622's catalogue")], globals())
ATT3 = ("gender", "generation", "number")
def batches(its):
    groups = collections.defaultdict(list)
    for n_, it in enumerate(its): groups[len(it["ids"])].append(n_)
    for Lq, idx in groups.items():
        for s0 in range(0, len(idx), BS): yield idx[s0:s0 + BS]
def states_all(its):
    """residual state after each block 0..L at each item's word position [n, L+1, D] (CPU)"""
    out = [None] * len(its)
    for ch in batches(its):
        cap = {}; hs = [layers[b].register_forward_hook(lambda m, a_, o, b=b: cap.__setitem__(b, (o[0] if isinstance(o, tuple) else o).float())) for b in range(L + 1)]
        try: model(torch.tensor([its[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1)
        finally: [h.remove() for h in hs]
        for k, n in enumerate(ch): out[n] = torch.stack([cap[b][k, its[n]["p"]] for b in range(L + 1)]).cpu()
    return torch.stack(out)
def mlp_inputs(its, b):
    """the MLP's (normalized) input at block b at each item's word position [n, D] (CPU)"""
    out = [None] * len(its)
    for ch in batches(its):
        cap = {}; h = layers[b].mlp.register_forward_pre_hook(lambda m, a_: cap.__setitem__("x", a_[0].float()))
        try: model(torch.tensor([its[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1)
        finally: h.remove()
        for k, n in enumerate(ch): out[n] = cap["x"][k, its[n]["p"]].cpu()
    return torch.stack(out)
# ---- the WDD vote on the MLP rows (cumulative over blocks), from discovery pairs; model moved off the GPU meanwhile
def omp_cpu(X, A, k=8, batch=128, ridge=1e-5):
    N, NA = X.shape[0], A.shape[0]; sel = torch.zeros(N, k, dtype=torch.long, device=X.device); eye = torch.eye(k, device=X.device)
    for s in range(0, N, batch):
        x = X[s:s + batch]; n = x.shape[0]; r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=X.device); taken = torch.zeros(n, NA, dtype=torch.bool, device=X.device)
        for step in range(k):
            corr = (r @ A.T).abs_().masked_fill_(taken, -1.0); pick = corr.argmax(-1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
            As = A[S]; G = As @ As.transpose(1, 2) + ridge * eye[:step + 1, :step + 1]; c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (c.transpose(1, 2) @ As)[:, 0]
        sel[s:s + n] = S; del taken, corr
    return sel
PAIRS = {}
for a in ATTRS:
    by = collections.defaultdict(dict)
    for it in ITEMS[a]:
        if it["disc"]: by[(it["lg"], it["pair"], it["t"])][it["side"]] = it
    PAIRS[a] = [(g[0], g[1]) for g in by.values() if 0 in g and 1 in g]
DIFF = {a: (states_all([p[1] for p in PAIRS[a]]) - states_all([p[0] for p in PAIRS[a]])) for a in ATTRS}   # pole 1 minus pole 0, [n, L+1, D]
Wrows = torch.cat([layers[b].mlp.down_proj.weight.detach().float().T.cpu() for b in range(L + 1)]); Wrows = Wrows / Wrows.norm(dim=1, keepdim=True).clamp_min(1e-12)
model.cpu(); torch.cuda.empty_cache(); Wg = Wrows.to(DEV); del Wrows
VOTE = {}
for a in ATTRS:
    VOTE[a] = {}
    for b in range(L + 1):
        X_ = DIFF[a][:, b].to(DEV); sel = omp_cpu(X_, Wg[: (b + 1) * DFF]); cnt = torch.bincount(sel.reshape(-1), minlength=(b + 1) * DFF); top = cnt.topk(10)
        VOTE[a][b] = [(int(i_) // DFF, int(i_) % DFF, float(c_) / X_.shape[0]) for i_, c_ in zip(top.indices.tolist(), top.values.tolist())]
del Wg; torch.cuda.empty_cache(); model.to(DEV); log("vote done")
def wdd_chain(a, thr=0.25):
    rows = []
    for b in range(L + 1):
        bb, r, sh = VOTE[a][b][0]
        if sh >= thr and (bb, r) not in rows: rows.append((bb, r))
    peak = max(range(L + 1), key=lambda b: VOTE[a][b][0][2]); single = VOTE[a][peak][0][:2]
    if single not in rows: rows.append(single)
    return rows, single
E622 = None
try:
    E622 = json.load(open(f"e622_catalogue_{CAT_OF[name]}.json"))["catalogue"] if name in CAT_OF else None
except FileNotFoundError: E622 = None
NEMB = model.get_input_embeddings().weight.shape[0]; PER = DFF + D
def e622_chain(a):
    if E622 is None: return None
    rows = []
    for lev in sorted(E622[a]["vote_by_block"], key=int):
        v = E622[a]["vote_by_block"][lev]; i_ = v["atom"] - NEMB; b, r = divmod(i_, PER)
        if v["type"] == 2 and v["share"] >= 0.25 and i_ >= 0 and r < DFF and (b, r) not in rows: rows.append((b, r))
    return rows
# ---- wikitext windows, loss, KL
import datasets
WT = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.split()) > 60]
WIN = []
for t in WT:
    ids = tok(t)["input_ids"]
    if len(ids) >= 128: WIN.append(ids[:128])
    if len(WIN) >= (4 if SMOKE else 12): break
WIN = torch.tensor(WIN, device=DEV); BASE = []
for s0 in range(0, WIN.shape[0], WB): BASE.append(model(WIN[s0:s0 + WB], use_cache=False).logits.float().log_softmax(-1).cpu().half())
def kl_off():
    kl, n = 0.0, 0
    for k, s0 in enumerate(range(0, WIN.shape[0], WB)):
        lp = model(WIN[s0:s0 + WB], use_cache=False).logits.float().log_softmax(-1); lb = BASE[k].to(DEV).float(); kl += float((lb.exp() * (lb - lp)).sum(-1).sum()); n += lp.shape[0] * lp.shape[1]; del lp, lb
    return kl / n
def W(b): return layers[b].mlp.down_proj.weight
class Negate:
    def __init__(self, rows): self.rows = list(dict.fromkeys(rows))
    def __enter__(self):
        self.saved = [(b, r, W(b)[:, r].clone()) for b, r in self.rows]
        for b, r, w in self.saved: W(b)[:, r] = -w
    def __exit__(self, *a_):
        for b, r, w in self.saved: W(b)[:, r] = w
POS = {}
def by_pole(a, heldonly=True):
    """held-out items of attribute a: share of each starting pole's items whose answer is the other pole, and the share answering pole 1"""
    its = [it for it in ITEMS[a] if not it["disc"]] if heldonly else ITEMS[a]; out = {0: [0, 0], 1: [0, 0]}; to1 = {0: [0, 0], 1: [0, 0]}
    for ch in batches(its):
        POS["p"] = torch.tensor([its[n]["p"] for n in ch], device=DEV)
        lg = model(torch.tensor([its[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1).logits[:, -1].float()
        for k, n in enumerate(ch):
            it = its[n]; pred = int(lg[k, it["cand"]].argmax()); s = it["side"]; out[s][0] += int(pred != s); out[s][1] += 1; to1[s][0] += int(pred == 1); to1[s][1] += 1
    return {f"flip_from_{s}": out[s][0] / max(out[s][1], 1) for s in (0, 1)} | {f"answer1_from_{s}": to1[s][0] / max(to1[s][1], 1) for s in (0, 1)}
# ==== SETUP END ====
# ---- per attribute: activations by pole, the chains
def acts(a, disc=True):
    its = [it for it in ITEMS[a] if it["disc"] == disc] if disc is not None else ITEMS[a]; return torch.stack([it["A"] for it in its]), torch.tensor([it["side"] for it in its])
CH = {}
for a in ATTRS:
    A_, s_ = acts(a); dm = A_[s_ == 1].mean(0) - A_[s_ == 0].mean(0); sd = A_.std(0) + 1e-6; dn = dm.abs() / sd
    wch, single = wdd_chain(a); M = len(wch)
    blk_score = dn.max(1).values; topb = blk_score.argsort(descending=True)[:M].tolist(); free = [(b, int(dn[b].argmax())) for b in sorted(topb)]
    naive = []
    for b, _ in wch:
        for r in dm[b].abs().argsort(descending=True).tolist():
            if (b, r) not in naive: naive.append((b, r)); break
    ach = torch.stack([A_[:, b, r] for b, r in wch], 1)   # [n, M]
    mag = {t: float(ach[s_ == t].abs().mean()) for t in (0, 1)}; marked = 1 if mag[1] >= mag[0] else 0
    per_row = [dict(row=f"{r}@b{b}", mean0=float(A_[s_ == 0, b, r].mean()), mean1=float(A_[s_ == 1, b, r].mean()), abs0=float(A_[s_ == 0, b, r].abs().mean()), abs1=float(A_[s_ == 1, b, r].abs().mean())) for b, r in wch]
    CH[a] = dict(wdd=wch, single=[single], e622=e622_chain(a), free=free, naive_in_wdd_blocks=naive, random=[[(b, int(rng.integers(DFF))) for b, _ in wch] for _ in range(2 if SMOKE else 3)],
                 marked=marked, mag=mag, mag_ratio=max(mag.values()) / max(min(mag.values()), 1e-9), per_row=per_row, vote_top={str(b): VOTE[a][b][:3] for b in range(L + 1)})
    e6 = CH[a]["e622"]; jac = (len(set(e6) & set(wch)) / len(set(e6) | set(wch))) if e6 else None
    log(f"{a}: WDD chain {[f'{r}@b{b}' for b, r in wch]} (e622 chain Jaccard {jac}); WDD-free chain {[f'{r}@b{b}' for b, r in free]}; |a| pole0 {mag[0]:.3f} pole1 {mag[1]:.3f} (marked pole {marked}, ratio {CH[a]['mag_ratio']:.1f})")
    CH[a]["e622_jaccard"] = jac
RES = dict(model=name, n_items={a: len(ITEMS[a]) for a in ATTRS}, n_kin=len(KIN), chains={a: {k: v for k, v in CH[a].items()} for a in ATTRS}, negation={}, composition={}, forcing={})
for a in ATTRS:
    RES["negation"][a] = {"none": by_pole(a)}
    for kind in ("single", "wdd", "e622", "free", "naive_in_wdd_blocks"):
        if CH[a][kind] is None: continue
        with Negate(CH[a][kind]): RES["negation"][a][kind] = by_pole(a) | {"kl": kl_off()}
    rr = []
    for rows in CH[a]["random"]:
        with Negate(rows): rr.append(by_pole(a))
    RES["negation"][a]["random"] = {k: float(np.mean([x[k] for x in rr])) for k in rr[0]}
    ng = RES["negation"][a]; log(f"{a} negation flips from pole0/pole1: " + "; ".join(f"{k} {ng[k]['flip_from_0']:.2f}/{ng[k]['flip_from_1']:.2f}" for k in ng))
# ---- composition by starting quadrant (kinship: index bit 1 = female, bit 2 = young)
def outcomes_by_q():
    lps, _ = forward_items(KIN); M_ = np.zeros((4, 4))
    for lp, it in zip(lps, KIN): M_[it["i"], int(lp.argmax()) ^ it["i"]] += 1
    return (M_ / np.maximum(M_.sum(1, keepdims=True), 1)).tolist()
G_, A2_ = CH["gender"], CH["generation"]; comp = {"none": outcomes_by_q()}
for k_, rows in (("both_single", G_["single"] + A2_["single"]), ("both_wdd", G_["wdd"] + A2_["wdd"]), ("both_free", G_["free"] + A2_["free"])):
    with Negate(rows): comp[k_] = outcomes_by_q()
RES["composition"] = comp; log("composition by quadrant (rows = starting item king/queen/prince/princess; cols = kept/gender/generation/both): " + "; ".join(f"{k} both-flip {[round(r[3], 2) for r in v]}" for k, v in comp.items()))
# ---- forcing the marked pole
class Pulse:
    """oracle: each chain row's activation at the item's word position set to its mean on the marked pole"""
    def __init__(self, rows, vals): self.by = collections.defaultdict(list); [self.by[b].append((r, v)) for (b, r), v in zip(rows, vals)]; self.h = []
    def __enter__(self):
        for b, rv in self.by.items():
            idx = torch.tensor([r for r, _ in rv], device=DEV); val = torch.tensor([v for _, v in rv], device=DEV)
            def hk(m, args, idx=idx, val=val):
                x = args[0].clone(); bi = torch.arange(x.shape[0], device=DEV); p = POS["p"]
                for k_ in range(idx.numel()): x[bi, p, idx[k_]] = val[k_]
                return (x,) + tuple(args[1:])
            self.h.append(layers[b].mlp.down_proj.register_forward_pre_hook(hk))
        return self
    def __exit__(self, *a_): [h.remove() for h in self.h]
class Floor:
    """every position: a <- s * max(|a|, f * tau), s the marked pole's sign, tau the mean |a| on the marked pole"""
    def __init__(self, rows, sg, tau, f): self.by = collections.defaultdict(list); [self.by[b].append((r, s, t)) for (b, r), s, t in zip(rows, sg, tau)]; self.f = f; self.h = []
    def __enter__(self):
        for b, rv in self.by.items():
            idx = torch.tensor([r for r, _, _ in rv], device=DEV); sg = torch.tensor([s for _, s, _ in rv], device=DEV); tau = torch.tensor([t for _, _, t in rv], device=DEV) * self.f
            def hk(m, args, idx=idx, sg=sg, tau=tau): x = args[0].clone(); x[..., idx] = sg * torch.maximum(x[..., idx].abs(), tau); return (x,) + tuple(args[1:])
            self.h.append(layers[b].mlp.down_proj.register_forward_pre_hook(hk))
        return self
    def __exit__(self, *a_): [h.remove() for h in self.h]
class Shift:
    """Gemini's weight-space input shift: gate and up rows of each chain row moved along the class-mean MLP input direction"""
    def __init__(self, deltas): self.deltas = deltas
    def __enter__(self):
        self.saved = []
        for (b, r), dg, du in self.deltas:
            mlp = layers[b].mlp; self.saved.append((b, r, mlp.gate_proj.weight[r].clone(), mlp.up_proj.weight[r].clone())); mlp.gate_proj.weight[r] += dg; mlp.up_proj.weight[r] += du
        return self
    def __exit__(self, *a_):
        for b, r, g, u in self.saved: layers[b].mlp.gate_proj.weight[r] = g; layers[b].mlp.up_proj.weight[r] = u
class Steer:
    def __init__(self, b, v): self.b, self.v = b, v
    def __enter__(self):
        def hk(m, args, out): return (out[0] + self.v,) + tuple(out[1:]) if isinstance(out, tuple) else out + self.v
        self.h = layers[self.b].register_forward_hook(hk); return self
    def __exit__(self, *a_): self.h.remove()
def shift_deltas(a, k):
    rows = CH[a]["wdd"]; mk = CH[a]["marked"]; out = []; its = [it for it in ITEMS[a] if it["disc"]]; sides = torch.tensor([it["side"] for it in its])   # discovery items only
    for b in sorted(set(b for b, _ in rows)):
        Xin = mlp_inputs(its, b); mu = Xin.mean(0); mu_hat = mu / mu.norm(); c = float((Xin @ mu_hat).mean())
        for bb, r in rows:
            if bb != b: continue
            g = layers[b].mlp.gate_proj.weight[r].detach().float().cpu(); u = layers[b].mlp.up_proj.weight[r].detach().float().cpu()
            pg, pu = Xin @ g, Xin @ u; dg = float(pg[sides == mk].mean() - pg[sides != mk].mean()); du = float(pu[sides == mk].mean() - pu[sides != mk].mean())
            out.append(((b, r), (k * dg / c * mu_hat).to(DEV), (k * du / c * mu_hat).to(DEV)))
    return out
ALPHAS = [0.25, 1, 4] if SMOKE else [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]; KS = [1, 2] if SMOKE else [0.5, 1, 2, 4]
def unmarked_moved(r, mk): return r[f"flip_from_{1 - mk}"]
for a in ATT3:
    rows = CH[a]["wdd"]; mk = CH[a]["marked"]; A_, s_ = acts(a); bm = CH[a]["single"][0][0]
    mean_m = [float(A_[s_ == mk, b, r].mean()) for b, r in rows]; sg = [float(np.sign(v)) or 1.0 for v in mean_m]; tau = [float(A_[s_ == mk, b, r].abs().mean()) for b, r in rows]
    F_ = {"base": by_pole(a)}
    with Pulse(rows, mean_m): F_["pulse_word_position"] = by_pole(a)
    for f in ((1.0,) if SMOKE else (0.5, 1.0)):
        with Floor(rows, sg, tau, f): F_[f"floor_{f}"] = by_pole(a) | {"kl": kl_off()}
    for k in KS:
        with Shift(shift_deltas(a, k)): F_[f"shift_{k}"] = by_pole(a) | {"kl": kl_off()}
    disc = [it for it in ITEMS[a] if it["disc"]]; Xs = states_all(disc)[:, bm]; sd_ = torch.tensor([it["side"] for it in disc]); v = (Xs[sd_ == mk].mean(0) - Xs[sd_ != mk].mean(0)).to(DEV)
    for al in ALPHAS:
        with Steer(bm, al * v): F_[f"steer_{al}"] = by_pole(a) | {"kl": kl_off()}
    RES["forcing"][a] = dict(marked=mk, results=F_)
    log(f"{a} forcing the marked pole {mk}: unmarked items moved: " + "; ".join(f"{k} {unmarked_moved(v_, mk):.2f}" + (f" (KL {v_['kl']:.4f})" if 'kl' in v_ else "") for k, v_ in F_.items()))
# ---- composition by marked forcing: gender and generation pulses together, and shifts together, on the kinship items
def kin_pos_outcomes(ctx):
    M_ = np.zeros((4, 4))
    with ctx:
        for ch in batches(KIN):
            POS["p"] = torch.tensor([KIN[n]["p"] for n in ch], device=DEV); lg = model(torch.tensor([KIN[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1).logits[:, -1].float()
            for k, n in enumerate(ch): it = KIN[n]; M_[it["i"], int(lg[k, it["cand"]].argmax()) ^ it["i"]] += 1
    return (M_ / np.maximum(M_.sum(1, keepdims=True), 1)).tolist()
class Multi:
    def __init__(self, *cs): self.cs = cs
    def __enter__(self): [c.__enter__() for c in self.cs]; return self
    def __exit__(self, *a_): [c.__exit__(*a_) for c in reversed(self.cs)]
def pulse_of(a): A_, s_ = acts(a); mk = CH[a]["marked"]; return Pulse(CH[a]["wdd"], [float(A_[s_ == mk, b, r].mean()) for b, r in CH[a]["wdd"]])
comp["pulse_gender"] = kin_pos_outcomes(pulse_of("gender")); comp["pulse_generation"] = kin_pos_outcomes(pulse_of("generation")); comp["pulse_both"] = kin_pos_outcomes(Multi(pulse_of("gender"), pulse_of("generation")))
comp["shift_both_2"] = kin_pos_outcomes(Multi(Shift(shift_deltas("gender", 2)), Shift(shift_deltas("generation", 2))))
log("marked-forcing composition, king's row (kept/gender/generation/both): " + "; ".join(f"{k} {[round(x, 2) for x in comp[k][0]]}" for k in ("pulse_gender", "pulse_generation", "pulse_both", "shift_both_2")))
RES["composition"] = comp
# ---- verdicts
ng = RES["negation"]; mkd = {a: CH[a]["marked"] for a in ATTRS}
def from_marked(a, k): return ng[a][k][f"flip_from_{mkd[a]}"]
def from_unmarked(a, k): return ng[a][k][f"flip_from_{1 - mkd[a]}"]
fr = RES["forcing"]; best_force = lambda a: max((k for k in fr[a]["results"] if k.startswith(("floor", "shift", "pulse"))), key=lambda k: unmarked_moved(fr[a]["results"][k], fr[a]["marked"]))
RES["verdicts"] = dict(M1=all(CH[a]["marked"] == 1 for a in ATT3), M2=all(CH[a]["mag_ratio"] >= 2 for a in ATT3), M3=all(from_marked(a, "wdd") >= 3 * max(from_unmarked(a, "wdd"), 0.01) for a in ATT3),
                       M4=comp["both_wdd"][3][3] >= 3 * max(comp["both_wdd"][0][3], comp["both_wdd"][1][3], comp["both_wdd"][2][3], 0.01),
                       M5=all(unmarked_moved(fr[a]["results"]["pulse_word_position"], fr[a]["marked"]) >= 0.3 for a in ATT3),
                       M6=sum(ng[a]["free"]["flip_from_0"] + ng[a]["free"]["flip_from_1"] >= ng[a]["wdd"]["flip_from_0"] + ng[a]["wdd"]["flip_from_1"] for a in ATTRS) >= 3,
                       M7=comp["pulse_both"][0][3] >= 0.25)
summ = (f"[{name}] markedness: marked pole by |a| " + ", ".join(f"{a} {CH[a]['marked']} (ratio {CH[a]['mag_ratio']:.1f})" for a in ATT3) + "; WDD-chain negation flips from marked/unmarked " + ", ".join(f"{a} {from_marked(a, 'wdd'):.2f}/{from_unmarked(a, 'wdd'):.2f}" for a in ATT3)
        + f"; both-chains double flips by quadrant {[round(r[3], 2) for r in comp['both_wdd']]}; marked pulse at the word moves unmarked " + ", ".join(f"{a} {unmarked_moved(fr[a]['results']['pulse_word_position'], fr[a]['marked']):.2f}" for a in ATT3) + f"; king to princess by both pulses {comp['pulse_both'][0][3]:.2f}; verdicts {RES['verdicts']} ({time.time() - t0:.0f}s)")
RES["summary"] = summ; RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"results/e661_markedness_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print(summ, flush=True); print("done", flush=True)
