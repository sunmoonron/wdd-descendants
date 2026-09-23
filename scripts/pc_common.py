"""Shared helpers for phase 2 (e357 onward): eager-attention loading, mean-ablation of attention heads and MLP
neurons by forward pre-hooks, induction and IOI data, attribution ranking of neurons, and the unit statistics
(pairwise interaction matrix, spectral clustering, adjusted Rand index, set non-additivity)."""
from wdd_common import *
import math, random
def unit(v): return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)

def load_eager(name, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf, fam = MODELS[name]
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, revision=revision, dtype=torch.float32, attn_implementation="eager", device_map=DEV).eval()
    return model, tok, fam

def attn_width(arch, l):
    w = arch.attn_lin(l).weight
    return w.shape[0] if arch.fam == "gpt2" else w.shape[1]

def capture_means(model, arch, ids):
    """per-layer mean over the batch, per position, of the attention-output input and the MLP down-projection input"""
    ca, cm = {}, {}
    hs = [arch.attn_lin(l).register_forward_pre_hook((lambda l_: lambda m, a: ca.__setitem__(l_, a[0].detach().float().mean(0)))(l)) for l in range(arch.NB)]
    hs += [arch.mlp_lin(l).register_forward_pre_hook((lambda l_: lambda m, a: cm.__setitem__(l_, a[0].detach().float().mean(0)))(l)) for l in range(arch.NB)]
    try: lg = model(ids).logits.float()
    finally: [h.remove() for h in hs]
    return ca, cm, lg

def ablate(model, arch, ids, heads=(), neurons=(), MA=None, MM=None):
    """mean-ablate the given heads (l, h) and neurons (l, i); returns logits"""
    NH = arch.NH; byl = {}
    for (l, h) in heads: byl.setdefault(("a", l), []).append(h)
    for (l, i) in neurons: byl.setdefault(("m", l), []).append(i)
    hs = []
    for (kind, l), items in byl.items():
        if kind == "a":
            def pre(m, a, l=l, items=items):
                x = a[0].clone(); B, T, W = x.shape; hd = W // NH; xv = x.view(B, T, NH, hd); mv = MA[l].to(x.dtype).view(T, NH, hd)
                for h in items: xv[:, :, h, :] = mv[None, :, h, :]
                return (x,) + tuple(a[1:])
            hs.append(arch.attn_lin(l).register_forward_pre_hook(pre))
        else:
            idx = torch.tensor(sorted(set(items)), device=DEV)
            def pre(m, a, l=l, idx=idx):
                x = a[0].clone(); x[:, :, idx] = MM[l].to(x.dtype)[None][:, :, idx]; return (x,) + tuple(a[1:])
            hs.append(arch.mlp_lin(l).register_forward_pre_hook(pre))
    try: lg = model(ids).logits.float()
    finally: [h.remove() for h in hs]
    return lg

def token_loss(lg, ids):
    lp = torch.log_softmax(lg[:, :-1], -1); return -lp.gather(2, ids[:, 1:, None])[..., 0].flatten()

def induction_batch(tok, c, n=32, half=128, seed=0):
    g = torch.Generator().manual_seed(seed); pool = torch.unique(c.s["eval_ids"].flatten()); r = pool[torch.randint(0, len(pool), (n, half), generator=g)]
    seq = torch.cat([r, r], 1); off = 0
    if tok.bos_token_id is not None: seq = torch.cat([torch.full((n, 1), tok.bos_token_id), seq], 1); off = 1
    return seq.to(DEV), off, half

def induction_loss(lg, ids, off, half, second=True):
    pos = torch.arange(off + (half if second else 0), off + (2 * half if second else half) - 1, device=DEV); lp = torch.log_softmax(lg[:, pos], -1)
    return -lp.gather(2, ids[:, pos + 1][..., None])[..., 0].flatten()

def attention_scores(model, ids, off, half, NB, NH):
    out = model(ids, output_attentions=True); att = out.attentions; j = torch.arange(1, half - 1, device=DEV); q = off + half + j; k = off + j + 1
    pref = torch.stack([att[l][:, :, q, k].mean((0, 2)) for l in range(NB)]); T = ids.shape[1]; qq = torch.arange(1, T, device=DEV)
    prev = torch.stack([att[l][:, :, qq, qq - 1].mean((0, 2)) for l in range(NB)]); del out, att; return pref.float(), prev.float()

def neuron_attr(model, arch, ids, metric_fn, MM):
    """first-order estimate of each neuron's mean-ablation effect on the mean metric: sum over tokens of (mean - act) * dM/dact"""
    acts = {}
    def mk(l):
        def pre(m, a):
            a[0].retain_grad(); acts[l] = a[0]; return None
        return pre
    with torch.enable_grad():
        hs = [arch.mlp_lin(l).register_forward_pre_hook(mk(l)) for l in range(arch.NB)]
        try:
            lg = model(ids).logits.float(); M = metric_fn(lg).mean(); M.backward()
        finally: [h.remove() for h in hs]
    sc = {l: ((MM[l][None] - a.detach().float()) * a.grad.float()).sum((0, 1)) for l, a in acts.items()}
    model.zero_grad(set_to_none=True); del acts; return sc

def ari(a, b):
    from math import comb
    a = list(a); b = list(b); n = len(a); ca = {}; cb = {}; cab = {}
    for x, y in zip(a, b): ca[x] = ca.get(x, 0) + 1; cb[y] = cb.get(y, 0) + 1; cab[(x, y)] = cab.get((x, y), 0) + 1
    s_ab = sum(comb(v, 2) for v in cab.values()); s_a = sum(comb(v, 2) for v in ca.values()); s_b = sum(comb(v, 2) for v in cb.values()); tot = comb(n, 2)
    exp = s_a * s_b / tot if tot else 0; mx = 0.5 * (s_a + s_b)
    return (s_ab - exp) / (mx - exp) if mx != exp else 0.0

def spectral(W, kmax=4, seed=0):
    n = W.shape[0]; W = W.clone().float(); W.fill_diagonal_(0); d = W.sum(1).clamp_min(1e-9); Dm = torch.diag(d.rsqrt()); Ls = torch.eye(n) - Dm @ W @ Dm; ev, V = torch.linalg.eigh(Ls)
    gaps = [(ev[k] - ev[k - 1]).item() for k in range(2, kmax + 1)]; k = 2 + int(torch.tensor(gaps).argmax()); X = V[:, :k]; X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
    g = torch.Generator().manual_seed(seed); best = None
    for rep in range(10):
        C = X[torch.randperm(n, generator=g)[:k]]
        for it in range(50): lab = torch.cdist(X, C).argmin(1); C = torch.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        inertia = ((X - C[lab]) ** 2).sum().item()
        if best is None or inertia < best[0]: best = (inertia, lab.clone())
    return best[1].tolist(), k, ev[:kmax + 1].tolist()

def pairwise(model, arch, ids, metric, base, comps, kind, MA, MM, E1):
    """interaction |E_ij - E_i - E_j| relative to the mean single norm, and the signed scalar version, over all pairs"""
    n = len(comps); I = torch.zeros(n, n); Sg = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            lg = ablate(model, arch, ids, heads=[comps[i], comps[j]] if kind == "head" else (), neurons=[comps[i], comps[j]] if kind == "neuron" else (), MA=MA, MM=MM); Eij = metric(lg) - base; lin = E1[comps[i]] + E1[comps[j]]
            I[i, j] = I[j, i] = ((Eij - lin).norm() / (0.5 * (E1[comps[i]].norm() + E1[comps[j]].norm())).clamp_min(1e-9)).item()
            Sg[i, j] = Sg[j, i] = ((Eij.mean() - lin.mean()) / (0.5 * (E1[comps[i]].mean().abs() + E1[comps[j]].mean().abs())).clamp_min(1e-9)).item()
    return I, Sg

def block_stats(I, Sg, ntop):
    n = I.shape[0]; iu = torch.triu_indices(n, n, 1); top = (iu[0] < ntop) & (iu[1] < ntop); rnd = (iu[0] >= ntop) & (iu[1] >= ntop); cross = ~top & ~rnd; v = I[iu[0], iu[1]]; s = Sg[iu[0], iu[1]]
    f = lambda m: float(v[m].mean()) if m.any() else float("nan"); g = lambda m: float(s[m].mean()) if m.any() else float("nan")
    return dict(I_top=f(top), I_random=f(rnd), I_cross=f(cross), signed_top=g(top), signed_random=g(rnd), superadditive_fraction_top=float((s[top] > 0.2).float().mean()) if top.any() else float("nan"), subadditive_fraction_top=float((s[top] < -0.2).float().mean()) if top.any() else float("nan"))

def set_nonadditivity(model, arch, ids, metric, base, members, kind, MA, MM, E1):
    lg = ablate(model, arch, ids, heads=members if kind == "head" else (), neurons=members if kind == "neuron" else (), MA=MA, MM=MM); ES = metric(lg) - base; lin = sum(E1[m] for m in members)
    return dict(nonadditivity=((ES - lin).norm() / sum(E1[m].norm() for m in members).clamp_min(1e-9)).item(), joint_over_sum=(ES.mean() / lin.mean()).item() if abs(lin.mean().item()) > 1e-9 else float("nan"), joint_effect=ES.mean().item())

NAMES = ["Mary", "John", "Tom", "James", "Dan", "Sid", "Martin", "Amy", "Alex", "Anna", "Chris", "Kate", "Paul", "Mark", "Sarah", "Lisa", "Michael", "David", "Robert", "Laura", "Emma", "Ryan", "Jack", "Ben", "Lucy", "Adam", "Rachel", "Peter", "Susan", "Daniel", "George", "Emily", "Kevin", "Steve", "Brian", "Linda", "Eric", "Karen", "Jane", "Joe"]
PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "house", "station", "park", "beach", "library", "market", "church", "museum"]
OBJECTS = ["drink", "book", "ring", "bone", "kiss", "snack", "computer", "apple", "flower", "gift", "letter", "ball", "key", "hat"]
TEMPLATES = ["When {X} and {Y} went to the {P}, {S} gave a {O} to", "After {X} and {Y} went to the {P}, {S} gave a {O} to", "While {X} and {Y} were working at the {P}, {S} gave a {O} to", "Then, {X} and {Y} had a long day at the {P}. {S} gave a {O} to"]

def single_token(tok, w):
    ids = tok.encode(" " + w, add_special_tokens=False); return ids[0] if len(ids) == 1 else None

def ioi_groups(tok, n_per=32, seed=0):
    rnd = random.Random(seed); names = [w for w in NAMES if single_token(tok, w) is not None]; places = [w for w in PLACES if single_token(tok, w) is not None]; objs = [w for w in OBJECTS if single_token(tok, w) is not None]; groups = []
    for t in TEMPLATES:
        for order in ("ABBA", "BABA"):
            rows = {}
            for _ in range(n_per * 4):
                A, B = rnd.sample(names, 2); X, Y = (A, B) if order == "ABBA" else (B, A)
                ids = tok.encode(t.format(X=X, Y=Y, S=B, P=rnd.choice(places), O=rnd.choice(objs)), add_special_tokens=False)
                if tok.bos_token_id is not None: ids = [tok.bos_token_id] + ids
                rows.setdefault(len(ids), []).append((ids, single_token(tok, A), single_token(tok, B)))
            L_ = max(rows, key=lambda k: len(rows[k])); sel = rows[L_][:n_per]
            groups.append((torch.tensor([r[0] for r in sel], device=DEV), torch.tensor([r[1] for r in sel], device=DEV), torch.tensor([r[2] for r in sel], device=DEV)))
    return groups, len(names)
