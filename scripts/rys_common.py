"""Shared helpers for e656-e658 (session 143): RYS-style layer repetition (Ng 2024; a contiguous block [i, j) run a
second time with no weight change, sequence 0..j-1, i..j-1, j..N-1) on small models, by putting the same layer modules
into the stack twice (transformers 5.x: GPT-NeoX loops over self.layers; Qwen2 and Llama slice by num_hidden_layers and
Qwen2 reads layer_types per index, so both are updated), plus hooks that edit what the repeated block sees on its second
traversal, and the probes (wikitext loss, few-shot single-digit addition, e560's capitals, weekday and month offsets).
Builds on bx_common (dictionary, OMP, rotation)."""
from bx_common import *
import collections
CAPS = None  # filled from e654_peel's list on first use


def stack(M):
    return M["model"].gpt_neox if M["fam"] == "neox" else M["model"].model


class Repeat:
    """context manager: run the model with layer sequence seq (a list of original layer indices)"""
    def __init__(self, M, seq): self.M, self.seq = M, seq
    def __enter__(self):
        st = stack(self.M); cfg = self.M["model"].config; self.orig = st.layers; self.n0 = cfg.num_hidden_layers; self.lt0 = getattr(cfg, "layer_types", None)
        st.layers = torch.nn.ModuleList([self.orig[k] for k in self.seq]); cfg.num_hidden_layers = len(self.seq)
        if self.lt0 is not None: cfg.layer_types = [self.lt0[k] for k in self.seq]
        return self
    def __exit__(self, *a):
        st = stack(self.M); cfg = self.M["model"].config; st.layers = self.orig; cfg.num_hidden_layers = self.n0
        if self.lt0 is not None: cfg.layer_types = self.lt0


def rys_seq(N, i, j): return list(range(0, j)) + list(range(i, j)) + list(range(j, N))


def _hs(args, kwargs): return args[0] if args else kwargs["hidden_states"]
def _set_hs(args, kwargs, x):
    if args: return (x,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs); kwargs["hidden_states"] = x; return args, kwargs
def _out(o): return o[0] if isinstance(o, tuple) else o
def _set_out(o, x): return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x


class Junction:
    """hooks for a repeated block [i, j): on the second traversal the block sees entry(x_i, x_j, mlp1) instead of x_j, and its
    increment is added to x_j (out = x_j + B(entry) - entry); entry=None is the plain repeat. mlp1 = the summed MLP writes of
    the block's first traversal. Works for one forward at a time (counts calls)."""
    def __init__(self, M, i, j, entry=None):
        self.M, self.i, self.j, self.entry = M, i, j, entry; self.h = []
    def __enter__(self):
        L = self.M["layers"]; self.calls = {}; self.mlp1 = None; self.xi = self.xj = self.ent = None
        def pre_i(mod, args, kwargs):
            c = self.calls.get("i", 0) + 1; self.calls["i"] = c; x = _hs(args, kwargs)
            if c == 1: self.xi = x.detach(); return None
            if c == 2:
                self.xj = x.detach()
                if self.entry is None: self.ent = x; return None
                self.ent = self.entry(self.xi, self.xj, self.mlp1); return _set_hs(args, kwargs, self.ent)
            return None
        def post_last(mod, args, kwargs, out):
            c = self.calls.get("last", 0) + 1; self.calls["last"] = c
            if c == 2 and self.entry is not None: return _set_out(out, self.xj + (_out(out) - self.ent))
            return None
        def mlp_hook(mod, args, out):
            if self.calls.get("i", 0) == 1 and self.calls.get("last", 0) == 0:
                y = out[0] if isinstance(out, tuple) else out; self.mlp1 = y.detach() if self.mlp1 is None else self.mlp1 + y.detach()
            return None
        self.h.append(L[self.i].register_forward_pre_hook(pre_i, with_kwargs=True)); self.h.append(L[self.j - 1].register_forward_hook(post_last, with_kwargs=True))
        for b in range(self.i, self.j): self.h.append(L[b].mlp.register_forward_hook(mlp_hook))
        return self
    def reset(self): self.calls = {}; self.mlp1 = None; self.xi = self.xj = self.ent = None
    def __exit__(self, *a):
        for h in self.h: h.remove()


class Inject:
    """single pass: replace the output of block j-1 by x_j + add(x_i, x_j) (x_i = the input of block i). The virtual repeat."""
    def __init__(self, M, i, j, add): self.M, self.i, self.j, self.add = M, i, j, add; self.h = []
    def __enter__(self):
        L = self.M["layers"]; self.xi = None
        def pre_i(mod, args, kwargs): self.xi = _hs(args, kwargs).detach(); return None
        def post_last(mod, args, kwargs, out): x = _out(out); return _set_out(out, x + self.add(self.xi, x))
        self.h.append(L[self.i].register_forward_pre_hook(pre_i, with_kwargs=True)); self.h.append(L[self.j - 1].register_forward_hook(post_last, with_kwargs=True)); return self
    def __exit__(self, *a):
        for h in self.h: h.remove()


def lm_loss(M, W, per_batch=None):
    """mean next-token cross-entropy over windows W (batches of 8); per_batch: a callable run before each forward (to reset hooks)"""
    tot, n = 0.0, 0
    for s0 in range(0, W.shape[0], 8):
        if per_batch: per_batch()
        ids = W[s0:s0 + 8]; lg = M["model"](ids, use_cache=False).logits.float()
        tot += float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")); n += ids[:, 1:].numel()
    return tot / n


def probes(M):
    """(name, prompt, answer) items whose answer's first token is stable under concatenation; returns buckets of same-length prompts"""
    global CAPS
    if CAPS is None:
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e654_peel.py")).read(); CAPS = eval(src[src.index("CAPS = [") + 7:src.index("]\nargs") + 1])
    tok = M["tok"]; items = []
    shots = "2+3=5\n4+1=5\n1+6=7\n3+3=6\n"
    for a in range(10):
        for b in range(10 - a):
            if f"{a}+{b}=" in shots: continue
            items.append(("addition", shots + f"{a}+{b}=", f"{a + b}"))
    for c, cap in CAPS: items.append(("capitals", f"The capital of {c} is", " " + cap))
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]; months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    words = {1: "One", 2: "Two", 3: "Three"}
    dshot = "Two days after Monday is Wednesday.\nOne day after Friday is Saturday.\nThree days after Sunday is Wednesday.\n"; mshot = "Two months after January is March.\nOne month after June is July.\nThree months after October is January.\n"
    for k in (1, 2, 3):
        for x, d in enumerate(days):
            q = f"{words[k]} day{'s' if k > 1 else ''} after {d} is"
            if q not in dshot: items.append(("weekdays", dshot + q, " " + days[(x + k) % 7]))
        for x, mo in enumerate(months):
            q = f"{words[k]} month{'s' if k > 1 else ''} after {mo} is"
            if q not in mshot: items.append(("months", mshot + q, " " + months[(x + k) % 12]))
    out = []
    for kind, p, a in items:
        pi = tok(p)["input_ids"]; fi = tok(p + a)["input_ids"]
        if fi[:len(pi)] != pi or len(fi) <= len(pi): continue
        out.append((kind, pi, fi[len(pi)]))
    buckets = collections.defaultdict(list)
    for n_, (kind, pi, t) in enumerate(out): buckets[len(pi)].append((n_, kind, pi, t))
    return out, buckets


def probe_acc(M, P, per_batch=None):
    """per-item correctness (next-token argmax == the answer's first token) for the probe buckets P = probes(M)"""
    items, buckets = P; ok = np.zeros(len(items), dtype=bool)
    for L_, b in buckets.items():
        for s0 in range(0, len(b), 32):
            if per_batch: per_batch()
            chunk = b[s0:s0 + 32]; ids = torch.tensor([c[2] for c in chunk], device=DEV); pred = M["model"](ids, use_cache=False).logits[:, -1].argmax(-1).tolist()
            for c, p_ in zip(chunk, pred): ok[c[0]] = (p_ == c[3])
    return ok


def kinds(P): return np.array([k for k, _, _ in P[0]])


def stopset(M, P, n=200):
    """e654's function words: the n most frequent wikitext-2 train tokens, minus every probe answer's first token"""
    V = M["WU"].shape[0]; uni = np.bincount(np.concatenate([np.array(x) for x in train_token_ids(M)]), minlength=V); ans = set(t for _, _, t in P[0])
    m = torch.zeros(V, dtype=torch.bool, device=DEV); m[torch.tensor([int(i) for i in np.argsort(-uni)[:n] if int(i) not in ans], device=DEV)] = True; return m


def probe_acc_both(M, P, stop, per_batch=None):
    """per-item correctness raw and with the function words masked (a repeat's gain beyond suppressing hedges)"""
    items, buckets = P; ok = np.zeros(len(items), dtype=bool); okm = np.zeros(len(items), dtype=bool)
    for L_, b in buckets.items():
        for s0 in range(0, len(b), 32):
            if per_batch: per_batch()
            chunk = b[s0:s0 + 32]; ids = torch.tensor([c[2] for c in chunk], device=DEV); lg = M["model"](ids, use_cache=False).logits[:, -1].float()
            pr = lg.argmax(-1).tolist(); pm = lg.masked_fill(stop[None], -1e9).argmax(-1).tolist()
            for c, a_, b_ in zip(chunk, pr, pm): ok[c[0]] = (a_ == c[3]); okm[c[0]] = (b_ == c[3])
    return ok, okm
