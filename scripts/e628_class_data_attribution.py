"""e628 (session 123): class-based data attribution with a causal handle. A word's class, the contexts where its row writes
above the floor, is the dictionary's own attribution of the feature to data, free of influence functions. Pythia-160m at
step 8000 is continued for 2,500 steps of 8 windows on a stream of Pile windows, and the rows the real run recruits between
steps 8000 and 10000 (words at 10000 that are not words at 8000, on a held-out eval set, blocks up to b) are split into two
groups A and B. Four streams: the full stream; the stream with every window containing a specific bigram of an A class
removed; the same for B; and the same number of random windows removed. Measured at the end for every recruit: whether
it is a word, its class size and its largest projection ratio against the floor; the recruitment rate of A and of B
under each stream; substitutes (another row whose class overlaps the recruit's reference class by Jaccard 0.5 or more);
the validation loss. Pre-registered in e628_prereg.json: D1 (0.4) a quarter or more of the recruits are words after the
full-stream continuation; D2 (0.5) removing A's contexts halves A's recruitment relative to the full stream while B's
stays within 20%, and symmetrically; D3 (0.5) random removal of the same number of windows changes neither; D4 (0.3)
removed classes get substitutes at a lower rate than under the full stream. Arguments: condition [--smoke]; conditions
full, removeA, removeB, random."""
import sys, os, time, copy, math, json, collections; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
from ma_common import out_of, block_states
from datasets import load_dataset
import wdd_common
COND = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); torch.set_grad_enabled(False); name = "pythia160"; b = 6; T = 256; STEPS = 6 if SMOKE else 2500; LR = 5e-5; NEV = 16 if SMOKE else 64; MAXW = 2000 if SMOKE else 45000
m8, tok, fam = load_model(name, revision="step8000"); arch = Arch(m8, fam); D = arch.D; DFF = arch.DFF; assert fam == "neox"
m10, _, _ = load_model(name, revision="step10000")
ds = load_dataset("NeelNanda/pile-10k", split="train")
def windows(nwin, skipdocs=0, maxdocs=None):
    wins, buf = [], []
    for k, ex in enumerate(ds):
        if k < skipdocs: continue
        if maxdocs is not None and k >= maxdocs: break
        buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
        while len(buf) >= T + 1 and len(wins) < nwin: wins.append(buf[:T + 1]); buf = buf[T + 1:]
        if len(wins) >= nwin: break
    return torch.tensor(wins, device=DEV)
TR = windows(MAXW, maxdocs=9000); EV = windows(NEV, skipdocs=9200); log(f"{name} step 8000, condition {COND}: {TR.shape[0]} training windows, {EV.shape[0]} eval windows ({time.time() - t0:.0f}s)")
def ce(lg, x): return torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
def val_loss(m):
    tot = 0.0
    for s0 in range(0, EV.shape[0], 8): x = EV[s0:s0 + 8]; tot += float(ce(m(x).logits.float(), x)) * x.shape[0]
    return tot / EV.shape[0]
def vocab(m):
    """words (top-256 rows by usage over blocks 0..b) and every row's class positions on the eval set, read at block b; returns words, classes (row -> positions), the ratio matrix's per-row max and the keep index"""
    X = block_states(m, Arch(m, fam), EV[:, :T], [b])[b].reshape(-1, D); keep = ~sinkmask(X); U = unitr(X[keep] - X[keep].mean(0)); A, _ = rows_of(Arch(m, fam), b); st = stats(U, A, K); words = set(torch.nonzero(wordset(st["usage"]))[:, 0].tolist()); R = st["ratio"].float(); kidx = torch.nonzero(keep)[:, 0]
    over = R > 1; nclass = over.sum(0); rmax = R.max(0).values
    return words, over, nclass, rmax, kidx
W8, O8, N8, M8_, K8 = vocab(m8); W10, O10, N10, M10_, K10 = vocab(m10); del m10; torch.cuda.empty_cache()
recruits = sorted(w for w in W10 - W8 if int(N10[w]) >= 10); log(f"vocabulary: {len(W8)} words at 8000, {len(W10)} at 10000; {len(W10 - W8)} recruits, {len(recruits)} with a class of 10 or more positions at 10000 ({time.time() - t0:.0f}s)")
# ---- each recruit's class contexts as bigrams on the eval set, kept if specific in the training stream
flat = EV[:, :T].reshape(-1); pos_tok = lambda i: (int(flat[(i // (T - 1)) * T + (i % (T - 1))]), int(flat[(i // (T - 1)) * T + (i % (T - 1)) + 1]))   # (previous token, token) at state position i % (T-1) + 1 of window i // (T-1)
def bigrams_of(w):
    idx = K10[O10[:, w]].tolist(); c = collections.Counter(pos_tok(i) for i in idx); return c
trn = TR[:, :T]; prev, cur = trn[:, :-1].reshape(-1), trn[:, 1:].reshape(-1); key = prev * 100000 + cur; win_of = torch.arange(trn.shape[0], device=DEV).repeat_interleave(T - 1)
def windows_with(bgs):
    """indices of training windows containing any of the bigrams"""
    if not bgs: return torch.zeros(0, dtype=torch.long, device=DEV)
    kk = torch.tensor([p * 100000 + c for p, c in bgs], device=DEV); hit = torch.isin(key, kk); return torch.unique(win_of[hit])
SIG = {}
for w in recruits:
    c = bigrams_of(w); spec = []
    for bg, n in c.most_common(12):
        share = windows_with([bg]).numel() / trn.shape[0]
        if share <= 0.02: spec.append(bg)
    SIG[w] = spec
usable = [w for w in recruits if len(SIG[w]) >= 2]; g = torch.Generator().manual_seed(1); perm = torch.randperm(len(usable), generator=g).tolist(); A_rows = sorted(usable[i] for i in perm[:len(usable) // 2]); B_rows = sorted(usable[i] for i in perm[len(usable) // 2:])
drop = {"full": torch.zeros(0, dtype=torch.long, device=DEV), "removeA": windows_with([bg for w in A_rows for bg in SIG[w]]), "removeB": windows_with([bg for w in B_rows for bg in SIG[w]])}
gr = torch.Generator().manual_seed(2); nrand = max(drop["removeA"].numel(), drop["removeB"].numel()); drop["random"] = torch.randperm(trn.shape[0], generator=gr)[:nrand].to(DEV)
log(f"{len(usable)} recruits with specific bigrams: A {len(A_rows)}, B {len(B_rows)}; windows dropped: removeA {drop['removeA'].numel()}, removeB {drop['removeB'].numel()}, random {drop['random'].numel()} of {trn.shape[0]} ({time.time() - t0:.0f}s)")
keepmask = torch.ones(trn.shape[0], dtype=torch.bool, device=DEV); keepmask[drop[COND]] = False; stream = TR[keepmask]
# ---- the continuation
m = m8; v0 = val_loss(m)
for p_ in m.parameters(): p_.requires_grad_(True)
m.train(); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0); gg = torch.Generator().manual_seed(0); order = torch.randperm(stream.shape[0], generator=gg)
with torch.enable_grad():
    for s in range(STEPS):
        idx = order[(s * 8) % stream.shape[0]:(s * 8) % stream.shape[0] + 8]; x = stream[idx]
        if x.shape[0] < 8: x = stream[torch.randint(0, stream.shape[0], (8,), generator=gg)]
        loss = ce(m(x).logits.float(), x); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % (2 if SMOKE else 500) == 0: log(f"step {s + 1}: ce {float(loss):.3f} ({time.time() - t0:.0f}s)")
m.eval()
for p_ in m.parameters(): p_.requires_grad_(False)
v1 = val_loss(m); W1, O1, N1, M1_, K1 = vocab(m)
# ---- the readout per recruit, and substitutes against the step-10000 reference classes
def jac(a, b_): return len(a & b_) / max(len(a | b_), 1)
ref_cls = {w: set(K10[O10[:, w]].tolist()) for w in usable}
def readout(w):
    cls1 = set(K1[O1[:, w]].tolist()); best = (None, 0.0)
    cand = torch.nonzero(N1 >= 10)[:, 0].tolist()
    for r in cand:
        if r == w: continue
        j = jac(ref_cls[w], set(K1[O1[:, r]].tolist()))
        if j > best[1]: best = (r, j)
    return dict(word=(w in W1), class_size=int(N1[w]), class_size_8000=int(N8[w]), class_size_10000=int(N10[w]), ratio_max=float(M1_[w]), ratio_max_8000=float(M8_[w]), ratio_max_10000=float(M10_[w]), class_jaccard_with_reference=jac(ref_cls[w], cls1), substitute=best[0], substitute_jaccard=best[1])
RO = {str(w): readout(w) for w in usable}
rate = lambda rows_: mean([float(RO[str(w)]["word"]) for w in rows_]) if rows_ else None; subr = lambda rows_: mean([float(RO[str(w)]["substitute_jaccard"] >= 0.5) for w in rows_]) if rows_ else None; csz = lambda rows_: mean([RO[str(w)]["class_size"] for w in rows_]) if rows_ else None
res = dict(condition=COND, block=b, steps=STEPS, lr=LR, n_training_windows=int(trn.shape[0]), n_dropped=int(drop[COND].numel()), n_recruits=len(recruits), n_usable=len(usable), A=A_rows, B=B_rows, signatures={str(w): SIG[w] for w in usable}, val=dict(before=v0, after=v1), recruitment=dict(A=rate(A_rows), B=rate(B_rows)), class_size=dict(A=csz(A_rows), B=csz(B_rows)), substitutes=dict(A=subr(A_rows), B=subr(B_rows)), readout=RO, words_after=len(W1), words_kept_from_8000=len(W1 & W8))
summ = (f"class-based data attribution ({COND}: {int(drop[COND].numel())} of {trn.shape[0]} windows dropped; {STEPS} steps): recruits usable {len(usable)} (A {len(A_rows)}, B {len(B_rows)}); words after the continuation A {rate(A_rows)}, B {rate(B_rows)}; mean class size A {csz(A_rows)}, B {csz(B_rows)} (at 10000: {mean([int(N10[w]) for w in A_rows]) if A_rows else None}, {mean([int(N10[w]) for w in B_rows]) if B_rows else None}); substitutes A {subr(A_rows)}, B {subr(B_rows)}; val {v0:.3f} -> {v1:.3f}; words kept from 8000 {len(W1 & W8)} of {len(W8)}")
log(summ); record(f"e628_attribution_{COND}" + ("_smoke" if SMOKE else ""), res, summ)
