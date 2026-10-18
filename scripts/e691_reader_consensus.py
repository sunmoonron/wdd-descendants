"""e691 (session 155): reader consensus against activation labels. The essay: a neuron's meaning is what its downstream
readers do with it ('become another reader'); auditors label neurons by their top-activating contexts. Two
similarities over neuron pairs at a block: by activation (the TF-IDF bag of current tokens of each neuron's top 0.1% of
positions, and the mean normalised input state there) and by readers (the vector over the next three blocks' neurons of
the exact mean contribution of the neuron's write to each reader's pre-activation at its class positions, in reader
standard deviations; and the weight-only version, write dot read through the norm's gain). Their rank correlation
over pairs, nearest-neighbour agreement, and the share of neurons whose activation neighbour is not among its reader
neighbours and vice versa. Arguments: --model M --block B [--smoke]. Pre-registered in e691_prereg.json."""
import argparse
from c11 import *
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e691_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 16 if SMOKE else 128; NNEU = 100 if SMOKE else 1000; READ = [b for b in range(B + 1, min(B + 4, NB))]; FRAC = 0.001
ids = windows_from(pile_ids(M), NWIN, T, seed=691).to(DEV); N = NWIN * (T - 1); CUR = ids[:, 1:].reshape(-1)
AB, XIN, AR = [], {b: [] for b in READ}, {b: [] for b in READ}; XB = []
with Capture(M, [B] + READ, ["act", "xin"], stop_after=max(READ)) as C:
    for s0 in range(0, NWIN, 8):
        run_base(M, ids[s0:s0 + 8]); AB.append(C.cap[("act", B)][:, 1:].half().reshape(-1, DFF)); XB.append(C.cap[("xin", B)][:, 1:].half().reshape(-1, D))
        for b in READ: XIN[b].append(C.cap[("xin", b)][:, 1:].float().reshape(-1, D)); AR[b].append(C.cap[("act", b)][:, 1:].half().reshape(-1, DFF))
AB = torch.cat(AB); XB = torch.cat(XB); XIN = {b: torch.cat(v) for b, v in XIN.items()}; AR = {b: torch.cat(v) for b, v in AR.items()}; W = write_w(M, B); nrw = W.norm(dim=0)
RT, SIG, PSTD, ASTD = {}, {}, {}, {}
for b in READ:
    Rd = read_w(M, b); key = "h" if "h" in Rd else "g"; RT[b] = read_tilde(M, b, Rd[key]); xn, sig = ln_apply(M, b, XIN[b]); SIG[b] = sig; PSTD[b] = (xn @ Rd[key].T + read_bias(M, b)[key]).std(0).clamp_min(1e-6); ASTD[b] = AR[b].float().std(0).clamp_min(1e-6)
    del XIN[b]
def causal_readers(k, top):
    """the downstream neurons' activation change, in their own std units, when neuron k is zeroed at its class positions (mean over the positions): what the readers do with it"""
    byw = collections.defaultdict(list)
    for i in top.tolist(): byw[i // (T - 1)].append((i % (T - 1) + 1, i))
    wins = sorted(byw); out = torch.zeros(len(READ) * DFF, device=DEV); cnt = 0
    for s0 in range(0, len(wins), 32):
        ws = wins[s0:s0 + 32]; rows_ = [(r, p, i) for r, w in enumerate(ws) for p, i in byw[w]]; rr = torch.tensor([x[0] for x in rows_], device=DEV); pp = torch.tensor([x[1] for x in rows_], device=DEV); ii = torch.tensor([x[2] for x in rows_], device=DEV)
        def hook(m, a):
            h = a[0].clone(); h[rr, pp, k] = 0.0; return (h,)
        hd = write_mod(M, B).register_forward_pre_hook(hook)
        try:
            with Capture(M, READ, ["act"], stop_after=max(READ)) as C: run_base(M, ids[torch.tensor(ws, device=DEV)])
        finally: hd.remove()
        for bi, b in enumerate(READ): out[bi * DFF:(bi + 1) * DFF] += ((C.cap[("act", b)][rr, pp].float() - AR[b][ii].float()) / ASTD[b]).sum(0)
        cnt += len(rows_)
    return out / max(cnt, 1)
ab = AB.float(); mx = ab.max(0).values; alive = torch.nonzero(mx > (1.0 if M["fam"] != "gated" else float(mx.quantile(0.5))))[:, 0].tolist(); NEU = sorted(random.Random(691 + B).sample(alive, min(NNEU, len(alive)))); K = max(10, int(N * FRAC))
L_(f"block {B} of {NB}, readers in blocks {READ}: {N} positions, {len(NEU)} neurons, class size {K}")
idf = torch.log(N / (torch.bincount(CUR, minlength=M["V"]).float() + 1)); XBc = XB.float(); XBc = XBc - XBc.mean(0)
SIGS = dict(act=[], state=[], read=[], weight=[], say=[])   # read: the causal reader response; weight: write dot read through the norm's gain, in reader pre-activation std units
for k in NEU:
    top = ab[:, k].topk(K).indices; c = torch.bincount(CUR[top], minlength=M["V"]).float() * idf; SIGS["act"].append(c / c.norm().clamp_min(1e-9)); s_ = XBc[top].mean(0); SIGS["state"].append(s_ / s_.norm().clamp_min(1e-9))
    d = W[:, k] / nrw[k].clamp_min(1e-9); wv = torch.cat([(RT[b] @ d) / PSTD[b] for b in READ]); rv = causal_readers(k, top)
    SIGS["read"].append(rv / rv.norm().clamp_min(1e-9)); SIGS["weight"].append(wv / wv.norm().clamp_min(1e-9)); sy = M["WU"] @ final_gain(M, d[None])[0]; SIGS["say"].append(sy / sy.norm().clamp_min(1e-9))
    if len(SIGS["act"]) % 200 == 0: L_(f"{len(SIGS['act'])} neurons' signatures")
SIM = {}
for key, vs in SIGS.items(): X = torch.stack(vs); SIM[key] = (X @ X.T).fill_diagonal_(-2)
n = len(NEU); iu = torch.triu_indices(n, n, 1).to(DEV); flat = {key: SIM[key][iu[0], iu[1]].cpu().numpy() for key in SIM}
if len(flat["act"]) > 200000: sel = np.random.default_rng(691).choice(len(flat["act"]), 200000, replace=False); flat = {key: v[sel] for key, v in flat.items()}
rho = {f"{a}~{b}": float(spearmanr(flat[a], flat[b]).correlation) for a, b in (("act", "read"), ("state", "read"), ("act", "state"), ("weight", "read"), ("say", "read"), ("act", "weight"), ("act", "say"))}
def nn_agree(a, b, top=10):
    na = SIM[a].argmax(1); tb = SIM[b].topk(top, dim=1).indices; return float((tb == na[:, None]).any(1).float().mean())
NN = {f"{a}->{b}": nn_agree(a, b) for a, b in (("act", "read"), ("read", "act"), ("state", "read"), ("read", "state"), ("act", "weight"), ("weight", "read"), ("read", "weight"))}
mis = float(((SIM["read"].topk(10, dim=1).indices == SIM["act"].argmax(1)[:, None]).any(1) | (SIM["act"].topk(10, dim=1).indices == SIM["read"].argmax(1)[:, None]).any(1)).logical_not().float().mean())
V = dict(C1_spearman_act_read_above_03=bool(rho["act~read"] > 0.3), C2_nn_agreement_above_03=bool(NN["act->read"] > 0.3), C3_weight_read_above_05=bool(rho["weight~read"] > 0.5))
res = dict(model=M["hf"], block=B, read_blocks=READ, n_neurons=n, positions=N, class_size=K, spearman=rho, nn_agreement=NN, chance_nn=10 / (n - 1), mismatch_share=mis, verdicts=V)
summ = f"[{name} block {B}] {n} neurons: Spearman over pairs, activation-tokens~readers {rho['act~read']:.3f}, state~readers {rho['state~read']:.3f}, tokens~state {rho['act~state']:.3f}, weight-only~readers {rho['weight~read']:.3f}, say~readers {rho['say~read']:.3f}; nearest neighbour by tokens in the readers' top-10 {NN['act->read']:.2f} (by state {NN['state->read']:.2f}; chance {10 / (n - 1):.3f}); readers' NN in tokens' top-10 {NN['read->act']:.2f}; mutual mismatch {mis:.2f}; verdicts {V}"
save(TAG, res, summ)
