"""e606 (session 112): the compression audit. Pythia-410m at block 12 (where 4-bit costs a third of a nat, not two),
six Pile domains (PubMed Abstracts, Github, Wikipedia, StackExchange, FreeLaw, USPTO), 16 held-out windows each.
Words on the pooled states, each word's domain profile (the share of its class in each domain), and for each
compression (8-bit per-channel, 4-bit in groups of 64, 4-bit in groups of 32, 3-bit in groups of 64, and magnitude
pruning of 30% and 50% of every attention and MLP weight matrix) the per-domain loss damage against the per-domain
silencing of the writers: the activation ratio at their classes and the still-writing share of the words whose
class is at least 60% one domain. The question is whether the damage a compression does to a domain is read off
which writers it silences, as the unlearning audit reads relearnability. Controls: the same measures on the words
of every other domain, and a random set of rows. Argument: --smoke.
Pre-registered (honest guesses):
 C1 (0.6) across the six compressions by six domains, the domain's loss damage correlates with its writers'
    activation ratio at Spearman -0.5 or beyond (more silencing, more damage);
 C2 (0.5) 4-bit in groups of 64 silences (activation below 0.8) fewer than a fifth of the writers of any domain;
 C3 (0.6) magnitude pruning at 50% silences more writers than 3-bit quantisation and damages the losses more."""
from s101_common import *
from datasets import load_dataset
import copy
SMOKE = "--smoke" in sys.argv; name = "pythia410"; t0 = time.time(); B = MID[name]; T = 256; NEV = 4 if SMOKE else 16; torch.set_grad_enabled(False)
model, tok, fam = load_model(name); arch = Arch(model, fam); DFF = arch.DFF; NB1 = B + 1
ds = load_dataset("NeelNanda/pile-10k", split="train")
DOMS = {"pubmed": "PubMed Abstracts", "github": "Github", "wikipedia": "Wikipedia (en)", "stackexchange": "StackExchange", "freelaw": "FreeLaw", "uspto": "USPTO Backgrounds"}
def windows(setname, nwin, skipdocs=0):
    wins, seen = [], 0
    for ex in ds:
        if ex["meta"]["pile_set_name"] != setname: continue
        seen += 1
        if seen <= skipdocs: continue
        ids = tok(ex["text"])["input_ids"]
        if len(ids) >= T + 1: wins.append(ids[:T + 1])
        if len(wins) >= nwin: break
    return torch.tensor(wins, device=DEV)
EVW = {d: windows(s, NEV, skipdocs=100) for d, s in DOMS.items()}; EV = torch.cat([EVW[d] for d in DOMS])[:, :T]; dom_of_seq = torch.cat([torch.full((EVW[d].shape[0],), i) for i, d in enumerate(DOMS)]); DL = list(DOMS)
log("windows: " + ", ".join(f"{d} {EVW[d].shape[0]}" for d in DOMS) + f" ({time.time() - t0:.0f}s)")
def lossof(m, ids, chunk=8):
    tot = 0.0
    for s0 in range(0, ids.shape[0], chunk):
        x = ids[s0:s0 + chunk]; lg = m(x[:, :T]).logits.float(); tot += float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:T].reshape(-1))) * x.shape[0]; del lg
    return tot / ids.shape[0]
def states(m):
    X = block_states(m, Arch(m, fam), EV, [B])[B].reshape(-1, arch.D); keep = ~sinkmask(X); A, _ = rows_of(Arch(m, fam), B); return X, keep, A
X0, keep0, A0 = states(model); U0 = unitr(X0[keep0] - X0[keep0].mean(0)); st0 = stats(U0, A0, K); w0 = torch.nonzero(wordset(st0["usage"]))[:, 0].tolist(); R0 = st0["ratio"].float(); kidx0 = torch.nonzero(keep0.cpu())[:, 0]
pos_dom = dom_of_seq.repeat_interleave(T - 1)[keep0.cpu()]; CLS = {w: torch.nonzero(R0[:, w] > 1)[:, 0] for w in w0}
prof = {w: torch.bincount(pos_dom[CLS[w]], minlength=len(DL)).float() / max(CLS[w].numel(), 1) for w in w0}
DW = {d: [w for w in w0 if CLS[w].numel() >= 5 and float(prof[w][i]) >= 0.6] for i, d in enumerate(DL)}; log("domain words: " + ", ".join(f"{d} {len(v)}" for d, v in DW.items()) + f" of {len(w0)} ({time.time() - t0:.0f}s)")
grnd = torch.Generator().manual_seed(5); allw = set(w0); nonword = [i for i in range(NB1 * DFF) if i not in allw]; RND = [nonword[int(i)] for i in torch.randperm(len(nonword), generator=grnd)[:40]]
WS = sorted(set(sum(DW.values(), [])) | set(RND)); byblock = {}
for w in WS: byblock.setdefault(w // DFF, []).append(w % DFF)
def acts(m):
    a2 = Arch(m, fam); cap = {}; hs = []
    for b, cols in byblock.items():
        colt = torch.tensor(cols, device=DEV)
        def mk(b, colt):
            def hk(mod, inp): cap.setdefault(b, []).append(inp[0][:, 1:, colt].detach().float())
            return hk
        hs.append(a2.layers[b].mlp.dense_4h_to_h.register_forward_pre_hook(mk(b, colt)))
    try:
        for s0 in range(0, EV.shape[0], 8): m(EV[s0:s0 + 8])
    finally: [h.remove() for h in hs]
    out = {}
    for b, cols in byblock.items():
        Ab = torch.cat(cap[b]).reshape(-1, len(cols))
        for j, c in enumerate(cols):
            w = b * DFF + c; full = (kidx0[CLS[w]] if w in CLS and CLS[w].numel() else kidx0[:0]).to(DEV); out[w] = float(Ab[full, j].mean()) if full.numel() else float(Ab[:, j].mean())
    return out
act0 = acts(model); L0 = {d: lossof(model, EVW[d]) for d in DL}; log("original losses: " + ", ".join(f"{d} {v:.3f}" for d, v in L0.items()))
def quantize(m, bits, group=None):
    m2 = copy.deepcopy(m); qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for n, mod in m2.named_modules():
            if isinstance(mod, torch.nn.Linear):
                W = mod.weight; o_, i_ = W.shape
                if group and i_ % group == 0:
                    Wg = W.reshape(o_, i_ // group, group); sc = Wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / qmax; mod.weight.copy_(((Wg / sc).round().clamp(-qmax - 1, qmax) * sc).reshape(o_, i_))
                else:
                    sc = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax; mod.weight.copy_((W / sc).round().clamp(-qmax - 1, qmax) * sc)
    return m2
def prune(m, frac):
    m2 = copy.deepcopy(m)
    with torch.no_grad():
        for n, p_ in m2.named_parameters():
            if p_.dim() == 2 and ("attention" in n or "mlp" in n):
                thr = p_.abs().flatten().kthvalue(int(frac * p_.numel())).values; p_.mul_((p_.abs() > thr).float())
    return m2
COMP = {"int8_per_channel": lambda m: quantize(m, 8), "int4_g64": lambda m: quantize(m, 4, 64), "int4_g32": lambda m: quantize(m, 4, 32), "int3_g64": lambda m: quantize(m, 3, 64), "prune30": lambda m: prune(m, 0.3), "prune50": lambda m: prune(m, 0.5)}
res = dict(n_words=len(w0), domain_words={d: len(v) for d, v in DW.items()}, loss0=L0, conditions={})
for k, fn in COMP.items():
    m2 = fn(model); X, keep, A = states(m2); U = unitr(X[keep] - X[keep].mean(0)); st = stats(U, A, K); R = st["ratio"].float(); posn = torch.nonzero(keep.cpu())[:, 0]; pmap = {int(p): i for i, p in enumerate(posn.tolist())}; am = acts(m2)
    c = dict(losses={d: lossof(m2, EVW[d]) for d in DL}, damage={d: lossof(m2, EVW[d]) - L0[d] for d in DL}, per_domain={})
    for d in DL:
        ws = DW[d]; ratios = [am[w] / act0[w] if abs(act0[w]) > 1e-6 else None for w in ws]; rr = [x for x in ratios if x is not None]; writes = []
        for w in ws:
            idx = torch.tensor([pmap[int(p)] for p in kidx0[CLS[w]].tolist() if int(p) in pmap]); writes.append(float((R[idx, w] > 1).float().mean() >= 0.5) if idx.numel() else 0.0)
        c["per_domain"][d] = dict(n=len(ws), act_ratio=med(rr) if rr else None, share_silenced=mean([float(x < 0.8) for x in rr]) if rr else None, still_writing=mean(writes) if writes else None)
    rr = [am[w] / act0[w] for w in RND if abs(act0[w]) > 1e-6]; c["random_rows_act_ratio"] = med(rr) if rr else None
    res["conditions"][k] = c; log(f"{k}: damage " + ", ".join(f"{d} {c['damage'][d]:+.3f}" for d in DL) + "; writers' activation " + ", ".join(f"{d} {c['per_domain'][d]['act_ratio'] if c['per_domain'][d]['act_ratio'] is None else round(c['per_domain'][d]['act_ratio'], 2)} (silenced {c['per_domain'][d]['share_silenced']}, writing {c['per_domain'][d]['still_writing']})" for d in DL) + f"; random rows {c['random_rows_act_ratio']} | {time.time() - t0:.0f}s")
    del m2, X, U, st, R; torch.cuda.empty_cache()
def spear(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 4: return None
    xx, yy = torch.tensor([p[0] for p in pairs], dtype=torch.float64), torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    return None if xx.std() == 0 or yy.std() == 0 else float(torch.corrcoef(torch.stack([xx.argsort().argsort().double(), yy.argsort().argsort().double()]))[0, 1])
pairs = [(c["damage"][d], c["per_domain"][d]) for c in res["conditions"].values() for d in DL]
res["spearman_damage_vs_activation"] = spear([p[0] for p in pairs], [p[1]["act_ratio"] for p in pairs]); res["spearman_damage_vs_silenced"] = spear([p[0] for p in pairs], [p[1]["share_silenced"] for p in pairs]); res["spearman_damage_vs_writing"] = spear([p[0] for p in pairs], [p[1]["still_writing"] for p in pairs])
within = {}
for k, c in res["conditions"].items(): within[k] = spear([c["damage"][d] for d in DL], [c["per_domain"][d]["act_ratio"] for d in DL])
res["spearman_within_compression"] = within
summ = (f"compression audit (Pythia-410m, {len(w0)} words, domain words " + ", ".join(f"{d} {len(v)}" for d, v in DW.items()) + f"): Spearman of per-domain damage with the writers' activation ratio {res['spearman_damage_vs_activation']} (with the share silenced {res['spearman_damage_vs_silenced']}, with still-writing {res['spearman_damage_vs_writing']}) over {len(pairs)} pairs; within each compression " + ", ".join(f"{k} {None if v is None else round(v, 2)}" for k, v in within.items()) + "; damage: " + "; ".join(f"{k} " + "/".join(f"{c['damage'][d]:+.2f}" for d in DL) for k, c in res["conditions"].items()) + f" | {time.time() - t0:.0f}s")
log(summ); record("e606_compression_audit" + ("_smoke" if SMOKE else ""), res, summ)
