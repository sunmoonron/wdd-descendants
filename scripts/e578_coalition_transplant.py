"""e578 (session 104): what has to travel for the computation to travel? e577 showed a transplanted direction becomes a
word without moving the recipient's predictions at its class. Here the producer coalition travels too. For thirty
entrants (words at step 16000, not at 8000, with three or more positions over the floor at 16000; some words are used often
without ever being over the floor) the class is its over-the-floor positions at 16000 and the coalition is
the k MLP neurons of blocks 0-12 whose writes contribute most (activation times projection on the entrant's direction,
summed over the class's positions) at 16000, the entrant's own neuron excluded. Conditions, each transplanted into the
step-8000 model (sufficiency) and, the same parts from step 8000 into the step-16000 model, the damage they do at the
class against elsewhere (necessity; this direction avoids the checkpoint-mixing confound of a later readout on
earlier states) (read row, bias and write column of every neuron in the set): the direction alone (the entrant's
write direction at the old norm); the coalition alone (k = 128); the coalition and the direction; a larger coalition
with the direction (k = 512); k = 128 random neurons with the coalition's block histogram (the size control). Per
entrant: whether the row is a word in the recipient (usage rank under 256), its S, the KL of the change at the class
positions and elsewhere, and the closure of the gap to the 16000 model at the class positions and elsewhere. Then
once, with all forty entrants at once: the readout (blocks 13-23, the final norm and the unembedding from 16000)
alone, and the readout with every entrant's coalition (k = 128) and direction. Pre-registered (probabilities are
honest guesses):
 K1 (0.6) the coalition alone makes fewer than half of the entrants words, but closes the gap at the class more
    than elsewhere (a positive specificity), unlike the random neurons;
 K2 (0.6) the coalition with the direction makes at least 0.7 of the entrants words, above the direction alone (0.54);
 K3 (0.7) no condition short of the readout closes more than a tenth of the gap at the class;
 K4 (0.5) the readout alone closes more than half of the gap everywhere with a specificity near zero."""
from s101_common import *
t0 = time.time(); B = 12; ids = pile_ids("pythia410"); NE = 30; KS = (128, 512)
m8, tok, fam = load_model("pythia410", revision="step8000"); m16, _, _ = load_model("pythia410", revision="step16000"); arch8, arch16 = Arch(m8, fam), Arch(m16, fam); DFF = arch8.DFF
S8 = lm_states("pythia410", B=B, ids=ids, model=m8); S16 = lm_states("pythia410", B=B, ids=ids, model=m16); keepc = S8["keep"] & S16["keep"]; kidx = torch.nonzero(keepc)[:, 0]
for S_ in (S8, S16): S_["U"] = unitr(S_["X"][keepc] - S_["X"][keepc].mean(0))
st8 = stats(S8["U"], S8["A"], K); st16 = stats(S16["U"], S16["A"], K); w8, w16 = wordset(st8["usage"]), wordset(st16["usage"]); ent_all = torch.nonzero(w16 & ~w8)[:, 0]
g = torch.Generator().manual_seed(0); ncls = (st16["ratio"][:, ent_all].float() > 1).sum(0); ent_ok = ent_all[ncls >= 3]; ent = ent_ok[torch.randperm(ent_ok.numel(), generator=g)[:NE]]; log(f"entrants {ent_all.numel()}, with a class of three or more positions at 16000: {ent_ok.numel()}; {ent.numel()} sampled")
# activations of the 16000 model's MLP neurons, blocks 0..B, at the kept positions
ACT = {}
hs = [arch16.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: ACT.__setitem__(b_, a[0].detach().float()[:, 1:].reshape(-1, a[0].shape[-1])[keepc]))(b)) for b in range(B + 1)]
class Stop(Exception): pass
def stop(m, i, o): raise Stop
hs.append(arch16.layers[B].register_forward_hook(stop))
try:
    with torch.no_grad(): m16(ids)
except Stop: pass
finally: [h.remove() for h in hs]
act16 = torch.cat([ACT[b] for b in range(B + 1)], 1); del ACT; torch.cuda.empty_cache()   # [N, m]
Rows16 = torch.cat([arch16.wdir(b) for b in range(B + 1)]); norms16 = Rows16.norm(dim=1); m = Rows16.shape[0]; blk = torch.arange(m, device=DEV) // DFF
def coalition(r, k):
    d = S16["A"][r]; cls = torch.nonzero(st16["ratio"][:, r].float() > 1)[:, 0].to(DEV); contrib = (act16[cls] * ((Rows16 @ d))[None]).clamp_min(0).sum(0); contrib[r] = -1; return contrib.topk(k).indices, cls
def transplant(rows, direction_row=None):
    saved = []
    with torch.no_grad():
        for r in rows.tolist():
            b, j = r // DFF, r % DFF; l8, l16 = arch8.layers[b].mlp, arch16.layers[b].mlp
            saved.append((b, j, l8.dense_4h_to_h.weight[:, j].clone(), l8.dense_h_to_4h.weight[j].clone(), l8.dense_h_to_4h.bias[j].clone()))
            l8.dense_4h_to_h.weight[:, j] = l16.dense_4h_to_h.weight[:, j]; l8.dense_h_to_4h.weight[j] = l16.dense_h_to_4h.weight[j]; l8.dense_h_to_4h.bias[j] = l16.dense_h_to_4h.bias[j]
        if direction_row is not None:
            for r in torch.as_tensor(direction_row).flatten().tolist():
                b, j = r // DFF, r % DFF; l8, l16 = arch8.layers[b].mlp, arch16.layers[b].mlp; saved.append((b, j, l8.dense_4h_to_h.weight[:, j].clone(), l8.dense_h_to_4h.weight[j].clone(), l8.dense_h_to_4h.bias[j].clone()))
                src = l16.dense_4h_to_h.weight[:, j]; l8.dense_4h_to_h.weight[:, j] = src / src.norm() * l8.dense_4h_to_h.weight[:, j].norm()
    def restore():
        with torch.no_grad():
            for b, j, wc, wr, bb in reversed(saved): arch8.layers[b].mlp.dense_4h_to_h.weight[:, j] = wc; arch8.layers[b].mlp.dense_h_to_4h.weight[j] = wr; arch8.layers[b].mlp.dense_h_to_4h.bias[j] = bb
    return restore
def logp(model):
    with torch.no_grad(): return model(ids).logits.float().log_softmax(-1)
def klm(a, b, pm): return float(((a.exp() * (a - b)).sum(-1))[pm].mean())
L8, L16 = logp(m8), logp(m16)
def masks(cls):
    full = torch.zeros(keepc.numel(), dtype=torch.bool); full[kidx.cpu()[cls.cpu()]] = True
    pc = torch.zeros(ids.shape, dtype=torch.bool); pc[:, 1:] = full.reshape(ids.shape[0], -1); po = torch.zeros_like(pc); po[:, 1:] = keepc.cpu().reshape(ids.shape[0], -1) & ~pc[:, 1:]
    assert int(pc.sum()) == int(cls.numel()) > 0, (int(pc.sum()), int(cls.numel())); return pc.to(DEV), po.to(DEV)
def measure(r, cls, tag):
    S_ = lm_states("pythia410", B=B, ids=ids, model=m8); U = unitr(S_["X"][keepc] - S_["X"][keepc].mean(0)); st = stats(U, S_["A"], K); rank = int((st["usage"] > st["usage"][r]).sum()); Lx = logp(m8); pc, po = masks(cls)
    g8c, g8o = klm(L16, L8, pc), klm(L16, L8, po); gxc, gxo = klm(L16, Lx, pc), klm(L16, Lx, po)
    return dict(word=float(rank < 256), S=float(st["S"][r]), usage_rank=rank, kl_change_class=klm(L8, Lx, pc), kl_change_other=klm(L8, Lx, po), closure_class=(g8c - gxc) / g8c, closure_other=(g8o - gxo) / g8o, gap_class=g8c)
COND = ["direction", "coalition_128", "coalition_128_direction", "coalition_512_direction", "random_128"]; res = dict(n_entrants=NE, per={c: [] for c in COND}, base=[])
for q, r in enumerate(ent.tolist()):
    co128, cls = coalition(r, 128); co512, _ = coalition(r, 512); hist = torch.bincount(blk[co128], minlength=B + 1); rnd = torch.cat([torch.nonzero((blk == b) & (torch.arange(m, device=DEV) != r))[:, 0][torch.randperm(DFF - (1 if b == r // DFF else 0), generator=g)[:int(hist[b])].to(DEV)] for b in range(B + 1) if hist[b] > 0])
    res["base"].append(dict(S_8000=float(st8["S"][r]), S_16000=float(st16["S"][r]), n_class=int(cls.numel())))
    for c, rows, drow in (("direction", torch.tensor([], dtype=torch.long), r), ("coalition_128", co128, None), ("coalition_128_direction", co128, r), ("coalition_512_direction", co512, r), ("random_128", rnd, None)):
        restore = transplant(rows, drow); out = measure(r, cls, c); restore(); res["per"][c].append(out)
    if q % 10 == 9: log(f"{q + 1} entrants: " + " | ".join(f"{c}: word {mean([o['word'] for o in res['per'][c]]):.2f}, closure class {med([o['closure_class'] for o in res['per'][c]]):+.3f} / other {med([o['closure_other'] for o in res['per'][c]]):+.3f}" for c in COND) + f" | {time.time() - t0:.0f}s")
agg = {c: dict(word=mean([o["word"] for o in v]), S=med([o["S"] for o in v]), kl_change_class=med([o["kl_change_class"] for o in v]), kl_change_other=med([o["kl_change_other"] for o in v]), closure_class=med([o["closure_class"] for o in v]), closure_other=med([o["closure_other"] for o in v]), specificity=med([o["closure_class"] - o["closure_other"] for o in v]), share_specific=mean([float(o["closure_class"] > o["closure_other"]) for o in v])) for c, v in res["per"].items()}
res["aggregate"] = agg
for c, a in agg.items(): log(f"{c}: word {a['word']:.2f}, S {a['S']:.2f}, KL change class/other {a['kl_change_class']:.4f}/{a['kl_change_other']:.4f}, closure class/other {a['closure_class']:+.3f}/{a['closure_other']:+.3f} (specific in {a['share_specific']:.2f})")
# the necessity direction: the step-8000 versions of the same parts placed into the 16000 model; damage at the class against elsewhere
def transplant_into16(rows, direction_row=None):
    saved = []
    with torch.no_grad():
        for r in rows.tolist():
            b, j = r // DFF, r % DFF; l16, l8 = arch16.layers[b].mlp, arch8.layers[b].mlp
            saved.append((b, j, l16.dense_4h_to_h.weight[:, j].clone(), l16.dense_h_to_4h.weight[j].clone(), l16.dense_h_to_4h.bias[j].clone()))
            l16.dense_4h_to_h.weight[:, j] = l8.dense_4h_to_h.weight[:, j]; l16.dense_h_to_4h.weight[j] = l8.dense_h_to_4h.weight[j]; l16.dense_h_to_4h.bias[j] = l8.dense_h_to_4h.bias[j]
        if direction_row is not None:
            for r in torch.as_tensor(direction_row).flatten().tolist():
                b, j = r // DFF, r % DFF; l16, l8 = arch16.layers[b].mlp, arch8.layers[b].mlp; saved.append((b, j, l16.dense_4h_to_h.weight[:, j].clone(), l16.dense_h_to_4h.weight[j].clone(), l16.dense_h_to_4h.bias[j].clone()))
                src = l8.dense_4h_to_h.weight[:, j]; l16.dense_4h_to_h.weight[:, j] = src / src.norm() * l16.dense_4h_to_h.weight[:, j].norm()
    def restore():
        with torch.no_grad():
            for b, j, wc, wr, bb in reversed(saved): arch16.layers[b].mlp.dense_4h_to_h.weight[:, j] = wc; arch16.layers[b].mlp.dense_h_to_4h.weight[j] = wr; arch16.layers[b].mlp.dense_h_to_4h.bias[j] = bb
    return restore
res["necessity"] = {c: [] for c in COND}
for q, r in enumerate(ent.tolist()):
    co128, cls = coalition(r, 128); co512, _ = coalition(r, 512); hist = torch.bincount(blk[co128], minlength=B + 1); rnd = torch.cat([torch.nonzero((blk == b) & (torch.arange(m, device=DEV) != r))[:, 0][torch.randperm(DFF - (1 if b == r // DFF else 0), generator=g)[:int(hist[b])].to(DEV)] for b in range(B + 1) if hist[b] > 0]); pc, po = masks(cls)
    for c, rows, drow in (("direction", torch.tensor([], dtype=torch.long), r), ("coalition_128", co128, None), ("coalition_128_direction", co128, r), ("coalition_512_direction", co512, r), ("random_128", rnd, None)):
        restore = transplant_into16(rows, drow); Ly = logp(m16); restore()
        S_ = None
        res["necessity"][c].append(dict(kl_change_class=klm(L16, Ly, pc), kl_change_other=klm(L16, Ly, po), toward_8000_class=(klm(L8, L16, pc) - klm(L8, Ly, pc)) / max(klm(L8, L16, pc), 1e-9), toward_8000_other=(klm(L8, L16, po) - klm(L8, Ly, po)) / max(klm(L8, L16, po), 1e-9)))
nec = {c: dict(kl_change_class=med([o["kl_change_class"] for o in v]), kl_change_other=med([o["kl_change_other"] for o in v]), toward_8000_class=med([o["toward_8000_class"] for o in v]), toward_8000_other=med([o["toward_8000_other"] for o in v]), share_specific=mean([float(o["kl_change_class"] > o["kl_change_other"]) for o in v])) for c, v in res["necessity"].items()}
res["necessity_aggregate"] = nec
for c, a in nec.items(): log(f"necessity, 8000 parts into 16000, {c}: KL change class/other {a['kl_change_class']:.4f}/{a['kl_change_other']:.4f} (larger at the class in {a['share_specific']:.2f}), moved toward 8000 at the class {a['toward_8000_class']:+.3f}, elsewhere {a['toward_8000_other']:+.3f}")
# the readout, once
allcls = torch.zeros(keepc.sum(), dtype=torch.bool)
for r in ent.tolist(): allcls |= (st16["ratio"][:, r].float() > 1)
pc, po = masks(torch.nonzero(allcls)[:, 0].to(DEV)); sd8 = {k: v.detach().clone() for k, v in m8.state_dict().items()}; sd16 = m16.state_dict()
def readout_on():
    with torch.no_grad():
        for k in sd8:
            if any(f"layers.{b}." in k for b in range(B + 1, arch8.NB)) or "final_layer_norm" in k or "embed_out" in k: m8.state_dict()[k].copy_(sd16[k])
def readout_off(): m8.load_state_dict(sd8)
g8c, g8o = klm(L16, L8, pc), klm(L16, L8, po); res["readout"] = {}
readout_on(); Lx = logp(m8); res["readout"]["readout_only"] = dict(closure_class=(g8c - klm(L16, Lx, pc)) / g8c, closure_other=(g8o - klm(L16, Lx, po)) / g8o); readout_off()
restores = [transplant(coalition(r, 128)[0], r) for r in ent.tolist()]; readout_on(); Lx = logp(m8); res["readout"]["readout_coalitions_directions"] = dict(closure_class=(g8c - klm(L16, Lx, pc)) / g8c, closure_other=(g8o - klm(L16, Lx, po)) / g8o); readout_off(); [rs() for rs in reversed(restores)]
restores = [transplant(coalition(r, 128)[0], r) for r in ent.tolist()]; Lx = logp(m8); res["readout"]["all_coalitions_directions_no_readout"] = dict(closure_class=(g8c - klm(L16, Lx, pc)) / g8c, closure_other=(g8o - klm(L16, Lx, po)) / g8o); [rs() for rs in reversed(restores)]
for k, v in res["readout"].items(): log(f"{k}: closure at the classes {v['closure_class']:+.3f}, elsewhere {v['closure_other']:+.3f}")
A_ = agg; Rd = res["readout"]
NC = res["necessity_aggregate"]
summ = (f"necessity (8000 parts into the 16000 model): KL change at the class / elsewhere direction {NC['direction']['kl_change_class']:.4f}/{NC['direction']['kl_change_other']:.4f}, coalition-128 {NC['coalition_128']['kl_change_class']:.4f}/{NC['coalition_128']['kl_change_other']:.4f}, both {NC['coalition_128_direction']['kl_change_class']:.4f}/{NC['coalition_128_direction']['kl_change_other']:.4f}, k=512 {NC['coalition_512_direction']['kl_change_class']:.4f}/{NC['coalition_512_direction']['kl_change_other']:.4f}, random {NC['random_128']['kl_change_class']:.4f}/{NC['random_128']['kl_change_other']:.4f} (larger at the class in {NC['coalition_128']['share_specific']:.2f} / {NC['coalition_512_direction']['share_specific']:.2f} / random {NC['random_128']['share_specific']:.2f}); moved toward 8000 at the class: {NC['direction']['toward_8000_class']:+.3f} / {NC['coalition_128']['toward_8000_class']:+.3f} / {NC['coalition_128_direction']['toward_8000_class']:+.3f} / {NC['coalition_512_direction']['toward_8000_class']:+.3f} / random {NC['random_128']['toward_8000_class']:+.3f} | " + f"coalition transplant into the 8000 model ({NE} entrants): word share direction / coalition-128 / coalition-128+direction / coalition-512+direction / random-128: {A_['direction']['word']:.2f} / {A_['coalition_128']['word']:.2f} / {A_['coalition_128_direction']['word']:.2f} / {A_['coalition_512_direction']['word']:.2f} / {A_['random_128']['word']:.2f}; "
        f"closure of the gap to 16000 at the class (elsewhere): {A_['direction']['closure_class']:+.3f} ({A_['direction']['closure_other']:+.3f}) / {A_['coalition_128']['closure_class']:+.3f} ({A_['coalition_128']['closure_other']:+.3f}) / {A_['coalition_128_direction']['closure_class']:+.3f} ({A_['coalition_128_direction']['closure_other']:+.3f}) / {A_['coalition_512_direction']['closure_class']:+.3f} ({A_['coalition_512_direction']['closure_other']:+.3f}) / {A_['random_128']['closure_class']:+.3f} ({A_['random_128']['closure_other']:+.3f}); "
        f"specific in {A_['coalition_128']['share_specific']:.2f} / {A_['coalition_512_direction']['share_specific']:.2f} / random {A_['random_128']['share_specific']:.2f} of entrants; the readout alone closes {Rd['readout_only']['closure_class']:+.3f} at the classes and {Rd['readout_only']['closure_other']:+.3f} elsewhere, with all coalitions and directions {Rd['readout_coalitions_directions']['closure_class']:+.3f} / {Rd['readout_coalitions_directions']['closure_other']:+.3f}; all coalitions and directions without the readout {Rd['all_coalitions_directions_no_readout']['closure_class']:+.3f} / {Rd['all_coalitions_directions_no_readout']['closure_other']:+.3f} | {time.time() - t0:.0f}s")
log(summ); record("e578_coalition_transplant", res, summ)
