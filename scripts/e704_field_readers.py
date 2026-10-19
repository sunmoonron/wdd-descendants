"""e704 (session 158): the field's directions, read the way rows were read. e693's sixteen principal directions of a
block's weak write carry most of what the weak activations do; here each direction is given the two readings the program
gave neurons: its causal downstream readers (the block's output pushed by one standard deviation of the direction's
coordinate at every position, the change of the next three blocks' neurons in their own standard deviations: how
concentrated the response is, how consistent its sign across positions) and its token image through the unembedding;
plus the direction's own class (the positions where its coordinate is in the top 1 percent: the one-token share, the sign
balance). Sixteen strong neurons of the block, pushed along their own write rows, are the reference. Arguments: --model M
--block B [--smoke]. Pre-registered in e701_e704_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e704_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else 64; NWH = 16 if SMOKE else 96; Q = 0.25; READ = [b for b in range(B + 1, min(B + 4, NB))]; NREF = 16; NDIR = 16
ids_all = wikitext_ids(M, "train"); half = len(ids_all) // 2; ids = windows_from(ids_all[:half], NWIN, T).to(DEV); ids_h = windows_from(ids_all[half:], NWH, T).to(DEV); N = NWIN * (T - 1); NH = NWH * (T - 1); W = write_w(M, B)
# ---------------- held-out: the weak threshold per neuron of block B and the weak write's principal directions
AH = []
with Capture(M, [B], ["act"], stop_after=B) as C:
    for s0 in range(0, NWH, 8): run_base(M, ids_h[s0:s0 + 8]); AH.append(C.cap[("act", B)][:, 1:].half().reshape(-1, DFF))
AH = torch.cat(AH); TAU = torch.cat([AH[:, c0:c0 + 512].float().abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)]); cov = torch.zeros(D, D, device=DEV)
for s in range(0, NH, 2048): a = AH[s:s + 2048].float(); dx = (a * (a.abs() < TAU[None])) @ W.T; cov += dx.T @ dx
ev, evec = torch.linalg.eigh(cov); U16 = evec[:, -NDIR:].flip(1).contiguous(); SD = (ev[-NDIR:].flip(0) / NH).sqrt(); del AH
# ---------------- evaluation: clean activations downstream, the block's activations, the weak write's coordinates
AB, AR = [], {b: [] for b in READ}
with Capture(M, [B] + READ, ["act"], stop_after=max(READ)) as C:
    for s0 in range(0, NWIN, 8):
        run_base(M, ids[s0:s0 + 8]); AB.append(C.cap[("act", B)][:, 1:].half().reshape(-1, DFF))
        for b in READ: AR[b].append(C.cap[("act", b)][:, 1:].half().reshape(-1, DFF))
AB = torch.cat(AB); AR = {b: torch.cat(v) for b, v in AR.items()}; ASTD = {b: AR[b].float().std(0).clamp_min(1e-6) for b in READ}; TOK = ids[:, 1:].reshape(-1); NXT = torch.cat([ids[:, 2:], torch.full((NWIN, 1), -1, device=DEV, dtype=torch.long)], 1).reshape(-1)
ab = AB.float(); Z = ((ab * (ab.abs() < TAU[None])) @ W.T) @ U16   # [N, 16] the weak write's coordinates at the evaluation rows
mx = ab.max(0).values; a100 = ab.topk(100, dim=0).values[-1]; floor_act = 1.0 if M["fam"] != "gated" else float(mx.quantile(0.1)); cand = torch.nonzero((mx > floor_act) & (a100 >= 0.25 * mx))[:, 0].tolist(); REF = sorted(random.Random(685 + B).sample(cand, min(NREF, len(cand))))
head = M["model"].get_output_embeddings().weight.detach().float(); g = getattr(M["fnorm"], "weight", None); WU = head * g.detach().float()[None] if g is not None else head; tok = M["tok"]
def readers(direction, eps):
    """push the block's output by eps along the unit direction at every position: per downstream neuron the mean change in its std units [len(READ) * DFF], and the per-row changes of the twenty largest"""
    d = direction / direction.norm().clamp_min(1e-9); diff = []
    def hook(m, a, out): return out + (eps * d).to(out.dtype)
    hd = write_mod(M, B).register_forward_hook(hook)
    try:
        with Capture(M, READ, ["act"], stop_after=max(READ)) as C:
            for s0 in range(0, NWIN, 8): run_base(M, ids[s0:s0 + 8]); diff.append(torch.cat([((C.cap[("act", b)][:, 1:].float().reshape(-1, DFF) - AR[b][s0 * (T - 1):(s0 + 8) * (T - 1)].float()) / ASTD[b]).half() for b in READ], 1))
    finally: hd.remove()
    diff = torch.cat(diff); mean = diff.float().mean(0); top = mean.abs().topk(20).indices; cons = float((torch.sign(diff[:, top].float()) == torch.sign(mean[top])[None]).float().mean())
    e = mean ** 2; conc = float(e.topk(max(1, e.numel() // 100)).values.sum() / e.sum().clamp_min(1e-12)); neff = float(e.sum() ** 2 / (e ** 2).sum().clamp_min(1e-12)); return dict(concentration_top1pct=conc, n_eff=neff, consistency_top20=cons, top_readers=[(int(i) // DFF, int(i) % DFF, float(mean[i])) for i in top[:10].tolist()], response_norm=float(mean.norm()))
def token_image(direction):
    d = direction / direction.norm().clamp_min(1e-9); lg = WU @ d; e = lg ** 2; up = lg.topk(5).indices.tolist(); dn = (-lg).topk(5).indices.tolist()
    return dict(top1pct_share=float(e.topk(max(1, e.numel() // 100)).values.sum() / e.sum().clamp_min(1e-12)), up=[tok.decode([i]) for i in up], down=[tok.decode([i]) for i in dn])
def own_class(z):
    K1 = max(20, N // 100); top = z.abs().topk(K1).indices; return dict(one_token=float(torch.bincount(TOK[top], minlength=M["V"]).max() / K1), positive_share=float((z > 0).float().mean()), top_positive_share=float((z[top] > 0).float().mean()), next_one_token=float(torch.bincount(NXT[top][NXT[top] >= 0], minlength=M["V"]).max() / max(1, int((NXT[top] >= 0).sum()))))
DIRS = []
for k in range(NDIR):
    d = U16[:, k]; r = readers(d, float(SD[k])); DIRS.append(dict(index=k, coord_std=float(SD[k]), energy_share=float(ev[-1 - k] / ev.sum()), readers=r, tokens=token_image(d), own=own_class(Z[:, k])))
    L_(f"direction {k}: coordinate std {float(SD[k]):.3f}, reader concentration {r['concentration_top1pct']:.2f} (N_eff {r['n_eff']:.0f}), consistency {r['consistency_top20']:.2f}; tokens up {DIRS[-1]['tokens']['up'][:3]} down {DIRS[-1]['tokens']['down'][:3]}; one-token share of its class {DIRS[-1]['own']['one_token']:.2f}")
REFS = []
for j in REF:
    d = W[:, j]; r = readers(d, float(ab[:, j].std() * d.norm())); REFS.append(dict(neuron=int(j), readers=r, tokens=token_image(d), own=own_class(ab[:, j])))
mdir = lambda key, sub: med([x[sub][key] for x in DIRS]); mref = lambda key, sub: med([x[sub][key] for x in REFS])
S_ = dict(n_directions=NDIR, n_reference_neurons=len(REFS), direction=dict(concentration=mdir("concentration_top1pct", "readers"), n_eff=mdir("n_eff", "readers"), consistency=mdir("consistency_top20", "readers"), token_top1pct=mdir("top1pct_share", "tokens"), one_token=mdir("one_token", "own"), positive_share=mdir("positive_share", "own"), top_positive_share=mdir("top_positive_share", "own")),
          reference=dict(concentration=mref("concentration_top1pct", "readers"), n_eff=mref("n_eff", "readers"), consistency=mref("consistency_top20", "readers"), token_top1pct=mref("top1pct_share", "tokens"), one_token=mref("one_token", "own")), first=dict(concentration=DIRS[0]["readers"]["concentration_top1pct"], n_eff=DIRS[0]["readers"]["n_eff"], tokens_up=DIRS[0]["tokens"]["up"], tokens_down=DIRS[0]["tokens"]["down"]))
V = dict(F1_directions_as_concentrated_as_neurons=bool(S_["direction"]["concentration"] >= 0.8 * S_["reference"]["concentration"]), F2_consistent_readers=bool(S_["direction"]["consistency"] >= 0.8), F3_first_direction_concentrated=bool(S_["first"]["n_eff"] < 0.01 * len(READ) * DFF), F4_lexical_classes=bool(S_["direction"]["one_token"] > 0.2), F5_signed_coordinates=bool(0.3 <= S_["direction"]["top_positive_share"] <= 0.7), F6_token_image_as_concentrated_as_neurons=bool(S_["direction"]["token_top1pct"] >= 0.8 * S_["reference"]["token_top1pct"]))
res = dict(model=M["hf"], block=B, n_blocks=NB, read_blocks=READ, positions=N, directions=DIRS, reference=REFS, summary=S_, verdicts=V)
summ = (f"[{name} block {B}] the weak write's 16 directions pushed by one std at every position: downstream reader response concentration (top 1% share) median {S_['direction']['concentration']:.2f} (16 strong neurons {S_['reference']['concentration']:.2f}), N_eff {S_['direction']['n_eff']:.0f} ({S_['reference']['n_eff']:.0f}), sign consistency of the twenty largest {S_['direction']['consistency']:.2f} ({S_['reference']['consistency']:.2f}); token image top-1% share {S_['direction']['token_top1pct']:.2f} ({S_['reference']['token_top1pct']:.2f}); "
        f"the directions' own classes: one-token share {S_['direction']['one_token']:.2f} (neurons {S_['reference']['one_token']:.2f}), positive share at the top {S_['direction']['top_positive_share']:.2f}; the first direction: N_eff {S_['first']['n_eff']:.0f}, tokens up {S_['first']['tokens_up'][:3]} down {S_['first']['tokens_down'][:3]}; verdicts {V}")
save(TAG, res, summ)
