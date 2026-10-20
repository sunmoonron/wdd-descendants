"""e712 (session 160): the edge program. Both camps count nodes; the exact ledger counts edges. For four hundred of a
block's neurons, at each one's top 1 percent of positions, the ledger's sixteen largest MLP inputs are the firing's wires (the attention outputs and the embedding are counted
aside);
an edge is an upstream unit paired with a neuron, its mass the contribution it carries over the neuron's firings where it
is a wire, its reuse the share of those firings. Counted: the units needed for four fifths of a firing's net drive and of its positive drive,
the share of each neuron's wire mass held by its stable wires (reuse of a half or more), the concentration of mass over
all active edges and its Zipf slope, and the share of stable edges held by the most-used upstream units.
Arguments: --model M --block B [--smoke]. Pre-registered in e707_e712_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e712_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else 64; NNEU = 8 if SMOKE else 400; NATT = n_att_feeding(M, B); ids = windows_from(wikitext_ids(M, "train"), NWIN, T).to(DEV); N = NWIN * (T - 1); NMLP = B * DFF; NU = NMLP + NATT + 1; K1 = max(20, N // 100); GATED = M["fam"] == "gated"
UP, AT, EM, XM, AB = [], [], [], [], []
with Capture(M, list(range(B + 1)), ["act", "att", "xin", "emb"]) as C:
    for s0 in range(0, NWIN, 8):
        run_base(M, ids[s0:s0 + 8])
        UP.append(torch.cat([C.cap[("act", b)][:, 1:].half() for b in range(B)], -1).reshape(-1, NMLP)); AT.append(torch.stack([C.cap[("att", b)][:, 1:].half() for b in range(NATT)], 2).reshape(-1, NATT, D))
        EM.append(C.cap[("emb", 0)][:, 1:].half().reshape(-1, D)); XM.append(C.cap[("xin", B)][:, 1:].float().reshape(-1, D)); AB.append(C.cap[("act", B)][:, 1:].half().reshape(-1, DFF))
S = dict(UP=torch.cat(UP), AT=torch.cat(AT), EM=torch.cat(EM), XM=torch.cat(XM), AB=torch.cat(AB)); del UP, AT, EM, XM, AB
Rd = read_w(M, B); key0 = "h" if "h" in Rd else "g"; RT = read_tilde(M, B, Rd[key0]); WUP = torch.cat([write_w(M, b).T for b in range(B)]); P_ = ln_params(M, B)
SIG = (S["XM"].var(-1, unbiased=False) + P_["eps"]).sqrt() if P_["centre"] else (S["XM"].pow(2).mean(-1) + P_["eps"]).sqrt(); ab = S["AB"].float()
def contrib(k, P):
    r = RT[k]; V = WUP @ r; s = SIG[P]; return torch.cat([S["UP"][P].float() * V[None] / s[:, None], (S["AT"][P].float() @ r) / s[:, None], ((S["EM"][P].float() @ r) / s)[:, None]], 1)
mx = ab.max(0).values; a100 = ab.topk(100, dim=0).values[-1]; floor_act = 1.0 if not GATED else float(mx.quantile(0.1)); cand = torch.nonzero((mx > floor_act) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(712 + B).sample(cand, min(NNEU, len(cand))))
L_(f"block {B}: {len(NEU)} neurons, {K1} firings each, {NU} upstream units ({NMLP} MLP neurons, {NATT} attention outputs, the embedding)")
EU, EN, EMASS, EREUSE = [], [], [], []; rows = []
for k in NEU:
    a = ab[:, k]; core = a.argsort(descending=True)[:K1]; Cm = contrib(k, core); tot = Cm.sum(1); srt = Cm.sort(1, descending=True).values; csum = srt.cumsum(1); pos = Cm.clamp_min(0); psum = pos.sum(1, keepdim=True)
    n80n = ((csum < 0.8 * tot[:, None]).sum(1) + 1).float(); n80p = ((pos.sort(1, descending=True).values.cumsum(1) < 0.8 * psum).sum(1) + 1).float(); att_share = float((Cm[:, NMLP:].abs().sum(1) / Cm.abs().sum(1).clamp_min(1e-12)).median())
    top = Cm[:, :NMLP].abs().topk(16, dim=1); units = top.indices.reshape(-1); mass = top.values.reshape(-1); uq, inv = torch.unique(units, return_inverse=True); cnt = torch.bincount(inv, minlength=uq.numel()).float(); ms = torch.bincount(inv, weights=mass, minlength=uq.numel())
    reuse = cnt / K1; stable = reuse >= 0.5; EU.append(uq); EN.append(torch.full_like(uq, k)); EMASS.append(ms); EREUSE.append(reuse)
    rows.append(dict(neuron=int(k), n80_net=float(n80n.median()), n80_pos=float(n80p.median()), n_units=int(uq.numel()), stable_count=int(stable.sum()), stable_mass_share=float(ms[stable].sum() / ms.sum().clamp_min(1e-12)), top_unit_reuse=float(reuse.max()), att_share=att_share))
EU, EN, EMASS, EREUSE = torch.cat(EU), torch.cat(EN), torch.cat(EMASS), torch.cat(EREUSE); NE = EU.numel(); srt = EMASS.sort(descending=True).values; csum = srt.cumsum(0) / srt.sum()
top10 = float(csum[max(int(0.1 * NE) - 1, 0)]); top1 = float(csum[max(int(0.01 * NE) - 1, 0)]); nz = min(10000, NE); lr_ = torch.log(torch.arange(1, nz + 1, device=DEV).float()); lm = torch.log(srt[:nz].clamp_min(1e-12)); zipf = float(-((lr_ - lr_.mean()) * (lm - lm.mean())).sum() / (lr_ - lr_.mean()).pow(2).sum())
stable = EREUSE >= 0.5; deg = torch.bincount(EU[stable], minlength=NMLP).float(); dsrt = deg.sort(descending=True).values; hub_share = float(dsrt[:max(int(0.01 * NMLP), 1)].sum() / dsrt.sum().clamp_min(1)); hub_count = int((deg >= 2).sum())
S_ = dict(n_neurons=len(NEU), firings_per_neuron=K1, upstream_units=NU, mlp_units=NMLP, active_edges=NE, n80_net=med([r["n80_net"] for r in rows]), n80_pos=med([r["n80_pos"] for r in rows]), units_per_neuron=med([r["n_units"] for r in rows]), stable_count=med([r["stable_count"] for r in rows]), stable_mass_share=med([r["stable_mass_share"] for r in rows]), top_unit_reuse=med([r["top_unit_reuse"] for r in rows]),
          top10pct_edge_mass=top10, top1pct_edge_mass=top1, zipf_slope=zipf, hub_1pct_share_of_stable=hub_share, units_stable_for_two_or_more=hub_count, att_mass_share=med([r["att_share"] for r in rows]))
V = dict(P1_top10pct_edges_half_mass=bool(top10 >= 0.5), P2_stable_wires_half_mass=bool(S_["stable_mass_share"] >= 0.5), P3_hubs_hold_third=bool(hub_share >= 0.3), P4_n80_net_under_32=bool(S_["n80_net"] <= 32))
res = dict(model=M["hf"], block=B, n_blocks=NB, summary=S_, verdicts=V, neurons=rows)
summ = (f"[{name} block {B}] {len(NEU)} neurons x {K1} firings: units for four fifths of the net drive {S_['n80_net']:.0f} and of the positive drive {S_['n80_pos']:.0f} (medians); attention and embedding carry {S_['att_mass_share']:.2f} of the absolute input mass; wires per neuron {S_['units_per_neuron']:.0f} distinct, {S_['stable_count']:.0f} stable (reuse >= 0.5) holding {S_['stable_mass_share']:.2f} of the wire mass; "
        f"{NE} active edges, the top 10 percent carry {top10:.2f} of the mass (top 1 percent {top1:.2f}, Zipf slope {zipf:.2f}); the top 1 percent of MLP units hold {hub_share:.2f} of the stable edges, {hub_count} units stable for two or more neurons; verdicts {V}")
save(TAG, res, summ)
