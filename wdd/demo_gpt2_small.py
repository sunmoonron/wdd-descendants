"""The ten-minute demo on GPT-2 small (one GPU, fp32): the floor and the rows at a position, the exact ledger of one neuron and its brake,
the weak field, the K-input simulation with and without the field, and read energy. Run: python -m wdd.demo_gpt2_small [--fast]"""
import sys, time, torch, numpy as np
from . import *
from .model import DEV
FAST = "--fast" in sys.argv; t0 = time.time(); say = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)
M = load("gpt2"); B = 6; T = 256; NWIN = 8 if FAST else 32; ids = windows_from(wikitext_ids(M, "test"), NWIN, T).to(DEV); hold = windows_from(wikitext_ids(M, "train"), NWIN, T).to(DEV); say(f"GPT-2 small, block {B}, {NWIN} windows of {T}")
# 1. the floor and the rows at one position
with Capture(M, [B], ["x"]) as C: run_base(M, ids[:1]); x = C.cap[("x", B)][0].float()
xc = x - x.mean(0); U = unitr(xc); RD, blk, neu, nr = rows_dictionary(M, B); cal = floor_calibration(U, RD); fl = floor_of(U, cal); proj = (U @ RD.T).abs(); over = (proj > fl[:, None]).sum(1)
say(f"rows over the identification floor per position: median {int(over.median())}, max {int(over.max())} of {RD.shape[0]} native rows; the floor is {float(fl.median()):.3f} in units of the state")
sel, cof = omp(U[64:80], RD, k=16); say(f"OMP with sixteen rows at sixteen positions leaves {float(1 - (torch.einsum('nk,nkd->nd', cof, RD[sel])).pow(2).sum(-1).mean() / U[64:80].pow(2).sum(-1).mean()):.2f} of the unit state's energy (the crowd)")
# 2. the ledger of one neuron and its brake
L = Ledger(M, B, ids, T=T); NEU = L.active_neurons(8); k = NEU[0]; say(f"ledger at block {B}: {L.NU} upstream units; {len(NEU)} active neurons sampled; the split's relative error at neuron {k}: {L.check(k):.1e}")
NC, NN = [], []
for kk in NEU:
    a = L.AB[:, kk].float(); order = a.argsort(descending=True); NC.append(L.net_ledger(kk, order[:max(20, L.N // 100)])); NN.append(L.net_ledger(kk, order[L.N // 2:][:400]))
say(f"median over {len(NEU)} neurons: at the core the sixteen largest inputs carry {med([r['wires_net_share'] for r in NC]):.2f} of the net drive and the crowd adds {med([r['crowd_over_wires'] for r in NC]):+.2f} of theirs ({med([r['n_eff'] for r in NC]):.0f} effective terms); off the class the crowd's net sum is {med([r['crowd_over_wires'] for r in NN]):+.2f} of the wires' (negative is the brake)")
# 3. the weak and strong fields of the upstream blocks, and the simulation with and without them
dirs, tau = field_directions(M, list(range(B)), hold); FW, FS = union(dirs, "weak"), union(dirs, "strong"); BOTH = torch.linalg.qr(torch.cat([FW, FS], 1))[0]
say(f"K-input simulation of neuron {k}"); sim = L.simulate(k, extra={"abs16_weak": crowd_on_field(L, FW), "abs16_both": crowd_on_field(L, BOTH)})
say("K-input simulation, KL over the ablation's at the firings / over all positions: " + ", ".join(f"{c} {sim[c]['ratio_strong']:.2f}/{sim[c]['ratio_all']:.2f}" for c in ("abs16", "abs16_weak", "abs16_both", "all")))
# 4. read energy: what the output reads of this position's state
pos = [(w, t) for w in range(min(4, NWIN)) for t in (40, 120)]; F, cnt = sampled_fisher(M, B, ids, pos, n_samples=256 if FAST else 1024); with_x = []
with Capture(M, [B], ["x", "act"]) as C:
    for w, t in pos: run_base(M, ids[w:w + 1]); xw = C.cap[("x", B)][0, t].float(); aw = C.cap[("act", B)][0, t].float(); with_x.append((xw, aw))
W = write_w(M, B); XS = torch.stack([v[0] for v in with_x]); AS = torch.stack([v[1] for v in with_x]); XBAR = XS.mean(0); XC = XS - XBAR; tauB = torch.cat([L.AB.float()[:, c0:c0 + 512].abs().quantile(0.75, dim=0) for c0 in range(0, M["DFF"], 512)]); wk = AS.abs() < tauB[None]
parts = {"content": XC, "weak write": (AS * wk) @ W.T, "strong write": (AS * ~wk) @ W.T}; say("read energy per unit energy against a random direction (median over positions): " + ", ".join(f"{nm} {float(grade(F, v).median()):.2f}" for nm, v in parts.items()) + f"; the weak field's directions: {float((torch.einsum('dk,pde,ek->p', FW, F, FW) / FW.shape[1] / (F.diagonal(dim1=1, dim2=2).sum(-1) / M['D'])).median()):.2f}")
Lb = blended_sqrt(F, lam=1.0)
selE, cofE = omp(XC, RD, k=16); RE = torch.einsum("nk,nkd->nd", cofE, RD[selE]); RF = torch.stack([omp_metric(XC[p:p + 1], RD, Lb[p], k=16)[0][0] for p in range(len(pos))])
say(f"sixteen rows by Euclidean pursuit leave {float(((XC - RE).pow(2).sum(-1) / XC.pow(2).sum(-1)).median()):.2f} of the energy and {float((read_energy(F, XC - RE) / read_energy(F, XC)).median()):.2f} of the read energy; by the blended pursuit {float(((XC - RF).pow(2).sum(-1) / XC.pow(2).sum(-1)).median()):.2f} and {float((read_energy(F, XC - RF) / read_energy(F, XC)).median()):.2f}")
say("done")
