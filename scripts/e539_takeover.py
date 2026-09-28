"""e539: is an exit a displacement, and by whom? e538b found the floor to be a one-way gate: entrants cross it, leavers
stay above it and lose usage and breadth. This reads the competition directly, from the e524 cache (unit centred states
at every kept position and the unit rows of blocks 0-12 at sixteen checkpoints, block 12, no model), with OMP over the
rows alone, for the native rows and for a fixed random dictionary of the same size against the same states. Recorded
per dictionary: e538b's profiles (the one-way gate under rows-only OMP, and whether a random dictionary shows it);
hysteresis (at equal S and breadth, a word's probability of staying against a non-word's of entering); at the positions
a leaver loses, its own projection over the floor and the sixteenth-largest projection at the position (the cutoff)
before and after, and which moved (the mirror at the positions an entrant gains); and the takeover audit: the atoms new
at a leaver's lost positions, their concentration (the effective number of takers over the number of lost positions,
the top taker's share), their class (established word, entrant, non-word), their cosine and neighbour rank to the
leaver, their projection at those positions before the exit, against a null of as many random positions; and the
atoms an entrant displaces at its gained positions, with the share that are leavers.
Pre-registered (honest guesses), native rows:
- the takers of a leaver's positions are mostly established words, weighted share 0.6 or more (0.6);
- the takeover is diffuse, the effective number of takers at least half the number of lost positions (0.5);
- the takers are not the leaver's neighbours: weighted cosine under 0.2, median neighbour rank over 100 (0.6);
- at the lost positions the cutoff moves more than the leaver's own projection at most positions, and the median
  cutoff change is positive (0.6); at the gained positions the entrant's own rise exceeds the cutoff's change (0.7);
- hysteresis: at equal S (1.1-1.3) and breadth a word's probability of staying exceeds a non-word's of entering by
  0.3 or more (0.7);
- the random dictionary shows less of the gate: its leavers over the floor at k=0 in fewer than 0.6 of cases (0.5).
Arguments: name."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
name = sys.argv[1]; B = 12; K = 16; CDIR = f"/workspace/wdd/cache/e524_{name}"; steps = list(range(1000, 16001, 1000)); T = len(steps)
ck = {n: torch.load(f"{CDIR}/step{n}.pt", map_location="cpu") for n in steps}; m, D = ck[steps[0]]["rows"].shape
g = torch.Generator(device=DEV).manual_seed(0); Arand = unitr(torch.randn(m, D, device=DEV, generator=g)); Arand_cpu = Arand.cpu()
H = {"native": [], "random": []}; An_cpu = []; agree = []; keepc = torch.stack([ck[n]["blocks"][B]["keep"] for n in steps]).all(0)
log(f"{name}: {int(keepc.sum())} positions kept at every checkpoint (sinks at any checkpoint dropped everywhere)")
for n in steps:
    bl = ck[n]["blocks"][B]; U = bl["U"][keepc].float().to(DEV); U = unitr(U); An = ck[n]["rows"].float().to(DEV); An_cpu.append(An.cpu())
    for d, A in (("native", An), ("random", Arand)): H[d].append(stats(U, A, K))
    agree.append(float(torch.corrcoef(torch.stack([H["native"][-1]["S"], bl["S"].float()]))[0, 1]))
    log(f"{name} step{n}: {U.shape[0]} positions; native words' median S {float(H['native'][-1]['S'][wordset(H['native'][-1]['usage'])].median()):.2f}, random {float(H['random'][-1]['S'][wordset(H['random'][-1]['usage'])].median()):.2f}; correlation of S with the cache's full-dictionary S {agree[-1]:.3f}"); del U, An; torch.cuda.empty_cache()
res = dict(model=name, block=B, steps=steps, S_agreement_with_cache=agree, by_dictionary={})
res["by_dictionary"]["native"] = full_analysis(H["native"], lambda t: An_cpu[t], "native", name)
res["by_dictionary"]["random"] = full_analysis(H["random"], lambda t: Arand_cpu, "random", name)
nat, rnd = res["by_dictionary"]["native"], res["by_dictionary"]["random"]; tk = nat["takeover"]["takers"]; dx = nat["decomposition"]["exits"]; dn = nat["decomposition"]["entries"]; hy = nat["hysteresis"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(takers_mostly_established=g_(tk["share_established"]) >= 0.6, takeover_diffuse=g_(tk["eff_over_lost"]) >= 0.5, takers_not_neighbours=g_(tk["cos_to_leaver"]) < 0.2 and g_(tk["neighbour_rank"]) > 100,
                     exit_is_displacement=g_(dx["cutoff_moved_more"]) >= 0.5 and g_(dx["d_cut"]) > 0, entry_is_own_rise=g_(dn["d_own"]) > abs(g_(dn["d_cut"])),
                     hysteresis_over_0_3=all(g_(hy[k]["matched_difference"]) >= 0.3 for k in ("1.1-1.2", "1.2-1.3")), random_less_gate=g_(rnd["profiles"]["fractions"]["leavers_over_floor_k0"]) < 0.6)
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name} block {B}: " + " | ".join(f"{d}: leavers over the floor at k=0 {f2(o['profiles']['fractions']['leavers_over_floor_k0'])}, S change across exits {f2(o['profiles']['delta_S_exit'])} / entries {f2(o['profiles']['delta_S_entry'])}; at lost positions own {f2(o['decomposition']['exits']['own_before'])}->{f2(o['decomposition']['exits']['own_after'])} against cutoff {f2(o['decomposition']['exits']['cut_before'])}->{f2(o['decomposition']['exits']['cut_after'])} (cutoff moved more at {f2(o['decomposition']['exits']['cutoff_moved_more'])}); at gained positions own {f2(o['decomposition']['entries']['own_before'])}->{f2(o['decomposition']['entries']['own_after'])} against cutoff {f2(o['decomposition']['entries']['cut_before'])}->{f2(o['decomposition']['entries']['cut_after'])}; takers: effective over lost {f2(o['takeover']['takers']['eff_over_lost'])} (null {f2(o['takeover']['takers_null']['eff_over_lost'])}), established/entrant/non-word {f2(o['takeover']['takers']['share_established'])}/{f2(o['takeover']['takers']['share_entrant'])}/{f2(o['takeover']['takers']['share_nonword'])}, cosine {f2(o['takeover']['takers']['cos_to_leaver'])} (null {f2(o['takeover']['takers_null']['cos_to_leaver'])}); hysteresis matched difference at 1.1-1.2/1.2-1.3 {f2(o['hysteresis']['1.1-1.2']['matched_difference'])}/{f2(o['hysteresis']['1.2-1.3']['matched_difference'])}" for d, o in res["by_dictionary"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e539_takeover_{name}", res, summ)
