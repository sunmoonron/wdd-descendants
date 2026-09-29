"""e585 (session 107): the relay. How responsibility for a class actually migrates between rows. For the 256 words at step
16000 with a class of ten or more positions (e582's cache: sixteen checkpoints, 24 sequences), at every checkpoint the
class's speaker is the row whose projection is over the floor at the most of the class's positions (ties by the mean
ratio); a handoff is a change of speaker between consecutive checkpoints; an overlap is a checkpoint at which two rows
are each over the floor at half the class or more; a gap is a checkpoint at which no row is. The class's own direction
(the unit mean of its unit states) is followed as the row-free invariant: its cosine between consecutive checkpoints
and with 16000, against the speaker's row identity and the speakers' directions. The final model (step 143000) is
added as a seventeenth point for the long jump. Pre-registered (probabilities are honest guesses):
 Y1 (0.6) most classes have more than one speaker over the sixteen checkpoints (a relay, not a tenure);
 Y2 (0.6) the class's own direction is more persistent than its speakers' identity: cosine with 16000 above 0.7 at
    the median from step 4000 on, while the speaker at 4000 is the speaker at 16000 for under half of the classes;
 Y3 (0.5) handoffs happen through overlaps more often than through gaps (responsibility is passed, not dropped)."""
from s101_common import *
t0 = time.time(); CD = "/workspace/wdd/cache/e582_pythia410"; steps = list(range(1000, 16001, 1000)); B = 12; ids = eval_ids("pythia410")[:24, :512].to(DEV)
ck = {n: torch.load(f"{CD}/step{n}.pt", map_location="cpu") for n in steps}
mF, _, fam = load_model("pythia410"); arch = Arch(mF, fam); XF = block_states(mF, arch, ids, [B], chunk=4)[B].reshape(-1, arch.D); keepF = ~sinkmask(XF); AF, nF = rows_of(arch, B); del mF; torch.cuda.empty_cache()
keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0) & keepF.cpu(); N = int(keepc.sum()); m = AF.shape[0]
U = {n: unitr((ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV)) for n in steps}; U["final"] = unitr(XF[keepc.to(DEV)] - XF[keepc.to(DEV)].mean(0)); A = {n: ck[n]["rows"].float().to(DEV) for n in steps}; A["final"] = AF; ALL = steps + ["final"]
RAT = {}
for n in ALL:
    st = stats(U[n], A[n], K); RAT[n] = st["ratio"]; 
    if n == 16000: W16 = wordset(st["usage"]); S16 = st["S"]
    del st; torch.cuda.empty_cache(); log(f"{n}: stats ({time.time() - t0:.0f}s)")
w16 = torch.nonzero(W16)[:, 0]; R16 = RAT[16000].float(); csize = (R16[:, w16] > 1).sum(0); tracked = w16[csize >= 10]; CLS = {int(w): torch.nonzero(R16[:, w] > 1)[:, 0] for w in tracked.tolist()}; log(f"{tracked.numel()} classes of ten or more positions")
def speakers(n, pos):
    r = RAT[n][pos].float().to(DEV); over = (r > 1).sum(0); mr = r.mean(0); score = over.float() * 1000 + mr; top = score.topk(3); return [(int(i), int(over[i]), float(mr[i])) for i in top.indices]
res = dict(n_classes=len(CLS), n_positions=N, per_class={}); hist = {}
for w, pos in CLS.items():
    k = pos.numel(); sp = {n: speakers(n, pos) for n in ALL}; dmean = {n: unitr(U[n][pos.to(DEV)].mean(0, keepdim=True))[0] for n in ALL}
    seq = [sp[n][0][0] if sp[n][0][1] >= k / 2 else None for n in steps]; distinct = len(set(x for x in seq if x is not None)); hand = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a is not None and b is not None and a != b)
    gaps = sum(1 for x in seq if x is None); overlaps = sum(1 for n in steps if sp[n][1][1] >= k / 2); hand_via_overlap = sum(1 for i, (a, b) in enumerate(zip(seq[:-1], seq[1:])) if a is not None and b is not None and a != b and (sp[steps[i]][1][1] >= k / 2 or sp[steps[i + 1]][1][1] >= k / 2))
    dcos = [float(dmean[a] @ dmean[b]) for a, b in zip(steps[:-1], steps[1:])]; dcos16 = {n: float(dmean[n] @ dmean[16000]) for n in ALL}; rowcos16 = {n: float(A[n][w] @ A[16000][w]) for n in ALL}
    same_speaker_as_16000 = {n: bool(seq[steps.index(n)] == w) if n in steps else bool(sp["final"][0][0] == w and sp["final"][0][1] >= k / 2) for n in ALL}
    res["per_class"][str(w)] = dict(size=k, speakers=seq, distinct=distinct, handoffs=hand, gaps=gaps, overlaps=overlaps, handoffs_via_overlap=hand_via_overlap, dir_cos_consecutive_median=med(dcos), dir_cos_16000={str(n): v for n, v in dcos16.items()}, row_cos_16000={str(n): v for n, v in rowcos16.items()}, same_speaker={str(n): v for n, v in same_speaker_as_16000.items()}, final_speaker=sp["final"][0][0], final_speaker_is_word=bool(sp["final"][0][0] == w))
P = res["per_class"].values()
res["summary"] = dict(distinct_median=med([p["distinct"] for p in P]), share_more_than_one=mean([float(p["distinct"] > 1) for p in P]), handoffs_median=med([p["handoffs"] for p in P]), gaps_median=med([p["gaps"] for p in P]), share_with_gap=mean([float(p["gaps"] > 0) for p in P]), overlaps_median=med([p["overlaps"] for p in P]),
                      handoffs_total=sum(p["handoffs"] for p in P), handoffs_via_overlap_total=sum(p["handoffs_via_overlap"] for p in P), dir_cos_consecutive=med([p["dir_cos_consecutive_median"] for p in P]),
                      dir_cos_16000_by_step={str(n): med([p["dir_cos_16000"][str(n)] for p in P]) for n in ALL}, row_cos_16000_by_step={str(n): med([p["row_cos_16000"][str(n)] for p in P]) for n in ALL}, same_speaker_by_step={str(n): mean([float(p["same_speaker"][str(n)]) for p in P]) for n in ALL}, final_speaker_is_word=mean([float(p["final_speaker_is_word"]) for p in P]))
Sm = res["summary"]
for n in ALL: log(f"{n}: class direction cos with 16000 {Sm['dir_cos_16000_by_step'][str(n)]:.2f}, the 16000 row's cos with itself then {Sm['row_cos_16000_by_step'][str(n)]:.2f}, same speaker as at 16000 {Sm['same_speaker_by_step'][str(n)]:.2f}")
summ = (f"the relay ({len(CLS)} classes, {N} positions, 16 checkpoints + final): distinct speakers per class {Sm['distinct_median']:.0f} at the median ({Sm['share_more_than_one']:.2f} of classes have more than one), handoffs {Sm['handoffs_median']:.0f}, gaps {Sm['gaps_median']:.0f} ({Sm['share_with_gap']:.2f} of classes have one), overlaps {Sm['overlaps_median']:.0f}; handoffs through an overlap {Sm['handoffs_via_overlap_total']} of {Sm['handoffs_total']}; "
        f"the class's own direction: consecutive cosine {Sm['dir_cos_consecutive']:.2f}, with 16000 at 4000/8000/12000 {Sm['dir_cos_16000_by_step']['4000']:.2f}/{Sm['dir_cos_16000_by_step']['8000']:.2f}/{Sm['dir_cos_16000_by_step']['12000']:.2f} and at the final model {Sm['dir_cos_16000_by_step']['final']:.2f}; the speaker at 4000/8000/12000 is the 16000 word for {Sm['same_speaker_by_step']['4000']:.2f}/{Sm['same_speaker_by_step']['8000']:.2f}/{Sm['same_speaker_by_step']['12000']:.2f}, at the final model {Sm['same_speaker_by_step']['final']:.2f} (the 16000 row's own direction there at cosine {Sm['row_cos_16000_by_step']['final']:.2f}) | {time.time() - t0:.0f}s")
log(summ); record("e585_relay", res, summ)
