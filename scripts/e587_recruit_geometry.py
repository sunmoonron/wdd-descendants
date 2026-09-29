"""e587 (session 107): is recruitment geometry or a lottery? For the tracked words of e584 (recruited between steps 3000
and 16000), two checkpoints before recruitment: the rank of the eventual row among all 53,248 rows by its cosine with
the class's own direction (the unit mean of the class's unit states, row-free), by its mean projection ratio over the
class's positions, and by the number of class positions it is over the floor at; the same five checkpoints before, and
at recruitment. If the eventual row is already the nearest available row two checkpoints ahead, recruitment is the
geometry's and predictable from the class alone. Pre-registered (probabilities are honest guesses):
 G1 (0.6) two checkpoints before recruitment the eventual row ranks in the top 10 of 53,248 by cosine with the class's
    direction for most words;
 G2 (0.5) five checkpoints before it still ranks in the top 100 for most (the recruit is chosen early);
 G3 (0.6) the rank by class-projection ratio predicts recruitment better than the rank by cosine (the states, not
    the direction alone, pick the row)."""
from s101_common import *
t0 = time.time(); CD = "/workspace/wdd/cache/e582_pythia410"; steps = list(range(1000, 16001, 1000)); T = len(steps)
ck = {n: torch.load(f"{CD}/step{n}.pt", map_location="cpu") for n in steps}; keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0); N = int(keepc.sum())
U = {n: unitr((ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV)) for n in steps}; A = {n: ck[n]["rows"].float().to(DEV) for n in steps}; m = A[steps[0]].shape[0]
ev = json.load(open("/workspace/wdd/results/e584_longitudinal.json"))["events"]; st16 = stats(U[16000], A[16000], K); R16 = st16["ratio"].float()
tracked = [(int(w), e["row"]) for w, e in ev.items() if e["row"] is not None and e["row"] >= 3]; log(f"{len(tracked)} words recruited at index 3 or later")
CAL = {n: floor_calibration(U[n], A[n]) for n in steps}
def ranks(n, w, pos):
    d = unitr(U[n][pos.to(DEV)].mean(0, keepdim=True))[0]; cos = A[n] @ d; L = floor_of(U[n][pos.to(DEV)], CAL[n]); r = (U[n][pos.to(DEV)] @ A[n].T).abs() / L[:, None]; mr = r.mean(0); over = (r > 1).sum(0).float()
    rk = lambda v: int((v > v[w]).sum()) + 1; return dict(rank_cos=rk(cos), rank_ratio=rk(mr), rank_over=rk(over), cos=float(cos[w]), ratio=float(mr[w]), over=int(over[w]))
res = dict(n=len(tracked), by_lead={}); LEADS = [0, 1, 2, 3, 5, 8]
for lead in LEADS:
    out = []
    for w, tr in tracked:
        i = tr - lead
        if i < 0: continue
        pos = torch.nonzero(R16[:, w] > 1)[:, 0]; out.append(ranks(steps[i], w, pos))
    if not out: continue
    r = dict(n=len(out), rank_cos_median=med([o["rank_cos"] for o in out]), rank_ratio_median=med([o["rank_ratio"] for o in out]), rank_over_median=med([o["rank_over"] for o in out]), top10_cos=mean([float(o["rank_cos"] <= 10) for o in out]), top10_ratio=mean([float(o["rank_ratio"] <= 10) for o in out]), top100_cos=mean([float(o["rank_cos"] <= 100) for o in out]), top100_ratio=mean([float(o["rank_ratio"] <= 100) for o in out]), top1_ratio=mean([float(o["rank_ratio"] == 1) for o in out]), cos_median=med([o["cos"] for o in out]), ratio_median=med([o["ratio"] for o in out]))
    res["by_lead"][lead] = r; log(f"{lead} checkpoints before recruitment (n {r['n']}): rank by cosine with the class direction {r['rank_cos_median']:.0f} (top 10 for {r['top10_cos']:.2f}, top 100 for {r['top100_cos']:.2f}), by class ratio {r['rank_ratio_median']:.0f} (top 10 {r['top10_ratio']:.2f}, top 100 {r['top100_ratio']:.2f}, first {r['top1_ratio']:.2f}), by count over the floor {r['rank_over_median']:.0f}; cosine {r['cos_median']:.2f}, ratio {r['ratio_median']:.2f}")
b = res["by_lead"]
summ = (f"recruitment as geometry ({len(tracked)} words): the eventual row's rank among {m} rows by cosine with the class's own direction, 2 / 5 / 8 checkpoints before recruitment: {b[2]['rank_cos_median']:.0f} / {b[5]['rank_cos_median']:.0f} / {b[8]['rank_cos_median']:.0f} (top 10 for {b[2]['top10_cos']:.2f} / {b[5]['top10_cos']:.2f} / {b[8]['top10_cos']:.2f}); by its mean ratio over the class {b[2]['rank_ratio_median']:.0f} / {b[5]['rank_ratio_median']:.0f} / {b[8]['rank_ratio_median']:.0f} (top 10 for {b[2]['top10_ratio']:.2f} / {b[5]['top10_ratio']:.2f} / {b[8]['top10_ratio']:.2f}; the first of all rows two before for {b[2]['top1_ratio']:.2f}); at recruitment rank by ratio {b[0]['rank_ratio_median']:.0f} | {time.time() - t0:.0f}s")
log(summ); record("e587_recruit_geometry", res, summ)
