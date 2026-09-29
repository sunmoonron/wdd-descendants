"""e579 (session 104): is the instrument useful before the labels are? In the toy sweep (e567, fourteen runs measured
every 250 steps) and the optimizer runs (e556), the moment the vocabulary statistic first shows the association
against the moment test accuracy first passes one half. Statistics: the words' S over the rotated words' S (the
native excess) crossing 1.5, and the own-over-rotation FVU advantage crossing 0.3; the lead in steps for every run
that generalises, and whether either statistic ever fires in a run that memorises. Pre-registered (probabilities are
honest guesses):
 U1 (0.6) the native excess crosses 1.5 before test accuracy crosses 0.5 in most generalising runs, by 250-1000 steps;
 U2 (0.7) it never crosses 1.5 in a memorising run (no false alarms on the training inputs alone)."""
from s101_common import *
import glob
t0 = time.time(); runs = {}
for f in sorted(glob.glob("/workspace/wdd/results/e567_dyn_*.json")): runs[os.path.basename(f)[9:-5]] = json.load(open(f))
def first(rows, key, thr): return next((r["step"] for r in rows if key(r) is not None and key(r) >= thr), None)
res = dict(runs={})
for v, d in runs.items():
    rows = d["log"]; gs = first(rows, lambda r: r["test_acc"], 0.5); ex = first(rows, lambda r: r["words_S"] / max(r["rotated_S"], 1e-6), 1.5); ad = first(rows, lambda r: r["advantage"], 0.3); fit = first(rows, lambda r: r["train_acc"], 0.99)
    maxex = max(r["words_S"] / max(r["rotated_S"], 1e-6) for r in rows); res["runs"][v] = dict(grok_step=gs, excess_step=ex, advantage_step=ad, fit_step=fit, final_test=d["final"]["test_acc"], max_excess=maxex, lead_excess=(gs - ex) if (gs is not None and ex is not None) else None, lead_advantage=(gs - ad) if (gs is not None and ad is not None) else None)
    r_ = res["runs"][v]; log(f"{v}: train fit at {fit}, test > 0.5 at {gs}, native excess >= 1.5 at {ex} (lead {r_['lead_excess']}), advantage >= 0.3 at {ad} (lead {r_['lead_advantage']}); max excess {maxex:.2f}, final test {r_['final_test']:.2f}")
gen = {v: r for v, r in res["runs"].items() if r["grok_step"] is not None}; mem = {v: r for v, r in res["runs"].items() if r["grok_step"] is None}
res["summary"] = dict(n_generalising=len(gen), n_memorising=len(mem), lead_excess_median=med([r["lead_excess"] for r in gen.values() if r["lead_excess"] is not None]), share_excess_leads=mean([float(r["lead_excess"] > 0) for r in gen.values() if r["lead_excess"] is not None]), share_excess_fires_gen=mean([float(r["excess_step"] is not None) for r in gen.values()]),
                      lead_advantage_median=med([r["lead_advantage"] for r in gen.values() if r["lead_advantage"] is not None]), share_advantage_leads=mean([float(r["lead_advantage"] > 0) for r in gen.values() if r["lead_advantage"] is not None]), false_alarms_excess=sum(r["excess_step"] is not None for r in mem.values()), false_alarms_advantage=sum(r["advantage_step"] is not None for r in mem.values()), max_excess_memorising=max(r["max_excess"] for r in mem.values()), min_max_excess_generalising=min(r["max_excess"] for r in gen.values()))
s = res["summary"]
summ = (f"lead time of the vocabulary statistic over test accuracy (toy sweep, {s['n_generalising']} generalising and {s['n_memorising']} memorising runs): the native excess (words' S over rotated words' S) crosses 1.5 in {s['share_excess_fires_gen']:.2f} of generalising runs, leading test accuracy's crossing of 0.5 in {s['share_excess_leads']:.2f} of them by {s['lead_excess_median']} steps at the median; false alarms in memorising runs {s['false_alarms_excess']} of {s['n_memorising']} (their maximum excess {s['max_excess_memorising']:.2f} against the generalising runs' minimum {s['min_max_excess_generalising']:.2f}); "
        f"the advantage crossing 0.3 leads in {s['share_advantage_leads']:.2f} by {s['lead_advantage_median']} steps, false alarms {s['false_alarms_advantage']} | {time.time() - t0:.0f}s")
log(summ); record("e579_lead_time", res, summ)
