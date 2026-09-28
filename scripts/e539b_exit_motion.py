"""e539b: whose motion narrows a leaver? e539 found that at the positions a leaver loses, its own projection over the
floor falls (0.92 to 0.79) while the sixteenth-largest projection at the position does not move, and the mirror for
entrants at the positions they gain (0.76 to 0.95): both events are the row's own alignment at its marginal positions
changing, not the field. Each change can be the row's motion, the state's motion or the floor's. From the e524 cache
(unit centred states at every position, unit rows, sixteen checkpoints, block 12, OMP over the rows alone as in e539),
for every clean exit at its lost positions, at its kept positions and at its peak (the position of its largest
projection), and for every clean entry at its gained positions and its peak: the cosine between state and row before
and after, and its change split into the part from the row's motion (the new row against the old state), the part
from the state's motion (the old row against the new state) and the remainder, plus the floor's change at those
positions. Medians over positions within an event, then over events.
Pre-registered (honest guesses):
- at the lost positions the row's motion accounts for more of the fall than the states' (0.6), as it did for the rise (e522);
- at a leaver's peak the cosine changes by less than a third of its change at the lost positions (0.6);
- the floor at the lost positions changes by less than 0.02 across the exit (0.7).
Arguments: name."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
name = sys.argv[1]; B = 12; K = 16; CDIR = f"/workspace/wdd/cache/e524_{name}"; steps = list(range(1000, 16001, 1000)); T = len(steps)
ck = {n: torch.load(f"{CDIR}/step{n}.pt", map_location="cpu") for n in steps}; keepc = torch.stack([ck[n]["blocks"][B]["keep"] for n in steps]).all(0)
Us = [unitr(ck[n]["blocks"][B]["U"][keepc].float()) for n in steps]; Ws = [ck[n]["rows"].float() for n in steps]; H = []
for i, n in enumerate(steps):
    st = stats(Us[i].to(DEV), Ws[i].to(DEV), K); del st["ratio"]; H.append(st); torch.cuda.empty_cache()
log(f"{name}: OMP over the rows at {T} checkpoints, {Us[0].shape[0]} positions")
words = [wordset(h["usage"]) for h in H]; ex, en = events(words)
def split(r, t, P):
    up, uc = Us[t - 1][P], Us[t][P]; wp, wc = Ws[t - 1][r], Ws[t][r]
    c_prev = (up @ wp).abs(); c_row = (up @ wc).abs(); c_state = (uc @ wp).abs(); c_cur = (uc @ wc).abs(); Lp, Lc = H[t - 1]["L"][P], H[t]["L"][P]
    return dict(cos_before=float(c_prev.median()), cos_after=float(c_cur.median()), d_total=float((c_cur - c_prev).median()), d_row=float((c_row - c_prev).median()), d_state=float((c_state - c_prev).median()), d_rest=float((c_cur - c_row - c_state + c_prev).median()), floor_ratio=float((Lc / Lp).median()), n=int(P.numel()))
def collect(events_, kind):
    acc = {}
    for r, t in events_:
        prev = (H[t - 1]["sel"] == r).any(1); cur = (H[t]["sel"] == r).any(1)
        sets = {"peak": torch.tensor([int(((Us[t - 1] @ Ws[t - 1][r]).abs() / H[t - 1]["L"]).argmax())])}
        if kind == "exit": sets["lost"] = torch.nonzero(prev & ~cur)[:, 0]; sets["kept"] = torch.nonzero(prev & cur)[:, 0]
        else: sets["gained"] = torch.nonzero(~prev & cur)[:, 0]; sets["held_before"] = torch.nonzero(prev & cur)[:, 0]
        for k, P in sets.items():
            if P.numel() == 0: continue
            d = split(r, t, P)
            for q, v in d.items(): acc.setdefault(k, {}).setdefault(q, []).append(v)
    return {k: {q: (med(v) if q != "n" else mean(v)) for q, v in d.items()} | {"n_events": len(d["n"])} for k, d in acc.items()}
res = dict(model=name, block=B, exits=collect(ex, "exit"), entries=collect(en, "entry"), n_clean_exits=len(ex), n_clean_entries=len(en))
f2 = lambda x: "n/a" if x is None else f"{x:.3f}"
for kind in ("exits", "entries"):
    log(f"{name} {kind} ({res['n_clean_' + kind]}): " + " | ".join(f"{k}: positions {o['n']:.0f}, cosine {f2(o['cos_before'])}->{f2(o['cos_after'])}, change {f2(o['d_total'])} = row {f2(o['d_row'])} + state {f2(o['d_state'])} + rest {f2(o['d_rest'])}, floor ratio {f2(o['floor_ratio'])}" for k, o in res[kind].items()))
lo, pk = res["exits"]["lost"], res["exits"]["peak"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(row_motion_dominates_fall=abs(g_(lo["d_row"])) > abs(g_(lo["d_state"])) and g_(lo["d_row"]) < 0, peak_moves_less_than_a_third=abs(g_(pk["d_total"])) < abs(g_(lo["d_total"])) / 3, floor_change_under_0_02=abs(g_(lo["floor_ratio"]) - 1) < 0.02)
summ = f"{name}: exits at lost positions cosine {f2(lo['cos_before'])}->{f2(lo['cos_after'])}, change {f2(lo['d_total'])} = row {f2(lo['d_row'])} + state {f2(lo['d_state'])} + rest {f2(lo['d_rest'])}; at the peak {f2(pk['cos_before'])}->{f2(pk['cos_after'])} (row {f2(pk['d_row'])}, state {f2(pk['d_state'])}); kept positions {f2(res['exits']['kept']['d_total'])} (row {f2(res['exits']['kept']['d_row'])}); entries at gained positions {f2(res['entries']['gained']['cos_before'])}->{f2(res['entries']['gained']['cos_after'])}, change {f2(res['entries']['gained']['d_total'])} = row {f2(res['entries']['gained']['d_row'])} + state {f2(res['entries']['gained']['d_state'])} + rest {f2(res['entries']['gained']['d_rest'])}; at the peak {f2(res['entries']['peak']['d_total'])} (row {f2(res['entries']['peak']['d_row'])}); floor ratio at lost positions {f2(lo['floor_ratio'])} | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e539b_exit_motion_{name}", res, summ)
