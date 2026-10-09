"""e618d (session 117): scoring the silencing controls (e618b) and the per-word localisation test (e618c) against the
pre-registered guesses in e618_prereg.json. Controls: for each condition at +2 the extra steps back within 0.1 nats after
silencing each set's inputs (censored at 101), medians per set overall and per method, the share of conditions where the
writers beat each control, and the union against each part. Per-word: per condition the Spearman between a word's
writer activation ratio and the word's recovery, against the difference-selected and random sets at the same positions
and the word's own rise; medians per method and the share of positive correlations; the pooled split of recovery between
words whose writers still fire and words whose writers went quiet. Pure python. Argument: --smoke."""
import json, glob, os, sys, statistics, datetime
SMOKE = "--smoke" in sys.argv; root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(root, "results")
SETS = ("wdd_forget", "diff_selected", "random", "active_retain", "wdd_retain", "magnitude_selected", "union_wdd_diff"); cens = lambda x: 101 if x is None else min(x, 101); med = statistics.median; f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; sg = lambda x: "n/a" if x is None else f"{x:+.0f}"
# ---- floors (e618e): the original model with each set silenced, no unlearning
FL = {}
for f in sorted(glob.glob(os.path.join(RES, "e618e_floors_*.json"))):
    if "smoke" in f: continue
    r = json.load(open(f)); FL.update({t: {s_: v["floor_forget"] for s_, v in fl.items()} for t, fl in r["floors"].items()})
def floor_rows(pattern, sets):
    """per silenced condition: the floor-corrected 20-step recovery of each set, the unsilenced recovery, and the nats left above the floor after 20 steps"""
    out = []
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        if "smoke" in f: continue
        r = json.load(open(f)); tag = os.path.basename(f)[:-5]; fl = FL.get(tag)
        if not fl: continue
        for k, c in r["conditions"].items():
            if not c.get("silenced") or c["recovery"] is None or c["loss_after_relearn20"] is None: continue
            row = dict(model=r["model"], domain=r["domain"], method=c["method"], seed=r["seed"], rec_alone=c["recovery"], excess_alone=c["loss_after_relearn20"] - r["loss0"]["forget"], rec_floor={}, excess={}, floor_rise={}, silence_rise={})
            for s_ in sets:
                v = c["silenced"].get(s_)
                if v is None or s_ not in fl or v["loss_after_relearn20"] is None: continue
                Lfs, L20s, flo = v["loss_forget"], v["loss_after_relearn20"], fl[s_]
                row["rec_floor"][s_] = (Lfs - L20s) / (Lfs - flo) if Lfs - flo > 0.1 else None; row["excess"][s_] = L20s - flo; row["floor_rise"][s_] = flo - r["loss0"]["forget"]; row["silence_rise"][s_] = Lfs - c["loss_forget"]
            out.append(row)
    return out
def floor_summary(rows, sets):
    S = [r for r in rows if all(r["rec_floor"].get(s_) is not None for s_ in sets)]
    if not S: return None
    return dict(n=len(S), median_recovery_unsilenced=med([r["rec_alone"] for r in S]), median_floor_corrected_recovery={s_: med([r["rec_floor"][s_] for r in S]) for s_ in sets}, median_nats_above_floor_after_20={s_: med([r["excess"][s_] for r in S]) for s_ in sets}, median_nats_above_original_after_20_unsilenced=med([r["excess_alone"] for r in S]), median_floor_rise={s_: med([r["floor_rise"][s_] for r in S]) for s_ in sets}, median_rise_from_silencing_after_unlearning={s_: med([r["silence_rise"][s_] for r in S]) for s_ in sets}, writers_below={s_: statistics.mean([r["rec_floor"]["wdd_forget"] < r["rec_floor"][s_] for r in S]) for s_ in sets if s_ != "wdd_forget"}, by_method={m: {s_: med([r["rec_floor"][s_] for r in S if r["method"] == m]) for s_ in sets} for m in sorted(set(r["method"] for r in S))})
# ---- controls
C = []
for f in sorted(glob.glob(os.path.join(RES, "e618b_controls_*.json"))):
    if ("smoke" in f) != SMOKE: continue
    r = json.load(open(f))
    for k, c in r["conditions"].items():
        if not c.get("silenced"): continue
        C.append(dict(model=r["model"], domain=r["domain"], method=c["method"], seed=r["seed"], back_alone=cens(c["steps_to_relearn"]), rec_alone=c["recovery"], rise=c["rise_forget"], delay={s: cens(c["silenced"][s]["steps_to_relearn"]) - cens(c["steps_to_relearn"]) for s in SETS if s in c["silenced"]}, loss_after_silence={s: c["silenced"][s]["loss_forget"] - c["loss_forget"] for s in SETS if s in c["silenced"]}, n_rows={s: c["silenced"][s]["n_rows"] for s in SETS if s in c["silenced"]}))
res = dict(n_controls=len(C), controls={}, perword={}, _exp="e618d_controls_perword_analysis" + ("_smoke" if SMOKE else ""))
if C:
    cc = [c for c in C if all(s in c["delay"] for s in SETS)]; uncens = [c for c in cc if c["back_alone"] < 101]
    for lab, S in (("all", cc), ("uncensored_alone", uncens)):
        if not S: continue
        res["controls"][lab] = dict(n=len(S), median_delay={s: med([c["delay"][s] for c in S]) for s in SETS}, mean_delay={s: statistics.mean([c["delay"][s] for c in S]) for s in SETS}, wdd_beats={s: statistics.mean([c["delay"]["wdd_forget"] > c["delay"][s] for c in S]) for s in SETS if s != "wdd_forget"}, union_beats_both=statistics.mean([c["delay"]["union_wdd_diff"] > max(c["delay"]["wdd_forget"], c["delay"]["diff_selected"]) for c in S]), median_loss_rise_from_silencing={s: med([c["loss_after_silence"][s] for c in S]) for s in SETS}, by_method={m: {s: med([c["delay"][s] for c in S if c["method"] == m]) for s in SETS} for m in sorted(set(c["method"] for c in S))}, by_domain={d: {s: med([c["delay"][s] for c in S if c["domain"] == d]) for s in SETS} for d in sorted(set(c["domain"] for c in S))})
    A = res["controls"]["all"]["median_delay"]; res["verdicts"] = dict(C1=A["active_retain"] < 0.5 * A["wdd_forget"] if A["wdd_forget"] > 0 else False, C2=A["wdd_retain"] <= 5, C3=A["union_wdd_diff"] > max(A["wdd_forget"], A["diff_selected"]))
# ---- per-word
P = []
for f in sorted(glob.glob(os.path.join(RES, "e618c_perword_*.json"))):
    if ("smoke" in f) != SMOKE: continue
    r = json.load(open(f)); pw = r.get("perword")
    if not pw or pw["n_words"] < 8: continue
    P.append(dict(model=r["model"], domain=r["domain"], method=r["method"], seed=r["seed"], n_words=pw["n_words"], sp=pw["spearman"], split=pw.get("split", {}), rows=pw["rows"]))
if P:
    keys = ("writers", "diff_at_positions", "random_at_positions", "rise", "class_size"); valid = lambda k: [p["sp"][k] for p in P if p["sp"].get(k) is not None]
    res["perword"] = dict(n_conditions=len(P), median_spearman={k: (med(valid(k)) if valid(k) else None) for k in keys}, share_positive={k: (statistics.mean([v > 0 for v in valid(k)]) if valid(k) else None) for k in keys}, writers_beat_diff=statistics.mean([p["sp"]["writers"] > p["sp"]["diff_at_positions"] for p in P if p["sp"].get("writers") is not None and p["sp"].get("diff_at_positions") is not None]) if any(p["sp"].get("writers") is not None and p["sp"].get("diff_at_positions") is not None for p in P) else None, by_method={m: {k: (med([p["sp"][k] for p in P if p["method"] == m and p["sp"].get(k) is not None]) if [p for p in P if p["method"] == m and p["sp"].get(k) is not None] else None) for k in keys} for m in sorted(set(p["method"] for p in P))}, split=dict(writers_firing=med([p["split"]["writers_firing"] for p in P if p["split"].get("writers_firing") is not None]) if P else None, writers_quiet=med([p["split"]["writers_quiet"] for p in P if p["split"].get("writers_quiet") is not None]) if P else None), n_words_median=med([p["n_words"] for p in P]))
    rows = [x for p in P for x in p["rows"]]; res["perword"]["pooled"] = dict(n_words=len(rows), recovery_mean=statistics.mean([x["recovery"] for x in rows]))
    w = res["perword"]["median_spearman"]; res.setdefault("verdicts", {}); res["verdicts"]["W1"] = (w["writers"] or 0) >= 0.3; res["verdicts"]["W2"] = (w["writers"] or 0) > (w["diff_at_positions"] or 0)
# ---- the floor-corrected view, for the e618b controls and for the e617 silencing conditions
SETS4 = ("wdd_forget", "diff_selected", "ratio_selected", "random")
res["floor_corrected"] = dict(n_floor_files=len(FL), controls=floor_summary(floor_rows("e618b_controls_*.json", SETS), SETS), heldout_160m=floor_summary([r for r in floor_rows("e617_heldout_*.json", SETS4) if r["model"] == "pythia160"], SETS4), heldout_qwen05=floor_summary([r for r in floor_rows("e617_heldout_*.json", SETS4) if r["model"] == "qwen05"], SETS4), heldout_410m=floor_summary([r for r in floor_rows("e617_heldout_*.json", SETS4) if r["model"] == "pythia410"], SETS4))
fc = res["floor_corrected"]["controls"]
if fc:
    m_ = fc["median_floor_corrected_recovery"]; res.setdefault("verdicts", {}); res["verdicts"]["C1_floor"] = (1 - m_["active_retain"]) < 0.5 * (1 - m_["wdd_forget"]) if (1 - m_["wdd_forget"]) > 0 else False; res["verdicts"]["C2_floor"] = m_["wdd_retain"] >= 0.95 * fc["median_recovery_unsilenced"]; res["verdicts"]["C3_floor"] = m_["union_wdd_diff"] < min(m_["wdd_forget"], m_["diff_selected"])
cs = res["controls"].get("all"); pw = res["perword"]
FCS = ("; floor-corrected (n " + str(fc["n"]) + " control conditions): 20-step recovery of the recoverable rise after silencing " + ", ".join(f"{s_} {f2(v)}" for s_, v in fc["median_floor_corrected_recovery"].items()) + f" against {f2(fc['median_recovery_unsilenced'])} unsilenced; nats above the floor after 20 steps " + ", ".join(f"{s_} {v:.2f}" for s_, v in fc["median_nats_above_floor_after_20"].items()) + f" (unsilenced {fc['median_nats_above_original_after_20_unsilenced']:.2f} above the original); the floor itself rises " + ", ".join(f"{s_} {v:+.2f}" for s_, v in fc["median_floor_rise"].items())) if fc else "; floor-corrected: no floors yet"
fh = res["floor_corrected"]["heldout_160m"]; FHS = ("; e617 silencing at 160m floor-corrected (n " + str(fh["n"]) + "): recovery " + ", ".join(f"{s_} {f2(v)}" for s_, v in fh["median_floor_corrected_recovery"].items()) + f" against {f2(fh['median_recovery_unsilenced'])} unsilenced; floor rise " + ", ".join(f"{s_} {v:+.2f}" for s_, v in fh["median_floor_rise"].items())) if fh else ""
summ = (f"controls (n {cs['n']}): median extra steps back after silencing " + ", ".join(f"{s} {sg(cs['median_delay'][s])}" for s in SETS) + "; writers beat " + ", ".join(f"{s} in {f2(v)}" for s, v in cs["wdd_beats"].items()) + f"; union beats both in {f2(cs['union_beats_both'])}; by method " + "; ".join(f"{m}: " + ", ".join(f"{s} {sg(v[s])}" for s in ("wdd_forget", "diff_selected", "active_retain", "wdd_retain", "magnitude_selected", "union_wdd_diff", "random")) for m, v in cs["by_method"].items()) if cs else "controls: none") + ("; per-word (n " + str(pw["n_conditions"]) + " conditions, median " + str(pw["n_words_median"]) + " words): median Spearman with per-word recovery " + ", ".join(f"{k} {f2(v)}" for k, v in pw["median_spearman"].items()) + "; share positive " + ", ".join(f"{k} {f2(v)}" for k, v in pw["share_positive"].items()) + f"; writers beat the difference-selected set in {f2(pw['writers_beat_diff'])}; recovery with writers firing {f2(pw['split']['writers_firing'])} against quiet {f2(pw['split']['writers_quiet'])}; by method " + "; ".join(f"{m}: writers {f2(v['writers'])}, diff {f2(v['diff_at_positions'])}, rise {f2(v['rise'])}" for m, v in pw["by_method"].items()) if pw else "; per-word: none") + FCS + FHS + ("; verdicts " + ", ".join(f"{k} {'confirmed' if v else 'refuted'}" for k, v in res.get("verdicts", {}).items()))
res["summary"] = summ; res["_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"); json.dump(res, open(os.path.join(RES, res["_exp"] + ".json"), "w"), indent=1)
if not SMOKE: open(os.path.join(RES, "FINDINGS_box8.log"), "a").write(f"{res['_time']} e618d_controls_perword_analysis: {summ}\n")
print(summ)
