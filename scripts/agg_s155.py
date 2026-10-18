"""agg_s155: one table per experiment from the e680-e691 result JSONs (laptop, numpy only). Usage: python agg_s155.py <results_dir>"""
import json, glob, os, sys, collections
import numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else "results"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import s155_load
def load(prefix):
    D = s155_load.load(R, prefix)
    for d in D.values():
        if "summary_stats" in d: d["summary"] = d["summary_stats"]
    return D
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); rx = x.argsort().argsort(); ry = y.argsort().argsort(); return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")
def tally(rows, keys):
    out = {}
    for k in keys: v = [r["verdicts"].get(k) for r in rows if "verdicts" in r and k in r["verdicts"]]; out[k] = f"{sum(bool(x) for x in v)}/{len(v)}"
    return out
f2 = lambda x: "n/a" if x is None else (f"{x:.2f}" if isinstance(x, (int, float)) else str(x))
print("=" * 100)
# e680
D = load("e680")
if D:
    print("e680 floor as a causal boundary: KL ratio native/random (own, foreign, rotated) at 0.25 .. 4 floors")
    for k, d in D.items():
        S = d["summary_by_condition"]; print(f"  {k:12s} own " + " ".join(f"{S['own'][a]['ratio_kl_p']:.2f}" for a in S["own"]) + " | foreign " + " ".join(f"{S['foreign'][a]['ratio_kl_p']:.2f}" for a in S["foreign"]) + " | rotated " + " ".join(f"{S['rotated'][a]['ratio_kl_p']:.2f}" for a in S["rotated"]) + f" | random KL " + " ".join(f"{S['random'][a]['kl_p']:.1e}" for a in S["random"]))
    print("  verdicts", tally(D.values(), ["C1_sub_floor_native_is_random", "C2_own_above_floor_exceeds_random", "C3_foreign_above_floor_is_random", "C4_rotated_is_random"]))
# e681
D = load("e681")
if D:
    print("\ne681 weak activations: KL (nats) keeping the strongest 25/10/5/1% | below 0.02/0.05/0.1/0.2/0.5 of max | mean-replaced quarter | dial/middle/detector/random third (quarter kept) | last block | flipped vs unflipped margin")
    for k, d in D.items():
        c = d["conditions"]; g = lambda key: c[key]["kl_mean"]
        print(f"  {k:10s} " + "/".join(f"{g(f'keep_{q}_all'):.3f}" for q in (0.25, 0.1, 0.05, 0.01)) + " | " + "/".join(f"{g(f'theta_{t}_all'):.3f}" for t in (0.02, 0.05, 0.1, 0.2, 0.5)) + f" | {g('keep_0.25_all_mean'):.3f} | {g('keep_0.25_dial'):.3f}/{g('keep_0.25_middle'):.3f}/{g('keep_0.25_detector'):.3f}/{g('keep_0.25_random_third'):.3f} | {g('keep_0.25_last_block'):.3f} | {f2(c['keep_0.25_all']['margin_flipped_median'])} vs {c['keep_0.25_all']['margin_unflipped_median']:.3f} | energy removed {c['keep_0.25_all']['removed_energy_fraction']:.3f}")
    print("  verdicts", tally(D.values(), ["W1_keep_quarter_kl_below_005", "W1b_keep_quarter_above_02", "W1c_theta01_kl_below_005", "W2_dial_cost_2x_detector", "W3_flipped_margin_half", "W4_last_block_per_neuron_2x", "W5_mean_cheaper_than_zero"]))
# e682
D = load("e682")
if D:
    print("\ne682 facts: related share (facts / random / delta) | codes (related / random) | Jaccard same value / same relation / different | codes among themselves | depth facts/codes | base overlap | max cos change | steps, acc")
    for k, d in D.items():
        S = d["summary"]; P = S["pairs_facts"]; print(f"  {k:10s} {S['facts_related_share']:.2f}/{S['facts_random_related_share']:.2f}/{S['facts_delta_related_share']:.2f} | {S['codes_related_share']:.2f}/{S['codes_random_related_share']:.2f} | {f2(P['same_value'])}/{f2(P['same_relation'])}/{f2(P['different_relation'])} | {f2(S['pairs_codes']['same_relation'])} | {S['facts_depth_median']:.2f}/{S['codes_depth_median']:.2f} | {S['facts_base_overlap']:.2f} | {1 - S['min_write_cos']:.4f} | {S['steps']}, {S['final_acc']}, codes {S['final_code']:.2f}")
    print("  verdicts", tally(D.values(), ["F1_writers_related_half", "F1b_random_below_01", "F2_same_value_jaccard_2x_different", "F3_codes_related_at_most_half_of_facts", "F3b_codes_share_writers_more_than_facts_same_relation", "F4_depth_in_last_third", "F5_no_rewritten_row"]))
# e683
D = load("e683")
if D:
    print("\ne683 growth topology: chain/old/jump (attribution) | expected chain, enrichment, share > 1.5 | state chain/jump | root vs final one-token | two-route: offshoot chain above core")
    for k, d in D.items():
        S = d["summary"]; print(f"  {k:12s} {S['median_chain_attr']:.2f}/{S['median_old_attr']:.2f}/{S['median_jump_attr']:.2f} | {S['median_expected_chain']:.2f}, {S['median_enrichment_attr']:.2f}, {S['share_enrichment_above_15']:.2f} | {S['median_chain_state']:.2f}/{S['median_jump_state']:.2f} | {S['median_root_one_token']:.2f} vs {S['median_final_one_token']:.2f} | n {S['n_two_route']}: {f2(S['share_off_chain_above_core'])}")
    print("  verdicts", tally(D.values(), ["G1_chain_enriched_70pct", "G2_jump_median_below_015", "G3_root_more_token_concentrated", "G4_offshoot_chain_above_core_60pct"]))
# e684
D = load("e684")
if D:
    print("\ne684 readers after recruitment: readers before -> after | word gain > control (count / top-10 share / mass) | alignment gain row vs class | specialists before -> after (null) | n words")
    for k, d in D.items():
        S = d["summary"]; print(f"  {k:12s} {S['median_readers_before']} -> {S['median_readers_after']} | {f2(S['share_word_gain_above_ctrl'])}/{f2(S['share_top10_gain_above_ctrl'])}/{f2(S['share_mass_gain_above_ctrl'])} | {f2(S['align_row_gain'])} vs {f2(S['align_class_gain'])} (row above class {f2(S['share_row_gain_above_class'])}) | {S['median_spec_before']} -> {S['median_spec_after']} ({S['median_spec_null_after']}; gain above null {f2(S['share_spec_gain_above_null'])}) | {S['n']}")
    print("  verdicts", tally(D.values(), ["R1_readers_gain_above_control_65pct", "R1b_top10_share_gain_above_control_65pct", "R1c_mass_gain_above_control_65pct", "R2_row_alignment_gain_above_class", "R3_specialists_gain_above_null_60pct"]))
# e685
D = load("e685")
if D:
    print("\ne685 readers' ledger: N_eff strong/weak | top-16 share | over-floor count, energy | KL ratio at K=1,4,16,64,256,1024,all (strong) | (all positions) | K to halve")
    for k, d in D.items():
        S = d["summary"]; ks = [str(x) for x in d["ks"]]; print(f"  {k:16s} {S['neff_strong']:.0f}/{S['neff_weak']:.0f} | {S['top16_share_strong']:.2f} | {f2(S['over_floor_count'])}, {f2(S['over_floor_energy'])} | " + " ".join(f"{S['sim_ratio_strong'][K]:.2f}" for K in ks) + " | " + " ".join(f"{S['sim_ratio_all'][K]:.2f}" for K in ks) + f" | {S.get('k_half_strong')} / {S.get('k_half_all')}")
    print("  verdicts", tally(D.values(), ["L0_all_inputs_reproduce", "L1_strong_needs_fewer_inputs", "L2_nick_ratio16_below_025", "L2_nanda_ratio16_above_05", "L3_over_floor_energy_above_half"]))
# e686
D = load("e686")
if D:
    print("\ne686 negative pole: per block (near share, median cos vs rotated-tail null, z, next-token overlap above null)")
    for k, d in D.items():
        print(f"  {k:10s} " + " | ".join(f"b{b}: {s['summary']['near_share']:.2f} ({s['summary']['median_cos']:+.2f} vs {s['summary']['null_rot_mean']:+.2f}, z {s['summary']['median_z_rot']:.1f}, nxt {s['summary']['next_overlap_above_null']:.2f})" for b, s in d["blocks"].items()))
    print("  verdicts", tally(D.values(), ["N1_near_above_half_at_middle", "N2_rises_with_depth", "N3_next_token_overlap_above_03"]))
# e687
D = load("e687")
if D:
    print("\ne687 explanation recall: per block (recall at precision 0.5 / 0.8 | precision at recall 0.9/0.7/0.5 | unseen-context positives | rule exists, size | one-token share)")
    for k, d in D.items():
        print(f"  {k:10s} " + " | ".join(f"b{b}: {s['summary']['median_recall_at_precision50']:.2f}/{s['summary']['median_recall_at_precision80']:.2f} | {s['summary']['median_precision_at_recall90']:.2f}/{s['summary']['median_precision_at_recall70']:.2f}/{s['summary']['median_precision_at_recall50']:.2f} | {s['summary']['median_unseen_share']:.2f} | {s['summary']['rule_exists_share']:.2f}, {s['summary']['median_rule_size_when_exists']} | {s['summary']['median_one_token_share']:.2f}" for b, s in d["blocks"].items()))
    print("  verdicts", tally(D.values(), ["X1_gao_precision90_median_below_03", "X1_nick_above_06", "X2_gao_rule_size_above_50_or_none", "X2_nick_rule_size_le10", "X3_surface_covers_half_at_precision_half"]))
# e688
D = load("e688")
if D:
    print("\ne688 plant a knot: mode seed | vote G/N before -> pin end -> release 1000/2000/4000/6000 | dials gender/number at pin end -> end | val before -> pin -> end")
    for k, d in D.items():
        tr = {t["label"]: t for t in d["trace"]}; v = lambda t, c: f"{t['vote'][c]['share']:.2f}"; lab = [l for l in ("before", "pin_end", "release_1000", "release_2000", "release_4000", "release_6000") if l in tr]
        print(f"  {k:14s} row {d['row']} | " + " -> ".join(f"{v(tr[l], 'G')}/{v(tr[l], 'N')}" for l in lab) + f" | {tr['pin_end']['dial']['gender']:+.3f}/{tr['pin_end']['dial']['number']:+.3f} -> {tr[lab[-1]]['dial']['gender']:+.3f}/{tr[lab[-1]]['dial']['number']:+.3f} | {tr['before']['val']:.3f} -> {tr['pin_end']['val']:.3f} -> {tr[lab[-1]]['val']:.3f}")
    knots = [d for k, d in D.items() if k.startswith("knot")]
    if knots:
        pe = sum(all(d["verdict_inputs"]["pin_end_votes"][c] >= 0.8 for c in ("G", "N")) for d in knots); shed = sum(any(d["verdict_inputs"]["end_votes"][c] < 0.5 for c in ("G", "N")) for d in knots); stable = sum(all(d["verdict_inputs"]["end_votes"][c] >= 0.8 for c in ("G", "N")) for d in knots)
        print(f"  knots: both installed at pin end {pe}/{len(knots)}; one shed by the end {shed}/{len(knots)}; both kept {stable}/{len(knots)}")
# e689
D = load("e689")
if D:
    print("\ne689 width: model d | rows over floor (per dim) | own share | crowd share | N_eff terms | top-3 norm share")
    fam = collections.defaultdict(list)
    for k, d in D.items():
        S = d["summary"]; print(f"  {k:10s} {d['d']:5d} | {S['over_floor_count_median']:6.0f} ({S['over_floor_per_dim']:.3f}) | {S['own_share_of_resultant_median']:.2f} | {S['crowd_share_median']:.2f} | {S['neff_terms_median']:6.0f} | {S['top3_norm_share_median']:.2f}")
        fam[k.rstrip("0123456789b")[:4]].append((d["d"], S["over_floor_count_median"], S["crowd_share_median"], S["neff_terms_median"]))
    for f, v in fam.items():
        if len(v) >= 3: v.sort(); print(f"  family {f}: Spearman with d: over-floor {spearman([x[0] for x in v], [x[1] for x in v]):+.2f}, crowd share {spearman([x[0] for x in v], [x[2] for x in v]):+.2f}, N_eff {spearman([x[0] for x in v], [x[3] for x in v]):+.2f}")
# e690
D = load("e690")
if D:
    print("\ne690 memorised: depth (mem / random / low-loss) | class entropy | field share | distinct writers")
    for k, d in D.items():
        m, r, l = d["memorised"], d["random"], d["low_loss"]; print(f"  {k:10s} {m['depth_median']:.2f}/{r['depth_median']:.2f}/{l['depth_median']:.2f} | {m['class_entropy_median']:.2f}/{r['class_entropy_median']:.2f}/{l['class_entropy_median']:.2f} | {f2(m['field_z_share_mean'])}/{f2(r['field_z_share_mean'])}/{f2(l['field_z_share_mean'])} | {m['distinct_median']}/{r['distinct_median']}/{l['distinct_median']} (n {m['n']}/{r['n']}/{l['n']})")
    print("  verdicts", tally(D.values(), ["M1_memorised_deeper_than_controls", "M2_class_entropy_not_higher", "M3_field_not_lower"]))
# e691
D = load("e691")
if D:
    print("\ne691 reader consensus: Spearman tokens~readers, state~readers, weight~readers, say~readers | NN tokens->readers (chance) | mismatch")
    for k, d in D.items():
        s = d["spearman"]; print(f"  {k:16s} {s['act~read']:+.3f}, {s['state~read']:+.3f}, {s['weight~read']:+.3f}, {s['say~read']:+.3f} | {d['nn_agreement']['act->read']:.2f} ({d['chance_nn']:.3f}) | {d['mismatch_share']:.2f}")
    print("  verdicts", tally(D.values(), ["C1_spearman_act_read_above_03", "C2_nn_agreement_above_03", "C3_weight_read_above_05"]))
