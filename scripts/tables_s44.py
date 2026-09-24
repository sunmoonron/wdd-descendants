"""Local: print the session-44 tables (e432-e443) from results/*.json."""
import json, os, glob
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
M5 = ["gpt2", "smollm2", "pythia410", "qwen05", "olmo1b"]
def j(n):
    p = os.path.join(R, n + ".json"); return json.load(open(p)) if os.path.exists(p) else None
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"
f3 = lambda x: "n/a" if x is None else f"{x:+.3f}"
print("e432/e437 sinks and M"); print("model | sinks | sink var share | M share all -> typical | overlap | Fisher share (chance) | Fisher/variance with -> without sinks")
for m in M5:
    a, h = j(f"e432_manatomy_{m}"), j(f"e437_sinkhygiene_{m}")
    if not a: continue
    au = a["audit"]; hb = h["b"] if h else None
    print(m, au["n_sinks_ev"], f2(au["sink_share_of_variance"]), f2(au["M_all_share_all"]), "->", f2(au["M_ns_share_ns"]), f2(au["overlap_M_all_M_ns"]),
          (f2(hb["M_ns"]["fisher"]) + f" ({hb['M_ns']['chance']:.3f})") if hb else "", (f2(hb["M_all"]["fisher_over_var_all"]) + " -> " + f2(hb["M_ns"]["fisher_over_var_typical"])) if hb else "")
print("\ne432 knee and retention"); print("model | M dL a=0 | knee M / pc9-16 / pc33-40 / rand8 / matched | cost per unit variance M / pc33-40 | whole-state dL | kept at L+1: M / rand8")
for m in M5:
    a = j(f"e432_manatomy_{m}")
    if not a: continue
    F = {k: v["_fit"] for k, v in a["scale"].items()}; b1 = str(a["track_blocks"][0])
    print(m, f3(F["M_ns"]["meas_a0"]), "/".join(f"{F[k]['knee_ratio_eps0.25']:.1f}" for k in ["M_ns", "pc9_16", "pc33_40", "rand8", "rand8_matched"]),
          f"{F['M_ns']['cost_per_variance_a0']:.2f}/{F['pc33_40']['cost_per_variance_a0']:.2f}", f3(a["ablate"]["whole_state_nonsink"]["dL_all"]),
          f"{a['restore']['M_ns@0.5'][b1]['in_subspace']:.2f}/{a['restore']['rand8@0.5'][b1]['in_subspace']:.2f}")
print("\ne433 identity"); print("model | token/position/both/block-0 explained: M ; whole | M parts removal dL (energy share): token / within-token")
for m in M5:
    a = j(f"e433_identity_{m}")
    if not a: continue
    E, P = a["explained"], a["parts"]
    print(m, "/".join(f2(E['M'][k]) for k in ["token", "position", "token_plus_position", "block0_state"]), ";", "/".join(f2(E['whole'][k]) for k in ["token", "position", "token_plus_position", "block0_state"]),
          "|", f"{P['token']['0.0']['dL']:+.3f} ({P['token']['energy_share_of_M']:.2f}) / {P['within_token']['0.0']['dL']:+.3f} ({P['within_token']['energy_share_of_M']:.2f})")
print("\ne438/e441 knee decomposition"); print("model | live | P | P+M | P+M+N | M-only | TV M/random")
for m in M5:
    a, b = j(f"e441_kneedecomp_{m}"), j(f"e438_kneesoftmax_{m}")
    if not a: continue
    R_ = a["runs"]
    print(m, " | ".join(f"{R_['M_' + c]['0.0']:+.3f} ({R_['M_' + c]['knee']:.2f})" for c in ["live", "P", "P_M", "P_M_N", "M"]), "|", f"{b['runs']['M_live']['tv_mean']:.3f}/{b['runs']['rand8_matched_live']['tv_mean']:.3f}" if b else "")
print("\ne434 retention at L+1 (medians)")
for m in M5:
    a = j(f"e434_kept_{m}")
    if not a: continue
    g = a["groups"]; print(m, " ".join(f"{k} {v['r1_median']:.2f}" for k, v in g.items()), "| partial r~usage|var", f2(a["own_relations"]["partial_r_usage_given_varshare"]))
print("\ne435 surrogates: own-over-rotation FVU advantage k16 (share of real) | Zipf real/gauss")
for m in M5:
    a = j(f"e435_surrogates_{m}")
    if not a: continue
    ad = a["advantage"]; r = ad["real"]["16"]["rot"]
    print(m, " ".join(f"{s} {ad[s]['16']['rot']:+.3f} ({ad[s]['16']['rot'] / r:.2f})" for s in ["real", "gauss", "shuffled", "lexical"]), "|", f2(a["usage"]["real"]["zipf"]), f2(a["usage"]["gauss"]["zipf"]))
print("\ne439 lexicon: gap k16 full / no lexicon / lexicon only | covA share full / no lexicon | names current token")
for n in [f"e439_lexicon_{m}_final" for m in M5] + [f"e439_lexicon_pythia410_{s}" for s in ["step1000", "step4000", "step16000"]]:
    a = j(n)
    if not a: continue
    G = a["gaps"]; print(n[12:], f"{G['full']['16']:+.3f} / {G['no_lexicon']['16']:+.3f} / {G['lexicon_only']['16']:+.3f}", "|", f2(a["covA_share"]["full"]["16"]), f2(a["covA_share"]["no_lexicon"]["16"]), "|", f2(a["naming"]["current_any"]))
print("\ne440 context part: non-lexicon own / rot / covA / mix8 (k16) -> shares | lexical part: own / lexicon / rot")
for m in M5:
    a = j(f"e440_contextvocab_{m}")
    if not a: continue
    C, X = a["parts"]["context"]["rec"], a["parts"]["lexical"]["rec"]; s = a["context_shares_k16"]
    print(m, f"var {a['parts']['context']['variance_share']:.2f}", "/".join(f2(C[k]['16']) for k in ["own_nolex", "rot_nolex", "covA_nolex", "mix8_nolex"]), "->", f2(s["covA"]), f2(s["mix8"]), "|", "/".join(f2(X[k]['16']) for k in ["own", "lexicon", "rot"]))
print("\ne442 light cone: recovery self own/rot, broadcast own/rot, all own/rot")
for m in M5:
    a = j(f"e442_lightcone_{m}")
    if not a: continue
    r = a["recovery"]; print(m, f"{r['own']['self_']:.2f}/{r['rot']['self_']:.2f}", f"{r['own']['broadcast']:.2f}/{r['rot']['broadcast']:.2f}", f"{r['own']['all']:.2f}/{r['rot']['all']:.2f}", "| adv self/bc", f"{a['advantage']['self_']:+.2f}/{a['advantage']['broadcast']:+.2f}")
print("\ne436 increments vs states: increment k16 own/rot/covA/mix8 -> shares | state shares")
for n in [f"e436_increment_{m}_final" for m in M5] + [f"e436_increment_pythia410_{s}" for s in ["step1000", "step4000", "step16000", "step143000"]]:
    a = j(n)
    if not a: continue
    c = a["increment"]["cells"]; s = a["increment"]["shares"]["16"]; ss = a["state"]["shares"]["16"]
    print(n[15:], "/".join(f2(c[v]['16']['rec']) for v in ["own", "rot", "covA", "mix8"]), "->", f2(s["covA"]), f2(s["mix8"]), "| state", f2(ss["covA"]), f2(ss["mix8"]), "| PR", f"{a['increment']['mlp_write_participation_ratio_median']:.0f}")
print("\ne443 block accents: k16 own/rot/covA/covA_block/mix8/mix8_block -> shares covA / covA_block / mix8 / mix8_block")
for n in [f"e443_blockaccents_{m}_final" for m in M5] + [f"e443_blockaccents_pythia410_{s}" for s in ["step1000", "step4000", "step16000"]]:
    a = j(n)
    if not a: continue
    r, s = a["rec"], a["shares"]
    print(n[18:], "/".join(f2(r[v]['16']) for v in ["own", "rot", "covA", "covA_block", "mix8", "mix8_block"]), "->", " / ".join(f2(s[v]['16']) for v in ["covA", "covA_block", "mix8", "mix8_block"]))
