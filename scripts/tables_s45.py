"""Local: session-45 tables (e444-e448) from results/*.json."""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
def j(n):
    p = os.path.join(R, n + ".json"); return json.load(open(p)) if os.path.exists(p) else None
f2 = lambda x: "  n/a" if x is None else f"{x:5.2f}"
print("e444 grokking: per variant, a time series (step: test acc | k4 FVU own/rot/covA/pca | emb Fourier top-5)")
for v in ["grok", "grok_s1", "nowd", "randlab", "frozenW", "frozenR"]:
    r = j(f"e444_grok_{v}")
    if not r: print(v, "missing"); continue
    print(f"== {v}: grok step {r['grok_step']}, FVU-advantage half step {r['adv_fvu_half_step']}, word-level half step {r['wordlevel_fvu_half_step']}, Fourier half step {r['fourier_half_step']}, "
          f"top-20 purity {r['mean_purity_top']:.2f} vs all {r['mean_purity_all_mlp']:.2f}, dominant freq in key {r['top_in_key']:.2f}, usage by type {r['usage_share_by_type']}")
    for row in r["log"]:
        if row["step"] % 1000 == 0 and row["step"] <= 12000 or row["step"] % 5000 == 0:
            print(f"  {row['step']:6d}: train {row['train_acc']:.2f} test {row['test_acc']:.2f} | FVU own {row['own_k4_fvu']:.2f} rot {row['rot_k4_fvu']:.2f} covA {row['covA_k4_fvu']:.2f} pca {row['pca_k4_fvu']:.2f} | "
                  f"rec own {f2(row['own_k4_train'])} covA {f2(row['covA_k4_train'])} pca {f2(row['pca_k4_train'])} | Fourier {row['emb_top5_fourier']:.2f} | wnorm {row['wnorm']:.1f}")
for n in ["e445_translate_160m_410m", "e445_translate_410m_160m", "e445_translate_410m_1b"]:
    r = j(n)
    if r: print(n, json.dumps(r["match"]), json.dumps({k: round(v, 3) for k, v in r["translate"].items()}))
for m in ["gpt2", "smollm2", "pythia410", "qwen05", "olmo1b"]:
    r = j(f"e446_meaning_{m}")
    if r: print("e446", m, " ; ".join(f"{u} {v['token_auc_median']:.3f}/{v['coh_cur_median']:+.3f}/{v['coh_next_median']:+.3f}/{v['selfcons_median']:.3f}" for u, v in r["units"].items()), "| rho usage~auc", round(r["native_rho_usage_vs_auc"], 2))
for lam in ["0.0", "0.3", "1.0", "3.0"]:
    r = j(f"e447_trainsd_gpt2_lam{lam}")
    if r: print("e447", lam, r["_exp"], "LM", round(r["test_lm_old"], 3), "->", round(r["test_lm_new"], 3), "gap16", {k: round(v, 3) for k, v in r["gap_new"].items()}, "old", {k: round(v, 3) for k, v in r["gap_old"].items()}, "cells", {c: {k: round(x, 2) for k, x in d.items()} for c, d in r["cells"].items()})
for m in ["qwen05", "smollm2"]:
    r = j(f"e448_interlingua_{m}")
    if r: print("e448", m, json.dumps({k: v for k, v in r["retrieval"].items()}), "shared", json.dumps(r["shared_words"].get(str(max(int(x) for x in r['shared_words'])), [])[:5]))
