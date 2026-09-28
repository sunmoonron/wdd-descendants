"""e566b (session 102): e566 found rows by index between Pythia-160m and its weight-seed variant (a different
initialization) at cosine 0.11, above the 0.07 of the deduped twin that shares the initialization. Either the naming
is not what it seems or trained rows share a common component that any index pairing inherits. Here: the rows at step
0 of main and weight-seed1 by index (1.0 if the initialization is shared), the final rows by index against the same
rows with the index shuffled (the common-component null), with the common component removed (each model's mean row
subtracted) and for the data-seed and deduped pairs; e557b's retention of the initialization re-measured with the
mean removed. Pre-registered: I5 (0.7) the shuffled-index cosine matches the by-index one for weight-seed1 (no shared
initialization) and falls below it for deduped and data-seed1 (shared); I6 (0.6) the step-0 rows of weight-seed1
differ from main's (cosine under 0.05)."""
from s101_common import *
t0 = time.time(); HF = {"main": "EleutherAI/pythia-160m", "deduped": "EleutherAI/pythia-160m-deduped", "weight-seed1": "EleutherAI/pythia-160m-weight-seed1", "data-seed1": "EleutherAI/pythia-160m-data-seed1"}
def rows_at(hf, rev=None):
    wdd_common.MODELS["tmp"] = (hf, "neox"); m, _, fam = load_model("tmp", revision=rev); A, n = rows_of(Arch(m, fam), 6); del m; torch.cuda.empty_cache(); return A.cpu()
R = {k: rows_at(hf) for k, hf in HF.items()}; R0 = {k: rows_at(HF[k], "step0") for k in ("main", "weight-seed1", "data-seed1")}
g = torch.Generator().manual_seed(0); perm = torch.randperm(R["main"].shape[0], generator=g)
def cs(P, Q): return float((P * Q).sum(1).median())
def centred(P): return unitr(P - P.mean(0))
res = {}
for k in ("deduped", "weight-seed1", "data-seed1"):
    res[k] = dict(by_index=cs(R["main"], R[k]), shuffled=cs(R["main"], R[k][perm]), by_index_centred=cs(centred(R["main"]), centred(R[k])), shuffled_centred=cs(centred(R["main"]), centred(R[k])[perm]), mean_row_norm=float(R[k].mean(0).norm()))
    log(f"main | {k}: by index {res[k]['by_index']:.3f}, shuffled index {res[k]['shuffled']:.3f}; mean row removed: by index {res[k]['by_index_centred']:.3f}, shuffled {res[k]['shuffled_centred']:.3f} (mean row norm {res[k]['mean_row_norm']:.3f})")
res["step0"] = {"main|weight-seed1": cs(R0["main"], R0["weight-seed1"]), "main|data-seed1": cs(R0["main"], R0["data-seed1"]), "retention_main": cs(R0["main"], R["main"]), "retention_main_centred": cs(R0["main"], centred(R["main"])), "retention_main_shuffled": cs(R0["main"], R["main"][perm]), "retention_weight_seed1": cs(R0["weight-seed1"], R["weight-seed1"])}
log("step 0: " + ", ".join(f"{k} {v:.3f}" for k, v in res["step0"].items()))
summ = (f"rows by index main|weight-seed1 {res['weight-seed1']['by_index']:.3f} against shuffled {res['weight-seed1']['shuffled']:.3f} (mean row removed {res['weight-seed1']['by_index_centred']:.3f} vs {res['weight-seed1']['shuffled_centred']:.3f}); main|data-seed1 {res['data-seed1']['by_index']:.3f} vs shuffled {res['data-seed1']['shuffled']:.3f} (centred {res['data-seed1']['by_index_centred']:.3f} vs {res['data-seed1']['shuffled_centred']:.3f}); main|deduped {res['deduped']['by_index']:.3f} vs {res['deduped']['shuffled']:.3f} (centred {res['deduped']['by_index_centred']:.3f} vs {res['deduped']['shuffled_centred']:.3f}); "
        f"step-0 rows main|weight-seed1 {res['step0']['main|weight-seed1']:.3f}, main|data-seed1 {res['step0']['main|data-seed1']:.3f}; retention of the initialization {res['step0']['retention_main']:.3f} (shuffled {res['step0']['retention_main_shuffled']:.3f}, centred {res['step0']['retention_main_centred']:.3f}) | {time.time() - t0:.0f}s")
log(summ); record("e566b_index_null", res, summ)
