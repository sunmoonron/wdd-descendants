"""e566d (session 102): e566c found every step-0 tensor of pythia-160m-weight-seed1 differs from main's, yet the trained
rows correspond by index at 0.11-0.18 (shuffled 0.00), which a different initialization cannot produce: permutation
symmetry leaves nothing for the data to align by index. Either the uploaded step 0 is not the initialization the run
started from, or something is shared. The early checkpoints decide: weight-seed1's rows at steps 512 and 1000 against
its own step 0 and against main's step 0 and main's steps 512 and 1000; the same for data-seed1 and main itself.
Pre-registered: I8 (0.6) weight-seed1's early rows correlate with main's step 0 and not with its own uploaded step 0."""
from s101_common import *
t0 = time.time(); HF = {"main": "EleutherAI/pythia-160m", "weight-seed1": "EleutherAI/pythia-160m-weight-seed1", "data-seed1": "EleutherAI/pythia-160m-data-seed1"}
def rows_at(hf, rev):
    wdd_common.MODELS["tmp"] = (hf, "neox"); m, _, fam = load_model("tmp", revision=rev); A = rows_of(Arch(m, fam), 6)[0].cpu(); del m; torch.cuda.empty_cache(); return A
R = {(k, rev): rows_at(hf, rev) for k, hf in HF.items() for rev in ("step0", "step512", "step1000")}
cs = lambda P, Q: float((P * Q).sum(1).median()); res = {}
for k in HF:
    for rev in ("step512", "step1000"):
        res[f"{k} {rev}"] = {"own step0": cs(R[(k, rev)], R[(k, "step0")]), "main step0": cs(R[(k, rev)], R[("main", "step0")]), f"main {rev}": cs(R[(k, rev)], R[("main", rev)])}
        log(f"{k} {rev}: vs own step 0 {res[f'{k} {rev}']['own step0']:.3f}, vs main step 0 {res[f'{k} {rev}']['main step0']:.3f}, vs main {rev} {res[f'{k} {rev}'][f'main {rev}']:.3f}")
w, d = res["weight-seed1 step512"], res["data-seed1 step512"]
summ = (f"weight-seed1 at step 512: rows vs its own uploaded step 0 {w['own step0']:.3f}, vs main's step 0 {w['main step0']:.3f}, vs main at 512 {w['main step512']:.3f}; data-seed1 at 512: own step 0 {d['own step0']:.3f}, main's step 0 {d['main step0']:.3f}, main at 512 {d['main step512']:.3f}; main at 512 vs its step 0 {res['main step512']['own step0']:.3f} | {time.time() - t0:.0f}s")
log(summ); record("e566d_init_check", res, summ)
