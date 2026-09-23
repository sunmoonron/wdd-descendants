"""e365b: drift of the functional coordinate across Pythia-410m training, in residual coordinates and basis-free. For
each checkpoint of e365: overlap with the final checkpoint of (i) the top-16 quotient subspace in residual coordinates,
(ii) the principal subspace of its logit image on the fixed vocabulary sketch, (iii) the column space of its linear
response map, against the same overlaps for random residual directions' images (the null: how much of the image
stability is just the unembedding's dominant directions) and against chance (16/256). Rows ordered by step, with the
induction metrics of the same run beside them."""
import os, glob, json, re, torch
RES = os.environ.get("WDD_RESULTS", "/workspace/wdd/results")
def orth(A): return torch.linalg.qr(A.float())[0]
def inside(A, B): A, B = orth(A), orth(B); return ((B.T @ A) ** 2).sum().item() / A.shape[1]
fs = sorted(glob.glob(f"{RES}/e365/*.pt"), key=lambda f: int(re.search(r"step(\d+)", f).group(1))); D = {int(re.search(r"step(\d+)", f).group(1)): torch.load(f) for f in fs}; T = max(D); fin = D[T]; rows = []
for s, d in sorted(D.items()):
    j = json.load(open(f"{RES}/e365_inddev_step{s}.json")) if os.path.exists(f"{RES}/e365_inddev_step{s}.json") else {}
    r = j.get("result", j)
    rows.append(dict(step=s, residual_overlap=inside(d["Q16"], fin["Q16"]), image_overlap=inside(d["Bq"], fin["Bq"]), response_map_overlap=inside(d["Mq"].T, fin["Mq"].T), random_image_overlap=inside(d["Br"], fin["Br"]), quotient_vs_final_random_image=inside(d["Bq"], fin["Br"]), gain_ratio=d["gain_q"] / max(d["gain_r"], 1e-12), induction_gain=r.get("induction_gain"), max_prefix=r.get("max_prefix_score"), quotient_dim=r.get("quotient_dim"), quotient_full=r.get("quotient_full")))
out = dict(final_step=T, chance=16 / 256, rows=rows)
json.dump(out, open(f"{RES}/e365b_image_drift.json", "w"), indent=1)
line = "e365b quotient drift, residual vs basis-free (overlap with final; chance 0.06): " + "; ".join(f"step{r['step']}: resid {r['residual_overlap']:.2f} image {r['image_overlap']:.2f} map {r['response_map_overlap']:.2f} rand-image {r['random_image_overlap']:.2f} gain x{r['gain_ratio']:.1f} ind {r['induction_gain'] if r['induction_gain'] is None else round(r['induction_gain'], 2)}" for r in rows)
open(f"{RES}/FINDINGS.log", "a").write(line + "\n"); print(line)
