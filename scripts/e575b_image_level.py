"""e575b (session 103): e575 found no shared partition between ViT-base and DeiT-base at the patch level (twins 0.01-0.09).
Is the partition shared at the image level? Each word's context set is taken as the set of images in which it is over
the floor at one patch or more (40 images), twins by Jaccard 0.25 against the rotated null; and a spatial check: each
word's set as the set of patch positions (of 196) it occupies in any image, twins likewise. If image-level twins are
high and patch-level ones are not, the two models partition the same images by different patches (spatial
implementations of the same classes); if both are low, the partition is not the data's for these two models."""
from s101_common import *
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
t0 = time.time(); NIMG = 40; JT = 0.25; B = 6
imgs = []; ds = load_dataset("ethz/food101", split="validation", streaming=True)
for ex in ds:
    imgs.append(ex["image"].convert("RGB"))
    if len(imgs) >= NIMG: break
def jac(P, Q):
    P = P.float(); Q = Q.float(); inter = P.T @ Q; return inter / (P.sum(0)[:, None] + Q.sum(0)[None] - inter).clamp_min(1)
def sets_for(name):
    proc = AutoImageProcessor.from_pretrained(name); model = AutoModel.from_pretrained(name, dtype=torch.float32).to(DEV).eval(); px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEV)
    L = model.layers if hasattr(model, "layers") else model.encoder.layer; D = model.config.hidden_size
    def dp(layer): return [m for m in layer.modules() if isinstance(m, torch.nn.Linear) and m.in_features == model.config.intermediate_size and m.out_features == D][0]
    cap = {}; h = L[B].register_forward_hook(lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    try:
        with torch.no_grad(): model(pixel_values=px)
    finally: h.remove()
    X = cap["x"][:, 1:].reshape(-1, D); keep = ~sinkmask(X); A = unitr(torch.cat([dp(L[b]).weight.detach().float().T for b in range(B + 1)])); U = unitr(X[keep] - X[keep].mean(0))
    st = stats(U, A, K); w = torch.nonzero(wordset(st["usage"]))[:, 0]; Ar = unitr(rotate(A, seed=7)); sr = stats(U, Ar, K); wr = torch.nonzero(wordset(sr["usage"]))[:, 0]
    full = torch.zeros(X.shape[0], w.numel(), dtype=torch.bool); full[keep.cpu()] = st["ratio"][:, w].float() > 1; fullr = torch.zeros(X.shape[0], wr.numel(), dtype=torch.bool); fullr[keep.cpu()] = sr["ratio"][:, wr].float() > 1
    NP = X.shape[0] // NIMG; img = lambda M: M.reshape(NIMG, NP, -1).any(1); patch = lambda M: M.reshape(NIMG, NP, -1).any(0)
    del model; torch.cuda.empty_cache(); return dict(patchpos=full, patchpos_r=fullr, image=img(full), image_r=img(fullr), spatial=patch(full), spatial_r=patch(fullr), NP=NP)
V, Dt = sets_for("google/vit-base-patch16-224"), sets_for("facebook/deit-base-patch16-224"); res = {}
for level in ("patchpos", "image", "spatial"):
    r = {}
    for a, b, SA, SB in (("vit", "deit", V, Dt), ("deit", "vit", Dt, V)):
        best = jac(SA[level], SB[level]).max(1).values; bestr = jac(SA[level], SB[level + "_r"]).max(1).values
        r[f"{a}->{b}"] = dict(share=float((best >= JT).float().mean()), median=float(best.median()), null_share=float((bestr >= JT).float().mean()), null_median=float(bestr.median()))
    r["median_set_size_vit"] = float(V[level].sum(0).float().median()); res[level] = r
    log(f"{level} sets (ViT median size {r['median_set_size_vit']:.0f}): ViT->DeiT {r['vit->deit']['share']:.2f} (median Jaccard {r['vit->deit']['median']:.2f}; null {r['vit->deit']['null_share']:.2f} / {r['vit->deit']['null_median']:.2f}), DeiT->ViT {r['deit->vit']['share']:.2f} (null {r['deit->vit']['null_share']:.2f})")
summ = (f"vision partition at three levels (ViT-base vs DeiT-base, {NIMG} images): patch positions {res['patchpos']['vit->deit']['share']:.2f} / {res['patchpos']['deit->vit']['share']:.2f} (nulls {res['patchpos']['vit->deit']['null_share']:.2f} / {res['patchpos']['deit->vit']['null_share']:.2f}); images {res['image']['vit->deit']['share']:.2f} / {res['image']['deit->vit']['share']:.2f} (nulls {res['image']['vit->deit']['null_share']:.2f} / {res['image']['deit->vit']['null_share']:.2f}; median Jaccard {res['image']['vit->deit']['median']:.2f} vs null {res['image']['vit->deit']['null_median']:.2f}); spatial (patch index over images) {res['spatial']['vit->deit']['share']:.2f} / {res['spatial']['deit->vit']['share']:.2f} (nulls {res['spatial']['vit->deit']['null_share']:.2f} / {res['spatial']['deit->vit']['null_share']:.2f}) | {time.time() - t0:.0f}s")
log(summ); record("e575b_image_level", res, summ)
