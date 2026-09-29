"""e575 (session 103): do two vision models agree on a partition? ViT-base and DeiT-base (same architecture, different
training recipes: supervised on ImageNet-21k then 1k against distillation on ImageNet-1k) on the same 40 food101
images: the 256 words of each at block 6 (rows 0-6), their over-the-floor patch sets, and the twin rate at Jaccard
0.25 in both directions against the rotated-dictionary null. The same for blocks 3 and 9. Pre-registered:
 V5 (0.55) the twin rate is at the language models' level (0.3-0.5) against a null under 0.05."""
from s101_common import *
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
t0 = time.time(); NIMG = 40; JT = 0.25
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
    out = {}
    for B in (3, 6, 9):
        cap = {}; h = L[B].register_forward_hook(lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
        try:
            with torch.no_grad(): model(pixel_values=px)
        finally: h.remove()
        X = cap["x"][:, 1:].reshape(-1, D); keep = ~sinkmask(X); A = unitr(torch.cat([dp(L[b]).weight.detach().float().T for b in range(B + 1)])); out[B] = dict(X=X.cpu(), keep=keep.cpu(), A=A)
    del model; torch.cuda.empty_cache(); return out
V, Dt = sets_for("google/vit-base-patch16-224"), sets_for("facebook/deit-base-patch16-224"); res = dict(blocks={})
for B in (3, 6, 9):
    keepc = V[B]["keep"] & Dt[B]["keep"]; S_ = {}
    for tag, M in (("vit", V[B]), ("deit", Dt[B])):
        U = unitr((M["X"][keepc] - M["X"][keepc].mean(0)).to(DEV)); st = stats(U, M["A"], K); w = torch.nonzero(wordset(st["usage"]))[:, 0]; Ar = unitr(rotate(M["A"], seed=7)); sr = stats(U, Ar, K); wr = torch.nonzero(wordset(sr["usage"]))[:, 0]
        S_[tag] = dict(over=st["ratio"][:, w].float() > 1, over_r=sr["ratio"][:, wr].float() > 1, S=float(st["S"][w].median())); torch.cuda.empty_cache()
    r = {}
    for a, b in (("vit", "deit"), ("deit", "vit")):
        best = jac(S_[a]["over"], S_[b]["over"]).max(1).values; bestr = jac(S_[a]["over"], S_[b]["over_r"]).max(1).values
        r[f"{a}->{b}"] = dict(share=float((best >= JT).float().mean()), median=float(best.median()), null_share=float((bestr >= JT).float().mean()), null_median=float(bestr.median()))
    r["n_positions"] = int(keepc.sum()); r["median_set_size"] = float(S_["vit"]["over"].sum(0).float().median()); res["blocks"][B] = r
    log(f"block {B} ({int(keepc.sum())} patch positions): ViT->DeiT twins {r['vit->deit']['share']:.2f} (null {r['vit->deit']['null_share']:.2f}, median Jaccard {r['vit->deit']['median']:.2f}), DeiT->ViT {r['deit->vit']['share']:.2f} (null {r['deit->vit']['null_share']:.2f}); median set size {r['median_set_size']:.0f}")
b6 = res["blocks"][6]
summ = (f"vision twins on {NIMG} food101 images: block 6 ViT->DeiT {b6['vit->deit']['share']:.2f} (null {b6['vit->deit']['null_share']:.2f}), DeiT->ViT {b6['deit->vit']['share']:.2f} (null {b6['deit->vit']['null_share']:.2f}); blocks 3/9 ViT->DeiT {res['blocks'][3]['vit->deit']['share']:.2f}/{res['blocks'][9]['vit->deit']['share']:.2f} (nulls {res['blocks'][3]['vit->deit']['null_share']:.2f}/{res['blocks'][9]['vit->deit']['null_share']:.2f}) | {time.time() - t0:.0f}s")
log(summ); record("e575_vision_twins", res, summ)
