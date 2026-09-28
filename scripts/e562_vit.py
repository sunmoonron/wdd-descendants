"""e562 (session 101): does a vision transformer have a native vocabulary? Every word statistic so far comes from language
models, where a word's context set is partly a token identity. A ViT has no discrete tokens at its input: patches of
natural images. Same instrument as sessions 89-100, rows only: the states after block B of google/vit-base-patch16-224
(12 blocks, width 768) on 40 natural images (196 patch tokens each, the class token dropped as position 0 is dropped in
language models), the dictionary the MLP write rows of blocks 0..B (3072 per block). Three dictionaries (native, rotated,
Gaussian with the rows' second moment) compared by the fraction of variance unexplained at K=16 and the top-256 usage
share; the 256 most used rows' extreme-value statistics (S over the floor, counts, breadth) against the rotated
dictionary's; at B=3, 6 and 9. Pythia-160m at block 6 (same depth and a similar width) is measured by the same code for
scale. Pre-registered (probabilities are honest guesses):
 V1 (0.8) the native rows beat the rotated rows in FVU at K=16 at the middle block;
 V2 (0.55) the 256 most used rows sit over the floor at the median (S >= 1.2) — words exist without tokens;
 V3 (0.6) the ViT's FVU advantage at the middle block is smaller than Pythia-160m's at its middle block;
 V4 (0.5) the ViT words are broader than Pythia's (breadth cnt75/cnt higher): no token identity to narrow them."""
from s101_common import *
from transformers import ViTModel, ViTImageProcessor
t0 = time.time(); name = sys.argv[1] if len(sys.argv) > 1 else "google/vit-base-patch16-224"; NIMG = 40; TAG = "" if name == "google/vit-base-patch16-224" else "_" + name.split("/")[-1].replace("-", "_")
from transformers import AutoImageProcessor, AutoModel
proc = AutoImageProcessor.from_pretrained(name); model = AutoModel.from_pretrained(name, dtype=torch.float32).to(DEV).eval()
imgs, src = [], None
from datasets import load_dataset
for cand, split, key in (("ethz/food101", "validation", "image"), ("uoft-cs/cifar10", "test", "img")):
    try:
        ds = load_dataset(cand, split=split, streaming=True)
        for ex in ds:
            imgs.append(ex[key].convert("RGB"))
            if len(imgs) >= NIMG: break
        src = cand; break
    except Exception as e: log(f"{cand} unavailable: {str(e)[:200]}"); imgs = []
if not imgs: raise SystemExit("no images")
log(f"{len(imgs)} images from {src} in {time.time() - t0:.0f}s")
px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEV)
res = dict(model=name, images=src, n_images=len(imgs), blocks={})
LAYERS = model.layers if hasattr(model, "layers") else (model.encoder.layer if hasattr(model, "encoder") else model.encoder.layers); D = model.config.hidden_size
def down_proj(layer): return [m for m in layer.modules() if isinstance(m, torch.nn.Linear) and m.in_features == model.config.intermediate_size and m.out_features == D][0]
log(f"{len(LAYERS)} layers; block 0 children {[n for n, _ in LAYERS[0].named_children()]}; down projection {down_proj(LAYERS[0])}")
def vit_states(B):
    cap = {}
    h = LAYERS[B].register_forward_hook(lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    try:
        with torch.no_grad(): model(pixel_values=px)
    finally: h.remove()
    X = cap["x"][:, 1:].reshape(-1, cap["x"].shape[-1]); keep = ~sinkmask(X)
    A = unitr(torch.cat([down_proj(LAYERS[b]).weight.detach().float().T for b in range(B + 1)]))
    return unitr(X[keep] - X[keep].mean(0)), A, keep
for B in (3, 6, 9):
    U, A, keep = vit_states(B); c = dict_compare(U, A, K); st = c.pop("_st")
    c["n_positions"] = int(U.shape[0]); c["n_rows"] = int(A.shape[0]); c["sinks_dropped"] = int((~keep).sum())
    res["blocks"][B] = c; log(summarize_compare(f"ViT block {B} ({U.shape[0]} positions, {A.shape[0]} rows)", c))
    del U, A; torch.cuda.empty_cache()
del model; torch.cuda.empty_cache()
S = lm_states("pythia160"); cp = dict_compare(S["U"], S["A"], K); cp.pop("_st"); res["pythia160_block6"] = cp; log(summarize_compare("Pythia-160m block 6", cp))
v6, p6 = res["blocks"][6], cp
summ = (f"ViT-base ({src}, {res['n_images']} images): block 6 FVU own/rot/covA {v6['own_fvu']:.3f}/{v6['rot_fvu']:.3f}/{v6['covA_fvu']:.3f} (advantage {v6['advantage_fvu']:.3f}; Pythia-160m block 6 {p6['advantage_fvu']:.3f}), "
        f"top-256 usage share own/rot {v6['own_top256_usage_share']:.2f}/{v6['rot_top256_usage_share']:.2f} (Pythia {p6['own_top256_usage_share']:.2f}/{p6['rot_top256_usage_share']:.2f}); "
        f"ViT words: median S {v6['words']['median_S']:.2f} (rotated {v6['rotated_words']['median_S']:.2f}), share over the floor {v6['words']['share_S_over_1']:.2f}, breadth {v6['words']['median_breadth']:.2f} (Pythia words median S {p6['words']['median_S']:.2f}, breadth {p6['words']['median_breadth']:.2f}); "
        f"rows over the floor anywhere {v6['words']['rows_over_floor_anywhere']} vs rotated {v6['rotated_words']['rows_over_floor_anywhere']} (Pythia {p6['words']['rows_over_floor_anywhere']} vs {p6['rotated_words']['rows_over_floor_anywhere']}); "
        f"blocks 3/6/9 advantage {res['blocks'][3]['advantage_fvu']:.3f}/{v6['advantage_fvu']:.3f}/{res['blocks'][9]['advantage_fvu']:.3f}, words' median S {res['blocks'][3]['words']['median_S']:.2f}/{v6['words']['median_S']:.2f}/{res['blocks'][9]['words']['median_S']:.2f} | {time.time() - t0:.0f}s")
summ = f"{name}: " + summ; log(summ); record(f"e562_vit{TAG}", res, summ)
