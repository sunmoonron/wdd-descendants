"""e564b (session 101): a control for e564. Fine-tuning Pythia-160m on TinyStories raised the Pile loss by 0.6 to 2.1
nats and kept 0.69-0.77 of the Pile word set. Is that more or less than a random perturbation of the weights doing the
same damage? Gaussian noise added to every weight matrix in proportion to its own scale (relative sigma 0.5%, 1%, 2%,
4%), the Pile loss, the Pile word set's Jaccard with the unperturbed model, the usage correlation, the rows' rotation
and the start words still over the floor, measured as in e564 (block 6, rows 0-6). Read against e564's three learning
rates at matched loss damage. Pre-registered (probabilities are honest guesses):
 N1 (0.55) at matched Pile-loss damage, fine-tuning keeps more of the vocabulary than noise (the training direction
    moves the states along the model's own words; noise moves them anywhere);
 N2 (0.7) noise of matched damage rotates the rows far more than fine-tuning did (fine-tuning's rows moved 1e-4)."""
from s101_common import *
t0 = time.time(); model, tok, fam = load_model("pythia160"); arch = Arch(model, fam); B = 6; pile = pile_ids("pythia160"); pile2 = pile_ids("pythia160", start=8)
sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
def lossof(ids):
    with torch.no_grad(): lg = model(ids).logits.float()
    return float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)))
def measure():
    S = lm_states("pythia160", B=B, ids=pile, model=model); st = stats(S["U"], S["A"], K); return dict(words=wordset(st["usage"]), usage=st["usage"], S=st["S"], A=S["A"].cpu(), loss=(lossof(pile) + lossof(pile2)) / 2)
h0 = measure(); res = dict(loss_start=h0["loss"], rows={}); g = torch.Generator(device=DEV).manual_seed(0)
for rel in (0.005, 0.01, 0.02, 0.04):
    with torch.no_grad():
        model.load_state_dict(sd)
        for k, v in model.state_dict().items():
            if v.is_floating_point() and v.dim() == 2: v.add_(torch.randn(v.shape, generator=g, device=DEV) * rel * v.std())
    h = measure(); w0, w = h0["words"], h["words"]; rot = 1 - (h0["A"] * h["A"]).sum(1)
    r = dict(loss=h["loss"], delta_loss=h["loss"] - h0["loss"], pile_words_jaccard=float((w & w0).sum() / (w | w0).sum()), usage_corr=float(torch.corrcoef(torch.stack([h0["usage"], h["usage"]]))[0, 1]), rot_words_median=float(rot[w0].median()), rot_all_median=float(rot.median()),
             share_start_words_over_floor=float((h["S"][w0] >= 1).float().mean()), median_S_start_words=float(h["S"][w0].median()))
    res["rows"][rel] = r; log(f"relative noise {rel}: loss {h0['loss']:.3f} -> {h['loss']:.3f} (+{r['delta_loss']:.2f}); Pile word set Jaccard {r['pile_words_jaccard']:.2f}, usage corr {r['usage_corr']:.2f}, rows' rotation words {r['rot_words_median']:.4f}, start words over the floor {r['share_start_words_over_floor']:.2f}")
ft = {}
for tag, f in (("2e-5", "e564_continual"), ("5e-6", "e564_continual_lr5e-6"), ("2e-6", "e564_continual_lr2e-6")):
    try: j = json.load(open(f"/workspace/wdd/results/{f}.json")); r = j["rows"]["300"]; ft[tag] = dict(delta_loss=r["loss_pile"] - j["rows"]["0"]["loss_pile"], jaccard=r["pile_words_jaccard"], rot=r["rot_words_median"], usage_corr=r["usage_corr_pile"])
    except Exception as e: log(f"no e564 result for lr {tag}: {e}")
res["finetune_reference"] = ft
summ = ("noise on Pythia-160m's weight matrices: " + "; ".join(f"sigma {rel}: loss +{r['delta_loss']:.2f}, word set Jaccard {r['pile_words_jaccard']:.2f}, usage corr {r['usage_corr']:.2f}, rows' rotation {r['rot_words_median']:.4f}" for rel, r in res["rows"].items())
        + " | fine-tuning (e564): " + "; ".join(f"lr {k}: loss +{v['delta_loss']:.2f}, Jaccard {v['jaccard']:.2f}, usage corr {v['usage_corr']:.2f}, rotation {v['rot']:.4f}" for k, v in ft.items()) + f" | {time.time() - t0:.0f}s")
log(summ); record("e564b_noise_control", res, summ)
