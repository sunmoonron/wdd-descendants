"""e569 (session 102): the stability hierarchy. Four levels at which a vocabulary can be stable under a perturbation:
row identity (the reference words' rows against the same rows after), dictionary membership (the word sets' Jaccard),
the usage distribution (correlation over all rows), and function: whether the reference words still describe the
perturbed states (the FVU of the perturbed states under the reference words' rows against the perturbed model's own
words, K=8) and whether they still matter to it (the loss cost of zeroing the reference words' write columns in the
perturbed model against zeroing its own words'). Perturbations: Gaussian weight noise at three scales and continued
training on TinyStories at two learning rates (Pythia-160m, block 6, rows 0-6); training itself between checkpoints
1000-2000, 8000-16000 and 16000-end (Pythia-410m, block 12, rows 0-12). Pre-registered (probabilities are honest
guesses):
 S1 (0.7) the levels order as row identity > usage > function > membership under noise and fine-tuning: membership
    is the fragile level, not function;
 S2 (0.6) under training between 8000 and 16000 the old words still describe the new states within 10% of the new
    words' FVU, though membership is 0.37;
 S3 (0.5) the old words' ablation cost in the new model is at least half the new words' own."""
from s101_common import *
from datasets import load_dataset
t0 = time.time(); torch.set_grad_enabled(False)
def lossof(model, ids):
    lg = model(ids).logits.float(); return float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)))
def snapshot(name, model, B, ids):
    S = lm_states(name, B=B, ids=ids, model=model); st = stats(S["U"], S["A"], K); return dict(X=S["X"], keep=S["keep"], A=S["A"], words=wordset(st["usage"]), usage=st["usage"], S=st["S"], arch=S["arch"], model=model, B=B)
def ablate_words(snap, rows):
    arch = snap["arch"]; DFF = arch.DFF; saved = []
    for r in rows.tolist():
        b, j = r // DFF, r % DFF; lin = arch.mlp_lin(b); saved.append((lin, j, lin.weight[:, j].clone())); lin.weight[:, j] = 0
    return lambda: [lin.weight.__setitem__((slice(None), j), w) for lin, j, w in saved]
def levels(ref, new, ids, tag):
    keepc = ref["keep"] & new["keep"]; Xn = new["X"][keepc]; Un = unitr(Xn - Xn.mean(0)); wr, wn = torch.nonzero(ref["words"])[:, 0], torch.nonzero(new["words"])[:, 0]
    l1 = float((ref["A"][wr] * new["A"][wr]).sum(1).median()); l2 = float((ref["words"] & new["words"]).sum() / (ref["words"] | new["words"]).sum()); l3 = float(torch.corrcoef(torch.stack([ref["usage"], new["usage"]]))[0, 1])
    tot = float(Un.pow(2).sum()); fvu = {}
    for k, Dd in (("ref_words_old_vectors", ref["A"][wr]), ("ref_words_new_vectors", new["A"][wr]), ("new_words", new["A"][wn])):
        sel, cof, err = omp(Un, Dd, 8, batch=1024, record_err=True); fvu[k] = float(err[:, -1].sum() / tot)
    base = lossof(new["model"], ids); cost = {}
    for k, rows in (("ref_words", wr), ("new_words", wn)):
        restore = ablate_words(new, rows); cost[k] = lossof(new["model"], ids) - base; restore()
    r = dict(row_identity=l1, membership=l2, usage=l3, fvu=fvu, function_describe=fvu["new_words"] / fvu["ref_words_old_vectors"], function_matter=cost["ref_words"] / max(cost["new_words"], 1e-6), ablation_cost=cost, share_ref_words_over_floor=float((new["S"][wr] >= 1).float().mean()), n_positions=int(keepc.sum()))
    log(f"{tag}: row identity {l1:.3f}, membership {l2:.2f}, usage {l3:.2f}, describe (new words' FVU / old words' FVU on the new states) {r['function_describe']:.2f} ({fvu['new_words']:.3f}/{fvu['ref_words_old_vectors']:.3f}; old words with their new vectors {fvu['ref_words_new_vectors']:.3f}), matter (old words' ablation cost / new words') {r['function_matter']:.2f} ({cost['ref_words']:+.3f}/{cost['new_words']:+.3f} nats), old words over the floor {r['share_ref_words_over_floor']:.2f}")
    return r
res = dict(pythia160={}, pythia410={})
# ---- Pythia-160m: noise and fine-tuning ----
model, tok, fam = load_model("pythia160"); B = 6; pile = pile_ids("pythia160"); ref = snapshot("pythia160", model, B, pile); sd = {k: v.detach().clone() for k, v in model.state_dict().items()}; g = torch.Generator(device=DEV).manual_seed(0)
for rel in (0.005, 0.02, 0.04):
    model.load_state_dict(sd)
    for k, v in model.state_dict().items():
        if v.is_floating_point() and v.dim() == 2: v.add_(torch.randn(v.shape, generator=g, device=DEV) * rel * v.std())
    new = snapshot("pythia160", model, B, pile); res["pythia160"][f"noise_{rel}"] = levels(ref, new, pile, f"noise {rel}"); res["pythia160"][f"noise_{rel}"]["delta_loss"] = lossof(model, pile) - lossof(ref["model"], pile) if False else None
model.load_state_dict(sd); L0 = lossof(model, pile)
def pack(split, nwin, T=256):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True); buf, wins = [], []
    for ex in ds:
        buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
        while len(buf) >= T + 1 and len(wins) < nwin: wins.append(buf[:T + 1]); buf = buf[T + 1:]
        if len(wins) >= nwin: break
    return torch.tensor(wins, device=DEV)
train = pack("train", 300 * 8)
for lr in (2e-6, 2e-5):
    model.load_state_dict(sd); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0); model.train()
    with torch.enable_grad():
        for step in range(300):
            ids = train[step * 8:(step + 1) * 8]; lg = model(ids[:, :256]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:256].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); new = snapshot("pythia160", model, B, pile); r = levels(ref, new, pile, f"fine-tune lr {lr:g}"); r["delta_loss"] = lossof(model, pile) - L0; res["pythia160"][f"finetune_{lr:g}"] = r
for rel in (0.005, 0.02, 0.04):   # loss deltas for the noise runs (recomputed cheaply)
    model.load_state_dict(sd); g2 = torch.Generator(device=DEV).manual_seed(0)
    for k, v in model.state_dict().items():
        if v.is_floating_point() and v.dim() == 2: v.add_(torch.randn(v.shape, generator=g2, device=DEV) * rel * v.std())
    res["pythia160"][f"noise_{rel}"]["delta_loss"] = lossof(model, pile) - L0
del model; torch.cuda.empty_cache()
# ---- Pythia-410m: training ----
B = 12; pile4 = pile_ids("pythia410"); snaps = {}
for rev in ("step1000", "step2000", "step8000", "step16000", None):
    mdl, _, _ = load_model("pythia410", revision=rev); snaps[rev or "final"] = snapshot("pythia410", mdl, B, pile4)
for a, b in (("step1000", "step2000"), ("step8000", "step16000"), ("step16000", "final")):
    r = levels(snaps[a], snaps[b], pile4, f"training {a} -> {b}"); r["delta_loss"] = lossof(snaps[b]["model"], pile4) - lossof(snaps[a]["model"], pile4); res["pythia410"][f"{a}->{b}"] = r
P1, P4 = res["pythia160"], res["pythia410"]; f2 = lambda x: f"{x:.2f}"
summ = ("stability hierarchy (row identity / membership / usage / describe / matter): " + "; ".join(f"{k} ({v['delta_loss']:+.2f} nats): {v['row_identity']:.3f} / {f2(v['membership'])} / {f2(v['usage'])} / {f2(v['function_describe'])} / {f2(v['function_matter'])}" for k, v in P1.items())
        + " | training: " + "; ".join(f"{k} ({v['delta_loss']:+.2f} nats): {v['row_identity']:.3f} / {f2(v['membership'])} / {f2(v['usage'])} / {f2(v['function_describe'])} / {f2(v['function_matter'])}" for k, v in P4.items()) + f" | {time.time() - t0:.0f}s")
log(summ); record("e569_stability_hierarchy", res, summ)
