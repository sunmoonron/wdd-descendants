"""e651 (2026-10-14, AI-safety detector evaluation): the coverage of single-row weight-edit detectors, read as a defender.
e623 built blind consistency detectors and found a sign flip invisible to them; e648b's read-write cosine would catch a
naive flip (it sends a row's read-write cosine from the block's bulk near -0.2 to +0.2). This standalone adds the read-write
cosine as a blind detector beside two others that need no dictionary machinery (the norm z-score for scaled rows, and the
activation-consistency cosine: does a write agree with the states where its neuron fires, minus its own contribution). The
defender's question: is there an edit that changes behaviour yet stays inside every detector's normal range? Each edit of a
block-4 row is scored by its behavioural effect (mean next-token KL from the unedited model on a wikitext corpus) and by
every detector's blind rank and threshold flag. Edits: double (control, caught by the norm), negate (baseline), a
norm-and-read-cosine-preserving reflection of the write about the read axis, and negate scaled to 0.25/0.5 (a sensitivity
sweep). Deliverable: the coverage table, which edits every detector misses and the behaviour those undetected edits carry.
Note vs e623: the dictionary-restricted word-chord detector is replaced by the activation-consistency detector over all
rows (the same statistic without the vocabulary restriction), since the floor machinery is not on this box. Standalone.
Arguments: model [--smoke]."""
import sys, os, time, json, math
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
MODELS = {"qwen05": "Qwen/Qwen2.5-0.5B", "smollm135": "HuggingFaceTB/SmolLM2-135M", "smollm360": "HuggingFaceTB/SmolLM2-360M", "olmo1b": "allenai/OLMo-1B-0724-hf", "pythia160": "EleutherAI/pythia-160m", "pythia410": "EleutherAI/pythia-410m"}
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); dev = "cuda:0" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODELS[name]); model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.float32).to(dev).eval()
fam = "neox" if hasattr(model, "gpt_neox") else "llama"; layers = model.gpt_neox.layers if fam == "neox" else model.model.layers
NB = len(layers); b = 4; D = model.config.hidden_size; TOPK = 32; NE = 3 if SMOKE else 8
mlp = lambda l: l.mlp
down = lambda: (mlp(layers[b]).dense_4h_to_h if fam == "neox" else mlp(layers[b]).down_proj).weight  # [D, DFF]
readw = (mlp(layers[b]).dense_h_to_4h.weight if fam == "neox" else mlp(layers[b]).up_proj.weight).detach().float()  # [DFF, D] row j = read
DFF = down().shape[1]; DOWN0 = down().detach().float().clone()
import datasets
lines = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.split()) > 40]
NW = 4 if SMOKE else 16; T = 192; PW = torch.stack([tok(t, return_tensors="pt").input_ids[0, :T] for t in lines[:NW] if tok(t, return_tensors="pt").input_ids.shape[1] >= T][:NW]).to(dev)
def logprobs():
    out = []
    for s0 in range(0, PW.shape[0], 4): out.append(model(PW[s0:s0 + 4]).logits[:, 1:].float().log_softmax(-1).cpu())
    return torch.cat(out).reshape(-1, model.config.vocab_size)
LP0 = logprobs()
def kl_effect(pos):
    """mean next-token KL(unedited || edited) over the given corpus positions (the edited neuron's top-firing tokens)"""
    LP1 = logprobs(); pe = LP0[pos].exp(); return float((pe * (LP0[pos] - LP1[pos])).sum(-1).mean())
def states_acts():
    """block-b output residual X [N,D] and the down_proj input activations A [N,DFF] over the corpus (positions 1..T-1)"""
    cap = {}
    h = (mlp(layers[b]).dense_4h_to_h if fam == "neox" else mlp(layers[b]).down_proj).register_forward_pre_hook(lambda m, a: cap.setdefault("a", []).append(a[0][:, 1:, :].detach().float().reshape(-1, DFF)))
    try:
        Xs = []
        for s0 in range(0, PW.shape[0], 4): out = model(PW[s0:s0 + 4], output_hidden_states=True); Xs.append(out.hidden_states[b + 1][:, 1:, :].detach().float().reshape(-1, D))
    finally: h.remove()
    return torch.cat(Xs), torch.cat(cap["a"])
X0, A0 = states_acts()
toppos = lambda j: A0[:, j].abs().topk(min(TOPK, A0.shape[0])).indices.cpu()
def detectors():
    W = down().detach().float(); nz = W.norm(dim=0); z = (nz - nz.mean()) / nz.std()
    wn = W / W.norm(dim=0, keepdim=True).clamp_min(1e-12); un = readw / readw.norm(dim=1, keepdim=True).clamp_min(1e-12); rw = (wn * un.T).sum(0)
    cons = torch.zeros(DFF, device=dev)
    for j0 in range(0, DFF, 512):
        js = torch.arange(j0, min(j0 + 512, DFF), device=dev); top = A0[:, js].abs().topk(TOPK, dim=0).indices
        for i, j in enumerate(js.tolist()):
            idx = top[:, i]; row = W[:, j]; rest = X0[idx] - A0[idx, j][:, None] * row[None]; cons[j] = torch.nn.functional.cosine_similarity(row[None], rest.mean(0, keepdim=True))[0]
    return z, rw, cons
z0, rw0, cons0 = detectors()
thr_cons = float(cons0.kthvalue(max(int(0.01 * DFF), 1)).values); thr_rw = float(rw0.kthvalue(max(int(0.99 * DFF), 1)).values)
print(f"[{name}] block {b}/{NB}: read-write cosine median {float(rw0.median()):+.3f}, 99th pct {thr_rw:+.3f}, share below -0.1 {float((rw0 < -0.1).float().mean()):.2f}; consistency median {float(cons0.median()):+.3f}, 1st pct {thr_cons:+.3f}", flush=True)
rng = np.random.default_rng(651); use = float(A0.abs().mean(0).cpu().numpy().std()); act = A0.abs().mean(0); word_like = torch.nonzero(act > act.median() * 3)[:, 0].tolist(); other = [j for j in range(DFF) if j not in set(word_like)]
pick_hi = [int(x) for x in np.array(word_like)[rng.permutation(len(word_like))[:NE]]] if len(word_like) >= NE else word_like
pick_lo = [int(x) for x in np.array(other)[rng.permutation(len(other))[:NE]]]
Wm = down()
def set_edit(j, how):
    w = DOWN0[:, j]
    if how == "double": Wm[:, j] = 2 * w
    elif how == "negate": Wm[:, j] = -w
    elif how == "reflect": u = readw[j] / readw[j].norm().clamp_min(1e-12); Wm[:, j] = -w + 2 * (w @ u) * u
    elif how.startswith("negate@"): eps = float(how.split("@")[1]); Wm[:, j] = (1 - 2 * eps) * w
rank_desc = lambda s, j: int((s.argsort(descending=True) == j).nonzero()[0, 0]) + 1
rank_asc = lambda s, j: int((s.argsort(descending=False) == j).nonzero()[0, 0]) + 1
HOWS = ["double", "negate", "reflect", "negate@0.25", "negate@0.5"]; res = dict(model=name, block=b, DFF=DFF, read_write_median=float(rw0.median()), read_write_p99=thr_rw, consistency_p01=thr_cons, share_rw_below_m01=float((rw0 < -0.1).float().mean()), edits=[])
for kind, picks in (("high_use", pick_hi), ("low_use", pick_lo)):
    for how in HOWS:
        for j in picks:
            pos = toppos(j); set_edit(j, how); kl = kl_effect(pos); z1, rw1, cons1 = detectors(); Wm[:, j] = DOWN0[:, j]
            ent = dict(kind=kind, edit=how, row=int(j), kl_effect=kl, z=float(z1[j]), rw=float(rw1[j]), cons=float(cons1[j]), rank_norm=rank_desc(z1.abs(), j), rank_rw=rank_desc(rw1, j), rank_cons=rank_asc(cons1, j),
                       flag_norm=bool(float(z1[j].abs()) > 3), flag_rw=bool(float(rw1[j]) > thr_rw), flag_cons=bool(float(cons1[j]) < thr_cons))
            ent["flag_any"] = bool(ent["flag_norm"] or ent["flag_rw"] or ent["flag_cons"]); res["edits"].append(ent)
        E = [e for e in res["edits"] if e["kind"] == kind and e["edit"] == how]; m = lambda key: float(np.mean([e[key] for e in E]))
        print(f"[{name}] {kind} {how}: KL {m('kl_effect'):.3f}; flag-any {m('flag_any'):.2f} (norm {m('flag_norm'):.2f} read-write {m('flag_rw'):.2f} consistency {m('flag_cons'):.2f}); read-write cos {m('rw'):+.3f}, consistency {m('cons'):+.3f} ({time.time()-t0:.0f}s)", flush=True)
S = {}
for kind in ("high_use", "low_use"):
    for how in HOWS:
        E = [e for e in res["edits"] if e["kind"] == kind and e["edit"] == how]; S[f"{kind}_{how}"] = {key: float(np.mean([e[key] for e in E])) for key in ("kl_effect", "flag_any", "flag_norm", "flag_rw", "flag_cons", "rw", "cons")}
res["summary"] = S
neg = [S[f"{k}_negate"] for k in ("high_use", "low_use")]; dbl = [S[f"{k}_double"] for k in ("high_use", "low_use")]; refl = [S[f"{k}_reflect"] for k in ("high_use", "low_use")]
res["verdicts"] = dict(V1=all(s["flag_rw"] >= 0.5 for s in neg), V2=all(s["flag_norm"] >= 0.5 for s in dbl), V3=all(s["flag_rw"] <= 0.2 for s in refl),
                       V4=all(s["flag_any"] <= 0.2 for s in refl), V5=all(S[f"{k}_reflect"]["kl_effect"] >= 0.5 * S[f"{k}_negate"]["kl_effect"] for k in ("high_use", "low_use")))
summ = f"detector coverage ({name} block {b}): " + "; ".join(f"{k} KL {S[k]['kl_effect']:.2f} flag-any {S[k]['flag_any']:.2f} (norm {S[k]['flag_norm']:.2f} rw {S[k]['flag_rw']:.2f} cons {S[k]['flag_cons']:.2f})" for k in S) + f"; verdicts {res['verdicts']}"
print(summ, flush=True); os.makedirs("results", exist_ok=True); json.dump(res, open(f"results/e651_detect_floor_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=float); print("done", f"{time.time()-t0:.0f}s")
