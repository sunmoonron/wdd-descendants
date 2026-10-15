"""e667b (session 150; Gemini 4's fix for e667's blind spot): a reflection of a write row about its read axis evades the
norm, the read-write cosine and, scored honestly (e667), mostly the activation-consistency check. Gemini's proposal: compare
the write's component orthogonal to the read axis, w_perp = w - (w.u)u, with the residual state before block b's MLP adds
anything (the block input plus attention), averaged over the row's top-32 firing positions: C_perp = cos(w_perp, mean x_pre).
Tampering with block b's write column cannot change x_pre or the firing positions, and the reflection maps w_perp to
-w_perp exactly, so C_perp flips sign exactly; whether that makes the row an outlier depends on where its clean C_perp sits.
e651's battery unchanged (four models, block 4, the same corpus, rows and edits); each edit scored blind on the tampered
model by C_perp (raw and with the corpus mean of x_pre removed), flagged two-sided outside the 0.5th-99.5th percentile of
the tampered model's own distribution (the same 1% false-flag budget as e651's one-sided checks) and one-sided below the
1st; e667's own-state consistency alongside. Arguments: model [--smoke]. Pre-registered in e667b_prereg.json."""
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
def states3():
    """over the corpus positions 1..T-1: block b's output X, its pre-MLP residual X - mlp_out (block input plus attention), and the down_proj inputs A"""
    cap = {"x": [], "m": [], "a": []}
    hs = [layers[b].register_forward_hook(lambda m, a, o: cap["x"].append((o[0] if isinstance(o, tuple) else o)[:, 1:].detach().float().reshape(-1, D))),
          mlp(layers[b]).register_forward_hook(lambda m, a, o: cap["m"].append((o[0] if isinstance(o, tuple) else o)[:, 1:].detach().float().reshape(-1, D))),
          (mlp(layers[b]).dense_4h_to_h if fam == "neox" else mlp(layers[b]).down_proj).register_forward_pre_hook(lambda m, a: cap["a"].append(a[0][:, 1:].detach().float().reshape(-1, DFF)))]
    try:
        for s0 in range(0, PW.shape[0], 4): model(PW[s0:s0 + 4])
    finally: [h.remove() for h in hs]
    X = torch.cat(cap["x"]); return X, X - torch.cat(cap["m"]), torch.cat(cap["a"])
X0, XP0, A0 = states3(); MU0 = XP0.mean(0)
toppos = lambda j: A0[:, j].abs().topk(min(TOPK, A0.shape[0])).indices.cpu()
U = readw / readw.norm(dim=1, keepdim=True).clamp_min(1e-12)
def cperp(XP, A, W, centred):
    out = torch.empty(DFF, device=dev); top = A.abs().topk(TOPK, dim=0).indices; mu = XP.mean(0) if centred else torch.zeros(D, device=dev)
    for j0 in range(0, DFF, 256):
        js = torch.arange(j0, min(j0 + 256, DFF), device=dev); idx = top[:, js]; m = (XP[idx] - mu).mean(0); w = W[:, js].T; wp = w - (w * U[js]).sum(1, keepdim=True) * U[js]
        out[js] = torch.nn.functional.cosine_similarity(wp, m, dim=1)
    return out
def consistency(X, A, W):
    out = torch.empty(DFF, device=dev); top = A.abs().topk(TOPK, dim=0).indices
    for j0 in range(0, DFF, 256):
        js = torch.arange(j0, min(j0 + 256, DFF), device=dev); idx = top[:, js]; w = W[:, js].T; rest = X[idx] - A[idx, js][..., None] * w[None]
        out[js] = torch.nn.functional.cosine_similarity(w, rest.mean(0), dim=1)
    return out
pctl = lambda v, q: float(v.kthvalue(max(int(q * DFF), 1)).values)
W0 = down().detach().float(); C0 = {c: cperp(XP0, A0, W0, c) for c in (False, True)}; cons0 = consistency(X0, A0, W0)
print(f"[{name}] block {b}/{NB}: clean C_perp raw median {float(C0[False].median()):+.3f} (0.5th / 99.5th pct {pctl(C0[False], 0.005):+.3f} / {pctl(C0[False], 0.995):+.3f}), centred median {float(C0[True].median()):+.3f} (0.5th / 99.5th {pctl(C0[True], 0.005):+.3f} / {pctl(C0[True], 0.995):+.3f})", flush=True)
rng = np.random.default_rng(651); act = A0.abs().mean(0); word_like = torch.nonzero(act > act.median() * 3)[:, 0].tolist(); other = [j for j in range(DFF) if j not in set(word_like)]
pick_hi = [int(x) for x in np.array(word_like)[rng.permutation(len(word_like))[:NE]]] if len(word_like) >= NE else word_like
pick_lo = [int(x) for x in np.array(other)[rng.permutation(len(other))[:NE]]]
Wm = down()
def set_edit(j, how):
    w = DOWN0[:, j]
    if how == "double": Wm[:, j] = 2 * w
    elif how == "negate": Wm[:, j] = -w
    elif how == "reflect": u = readw[j] / readw[j].norm().clamp_min(1e-12); Wm[:, j] = -w + 2 * (w @ u) * u
rank_desc = lambda s, j: int((s.argsort(descending=True) == j).nonzero()[0, 0]) + 1
pr_clean = lambda c, j: float((C0[c] < C0[c][j]).float().mean())   # the row's clean percentile
HOWS = ["double", "negate", "reflect"]; res = dict(model=name, block=b, DFF=DFF, clean=dict(raw_median=float(C0[False].median()), raw_p005=pctl(C0[False], 0.005), raw_p995=pctl(C0[False], 0.995), cent_median=float(C0[True].median()), cent_p005=pctl(C0[True], 0.005), cent_p995=pctl(C0[True], 0.995)), picks=dict(high_use=pick_hi, low_use=pick_lo), edits=[])
for kind, picks in (("high_use", pick_hi), ("low_use", pick_lo)):
    for how in HOWS:
        for j in picks:
            pos = toppos(j); set_edit(j, how); kl = kl_effect(pos); X1, XP1, A1 = states3(); W1 = down().detach().float().clone(); Wm[:, j] = DOWN0[:, j]
            C1 = {c: cperp(XP1, A1, W1, c) for c in (False, True)}; cons1 = consistency(X1, A1, W1); others = torch.ones(DFF, dtype=torch.bool, device=dev); others[j] = False
            ent = dict(kind=kind, edit=how, row=int(j), kl_effect=kl, use_ratio=float(act[j] / act.median()), xpre_unchanged=bool(torch.allclose(XP1, XP0, atol=1e-3)), others_unchanged=float((C1[False][others] - C0[False][others]).abs().max()))
            for c, lab in ((False, "raw"), (True, "cent")):
                v, v0 = float(C1[c][j]), float(C0[c][j]); lo, hi, lo1 = pctl(C1[c], 0.005), pctl(C1[c], 0.995), pctl(C1[c], 0.01)
                ent.update({f"c_{lab}": v, f"c_{lab}_clean": v0, f"c_{lab}_clean_pct": pr_clean(c, j), f"flag2_{lab}": bool(v < lo or v > hi), f"flag1_{lab}": bool(v < lo1), f"rank_{lab}": rank_desc((C1[c] - C1[c].median()).abs(), j)})
            ent["cons_own"] = float(cons1[j]); ent["flag_cons_own"] = bool(float(cons1[j]) < pctl(cons1, 0.01)); res["edits"].append(ent)
        E = [e for e in res["edits"] if e["kind"] == kind and e["edit"] == how]; m = lambda key: float(np.mean([e[key] for e in E]))
        print(f"[{name}] {kind} {how}: KL {m('kl_effect'):.3f}; C_perp raw flagged two-sided {m('flag2_raw'):.2f}, below 1st pct {m('flag1_raw'):.2f}; centred two-sided {m('flag2_cent'):.2f}; own-state consistency {m('flag_cons_own'):.2f}; clean C_perp raw {m('c_raw_clean'):+.3f} -> {m('c_raw'):+.3f} (clean percentile {m('c_raw_clean_pct'):.2f}) ({time.time() - t0:.0f}s)", flush=True)
S = {}
for kind in ("high_use", "low_use"):
    for how in HOWS:
        E = [e for e in res["edits"] if e["kind"] == kind and e["edit"] == how]; S[f"{kind}_{how}"] = {key: float(np.mean([e[key] for e in E])) for key in ("kl_effect", "flag2_raw", "flag1_raw", "flag2_cent", "flag1_cent", "flag_cons_own", "c_raw", "c_raw_clean", "c_raw_clean_pct", "c_cent", "c_cent_clean")} | {"n": len(E)}
res["summary"] = S; res["exact"] = dict(xpre_unchanged=all(e["xpre_unchanged"] for e in res["edits"]), others_max_change=max(e["others_unchanged"] for e in res["edits"]),
                                        flip_exact=max(abs(e["c_raw"] + e["c_raw_clean"]) for e in res["edits"] if e["edit"] in ("negate", "reflect")))
summ = f"[{name}] C_perp: clean raw median {res['clean']['raw_median']:+.3f}; " + "; ".join(f"{k} KL {v['kl_effect']:.3f} flagged two-sided raw {v['flag2_raw']:.2f} / centred {v['flag2_cent']:.2f} / own-state consistency {v['flag_cons_own']:.2f}" for k, v in S.items()) + f"; exactness {res['exact']}"
print(summ, flush=True); os.makedirs("results", exist_ok=True); json.dump(res, open(f"results/e667b_cperp_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("done", f"{time.time() - t0:.0f}s", flush=True)
