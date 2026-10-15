"""e667 (session 149; found while building the live demo): e651 scored its blind activation-consistency detector with
the clean model's states. Its X0, A0 come from the unedited model once, before any edit, and every detector pass after an
edit reuses them, so the edited write is compared with states that still contain the original write: the 'rest' is
X0 - a w_new = (X - a w_old) + a (w_old - w_new), which adds an artifact pointing against the edit. A blind auditor has
only the tampered model, whose own states give rest = X - a w_old exactly (the edit at block b's write leaves block b's
activations unchanged). Here e651's battery is rerun unchanged (same models, corpus, rows, edits, thresholds) and every
edit is scored both ways: consistency with the clean states (e651's, for reproduction) and with the edited model's own
states (the blind auditor's), its threshold the 1st percentile of the edited model's own distribution. The norm z-score
and the read-write cosine need no states and are unchanged. Arguments: model [--smoke]. Pre-registered in
e667_prereg.json."""
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
def consistency(X, A, W):
    """vectorized e651 statistic: for every row j, the cosine of its write with the mean over its top-32 firing positions of the state minus its own write"""
    out = torch.empty(DFF, device=dev); top = A.abs().topk(TOPK, dim=0).indices
    for j0 in range(0, DFF, 256):
        js = torch.arange(j0, min(j0 + 256, DFF), device=dev); idx = top[:, js]; w = W[:, js].T; rest = X[idx] - A[idx, js][..., None] * w[None]
        out[js] = torch.nn.functional.cosine_similarity(w, rest.mean(0), dim=1)
    return out
def detectors(X=None, A=None):
    W = down().detach().float(); nz = W.norm(dim=0); z = (nz - nz.mean()) / nz.std()
    wn = W / W.norm(dim=0, keepdim=True).clamp_min(1e-12); un = readw / readw.norm(dim=1, keepdim=True).clamp_min(1e-12); rw = (wn * un.T).sum(0)
    return z, rw, consistency(X0 if X is None else X, A0 if A is None else A, W)
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
            pos = toppos(j); set_edit(j, how); kl = kl_effect(pos); z1, rw1, cons1 = detectors(); Xe, Ae = states_acts(); cons_own = consistency(Xe, Ae, down().detach().float()); thr_own = float(cons_own.kthvalue(max(int(0.01 * DFF), 1)).values); Wm[:, j] = DOWN0[:, j]
            ent = dict(kind=kind, edit=how, row=int(j), kl_effect=kl, z=float(z1[j]), rw=float(rw1[j]), cons=float(cons1[j]), cons_own=float(cons_own[j]), rank_norm=rank_desc(z1.abs(), j), rank_rw=rank_desc(rw1, j), rank_cons=rank_asc(cons1, j), rank_cons_own=rank_asc(cons_own, j),
                       flag_norm=bool(float(z1[j].abs()) > 3), flag_rw=bool(float(rw1[j]) > thr_rw), flag_cons=bool(float(cons1[j]) < thr_cons), flag_cons_own=bool(float(cons_own[j]) < thr_own), acts_unchanged=bool(torch.allclose(Ae, A0, atol=1e-4)))
            ent["flag_any"] = bool(ent["flag_norm"] or ent["flag_rw"] or ent["flag_cons"]); ent["flag_any_own"] = bool(ent["flag_norm"] or ent["flag_rw"] or ent["flag_cons_own"]); res["edits"].append(ent)
        E = [e for e in res["edits"] if e["kind"] == kind and e["edit"] == how]; m = lambda key: float(np.mean([e[key] for e in E]))
        print(f"[{name}] {kind} {how}: KL {m('kl_effect'):.3f}; flag-any clean states {m('flag_any'):.2f} / own states {m('flag_any_own'):.2f} (norm {m('flag_norm'):.2f} read-write {m('flag_rw'):.2f} consistency clean {m('flag_cons'):.2f} own {m('flag_cons_own'):.2f}); consistency {m('cons'):+.3f} / {m('cons_own'):+.3f} ({time.time()-t0:.0f}s)", flush=True)
S = {}
for kind in ("high_use", "low_use"):
    for how in HOWS:
        E = [e for e in res["edits"] if e["kind"] == kind and e["edit"] == how]; S[f"{kind}_{how}"] = {key: float(np.mean([e[key] for e in E])) for key in ("kl_effect", "flag_any", "flag_any_own", "flag_norm", "flag_rw", "flag_cons", "flag_cons_own", "rw", "cons", "cons_own", "acts_unchanged")}
res["summary"] = S
neg = [S[f"{k}_negate"] for k in ("high_use", "low_use")]; dbl = [S[f"{k}_double"] for k in ("high_use", "low_use")]; refl = [S[f"{k}_reflect"] for k in ("high_use", "low_use")]
E651 = {"pythia160": 0.67, "pythia410": 0.80, "qwen05": 0.12, "smollm135": 1.00}   # e651's recorded consistency flag rate on high-use reflections
hr = S["high_use_reflect"]; hn = S["high_use_negate"]
res["verdicts"] = dict(C1=abs(hr["flag_cons"] - E651.get(name, hr["flag_cons"])) <= 0.15, C2_rates=dict(clean=hr["flag_cons"], own=hr["flag_cons_own"], negate_clean=hn["flag_cons"], negate_own=hn["flag_cons_own"]),
                       C3=min(hr["flag_any_own"], hn["flag_any_own"]) < 0.5, acts_unchanged=all(e["acts_unchanged"] for e in res["edits"]))
summ = f"detector coverage with clean against own states ({name} block {b}): " + "; ".join(f"{k} KL {S[k]['kl_effect']:.3f} flag-any {S[k]['flag_any']:.2f}/{S[k]['flag_any_own']:.2f} (norm {S[k]['flag_norm']:.2f} rw {S[k]['flag_rw']:.2f} cons {S[k]['flag_cons']:.2f}/{S[k]['flag_cons_own']:.2f})" for k in S) + f"; verdicts {res['verdicts']}"
print(summ, flush=True); os.makedirs("results", exist_ok=True); json.dump(res, open(f"results/e667_detect_own_states_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("done", f"{time.time()-t0:.0f}s")
