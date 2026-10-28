"""e733: priced atoms on the Dell. Pythia-410m in fp32 on one Quadro P4000, one MLP layer (12 primary): the native rows (the 4,096
down-projection columns) and EleutherAI's 65,536-feature TopK SAE of the same MLP output, priced by the same sampled-Fisher gradient at
the MLP output (g = d sum_t log p_t(y_t) / d mlp_out, y_t sampled from the model): native S2_n = mean (g.w_n)^2, R_dir = S2/|w|^2,
R_nat = S2 q999(|a|)^2, R_abl = S2 var(a); feature S2_f = mean (g.d_f)^2, R_abl_f = S2_f var(a_f). Then the cost of removal on held-out
text for both dictionaries (top-k by R_abl, top-k by energy, bottom-k by R_abl, random k; k in 16, 64, 256, 1024; the SAE's own
reconstruction cost), single-atom removal against R_abl for 40 rows and 40 features, and the three questions (hear, say, cost) for the
top ten of each. Pre-registered in results/e733_prereg.json. Usage: python e733_control.py [--smoke] [--layer 12]"""
import os, sys, json, time, math, random, glob
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
T0 = time.time(); SMOKE = "--smoke" in sys.argv
def arg(name, default): return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
L = int(arg("--layer", "12")); TAG = f"e733_pythia410m_L{L}" + ("_smoke" if SMOKE else ""); OUT = os.environ.get("WDD_OUT", "results"); os.makedirs(OUT, exist_ok=True)
MODEL_DIR = os.environ.get("E733_MODEL_DIR", "EleutherAI/pythia-410m"); SAE_REPO = "EleutherAI/sae-pythia-410m-65k"; dev = "cuda"; SEED = 7330
WLEN = 512; NWIN = 6 if SMOKE else 48; NCOST = 3 if SMOKE else 16; NS = 2; KS = [16, 64] if SMOKE else [16, 64, 256, 1024]
NSINGLE = 6 if SMOKE else 20; NSEEDS = 1 if SMOKE else 3; NCOST_SINGLE = 2 if SMOKE else 8; NTOP = 10
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
torch.manual_seed(SEED); RNG = random.Random(SEED)
tok = AutoTokenizer.from_pretrained(MODEL_DIR); model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).to(dev).eval(); model.requires_grad_(False)
base = getattr(model, "gpt_neox", None) or model.model; blocks = base.layers; mlp = blocks[L].mlp; ch = dict(mlp.named_children())
UP = next(ch[k] for k in ("dense_h_to_4h", "up_proj", "fc1", "c_fc") if k in ch); DOWN = next(ch[k] for k in ("dense_4h_to_h", "down_proj", "fc2", "c_proj") if k in ch)
LN2 = next(getattr(blocks[L], k) for k in ("post_attention_layernorm", "post_attention_norm", "ln_2", "post_attention_layer_norm") if hasattr(blocks[L], k))
FLN = next(getattr(base, k) for k in ("final_layer_norm", "norm", "ln_f") if hasattr(base, k))
W_down = DOWN.weight.detach(); W_in = UP.weight.detach(); G2 = LN2.weight.detach(); GF = FLN.weight.detach(); U = model.get_output_embeddings().weight.detach(); EMB = model.get_input_embeddings().weight.detach()
print("modules:", type(UP).__name__, tuple(W_in.shape), type(DOWN).__name__, tuple(W_down.shape), type(LN2).__name__, type(FLN).__name__, flush=True)
D, NN = W_down.shape; V = U.shape[0]; WN = W_down.norm(dim=0)
cfg = json.load(open(hf_hub_download(SAE_REPO, f"layers.{L}.mlp/cfg.json"))); SW = load_file(hf_hub_download(SAE_REPO, f"layers.{L}.mlp/sae.safetensors"))
W_enc, b_enc, W_dec, b_dec = (SW[k].to(dev) for k in ("encoder.weight", "encoder.bias", "W_dec", "b_dec")); K = cfg["k"]; NF = W_dec.shape[0]
log(f"model {len(blocks)} layers d {D} rows {NN} vocab {V}; layer {L}; SAE {NF} latents k {K} dec norms {W_dec.norm(dim=1).mean():.3f}; smoke {SMOKE}")
@torch.no_grad()
def sae_latents(x):
    pre = F.relu((x - b_dec) @ W_enc.T + b_enc); vals, idx = pre.topk(K, dim=-1); return torch.zeros_like(pre).scatter_(1, idx, vals)
@torch.no_grad()
def sae_recon(x): return sae_latents(x) @ W_dec + b_dec
FL = {"fisher": False}; CAP = {}; STATS = {"on": False, "a": None}; ABL = {"native": None, "sae": None, "recon": False}
def mlp_hook(m, args, o):
    if FL["fisher"]: o2 = o.detach().requires_grad_(True); CAP["o"] = o2; return o2
    if ABL["sae"] is not None: x = o[0]; lat = sae_latents(x); idx = ABL["sae"]; return (x - lat[:, idx] @ W_dec[idx])[None]
    if ABL["recon"]: return sae_recon(o[0])[None]
    return o
mlp.register_forward_hook(mlp_hook)
def down_pre(m, args):
    x = args[0]
    if STATS["on"]: STATS["a"] = x[0].detach()
    if ABL["native"] is not None: x = x.clone(); x[..., ABL["native"]] = 0; return (x,)
DOWN.register_forward_pre_hook(down_pre)
# ---------------- text windows ----------------
def load_split(split):
    try:
        import datasets; ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split); lines = ds["text"]
    except Exception as e:
        import pyarrow.parquet as pq; files = glob.glob(os.path.join(os.environ.get("HF_HOME", ""), "hub", "datasets--Salesforce--wikitext", "snapshots", "*", "wikitext-103-raw-v1", f"{split}-*.parquet")); lines = pq.read_table(files[0]).column("text").to_pylist(); log("parquet fallback", split, str(e)[:80])
    return "\n".join(t for t in lines if t.strip())
VAL = tok(load_split("validation"), return_tensors="pt", add_special_tokens=False).input_ids[0]; TEST = tok(load_split("test"), return_tensors="pt", add_special_tokens=False).input_ids[0]
def windows(ids, n): return [ids[s:s + WLEN][None].to(dev) for s in (RNG.randrange(0, len(ids) - WLEN - 1) for _ in range(n))]
FWINS = windows(VAL, NWIN); CWINS = windows(TEST, NCOST); log(f"tokens validation {len(VAL)} test {len(TEST)}; windows fisher {NWIN} cost {NCOST}")
# ---------------- the price: one gradient for both dictionaries ----------------
def fisher_pass():
    S2n = torch.zeros(NN, device=dev); S2f = torch.zeros(NF, device=dev); cnt = 0; npos = 0   # cnt counts sampled positions (NS per window) for the Fisher sums; npos counts positions once for the activation statistics
    st = {"n_sum": torch.zeros(NN, device=dev), "n_sq": torch.zeros(NN, device=dev), "n_top": torch.zeros(NN, 32, device=dev), "f_sum": torch.zeros(NF, device=dev), "f_sq": torch.zeros(NF, device=dev), "f_max": torch.zeros(NF, device=dev), "f_cnt": torch.zeros(NF, device=dev)}
    for wi, ids in enumerate(FWINS):
        FL["fisher"] = True; STATS["on"] = True; out = model(ids); FL["fisher"] = False; STATS["on"] = False
        logp = F.log_softmax(out.logits[0].float(), dim=-1); p = logp.exp(); T = ids.shape[1]
        with torch.no_grad():
            a = STATS["a"]; lat = sae_latents(CAP["o"].detach()[0])
            npos += T; st["n_sum"] += a.sum(0); st["n_sq"] += (a * a).sum(0); st["n_top"] = torch.cat([st["n_top"], a.abs().T], 1).topk(32, dim=1).values
            st["f_sum"] += lat.sum(0); st["f_sq"] += (lat * lat).sum(0); st["f_max"] = torch.maximum(st["f_max"], lat.max(0).values); st["f_cnt"] += (lat > 0).sum(0)
        for s in range(NS):
            y = torch.multinomial(p, 1).squeeze(1); CAP["o"].grad = None
            logp[torch.arange(T, device=dev), y].sum().backward(retain_graph=(s < NS - 1))
            with torch.no_grad(): g = CAP["o"].grad[0]; pn = g @ W_down; S2n += (pn * pn).sum(0); pf = g @ W_dec.T; S2f += (pf * pf).sum(0); cnt += T
        CAP.clear(); del out, logp, p
        if wi % 8 == 0 or wi == NWIN - 1: log(f"fisher window {wi + 1}/{NWIN}")
    return S2n / cnt, S2f / cnt, st, cnt, npos
S2n, S2f, ST, CNT, NPOS = fisher_pass()
with torch.no_grad():
    k999 = max(1, min(32, round(0.001 * NPOS))); Q999 = ST["n_top"][:, k999 - 1]; MEANn = ST["n_sum"] / NPOS; VARn = (ST["n_sq"] / NPOS - MEANn ** 2).clamp_min(0); En = ST["n_sq"] / NPOS * WN ** 2
    R_dir = S2n / WN ** 2; R_nat = S2n * Q999 ** 2; R_abl = S2n * VARn
    MEANf = ST["f_sum"] / NPOS; VARf = (ST["f_sq"] / NPOS - MEANf ** 2).clamp_min(0); Ef = ST["f_sq"] / NPOS; R_abl_f = S2f * VARf; ALIVE = (ST["f_cnt"] > 0)
log(f"priced: native R_abl median {R_abl.median():.3e} max {R_abl.max():.3e}; SAE alive {int(ALIVE.sum())}/{NF}, R_abl_f median(alive) {R_abl_f[ALIVE].median():.3e} max {R_abl_f.max():.3e}; energy native sum {En.sum():.3f} SAE sum {Ef.sum():.3f}")
# ---------------- the cost of removal ----------------
@torch.no_grad()
def base_logp(wins):
    out = []
    for ids in wins: o = model(ids); out.append((ids, F.log_softmax(o.logits[0].float(), dim=-1).half().cpu()))
    return out
BASE = base_logp(CWINS)
@torch.no_grad()
def cost(base=None):
    base = base or BASE; kls, dn = [], []
    for ids, lpb in base:
        lpb = lpb.to(dev).float(); lp = F.log_softmax(model(ids).logits[0].float(), dim=-1); y = ids[0, 1:]; ar = torch.arange(len(y), device=dev)
        kls.append((lpb.exp() * (lpb - lp)).sum(-1)[:-1].mean().item()); dn.append(((-lp[:-1][ar, y]).mean() - (-lpb[:-1][ar, y]).mean()).item())
    return float(np.mean(kls)), float(np.mean(dn))
def clear(): ABL["native"] = None; ABL["sae"] = None; ABL["recon"] = False
def run_native(idx):
    clear(); ABL["native"] = torch.as_tensor(idx, device=dev); r = cost(); clear(); return r
def run_sae(idx):
    clear(); ABL["sae"] = torch.as_tensor(idx, device=dev); r = cost(); clear(); return r
clear(); ABL["recon"] = True; RECON = cost(); clear(); log(f"SAE reconstruction cost kl {RECON[0]:.4f} dnll {RECON[1]:.4f}")
order_n = {"top_R_abl": torch.argsort(R_abl, descending=True), "top_energy": torch.argsort(En, descending=True), "bottom_R_abl": torch.argsort(R_abl), "top_R_nat": torch.argsort(R_nat, descending=True)}
alive_idx = torch.nonzero(ALIVE).squeeze(1); R_abl_alive = R_abl_f.clone(); R_abl_alive[~ALIVE] = -1
order_f = {"top_R_abl": torch.argsort(R_abl_alive, descending=True), "top_energy": torch.argsort(Ef, descending=True), "bottom_R_abl": alive_idx[torch.argsort(R_abl_f[ALIVE])]}
COSTS = {"native": {}, "sae": {}, "sae_recon": {"kl": RECON[0], "dnll": RECON[1]}}
for k in KS:
    for name, od in order_n.items():
        idx = od[:k].tolist(); kl, dn = run_native(idx); COSTS["native"][f"{name}_k{k}"] = {"kl": kl, "dnll": dn, "energy": float(En[idx].sum())}; log(f"native {name:12s} k {k:4d} kl {kl:.4f} dnll {dn:.4f} energy {En[idx].sum():.4f}")
    rs = []
    for sd in range(NSEEDS):
        rr = random.Random(SEED + sd); idx = rr.sample(range(NN), k); kl, dn = run_native(idx); rs.append((kl, dn, float(En[idx].sum())))
    COSTS["native"][f"random_k{k}"] = {"kl": float(np.mean([r[0] for r in rs])), "dnll": float(np.mean([r[1] for r in rs])), "energy": float(np.mean([r[2] for r in rs])), "seeds": rs}; log(f"native random       k {k:4d} kl {COSTS['native'][f'random_k{k}']['kl']:.4f}")
    for name, od in order_f.items():
        idx = od[:k].tolist(); kl, dn = run_sae(idx); COSTS["sae"][f"{name}_k{k}"] = {"kl": kl, "dnll": dn, "energy": float(Ef[idx].sum())}; log(f"sae    {name:12s} k {k:4d} kl {kl:.4f} dnll {dn:.4f} energy {Ef[idx].sum():.4f}")
    rs = []
    for sd in range(NSEEDS):
        rr = random.Random(SEED + sd); idx = rr.sample(alive_idx.tolist(), k); kl, dn = run_sae(idx); rs.append((kl, dn, float(Ef[idx].sum())))
    COSTS["sae"][f"random_k{k}"] = {"kl": float(np.mean([r[0] for r in rs])), "dnll": float(np.mean([r[1] for r in rs])), "energy": float(np.mean([r[2] for r in rs])), "seeds": rs}; log(f"sae    random       k {k:4d} kl {COSTS['sae'][f'random_k{k}']['kl']:.4f}")
# ---------------- single atoms against the price ----------------
BASE_S = BASE[:NCOST_SINGLE]
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")
rr = random.Random(SEED + 11); single_n = order_n["top_R_abl"][:NSINGLE].tolist() + rr.sample(range(NN), NSINGLE); single_f = order_f["top_R_abl"][:NSINGLE].tolist() + rr.sample(alive_idx.tolist(), NSINGLE)
SING = {"native": [], "sae": []}
for n in single_n: clear(); ABL["native"] = torch.as_tensor([n], device=dev); kl, dn = cost(BASE_S); clear(); SING["native"].append({"row": n, "R_abl": float(R_abl[n]), "R_nat": float(R_nat[n]), "energy": float(En[n]), "kl": kl, "dnll": dn})
for f in single_f: clear(); ABL["sae"] = torch.as_tensor([f], device=dev); kl, dn = cost(BASE_S); clear(); SING["sae"].append({"feature": f, "R_abl": float(R_abl_f[f]), "energy": float(Ef[f]), "count": int(ST["f_cnt"][f]), "kl": kl, "dnll": dn})
SP = {"native": spearman([s["R_abl"] for s in SING["native"]], [s["kl"] for s in SING["native"]]), "sae": spearman([s["R_abl"] for s in SING["sae"]], [s["kl"] for s in SING["sae"]]),
      "native_energy_vs_kl": spearman([s["energy"] for s in SING["native"]], [s["kl"] for s in SING["native"]]), "R_abl_vs_energy_all_rows": spearman(R_abl.cpu().numpy(), En.cpu().numpy())}
log(f"single-atom Spearman: native R_abl {SP['native']:.3f} (energy {SP['native_energy_vs_kl']:.3f}); SAE R_abl {SP['sae']:.3f}; R_abl vs energy over all rows {SP['R_abl_vs_energy_all_rows']:.3f}")
# ---------------- the three questions for the top ten ----------------
def toks(ids): return [tok.decode([i]) for i in ids]
@torch.no_grad()
def say(vec): z = U @ (GF * vec); return {"promotes": toks(z.topk(NTOP).indices.tolist()), "suppresses": toks((-z).topk(NTOP).indices.tolist())}
@torch.no_grad()
def hear_emb(vec): z = EMB @ vec; return toks(z.topk(NTOP).indices.tolist())
top_n = order_n["top_R_abl"][:NTOP].tolist(); top_f = order_f["top_R_abl"][:NTOP].tolist(); CTX = {"native": {n: [] for n in top_n}, "sae": {f: [] for f in top_f}}
with torch.no_grad():
    for wi, ids in enumerate(FWINS):
        STATS["on"] = True; FL["fisher"] = True; model(ids); FL["fisher"] = False; STATS["on"] = False; a = STATS["a"]; lat = sae_latents(CAP["o"].detach()[0]); CAP.clear()
        for n in top_n:
            v, pos = a[:, n].abs().topk(3); CTX["native"][n] += [(float(a[p, n]), wi, int(p)) for p in pos.tolist()]
        for f in top_f:
            v, pos = lat[:, f].topk(3); CTX["sae"][f] += [(float(lat[p, f]), wi, int(p)) for p in pos.tolist() if lat[p, f] > 0]
def contexts(lst, nctx=5):
    out = []
    for val, wi, p in sorted(lst, key=lambda t: -abs(t[0]))[:nctx]:
        ids = FWINS[wi][0]; out.append({"activation": round(val, 3), "context": tok.decode(ids[max(0, p - 8):p].tolist()) + " [[" + tok.decode([int(ids[p])]) + "]]"})
    return out
THREE = {"native": [], "sae": []}
single_kl_n = {s["row"]: s["kl"] for s in SING["native"]}; single_kl_f = {s["feature"]: s["kl"] for s in SING["sae"]}
for n in top_n: THREE["native"].append({"row": n, "R_abl": float(R_abl[n]), "R_dir": float(R_dir[n]), "q999": float(Q999[n]), "energy": float(En[n]), "single_kl": single_kl_n.get(n), "say": say(W_down[:, n]), "hear_embedding": hear_emb(G2 * W_in[n]), "hear_contexts": contexts(CTX["native"][n])})
for f in top_f: THREE["sae"].append({"feature": f, "R_abl": float(R_abl_f[f]), "R_dir": float(S2f[f]), "count": int(ST["f_cnt"][f]), "max": float(ST["f_max"][f]), "energy": float(Ef[f]), "single_kl": single_kl_f.get(f), "say": say(W_dec[f]), "hear_embedding": hear_emb(W_enc[f]), "hear_contexts": contexts(CTX["sae"][f])})
# ---------------- verdicts ----------------
def c(d, k): return d[k]["kl"]
V_ = {}
if not SMOKE:
    V_["P1"] = SP["native"] >= 0.5; V_["P2"] = SP["sae"] >= 0.5; V_["P3"] = SP["R_abl_vs_energy_all_rows"] < 0.9
    V_["C1"] = c(COSTS["sae"], "top_R_abl_k64") >= 2 * c(COSTS["native"], "top_R_abl_k64")
    pe_s = c(COSTS["sae"], "top_R_abl_k64") / max(1e-9, COSTS["sae"]["top_R_abl_k64"]["energy"]); pe_n = c(COSTS["native"], "top_R_abl_k64") / max(1e-9, COSTS["native"]["top_R_abl_k64"]["energy"]); V_["C2"] = pe_s >= pe_n
    V_["C3"] = RECON[0] > c(COSTS["native"], "top_R_abl_k256")
    V_["C4"] = c(COSTS["native"], "random_k256") <= 0.25 * c(COSTS["native"], "top_R_abl_k256") and c(COSTS["native"], "bottom_R_abl_k256") <= 0.1 * c(COSTS["native"], "top_R_abl_k256")
    V_["C5"] = c(COSTS["sae"], "random_k256") <= 0.25 * c(COSTS["sae"], "top_R_abl_k256")
    log("verdicts", V_)
RES = {"tag": TAG, "smoke": SMOKE, "layer": L, "model": MODEL_DIR, "sae": SAE_REPO, "cfg": cfg, "positions": NPOS, "sampled_positions": CNT, "windows": {"fisher": NWIN, "cost": NCOST, "single": NCOST_SINGLE, "wlen": WLEN}, "sae_alive": int(ALIVE.sum()),
       "price_summary": {"native": {"R_abl_median": float(R_abl.median()), "R_abl_max": float(R_abl.max()), "energy_total": float(En.sum())}, "sae": {"R_abl_median_alive": float(R_abl_f[ALIVE].median()), "R_abl_max": float(R_abl_f.max()), "energy_total": float(Ef.sum())}},
       "costs": COSTS, "single": SING, "spearman": SP, "three_questions": THREE, "verdicts": V_, "seconds": round(time.time() - T0, 1)}
json.dump(RES, open(os.path.join(OUT, TAG + ".json"), "w"), indent=1, ensure_ascii=False)
np.savez(os.path.join(OUT, TAG + "_prices.npz"), S2n=S2n.cpu().numpy(), R_dir=R_dir.cpu().numpy(), R_nat=R_nat.cpu().numpy(), R_abl=R_abl.cpu().numpy(), En=En.cpu().numpy(), Q999=Q999.cpu().numpy(), VARn=VARn.cpu().numpy(), WN=WN.cpu().numpy(), S2f=S2f.cpu().numpy(), R_abl_f=R_abl_f.cpu().numpy(), Ef=Ef.cpu().numpy(), f_cnt=ST["f_cnt"].cpu().numpy())
log("saved", TAG)
