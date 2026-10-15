"""e673 (session 152; Gemini 4's "edge featurizer" pitch, read against Cammarata 2026): does a neuron's consumption by the
downstream gates predict what removing it costs better than its node prominence? The essay's Edge Featurizers figure factors
25,000 ImageNet attribution graphs into 96 edge motifs and 96 activation factors; removing a motif's neurons costs a median
0.86 logits of class margin against 0.03 for an activation factor, usage-damage correlation median 0.35 (the teapot motif
0.80): groups of neurons in a vision model. Gemini's transplant scores single neurons of a language model. e568's prompts
on Pythia-410m (160 real countries and 60 invented ones in "The capital of X is", 30 real and 30 invented people in "The
birthplace of X is"); at the last position every neuron of blocks 12, 17 and 22 scored by activation, write size |a| |w_out|,
node prominence (Gemini's |a (w_out . u)| with u the unembedding of the model's top-1 token, and the exact direct effect
through the final LayerNorm), edge consumption (Gemini's sum over every downstream neuron k of |a (w_out . w_in,k) phi'(h_k)|,
and the same through each block's LayerNorm), the one-hop signed sum weighted by each downstream neuron's own gradient, and
gradient times activation (AtP). Then each neuron zeroed at the last position (the prefix cached, the last token recomputed):
damage = the drop in the log probability of the model's own top-1 token (absolute, primary), the KL divergence, and for real
countries the drop for the true capital. Spearman across the 4,096 neurons of each block with a bootstrap over neurons; the
40 most damaging neurons against each score's top 40; e568's bystander row (block 17, neuron 3002). Arguments: [--smoke].
Pre-registered in e673_prereg.json."""
import sys, os, time, json, math, collections
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False); DEV = "cpu" if "--cpu" in sys.argv else "cuda:0"; t0 = time.time(); SMOKE = "--smoke" in sys.argv
if DEV == "cpu": torch.set_num_threads(40)
HF = "EleutherAI/pythia-410m"; BLOCKS = [17] if SMOKE else [12, 17, 22]; MAXSEQ = 1000; NBOOT = 200 if SMOKE else 1000; TOPK = 10 if SMOKE else 40; BYSTANDER = (17, 3002)
def log(s): print(f"[e673] {s} ({time.time() - t0:.0f}s)", flush=True)
PRM = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e673_prompts.json")))
tok = AutoTokenizer.from_pretrained(HF); model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32).to(DEV).eval()
for p_ in model.parameters(): p_.requires_grad_(False)
L = model.gpt_neox.layers; NL = len(L); DFF = model.config.intermediate_size; D = model.config.hidden_size; UNEMB = model.get_output_embeddings(); WU = UNEMB.weight
PROMPTS = [("The capital of %s is" % c, "real countries", " " + cap) for c, cap in PRM["caps"]] + [("The capital of %s is" % c, "invented countries", None) for c in PRM["fake_countries"]] \
    + [("The birthplace of %s is" % p, "real people", None) for p in PRM["real_people"]] + [("The birthplace of %s is" % p, "invented people", None) for p in PRM["fake_people"]]
if SMOKE: PROMPTS = PROMPTS[::6]
NP = len(PROMPTS); IDS = [tok(p[0])["input_ids"] for p in PROMPTS]; GRP = collections.defaultdict(list)
for i, x in enumerate(IDS): GRP[len(x)].append(i)
GOLD = torch.tensor([tok(p[2])["input_ids"][0] if p[2] else -1 for p in PROMPTS])
NEU = list(range(256)) + [BYSTANDER[1]] if SMOKE else list(range(DFF)); NEUt = torch.tensor(NEU, device=DEV)
log(f"{NP} prompts in {len(GRP)} length groups; blocks {BLOCKS}; {len(NEU)} neurons per block")
# ---------------- the clean pass, with gradients of log p(top-1) at the last position
XS = torch.zeros(NL, NP, D, device=DEV); A = torch.zeros(NL, NP, DFF, device=DEV); H = torch.zeros(NL, NP, DFF, device=DEV); G = torch.zeros(NL, NP, DFF, device=DEV); SIG = torch.zeros(NL, NP, device=DEV); GDIR = torch.zeros(NP, D, device=DEV)
LP0 = torch.zeros(NP, WU.shape[0], device=DEV); TGT = torch.zeros(NP, dtype=torch.long, device=DEV)
for ln_, idx in GRP.items():
    x = torch.tensor([IDS[i] for i in idx], device=DEV); cap = {}; hs = []
    for l in range(NL):
        hs.append(L[l].mlp.dense_h_to_4h.register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("h", l), o)))
        hs.append(L[l].mlp.dense_4h_to_h.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(("a", l), a[0])))
        hs.append(L[l].post_attention_layernorm.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(("x", l), a[0].detach())))
        hs.append(L[l].register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("out", l), (o[0] if isinstance(o, tuple) else o).detach())))
    hs.append(model.gpt_neox.final_layer_norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("xL", a[0])))
    try:
        with torch.enable_grad():
            emb = model.gpt_neox.embed_in(x).detach().requires_grad_(True)   # a leaf, so the graph is built with frozen weights
            hl = model.gpt_neox(inputs_embeds=emb).last_hidden_state[:, -1]; lp = UNEMB(hl).float().log_softmax(-1); tg = lp.argmax(-1)
            gr = torch.autograd.grad(lp.gather(1, tg[:, None]).sum(), [cap[("a", l)] for l in range(NL)] + [cap["xL"]])
    finally: [h.remove() for h in hs]
    it = torch.tensor(idx, device=DEV)
    for l in range(NL):
        XS[l, it] = cap[("out", l)][:, -1]; A[l, it] = cap[("a", l)][:, -1].detach(); H[l, it] = cap[("h", l)][:, -1].detach(); G[l, it] = gr[l][:, -1]
        xx = cap[("x", l)][:, -1]; SIG[l, it] = (xx.var(-1, unbiased=False) + L[l].post_attention_layernorm.eps).sqrt()
    GDIR[it] = gr[NL][:, -1]; LP0[it] = lp.detach(); TGT[it] = tg
log(f"clean pass: top-1 = the true capital's first token for {float((TGT.cpu() == GOLD)[GOLD >= 0].float().mean()):.2f} of real countries")
def dact(h, act):
    with torch.enable_grad():
        hh = h.detach().clone().requires_grad_(True); (g,) = torch.autograd.grad(act(hh).sum(), hh)
    return g
DPHI = torch.stack([dact(H[l], L[l].mlp.act) for l in range(NL)])
# ---------------- predictors for each neuron of a block, per prompt [DFF, NP]
def predictors(b):
    WoT = L[b].mlp.dense_4h_to_h.weight.T.float(); Ab = A[b].T; aa = Ab.abs()   # [DFF, D], [DFF, NP]
    U = XS[b] - XS[b].mean(0, keepdim=True); U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-9)   # the state after the block, centred over the prompts (WDD's row score, no activation in it)
    out = dict(projection=((WoT / WoT.norm(dim=1, keepdim=True)) @ U.T).abs(), activation=aa, write=aa * WoT.norm(dim=1, keepdim=True), node_u=(Ab * (WoT @ WU[TGT].T)).abs(), node_direct=(Ab * (WoT @ GDIR.T)).abs(), atp=(Ab * G[b].T).abs())
    e_lit = torch.zeros(DFF, NP, device=DEV); e_ln = torch.zeros(DFF, NP, device=DEV); hop = torch.zeros(DFF, NP, device=DEV)
    for l in range(b + 1, NL):
        Win = L[l].mlp.dense_h_to_4h.weight.float(); R = Win * L[l].post_attention_layernorm.weight.float()[None]; R = R - R.mean(1, keepdim=True)
        Ml = WoT @ Win.T; Mr = WoT @ R.T; dp = DPHI[l].T   # [DFF_b, DFF_l], [DFF_l, NP]
        e_lit += Ml.abs() @ dp.abs(); e_ln += (Mr.abs() @ dp.abs()) / SIG[l][None]; hop += (Mr @ (dp * G[l].T)) / SIG[l][None]; del Ml, Mr
    out.update(edge=aa * e_lit, edge_ln=aa * e_ln, hop=(Ab * hop).abs())
    return out
# ---------------- damage: each neuron zeroed at the last position, prefix cached
def damage(b):
    dl = torch.zeros(DFF, NP, device=DEV); kl = torch.zeros(DFF, NP, device=DEV); dg = torch.full((DFF, NP), float("nan"), device=DEV); check = []
    for ln_, idx in GRP.items():
        it = torch.tensor(idx, device=DEV); n = len(idx); NB = max(1, min(64, MAXSEQ // n)); x = torch.tensor([IDS[i] for i in idx], device=DEV)
        out = model.gpt_neox(input_ids=x[:, :-1], use_cache=True); cache = out.past_key_values; cache.batch_repeat_interleave(NB); last = x[:, -1:].repeat_interleave(NB, 0)
        tg = TGT[it].repeat_interleave(NB, 0); gold = GOLD[idx].to(DEV).repeat_interleave(NB, 0)
        rows = torch.arange(n * NB, device=DEV); slot = rows % NB
        for c0 in range(-NB, len(NEU), NB):   # the first pass zeroes nothing: the baseline on the same cached path (and a check against the full clean pass)
            js = NEU[c0:c0 + NB] if c0 >= 0 else []; js = torch.tensor(js + [-1] * (NB - len(js)), device=DEV); live = js >= 0; neur = js.clamp_min(0)[slot]
            def hook(m, a):
                h = a[0].clone(); keep = live[slot]; h[rows[keep], -1, neur[keep]] = 0.0; return (h,)
            hd = L[b].mlp.dense_4h_to_h.register_forward_pre_hook(hook)
            try: hl = model.gpt_neox(input_ids=last, past_key_values=cache, use_cache=True).last_hidden_state[:, -1]
            finally: hd.remove(); cache.crop(-1)
            lp = UNEMB(hl).float().log_softmax(-1)
            if c0 < 0: lp0 = lp; p0 = lp0.exp(); check.append(float((lp - LP0[it].repeat_interleave(NB, 0)).abs().max())); continue
            d_t = (lp0.gather(1, tg[:, None]) - lp.gather(1, tg[:, None]))[:, 0]; k_ = (p0 * (lp0 - lp)).sum(-1)
            d_g = torch.where(gold >= 0, lp0.gather(1, gold.clamp_min(0)[:, None])[:, 0] - lp.gather(1, gold.clamp_min(0)[:, None])[:, 0], torch.full_like(d_t, float("nan")))
            m = live[slot]; jj = neur[m]; ii = it.repeat_interleave(NB)[m]
            dl[jj, ii] = d_t[m]; kl[jj, ii] = k_[m]; dg[jj, ii] = d_g[m]
        del cache, lp0, p0
    return {k: v[NEUt] for k, v in dict(dlogp=dl, abs_dlogp=dl.abs(), kl=kl.clamp_min(0), dgold=dg).items()}, max(check)
def rank(v): r = np.empty(len(v)); r[np.argsort(v, kind="stable")] = np.arange(len(v)); return r
def spearman(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def boot_diff(a, b, y, seed):
    rng = np.random.default_rng(seed); n = len(y); out = []
    for _ in range(NBOOT):
        s = rng.integers(0, n, n); out.append(spearman(a[s], y[s]) - spearman(b[s], y[s]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
PRED = ["projection", "activation", "write", "node_u", "node_direct", "edge", "edge_ln", "hop", "atp"]; RES = dict(model=HF, n_prompts=NP, blocks={}, smoke=SMOKE)
grp_of = np.array([p[1] for p in PROMPTS])
for b in BLOCKS:
    pr = {k: v[NEUt] for k, v in predictors(b).items()}; dm, check = damage(b)
    Y = {k: v.mean(1).cpu().numpy() for k, v in dm.items() if k != "dgold"}; Y["dgold"] = torch.nanmean(dm["dgold"], 1).cpu().numpy()
    X = {k: v.mean(1).cpu().numpy() for k, v in pr.items()}
    rho = {t: {k: spearman(X[k], Y[t]) for k in PRED} for t in ("abs_dlogp", "kl", "dlogp")}
    rho_prompt = {k: float(np.nanmedian([spearman(pr[k][:, i].cpu().numpy(), dm["abs_dlogp"][:, i].cpu().numpy()) for i in range(NP)])) for k in PRED}
    topd = set(np.argsort(-Y["abs_dlogp"])[:TOPK].tolist()); prec = {k: len(topd & set(np.argsort(-X[k])[:TOPK].tolist())) / TOPK for k in PRED}
    ci = dict(edge_ln_vs_node_direct=boot_diff(X["edge_ln"], X["node_direct"], Y["abs_dlogp"], b), edge_vs_node_u=boot_diff(X["edge"], X["node_u"], Y["abs_dlogp"], b + 100), hop_vs_node_direct=boot_diff(X["hop"], X["node_direct"], Y["abs_dlogp"], b + 200),
              edge_ln_vs_write=boot_diff(X["edge_ln"], X["write"], Y["abs_dlogp"], b + 300), edge_ln_vs_write_kl=boot_diff(X["edge_ln"], X["write"], Y["kl"], b + 400))
    by_group = {g: {k: spearman(pr[k][:, torch.tensor(np.nonzero(grp_of == g)[0], device=DEV)].mean(1).cpu().numpy(), dm["abs_dlogp"][:, torch.tensor(np.nonzero(grp_of == g)[0], device=DEV)].mean(1).cpu().numpy()) for k in ("node_direct", "edge_ln", "hop", "atp")} for g in sorted(set(grp_of))}
    R = dict(cache_check_max_abs=check, rho=rho, rho_within_prompt_median=rho_prompt, top40_overlap=prec, ci=ci, by_group=by_group, damage_median=float(np.median(Y["abs_dlogp"])), damage_max=float(Y["abs_dlogp"].max()), damage_top=[[NEU[int(j)], float(Y["abs_dlogp"][j])] for j in np.argsort(-Y["abs_dlogp"])[:10]])
    if b == BYSTANDER[0]:
        j = NEU.index(BYSTANDER[1]); inv = torch.tensor(np.nonzero(grp_of == "invented countries")[0], device=DEV); pct = lambda v, jj: float((v < v[jj]).mean())
        R["bystander"] = dict(neuron=j, all_prompts={k: pct(X[k], j) for k in PRED} | {"damage_abs_dlogp": pct(Y["abs_dlogp"], j), "damage_kl": pct(Y["kl"], j)},
                              invented_countries={k: pct(pr[k][:, inv].mean(1).cpu().numpy(), j) for k in PRED} | {"damage_abs_dlogp": pct(dm["abs_dlogp"][:, inv].mean(1).cpu().numpy(), j)},
                              activation_invented=float(A[b, inv, BYSTANDER[1]].mean()), damage_invented=float(dm["abs_dlogp"][j, inv].mean()))
    RES["blocks"][b] = R
    log(f"block {b} (cache check {check:.1e}): Spearman with |log-probability drop| across {len(NEU)} neurons: " + ", ".join(f"{k} {rho['abs_dlogp'][k]:.2f}" for k in PRED)
        + f"; with KL: " + ", ".join(f"{k} {rho['kl'][k]:.2f}" for k in PRED) + f"; edge_ln minus node_direct 95% CI {ci['edge_ln_vs_node_direct']}, edge minus node_u {ci['edge_vs_node_u']}, edge_ln minus write {ci['edge_ln_vs_write']} (KL {ci['edge_ln_vs_write_kl']}); top-{TOPK} overlap " + ", ".join(f"{k} {prec[k]:.2f}" for k in PRED)
        + f"; within-prompt median: " + ", ".join(f"{k} {rho_prompt[k]:.2f}" for k in PRED) + f"; damage median {R['damage_median']:.1e}, max {R['damage_max']:.2f}")
    if "bystander" in R: log(f"bystander block {b} neuron {BYSTANDER[1]}: percentiles over all prompts {R['bystander']['all_prompts']}; invented countries {R['bystander']['invented_countries']}; mean activation there {R['bystander']['activation_invented']:.3f}, damage {R['bystander']['damage_invented']:.2e}")
    del pr, dm; torch.cuda.empty_cache()
RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S"); RES["seconds"] = time.time() - t0
os.makedirs("results", exist_ok=True); json.dump(RES, open(f"results/e673_edge_vs_node{'_smoke' if SMOKE else ''}.json", "w"), indent=1); log("done"); print("done", flush=True)
