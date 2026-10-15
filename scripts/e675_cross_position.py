"""e675 (session 153; Gemini 4's reading of e670 against e673): "neuron activations die at the token where they fire,
whereas their write directions persist in the residual stream after attention moves the state to a summary token", so
WDD's write dictionary would be a cross-position forensic trace: given only a summary state, it recovers which upstream
neurons wrote the content at earlier tokens. e670's echo was not that: it ranked rows against the final-position pole
difference, a direction built from the labels of many items. Two tests of the label-free claim.
Part A, Pythia-410m, e568's 280 prompts (as e673), blocks 12, 17 and 22: every neuron zeroed at all positions before the
last (its earlier writes removed, the final position untouched), damage = the absolute drop in log probability of the
model's own top-1 token at the last position, and KL; predictors: the activation-free projection of the final state on
the neuron's write direction (the state after the block, and the last layer's state, each centred over the prompts),
the earlier positions' write size (sum over earlier positions of |a| times the write row's norm), gradient times
activation summed over the earlier positions (absolute sum and absolute of the signed sum), and the neuron's own
final-position activation. Spearman across the 4,096 neurons of each block.
Part B, Qwen2.5-0.5B, e670's translation items (wdd_demo's harness), gender and number: every MLP row of blocks 0..12
ranked for each item by the label-free projection of that item's own final-position state at block 12 (centred over all
items, no labels); the rank of the attribute's oracle rows (e664b at the word) under it; then for each held-out item its
own top 32 rows negated and the flip measured, against e670's supervised echo (one global top 32 from the labelled pole
difference), the oracle's top 32 and random rows. Arguments: --part A|B [--smoke]. Pre-registered in e675_prereg.json."""
import sys, os, time, json, math, random, collections, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(4); DEV = "cuda:0"; t0 = time.time()
def log(s): print(f"[e675{A_.part}] {s} ({time.time() - t0:.0f}s)", flush=True)
def rank(v): r = np.empty(len(v)); r[np.argsort(v, kind="stable")] = np.arange(len(v)); return r
def spearman(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def boot_diff(a, b, y, seed, n=1000):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y)); out.append(spearman(a[s], y[s]) - spearman(b[s], y[s]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
RES = dict(part=A_.part, smoke=SMOKE)
if A_.part == "A":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF = "EleutherAI/pythia-410m"; BLOCKS = [17] if SMOKE else [12, 17, 22]; MAXSEQ = 1500
    PRM = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e673_prompts.json")))
    tok = AutoTokenizer.from_pretrained(HF); model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32).to(DEV).eval()
    for p_ in model.parameters(): p_.requires_grad_(False)
    L = model.gpt_neox.layers; NL = len(L); DFF = model.config.intermediate_size; D = model.config.hidden_size; UNEMB = model.get_output_embeddings()
    PROMPTS = [("The capital of %s is" % c) for c, _ in PRM["caps"]] + [("The capital of %s is" % c) for c in PRM["fake_countries"]] + [("The birthplace of %s is" % p) for p in PRM["real_people"]] + [("The birthplace of %s is" % p) for p in PRM["fake_people"]]
    if SMOKE: PROMPTS = PROMPTS[::6]
    NP = len(PROMPTS); IDS = [tok(p)["input_ids"] for p in PROMPTS]; GRP = collections.defaultdict(list)
    for i, x in enumerate(IDS): GRP[len(x)].append(i)
    NEU = list(range(512)) if SMOKE else list(range(DFF)); NEUt = torch.tensor(NEU, device=DEV)
    log(f"{NP} prompts, {len(GRP)} length groups, blocks {BLOCKS}, {len(NEU)} neurons per block")
    # the clean pass with gradients of log p(top-1) at the last position with respect to every activation at every position
    AE = {b: torch.zeros(NP, DFF, device=DEV) for b in BLOCKS}; AF = {b: torch.zeros(NP, DFF, device=DEV) for b in BLOCKS}; GE = {b: torch.zeros(NP, DFF, device=DEV) for b in BLOCKS}; GS = {b: torch.zeros(NP, DFF, device=DEV) for b in BLOCKS}
    XF = {b: torch.zeros(NP, D, device=DEV) for b in BLOCKS + ["last"]}; TGT = torch.zeros(NP, dtype=torch.long, device=DEV)
    for ln_, idx in GRP.items():
        x = torch.tensor([IDS[i] for i in idx], device=DEV); cap = {}; hs = []
        for b in BLOCKS:
            hs.append(L[b].mlp.dense_4h_to_h.register_forward_pre_hook(lambda m, a, b=b: cap.__setitem__(("a", b), a[0])))
            hs.append(L[b].register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(("x", b), (o[0] if isinstance(o, tuple) else o).detach())))
        hs.append(model.gpt_neox.final_layer_norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("xl", a[0].detach())))
        try:
            with torch.enable_grad():
                emb = model.gpt_neox.embed_in(x).detach().requires_grad_(True)
                hl = model.gpt_neox(inputs_embeds=emb).last_hidden_state[:, -1]; lp = UNEMB(hl).float().log_softmax(-1); tg = lp.argmax(-1)
                gr = torch.autograd.grad(lp.gather(1, tg[:, None]).sum(), [cap[("a", b)] for b in BLOCKS])
        finally: [h.remove() for h in hs]
        it = torch.tensor(idx, device=DEV)
        for k, b in enumerate(BLOCKS):
            a = cap[("a", b)].detach(); g = gr[k]
            AE[b][it] = a[:, :-1].abs().sum(1); AF[b][it] = a[:, -1].abs(); GE[b][it] = (a[:, :-1] * g[:, :-1]).abs().sum(1); GS[b][it] = (a[:, :-1] * g[:, :-1]).sum(1).abs(); XF[b][it] = cap[("x", b)][:, -1]
        XF["last"][it] = cap["xl"][:, -1]; TGT[it] = tg
    unit = lambda X: (X - X.mean(0, keepdim=True)) / (X - X.mean(0, keepdim=True)).norm(dim=1, keepdim=True).clamp_min(1e-9)
    def damage(b):
        """each neuron zeroed at every position before the last; the last position's top-1 log probability and KL"""
        dl = torch.zeros(DFF, NP, device=DEV); kl = torch.zeros(DFF, NP, device=DEV)
        for ln_, idx in GRP.items():
            it = torch.tensor(idx, device=DEV); n = len(idx); NB = max(1, min(64, MAXSEQ // n)); x = torch.tensor([IDS[i] for i in idx], device=DEV).repeat_interleave(NB, 0)
            tg = TGT[it].repeat_interleave(NB, 0); rows = torch.arange(n * NB, device=DEV); slot = rows % NB
            for c0 in range(-NB, len(NEU), NB):
                js = NEU[c0:c0 + NB] if c0 >= 0 else []; js = torch.tensor(js + [-1] * (NB - len(js)), device=DEV); live = js >= 0; neur = js.clamp_min(0)[slot]
                def hook(m, a):
                    h = a[0].clone(); keep = live[slot]; r_ = rows[keep]; h[r_, :-1, neur[keep]] = 0.0; return (h,)
                hd = L[b].mlp.dense_4h_to_h.register_forward_pre_hook(hook)
                try: lp = UNEMB(model.gpt_neox(input_ids=x).last_hidden_state[:, -1]).float().log_softmax(-1)
                finally: hd.remove()
                if c0 < 0: lp0 = lp; p0 = lp.exp(); continue
                m = live[slot]; jj = neur[m]; ii = it.repeat_interleave(NB)[m]
                dl[jj, ii] = (lp0.gather(1, tg[:, None]) - lp.gather(1, tg[:, None]))[:, 0][m]; kl[jj, ii] = (p0 * (lp0 - lp)).sum(-1).clamp_min(0)[m]
        return dl[NEUt].abs().mean(1).cpu().numpy(), kl[NEUt].mean(1).cpu().numpy()
    RES["blocks"] = {}
    for b in BLOCKS:
        Wo = L[b].mlp.dense_4h_to_h.weight.detach().float(); wn = Wo.norm(dim=0); Wu = (Wo / wn[None]).T   # [DFF, D] unit write directions
        X = {"projection_block": (Wu @ unit(XF[b]).T).abs().mean(1), "projection_last": (Wu @ unit(XF["last"]).T).abs().mean(1), "write_earlier": (AE[b] * wn[None]).mean(0), "atp_earlier": GE[b].mean(0), "atp_earlier_signed": GS[b].mean(0), "activation_final": AF[b].mean(0)}
        X = {k: v[NEUt].cpu().numpy() for k, v in X.items()}; Y, Ykl = damage(b)
        rho = {k: spearman(v, Y) for k, v in X.items()}; rho_kl = {k: spearman(v, Ykl) for k, v in X.items()}
        ci = dict(projection_block_vs_write_earlier=boot_diff(X["projection_block"], X["write_earlier"], Y, b, 200 if SMOKE else 1000), projection_last_vs_write_earlier=boot_diff(X["projection_last"], X["write_earlier"], Y, b + 1, 200 if SMOKE else 1000))
        RES["blocks"][str(b)] = dict(rho=rho, rho_kl=rho_kl, ci=ci, damage_median=float(np.median(Y)), damage_max=float(Y.max()), kl_median=float(np.median(Ykl)))
        log(f"block {b}: Spearman with the cross-position damage (|log-probability drop| when the neuron is removed at every earlier position) " + ", ".join(f"{k} {v:.2f}" for k, v in rho.items()) + f"; with KL " + ", ".join(f"{k} {v:.2f}" for k, v in rho_kl.items()) + f"; projection minus write size 95% CI {ci['projection_block_vs_write_earlier']} (last layer {ci['projection_last_vs_write_earlier']}); damage median {np.median(Y):.1e}, max {Y.max():.2f}")
else:
    sys.argv = [sys.argv[0], "--auto", "--no-color"]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import wdd_demo as wd
    if SMOKE: wd.LANGS = ("fr",)
    lm = wd.LM("Qwen/Qwen2.5-0.5B"); L, DFF = lm.L, lm.DFF; K = 32; NR_ = (L + 1) * DFF
    WN = [lm.W(b).detach().float() for b in range(L + 1)]; WU = torch.stack([w / w.norm(dim=0, keepdim=True).clamp_min(1e-12) for w in WN])   # [L+1, D, DFF]
    def capture_end(its):
        groups = collections.defaultdict(list)
        for n, it in enumerate(its): groups[len(it["ids"])].append(n)
        for idx in groups.values():
            for s0 in range(0, len(idx), lm.BS):
                ch = idx[s0:s0 + lm.BS]; cap = {}; hd = lm.layers[L].register_forward_hook(lambda m, a, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[:, -1].float()))
                try: lm.model(torch.tensor([its[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1)
                finally: hd.remove()
                for k, n in enumerate(ch): its[n]["Xe"] = cap["x"][k]
    RES["attrs"] = {}
    for a in wd.POLES:
        its = lm.items(a); SC = lm.scan(its); capture_end(its); disc = [it for it in its if it["disc"]]; held = [it for it in its if not it["disc"]]
        Xe = torch.stack([it["Xe"] for it in its]); mu = Xe.mean(0); s = torch.tensor([it["side"] for it in disc])
        Xd = torch.stack([it["Xe"] for it in disc]); md = Xd[s == 1].mean(0) - Xd[s == 0].mean(0); echo = (torch.einsum("bdr,d->br", WU, md / md.norm())).abs()   # e670's supervised echo
        top = lambda S, k=K: [divmod(int(i), DFF) for i in S.reshape(-1).argsort(descending=True)[:k].tolist()]
        def proj(it): u = it["Xe"] - mu; u = u / u.norm().clamp_min(1e-12); return torch.einsum("bdr,d->br", WU, u).abs()
        oracle = SC["order"]; o1 = oracle[0]; o8 = oracle[:8]
        rk = lambda S, br: int((S.reshape(-1) > S[br[0], br[1]]).sum()) + 1
        ranks1 = [rk(proj(it), o1) for it in its]; ranks8 = [float(np.median([rk(proj(it), br) for br in o8])) for it in its]
        jac = [len(set(top(proj(it))) & set(map(tuple, oracle[:K]))) / len(set(top(proj(it))) | set(map(tuple, oracle[:K]))) for it in its]
        # flips: each held-out item with its own top 32 negated (label-free), against one global set (echo, oracle, random)
        def flip_items(rows_of):
            out = {0: [], 1: []}
            for it in held:
                with wd.Negate(lm, rows_of(it)): lm.run([it])
                out[it["side"]].append(it["pred"] != it["side"])
            lm.run(held); return {k: float(np.mean(v)) if v else None for k, v in out.items()}
        rg = random.Random(675); rnd = [(rg.randrange(L + 1), rg.randrange(DFF)) for _ in range(K)]
        F = {"per_item_projection": flip_items(lambda it: top(proj(it))), "echo_supervised": flip_items(lambda it: top(echo)), "oracle": flip_items(lambda it: [tuple(x) for x in oracle[:K]]), "random": flip_items(lambda it: rnd)}
        RES["attrs"][a] = dict(n_items=len(its), n_held=len(held), oracle_top=list(o1), rank_oracle_top_median=float(np.median(ranks1)), rank_oracle_top_share_top100=float(np.mean([r <= 100 for r in ranks1])), rank_oracle8_median=float(np.median(ranks8)),
                               rank_oracle_top_under_echo=rk(echo, o1), jaccard32_median=float(np.median(jac)), flips=F, n_rows=NR_)
        log(f"{a}: the oracle's top row (block {o1[0]}, row {o1[1]}) ranks {np.median(ranks1):.0f} of {NR_} under each item's own label-free projection (median; top 100 for {np.mean([r <= 100 for r in ranks1]):.2f} of items), {rk(echo, o1)} under the supervised echo; "
            f"top-32 Jaccard with the oracle {np.median(jac):.2f}; flips of held-out marked items when 32 rows are negated: " + ", ".join(f"{k} {v[1]:.2f} (unmarked {v[0]:.2f})" for k, v in F.items()))
RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S"); RES["seconds"] = time.time() - t0
os.makedirs("/workspace/wdd/results", exist_ok=True); json.dump(RES, open(f"/workspace/wdd/results/e675{A_.part}_cross_position{'_smoke' if SMOKE else ''}.json", "w"), indent=1); log("done"); print("done", flush=True)
