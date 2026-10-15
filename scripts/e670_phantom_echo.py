"""e670 (session 151; Gemini 4's "phantom echo"): e664b's direct-contribution localizer is handed the word's position p
(the noun being translated), where the attribute rows fire. At the prompt's final position, where the answer is
predicted, those rows are silent, so a local direct contribution there cannot see them; Gemini asks whether the
final-position state difference, carried there by attention, still projects onto their write rows. Qwen2.5-0.5B, the
translation items of e622/e659 (wdd_demo's harness), gender and number. Four rankings of every MLP row of blocks 0..L:
the oracle (e664b's direct contribution at p); the final position's own direct contribution (its activation change times
its write's cosine with the final-position pole difference at its block); the phantom echo (|cosine| of each write with
the final-position pole difference at block L, no activation used); random rows. Measured: how silent the oracle's rows
are at the final position; where the oracle's top row ranks under the echo; the held-out flips and wikitext KL when the
top 8 or 32 rows of each ranking are negated. Arguments: [--smoke]. Pre-registered in e670_prereg.json."""
import sys, time, json, random, collections
SMOKE = "--smoke" in sys.argv; sys.argv = [sys.argv[0], "--auto", "--no-color"]
import numpy as np, torch
import wdd_demo as wd
torch.set_grad_enabled(False); t0 = time.time(); DEV = wd.DEV
def log(s): print(f"[e670] {s} ({time.time() - t0:.0f}s)", flush=True)
lm = wd.LM("Qwen/Qwen2.5-0.5B"); IT = {a: lm.items(a) for a in wd.POLES}; SC = {a: lm.scan(IT[a]) for a in wd.POLES}; lm.wiki_setup(n=4 if SMOKE else 12)
L, DFF = lm.L, lm.DFF; KS = [8] if SMOKE else [8, 32]
def capture_end(its):
    """MLP activations and block outputs at each item's final position, blocks 0..L"""
    groups = collections.defaultdict(list)
    for n, it in enumerate(its): groups[len(it["ids"])].append(n)
    for idx in groups.values():
        for s0 in range(0, len(idx), lm.BS):
            ch = idx[s0:s0 + lm.BS]; cap, hs = {}, []
            for b in range(L + 1):
                hs.append(lm.layers[b].mlp.down_proj.register_forward_pre_hook(lambda m, a, b=b: cap.__setitem__(("a", b), a[0][:, -1].float())))
                hs.append(lm.layers[b].register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(("x", b), (o[0] if isinstance(o, tuple) else o)[:, -1].float())))
            try: lm.model(torch.tensor([its[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1)
            finally: [h.remove() for h in hs]
            for k, n in enumerate(ch): its[n]["Ae"] = torch.stack([cap[("a", b)][k] for b in range(L + 1)]).cpu(); its[n]["Xe"] = torch.stack([cap[("x", b)][k] for b in range(L + 1)]).cpu()
WN = [lm.W(b).detach().float().cpu() for b in range(L + 1)]; WU = torch.stack([w / w.norm(dim=0, keepdim=True).clamp_min(1e-12) for w in WN])   # [L+1, D, DFF] unit write columns
RES = dict(model=lm.hf, smoke=SMOKE, attrs={})
for a in wd.POLES:
    disc = [it for it in IT[a] if it["disc"]]; capture_end(disc); s = torch.tensor([it["side"] for it in disc])
    Ap = torch.stack([it["A"] for it in disc]); Ae = torch.stack([it["Ae"] for it in disc]); Xe = torch.stack([it["Xe"] for it in disc])
    dap = Ap[s == 1].mean(0) - Ap[s == 0].mean(0); dae = Ae[s == 1].mean(0) - Ae[s == 0].mean(0)   # [L+1, DFF]
    mde = Xe[s == 1].mean(0) - Xe[s == 0].mean(0); dhe = mde / mde.norm(dim=1, keepdim=True).clamp_min(1e-12)   # final-position pole difference per block
    end_local = torch.stack([dae[b] * (WU[b].T @ dhe[b]) for b in range(L + 1)]); echo = torch.stack([(WU[b].T @ dhe[L]).abs() for b in range(L + 1)])
    top = lambda S: [divmod(int(i), DFF) for i in S.reshape(-1).argsort(descending=True)[:64].tolist()]
    R_ = {"oracle_p": SC[a]["order"][:64], "end_local": top(end_local), "echo": top(echo)}
    rg = random.Random(670); R_.update({f"random{q}": [(rg.randrange(L + 1), rg.randrange(DFF)) for _ in range(64)] for q in range(1 if SMOKE else 3)})
    o8 = R_["oracle_p"][:8]; silence = [float(abs(dae[b, r]) / max(abs(dap[b, r]), 1e-9)) for b, r in o8]
    rank_in = lambda S, br: int((S.reshape(-1) > S[br[0], br[1]]).sum()) + 1
    t1 = dict(silence_ratio_median=float(np.median(silence)), silence_ratios=silence, oracle_top=list(R_["oracle_p"][0]), oracle_top_rank_echo=rank_in(echo, R_["oracle_p"][0]), oracle_top_rank_end_local=rank_in(end_local, R_["oracle_p"][0]),
              oracle8_median_rank_echo=float(np.median([rank_in(echo, br) for br in o8])), jaccard32={k: len(set(map(tuple, v[:32])) & set(map(tuple, R_["oracle_p"][:32]))) / len(set(map(tuple, v[:32])) | set(map(tuple, R_["oracle_p"][:32]))) for k, v in R_.items()},
              mean_block32={k: float(np.mean([b for b, _ in v[:32]])) for k, v in R_.items()})
    log(f"{a}: oracle's top-8 rows at the final position move {t1['silence_ratio_median']:.2f} of their activation change at the word (median); the oracle's top row (block {o8[0][0]}, row {o8[0][1]}) ranks {t1['oracle_top_rank_echo']} under the echo, {t1['oracle_top_rank_end_local']} under the final position's own direct contribution, of {(L + 1) * DFF}; top-32 Jaccard with the oracle " + ", ".join(f"{k} {v:.2f}" for k, v in t1["jaccard32"].items()))
    t2 = {}
    for k in KS:
        for name, rows in R_.items():
            with wd.Negate(lm, rows[:k]): f = lm.flips(IT[a]); kl = lm.cost()[1]
            t2[f"{name}_{k}"] = dict(flip_marked=f[1], flip_unmarked=f[0], kl=kl)
        log(f"{a} k={k}: flips marked->unmarked " + ", ".join(f"{n_} {t2[n_ + '_' + str(k)]['flip_marked']:.2f} (KL {t2[n_ + '_' + str(k)]['kl']:.4f})" for n_ in R_))
    RES["attrs"][a] = dict(t1=t1, t2=t2, rankings={k: [list(x) for x in v[:32]] for k, v in R_.items()})
RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S"); RES["seconds"] = time.time() - t0
json.dump(RES, open(f"results/e670_phantom_echo{'_smoke' if SMOKE else ''}.json", "w"), indent=1); log("done"); print("done", flush=True)
