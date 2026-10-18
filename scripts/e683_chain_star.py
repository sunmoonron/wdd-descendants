"""e683 (session 155): chain, star or jump, the topology of a neuron's growth. Recursive generalization predicts a
branch: each new context a neuron takes on is adjacent to a recent offshoot ('iy' to 'diy' to 'do it yourself'), not
to the core, and never unrelated; arbitrary packing predicts jumps. On Pythia-160m (seeds 0, 1, 2), block 6, the
e666/e629 corpus (24 Pile sequences of 512), checkpoints 1000 to 16000: a neuron's class at a checkpoint is its top 1%
of positions by activation (rank-based, e679's lesson); an entrant is a position in the class that was not in it at the
previous checkpoint; its nearest prior member is found by the cosine of their input attributions (the exact split of
the pre-activation into upstream units at that checkpoint, e676) and, as a second space, of their block-6 states. The
entrant is a jump if that cosine is below the 95th percentile of its cosines with random non-class positions; otherwise
chain if the nearest member itself entered within the last two checkpoints, old if earlier. Chain enrichment = the
chain share over its expectation under a random choice of neighbour. Also the root (the top-20 at step 1000) against
the final top-20 by one-token share, and, for two-route neurons at 16000 (e672's cluster null), whether the minor
route's members entered by chain more than the core's. Arguments: --tag pythia160|pythia160s1|pythia160s2 [--smoke].
Pre-registered in e683_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
tag = A_.tag; TAG = f"e683_{tag}" + ("_smoke" if SMOKE else ""); L_ = lambda s: log(TAG, s); B = 6; NSEQ = 8 if SMOKE else 24; T = 512; NNEU = 20 if SMOKE else 200
LATE = [1000, 4000, 8000, 16000] if SMOKE else list(range(1000, 16001, 1000)); NRAND = 100; RECENT = 2
ids = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["pythia410"]["eval_ids"][:NSEQ, :T].to(DEV)
def forward(M):
    """block-6 output states, block-6 activations, upstream MLP activations (0-5), attention outputs (0-5), the block-0 input, block-6's norm input; positions 1.."""
    out = collections.defaultdict(list)
    with Capture(M, list(range(B + 1)), ["act", "x", "att", "xin", "emb"], stop_after=B) as C:
        for s0 in range(0, NSEQ, 4):
            run_base(M, ids[s0:s0 + 4]); D = M["D"]
            out["x6"].append(C.cap[("x", B)][:, 1:].float().reshape(-1, D)); out["a6"].append(C.cap[("act", B)][:, 1:].half().reshape(-1, M["DFF"])); out["xm"].append(C.cap[("xin", B)][:, 1:].float().reshape(-1, D))
            out["up"].append(torch.cat([C.cap[("act", b)][:, 1:].half() for b in range(B)], -1).reshape(-1, B * M["DFF"])); out["at"].append(torch.stack([C.cap[("att", b)][:, 1:].half() for b in range(B)], 2).reshape(-1, B, D)); out["em"].append(C.cap[("emb", 0)][:, 1:].half().reshape(-1, D))
    return {k: torch.cat(v) for k, v in out.items()}
def attribution(M, j, P, S):
    """[len(P), B*DFF + B + 1] exact split of neuron j's pre-activation at positions P into upstream MLP units, attention outputs and the embedding (constants left out)"""
    Rt = read_tilde(M, B, read_w(M, B)["h"][j:j + 1])[0]; WUP = torch.cat([write_w(M, b).T for b in range(B)]); V = WUP @ Rt; xm = S["xm"][P]; sig = (xm.var(-1, unbiased=False) + ln_params(M, B)["eps"]).sqrt()
    return torch.cat([S["up"][P].float() * V[None] / sig[:, None], (S["at"][P].float() @ Rt) / sig[:, None], ((S["em"][P].float() @ Rt) / sig)[:, None]], 1)
M = load(tag, revision=f"step{LATE[-1]}"); S = forward(M); N = S["x6"].shape[0]; keep = ~sinkmask(S["x6"]); K = max(10, int(keep.sum()) // 100); DFF = M["DFF"]
a6 = S["a6"].float(); mx = a6.max(0).values; a100 = a6.topk(100, dim=0).values[-1]; cand = torch.nonzero((mx > 1.0) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(683).sample(cand, min(NNEU, len(cand))))
L_(f"{tag}: {N} positions ({int(keep.sum())} typical), class size {K}, {len(cand)} candidate neurons, {len(NEU)} sampled")
# the final top-100 firings, their two-route split (e672's null) and the final top-20 one-token share
FIN = {}
for j in NEU:
    tv, ti = (a6[:, j] * keep).topk(100); Cj = attribution(M, j, ti, S); units = Cj.abs().sum(0).topk(256).indices; Xr = Cj[:, units].cpu(); Xn = unitr(Xr); lab = kmeans(Xn, 2); sil = silhouette2(Xn, lab); nul = null_silhouettes(Xr, j)
    n1 = int(lab.sum()); core = 1 if n1 >= 50 else 0; FIN[j] = dict(idx=ti.cpu(), is_core=(lab == core), multi=bool(sil > float(np.percentile(nul, 95)) and int((lab != core).sum()) >= 10), top20=ti[:20].cpu())
del M; torch.cuda.empty_cache()
CUR = ids[:, 1:].reshape(-1).cpu(); KEEP_IDX = set(torch.nonzero(keep)[:, 0].tolist())
one_tok = lambda P: max(collections.Counter(CUR[P].tolist()).values()) / len(P)
PREV = {j: None for j in NEU}; ENTRY = {j: {} for j in NEU}; EV = {j: [] for j in NEU}; ROOT = {}; rng = np.random.default_rng(683)
for ti_, n in enumerate(LATE):
    M = load(tag, revision=f"step{n}"); S = forward(M); a6 = S["a6"].float(); X6c = S["x6"] - S["x6"][keep].mean(0); U6 = unitr(X6c)
    for j in NEU:
        cls = set((a6[:, j] * keep).topk(K).indices.tolist())
        if ti_ == 0:
            ROOT[j] = one_tok((a6[:, j] * keep).topk(20).indices.cpu())
            for p in cls: ENTRY[j][p] = -1   # founders
            PREV[j] = cls; continue
        prev = PREV[j]; new = sorted(cls - prev); old = sorted(prev)
        if new and old:
            P = torch.tensor(new + old, device=DEV); C = attribution(M, j, P, S); Cn = unitr(C); Cs = Cn[: len(new)] @ Cn[len(new):].T   # entrants x prior members
            Us = U6[P]; Ss = Us[: len(new)] @ Us[len(new):].T
            pool = np.array(sorted(KEEP_IDX - cls - prev)); nonc = torch.tensor(rng.choice(pool, NRAND, replace=False), device=DEV); Cr = unitr(attribution(M, j, nonc, S)); Ur = U6[nonc]
            thr_c = (Cn[: len(new)] @ Cr.T).quantile(0.95, dim=1); thr_s = (Us[: len(new)] @ Ur.T).quantile(0.95, dim=1)
            best_c, arg_c = Cs.max(1); best_s, arg_s = Ss.max(1); ages = torch.tensor([ti_ - ENTRY[j][q] if ENTRY[j][q] >= 0 else 10 ** 6 for q in old], device=DEV)
            n_recent = int((ages <= RECENT).sum()); exp_chain = n_recent / len(old)
            for i, p in enumerate(new):
                kc = "jump" if best_c[i] < thr_c[i] else ("chain" if ages[arg_c[i]] <= RECENT else "old"); ks = "jump" if best_s[i] < thr_s[i] else ("chain" if ages[arg_s[i]] <= RECENT else "old")
                EV[j].append(dict(t=ti_, pos=p, attr=kc, state=ks, cos=float(best_c[i]), thr=float(thr_c[i]), exp_chain=exp_chain))
        for p in new: ENTRY[j][p] = ti_
        PREV[j] = cls
    del M, S, a6, X6c, U6; torch.cuda.empty_cache(); L_(f"checkpoint {n}: entrants so far {sum(len(v) for v in EV.values())}")
rows = {}
for j in NEU:
    ev = EV[j]; n = len(ev)
    if n < 5: continue
    sh = lambda key, val: mean([float(e[key] == val) for e in ev]); exp_c = mean([e["exp_chain"] for e in ev]); chain_a, jump_a, chain_s, jump_s = sh("attr", "chain"), sh("attr", "jump"), sh("state", "chain"), sh("state", "jump")
    fin = FIN[j]; kinds = {int(p): None for p in fin["idx"].tolist()}
    for e in ev:
        if e["pos"] in kinds: kinds[e["pos"]] = e["attr"]
    core_members = [kinds[int(p)] for p, c in zip(fin["idx"].tolist(), fin["is_core"].tolist()) if c]; off_members = [kinds[int(p)] for p, c in zip(fin["idx"].tolist(), fin["is_core"].tolist()) if not c]
    cs = lambda L: mean([float(k == "chain") for k in L if k is not None])
    rows[j] = dict(n_entrants=n, chain_attr=chain_a, old_attr=sh("attr", "old"), jump_attr=jump_a, expected_chain=exp_c, enrichment_attr=chain_a / max(exp_c, 1e-9), chain_state=chain_s, jump_state=jump_s, enrichment_state=chain_s / max(exp_c, 1e-9),
                   root_one_token=ROOT[j], final_one_token=one_tok(fin["top20"]), multi=fin["multi"], core_chain=cs(core_members), off_chain=cs(off_members), n_final_entered=sum(v is not None for v in kinds.values()))
R = list(rows.values()); two = [r for r in R if r["multi"] and r["core_chain"] is not None and r["off_chain"] is not None]
S_ = dict(n=len(R), median_chain_attr=med([r["chain_attr"] for r in R]), median_old_attr=med([r["old_attr"] for r in R]), median_jump_attr=med([r["jump_attr"] for r in R]), median_expected_chain=med([r["expected_chain"] for r in R]), median_enrichment_attr=med([r["enrichment_attr"] for r in R]),
          share_enrichment_above_15=mean([float(r["enrichment_attr"] > 1.5) for r in R]), share_chain_above_jump=mean([float(r["chain_attr"] > r["jump_attr"]) for r in R]), median_chain_state=med([r["chain_state"] for r in R]), median_jump_state=med([r["jump_state"] for r in R]), median_enrichment_state=med([r["enrichment_state"] for r in R]),
          median_root_one_token=med([r["root_one_token"] for r in R]), median_final_one_token=med([r["final_one_token"] for r in R]), share_root_more_concentrated=mean([float(r["root_one_token"] > r["final_one_token"]) for r in R]), n_two_route=len(two), share_off_chain_above_core=mean([float(r["off_chain"] > r["core_chain"]) for r in two]) if two else None,
          median_off_chain=med([r["off_chain"] for r in two]) if two else None, median_core_chain=med([r["core_chain"] for r in two]) if two else None)
V = dict(G1_chain_enriched_70pct=bool(S_["share_enrichment_above_15"] >= 0.7), G2_jump_median_below_015=bool(S_["median_jump_attr"] < 0.15), G3_root_more_token_concentrated=bool(S_["median_root_one_token"] > S_["median_final_one_token"]), G4_offshoot_chain_above_core_60pct=bool(S_["share_off_chain_above_core"] is not None and S_["share_off_chain_above_core"] >= 0.6))
res = dict(tag=tag, block=B, checkpoints=LATE, positions=N, class_size=K, n_neurons=len(R), summary=S_, verdicts=V, neurons={str(j): r for j, r in rows.items()})
summ = (f"[{tag}] growth topology, {len(R)} neurons, {sum(r['n_entrants'] for r in R)} entrants: attribution space chain {S_['median_chain_attr']:.2f} / old {S_['median_old_attr']:.2f} / jump {S_['median_jump_attr']:.2f} (expected chain {S_['median_expected_chain']:.2f}, enrichment {S_['median_enrichment_attr']:.2f}, above 1.5 for {S_['share_enrichment_above_15']:.2f}); "
        f"state space chain {S_['median_chain_state']:.2f} / jump {S_['median_jump_state']:.2f} (enrichment {S_['median_enrichment_state']:.2f}); root one-token share {S_['median_root_one_token']:.2f} vs final {S_['median_final_one_token']:.2f}; two-route neurons {S_['n_two_route']}: offshoot chain share above core's in {S_['share_off_chain_above_core']}; verdicts {V}")
save(TAG, res, summ)
