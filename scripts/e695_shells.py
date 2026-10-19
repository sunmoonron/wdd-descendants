"""e695 (session 156): the residue of recursive generalization in the final model, read through the ledger. The essay says a
neuron's class grew outward from a local origin; e683 found the arrivals adjacent but unordered over Pythia-160m's
checkpoints. Without checkpoints, growth by generalization leaves residue: the class, taken as nested shells of activation
rank (the top 1 percent as the core, then 1 to 5, 5 to 20, 20 to 50 percent, and the bottom half as the null), should be
connected at every level, each shell adjacent in ledger space (the exact split of the neuron's drive into upstream units)
and in state space to the core, until a frontier; the weak shells should be driven by the core's root inputs; the seed
should survive as a lexically narrow, high-activation core; the periphery should lean on deeper inputs; the read row should
be a sum of sub-class directions; and the shells' next tokens should overlap the core's. Nanda's reading of weak activations
(other features leaking through) predicts a break: outer shells unrelated to the core in ledger space, driven by other
inputs. Arguments: --model M --block B [--smoke]. Pre-registered in e695_prereg.json; the neurons are e685's sample."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e695_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF, VOC = M["NB"], M["D"], M["DFF"], M["V"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else 128; NNEU = 6 if SMOKE else 100; NATT = n_att_feeding(M, B); NSUB = 400; NTOP = 64; SHELLS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.5)]; SNAME = ["core", "s1_5", "s5_20", "s20_50"]
ids = windows_from(wikitext_ids(M, "train"), NWIN, T).to(DEV); N = NWIN * (T - 1); NMLP = B * DFF; NU = NMLP + NATT + 1
UP, AT, EM, XM, AB = [], [], [], [], []
with Capture(M, list(range(B + 1)), ["act", "att", "xin", "emb"]) as C:
    for s0 in range(0, NWIN, 8):
        run_base(M, ids[s0:s0 + 8])
        UP.append(torch.cat([C.cap[("act", b)][:, 1:].half() for b in range(B)], -1).reshape(-1, NMLP) if B else torch.zeros(8 * (T - 1), 0, device=DEV, dtype=torch.half)); AT.append(torch.stack([C.cap[("att", b)][:, 1:].half() for b in range(NATT)], 2).reshape(-1, NATT, D))
        EM.append(C.cap[("emb", 0)][:, 1:].half().reshape(-1, D)); XM.append(C.cap[("xin", B)][:, 1:].float().reshape(-1, D)); AB.append(C.cap[("act", B)][:, 1:].half().reshape(-1, DFF))
UP, AT, EM, XM, AB = torch.cat(UP), torch.cat(AT), torch.cat(EM), torch.cat(XM), torch.cat(AB); TOK = ids[:, 1:].reshape(-1); NXT = torch.cat([ids[:, 2:], torch.full((NWIN, 1), -1, device=DEV, dtype=torch.long)], 1).reshape(-1)
Rd = read_w(M, B); RB = read_bias(M, B); RT = {k: read_tilde(M, B, Rd[k]) for k in Rd}; WUP = torch.cat([write_w(M, b).T for b in range(B)]) if B else torch.zeros(0, D, device=DEV); P_ = ln_params(M, B)
SIG = (XM.var(-1, unbiased=False) + P_["eps"]).sqrt() if P_["centre"] else (XM.pow(2).mean(-1) + P_["eps"]).sqrt(); key0 = "h" if M["fam"] != "gated" else "g"; U = unitr(XM - XM.mean(0)); MU_ALL = XM.mean(0)
BLK = torch.arange(NMLP, device=DEV) // DFF if NMLP else torch.zeros(0, device=DEV, dtype=torch.long)
def contrib(k, P):
    r = RT[key0][k]; V = WUP @ r; s = SIG[P]; return torch.cat([UP[P].float() * V[None] / s[:, None], (AT[P].float() @ r) / s[:, None], ((EM[P].float() @ r) / s)[:, None]], 1)
ab = AB.float(); mx = ab.max(0).values; a100 = ab.topk(100, dim=0).values[-1]; floor_act = 1.0 if M["fam"] != "gated" else float(mx.quantile(0.1)); cand = torch.nonzero((mx > floor_act) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(685 + B).sample(cand, min(NNEU, len(cand))))
gen = torch.Generator(device=DEV).manual_seed(695 + B); K1 = max(20, N // 100); L_(f"block {B} of {NB}: {N} positions, {NU} upstream units, core size {K1}, {len(NEU)} neurons")
def sample(idx, n): return idx[torch.randperm(idx.numel(), device=DEV, generator=gen)[:n]] if idx.numel() > n else idx
def maxcos(A, Bm): return (unitr(A) @ unitr(Bm).T).max(1).values   # per row of A, the largest cosine to any row of Bm
def dist(toks): v = toks[toks >= 0]; p = torch.bincount(v, minlength=VOC).float(); return p / p.sum().clamp_min(1)
rows = {}
for k in NEU:
    a = ab[:, k]; order = a.argsort(descending=True); S = {nm: order[int(lo * N):int(hi * N)] if nm != "core" else order[:K1] for nm, (lo, hi) in zip(SNAME, SHELLS)}; S["s1_5"] = order[K1:int(0.05 * N)]
    core = S["core"]; null = sample(order[int(0.5 * N):], NSUB); Cc = contrib(k, core); Cn = contrib(k, null); Uc = U[core]
    cos_null = maxcos(Cn, Cc); thr = float(cos_null.quantile(0.95)); cos_null_s = maxcos(U[null], Uc); thr_s = float(cos_null_s.quantile(0.95))
    top_core = Cc.abs().topk(NTOP, dim=1).indices; cnt = torch.bincount(top_core.reshape(-1), minlength=NU); root = torch.nonzero(cnt >= 0.5 * len(core))[:, 0]; rootmask = torch.zeros(NU, dtype=torch.bool, device=DEV); rootmask[root] = True
    def root_share(Cx): pos = Cx.clamp_min(0); return float(((Cx * rootmask[None]).sum(1) / Cx.sum(1).abs().clamp_min(1e-9) * torch.sign(Cx.sum(1))).median()), float(((pos * rootmask[None]).sum(1) / pos.sum(1).clamp_min(1e-9)).median())
    def depth(Cx): pos = Cx[:, :NMLP].clamp_min(0); return float(((pos * BLK[None]).sum(1) / pos.sum(1).clamp_min(1e-9) / max(B - 1, 1)).median()) if NMLP else None
    def jac(Cx): tp = Cx.abs().topk(NTOP, dim=1).indices; inter = rootmask[tp].float().sum(1); return float((inter / (NTOP + root.numel() - inter).clamp_min(1)).median())
    def net(Cx):
        """the 16 largest inputs' share of the net sum, and the crowd's net sum over the 16 largest (signed along the 16 largest: negative = the crowd opposes them)"""
        tp = Cx.abs().topk(16, dim=1); t16 = Cx.gather(1, tp.indices).sum(1); tot = Cx.sum(1); return float((t16 / tot.abs().clamp_min(1e-9) * torch.sign(tot)).median()), float(((tot - t16) * torch.sign(t16) / t16.abs().clamp_min(1e-9)).median())
    p_core = dist(NXT[core]); p_null = dist(NXT[null]); ov_null = float(torch.minimum(p_null, p_core).sum())
    one = lambda idx: float(torch.bincount(TOK[idx], minlength=VOC).max() / idx.numel()) if idx.numel() else None
    sh = {}; prev_C, prev_U = Cc, Uc
    for nm in SNAME:
        P = sample(S[nm], NSUB) if nm != "core" else core; Cx = contrib(k, P) if nm != "core" else Cc
        cm = maxcos(Cx, Cc) if nm != "core" else None; cs = maxcos(U[P], Uc) if nm != "core" else None
        rs_signed, rs_pos = root_share(Cx); n16, c16 = net(Cx); d = dict(n=int(S[nm].numel()), root_share=rs_signed, root_share_positive=rs_pos, root_jaccard=jac(Cx), depth=depth(Cx), one_token=one(S[nm]), act_median=float(a[S[nm]].median()) if S[nm].numel() else None, net_top16_share=n16, crowd_over_top16=c16,
                                                     next_overlap_above_null=float(torch.minimum(dist(NXT[P]), p_core).sum()) - ov_null if nm != "core" else None)
        if nm != "core":
            d.update(adjacent_ledger=float((cm > thr).float().mean()), cos_ledger=float(cm.median()), adjacent_state=float((cs > thr_s).float().mean()), cos_state=float(cs.median()))
            cp = maxcos(Cx, prev_C); d["nearer_previous_shell"] = float((cp > cm).float().mean()); prev_C, prev_U = Cx, U[P]
        sh[nm] = d
    n16n, c16n = net(Cn); rsn, rspn = root_share(Cn); sh["null"] = dict(n=int(null.numel()), net_top16_share=n16n, crowd_over_top16=c16n, root_share=rsn, root_share_positive=rspn, depth=depth(Cn), act_median=float(a[null].median()))
    sh["core"]["one_token_top20"] = one(order[:20]); sh["core"]["cos_null_95"] = thr; sh["core"]["cos_state_null_95"] = thr_s; sh["core"]["root_size"] = int(root.numel())
    # the read row as a sum of steps: the class direction against the differences between sub-classes of the core
    r = RT[key0][k]; r2 = float(r @ r); Xc = XM[core]; mu_c = Xc.mean(0); dc = mu_c - MU_ALL; lab = kmeans(Xc - mu_c, 3, seed=k).to(DEV); mus = [Xc[lab == j].mean(0) - mu_c for j in range(3) if int((lab == j).sum()) >= 3]
    def explained(dirs):
        Q, _ = torch.linalg.qr(torch.stack(dirs, 1)); return float(((Q.T @ r) ** 2).sum() / r2)
    f1 = explained([dc]); f3 = explained([dc] + mus[:2]) if len(mus) >= 2 else None; rnd = torch.randn(D, 2, device=DEV, generator=gen); fn = explained([dc, rnd[:, 0], rnd[:, 1]])
    rows[k] = dict(shells=sh, read_row=dict(class_direction=f1, with_subclass_steps=f3, with_random_pair=fn, gain=(f3 - f1) if f3 is not None else None, null_gain=fn - f1))
    if len(rows) % 20 == 0: L_(f"{len(rows)} neurons; last: adjacent shares ledger {sh['s1_5']['adjacent_ledger']:.2f}/{sh['s5_20']['adjacent_ledger']:.2f}/{sh['s20_50']['adjacent_ledger']:.2f}, root share {sh['core']['root_share']:.2f}/{sh['s1_5']['root_share']:.2f}/{sh['s5_20']['root_share']:.2f}")
R = list(rows.values()); g = lambda nm, key: med([r["shells"][nm].get(key) for r in R])
S_ = {nm: {key: g(nm, key) for key in ("root_share", "root_share_positive", "root_jaccard", "depth", "one_token", "act_median", "next_overlap_above_null", "adjacent_ledger", "cos_ledger", "adjacent_state", "cos_state", "nearer_previous_shell", "net_top16_share", "crowd_over_top16")} for nm in SNAME}
S_["null"] = {key: g("null", key) for key in ("net_top16_share", "crowd_over_top16", "root_share", "root_share_positive", "depth", "act_median")}
S_["core"]["one_token_top20"] = g("core", "one_token_top20"); S_["core"]["root_size"] = g("core", "root_size"); S_["read_row"] = {key: med([r["read_row"][key] for r in R]) for key in ("class_direction", "with_subclass_steps", "with_random_pair", "gain", "null_gain")}
frontier = next((nm for nm in SNAME[1:] if (S_[nm]["adjacent_ledger"] or 0) < 0.5), None); S_["frontier_shell"] = frontier; S_["n"] = len(R)
V = dict(R1_shell1_adjacent_above_08=bool((S_["s1_5"]["adjacent_ledger"] or 0) > 0.8), R2_shell2_adjacent_above_05=bool((S_["s5_20"]["adjacent_ledger"] or 0) > 0.5), R3_frontier_within_firing_range=bool(frontier in ("s1_5", "s5_20", "s20_50")),
         R4_shell1_root_share_half_core=bool((S_["s1_5"]["root_share"] or 0) >= 0.5 * (S_["core"]["root_share"] or 0)), R4b_shell2_root_share_half_core=bool((S_["s5_20"]["root_share"] or 0) >= 0.5 * (S_["core"]["root_share"] or 0)),
         R5_periphery_deeper=bool(S_["s5_20"]["depth"] is not None and S_["core"]["depth"] is not None and S_["s5_20"]["depth"] - S_["core"]["depth"] >= 0.05), R6_root_vestige=bool((S_["core"]["one_token_top20"] or 0) >= 2 * (S_["core"]["one_token"] or 1)),
         R7_read_row_of_steps=bool(S_["read_row"]["gain"] is not None and S_["read_row"]["gain"] - S_["read_row"]["null_gain"] >= 0.1), R8_shell1_next_token_above_null=bool((S_["s1_5"]["next_overlap_above_null"] or 0) > 0.1), R9_chain_of_shells=bool((S_["s5_20"]["nearer_previous_shell"] or 0) > 0.5),
         R10_crowd_suppresses_off_class=bool(S_["null"]["crowd_over_top16"] is not None and S_["core"]["crowd_over_top16"] is not None and S_["null"]["crowd_over_top16"] < -0.5 and abs(S_["core"]["crowd_over_top16"]) < 0.5))
res = dict(model=M["hf"], block=B, n_blocks=NB, positions=N, upstream_units=NU, core_size=K1, shells=SHELLS, summary=S_, verdicts=V, neurons={str(k): v for k, v in rows.items()})
f2_ = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = (f"[{name} block {B}] {len(R)} neurons, shells by activation rank (core = top 1%): adjacent to the core in ledger space (above the bottom half's 95th percentile) s1-5 {f2_(S_['s1_5']['adjacent_ledger'])}, s5-20 {f2_(S_['s5_20']['adjacent_ledger'])}, s20-50 {f2_(S_['s20_50']['adjacent_ledger'])} (in state space {f2_(S_['s1_5']['adjacent_state'])}/{f2_(S_['s5_20']['adjacent_state'])}/{f2_(S_['s20_50']['adjacent_state'])}); frontier {frontier}; "
        f"nearer the previous shell than the core {f2_(S_['s5_20']['nearer_previous_shell'])}/{f2_(S_['s20_50']['nearer_previous_shell'])}; the core's root inputs ({f2_(S_['core']['root_size'])} units) carry {f2_(S_['core']['root_share'])} of the core's drive, {f2_(S_['s1_5']['root_share'])} / {f2_(S_['s5_20']['root_share'])} / {f2_(S_['s20_50']['root_share'])} of the shells'; "
        f"contribution-weighted upstream depth core {f2_(S_['core']['depth'])} to s5-20 {f2_(S_['s5_20']['depth'])}; one-token share top-20 {f2_(S_['core']['one_token_top20'])}, core {f2_(S_['core']['one_token'])}, s1-5 {f2_(S_['s1_5']['one_token'])}; next-token overlap with the core above the null s1-5 {f2_(S_['s1_5']['next_overlap_above_null'])}, s5-20 {f2_(S_['s5_20']['next_overlap_above_null'])}; "
        f"net share of the 16 largest inputs core/s1-5/s5-20/s20-50/bottom half {f2_(S_['core']['net_top16_share'])}/{f2_(S_['s1_5']['net_top16_share'])}/{f2_(S_['s5_20']['net_top16_share'])}/{f2_(S_['s20_50']['net_top16_share'])}/{f2_(S_['null']['net_top16_share'])}, the crowd's net sum over them (signed) {f2_(S_['core']['crowd_over_top16'])}/{f2_(S_['s1_5']['crowd_over_top16'])}/{f2_(S_['s5_20']['crowd_over_top16'])}/{f2_(S_['s20_50']['crowd_over_top16'])}/{f2_(S_['null']['crowd_over_top16'])}; "
        f"read row explained by the class direction {f2_(S_['read_row']['class_direction'])}, with two sub-class steps {f2_(S_['read_row']['with_subclass_steps'])} (random pair {f2_(S_['read_row']['with_random_pair'])}); verdicts {V}")
save(TAG, res, summ)
