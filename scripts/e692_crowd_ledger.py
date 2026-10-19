"""e692 (session 156): the crowd ledger. e685 found a mid-network neuron's input to be a crowd of thousands of upstream
writes, and e685b that the crowd's corpus mean recovers only a seventh to a quarter of a sixteen-input simulation's error.
Here the crowd is summarised without training, rung by rung, and each rung is scored as e685b scored the mean: the neuron
driven by its K largest inputs plus the summary, swapped in at every position, the downstream KL at its strong firings and
over all positions relative to mean-ablating the neuron. Rungs: nothing; the corpus mean; the per-token mean of the crowd
(a lookup table built on held-out text); the per-bigram mean with backoff; the crowd of the held-out position with the
same token whose K largest inputs are nearest (is the crowd a function of the sparse part?); a random held-out position
with the same token (the resample control); and the sixteen guilds with the largest crowd sums at the position, a guild
being a cluster of upstream units by write row (weights only), by co-activation on held-out text, or random of the same
sizes. Also the crowd's species at the strong firings (lexical, the per-token mean of each upstream unit's activation,
against contextual; aligned, gain above the median, against unaligned; attention; embedding), the coherence of the crowd's
largest contributors with the neuron's class, and the variance of the crowd each rung explains. Arguments: --model M
--block B [--smoke]. Pre-registered in e692_e694_prereg.json; the neurons are e685b's (same seed)."""
import argparse
from c11 import *
F_ = torch.nn.functional
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e692_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF, VOC = M["NB"], M["D"], M["DFF"], M["V"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else 64; NWH = 16 if SMOKE else 128; NNEU = 4 if SMOKE else 40; NATT = n_att_feeding(M, B); NSTRONG = 160; NG = 64 if SMOKE else 256; TOPG = 16; NCOH = 200; KS = [16, 64]; MAXC = 512
RUNGS16 = ["none", "mean", "token", "bigram", "knn", "resample", "guild_write", "guild_coact", "guild_random", "guild_write_a", "guild_coact_a", "guild_random_a", "spca16", "spca64"]; RUNGS64 = ["none", "token", "guild_write", "guild_coact", "spca16"]; GUILDS = ("guild_write", "guild_coact", "guild_random")
ids_all = wikitext_ids(M, "train"); half = len(ids_all) // 2
ids = windows_from(ids_all[:half], NWIN, T).to(DEV); ids_h = windows_from(ids_all[half:], NWH, T).to(DEV); N = NWIN * (T - 1); NH = NWH * (T - 1); NMLP = B * DFF; NU = NMLP + NATT + 1
def capture(ids_, nwin, need_lp):
    """activation rows are positions 1..T-1 of each window; TOK is the token at the row, PREV the one before it"""
    UP, AT, EM, XM, AB, LP = [], [], [], [], [], []
    with Capture(M, list(range(B + 1)), ["act", "att", "xin", "emb"]) as C:
        for s0 in range(0, nwin, 8):
            o = M["base"](input_ids=ids_[s0:s0 + 8], use_cache=False)
            if need_lp: LP.append(logprobs_at(M, o.last_hidden_state[:, :-1]))   # fp32 (see e680)
            UP.append(torch.cat([C.cap[("act", b)][:, 1:].half() for b in range(B)], -1).reshape(-1, NMLP) if B else torch.zeros(8 * (T - 1), 0, device=DEV, dtype=torch.half)); AT.append(torch.stack([C.cap[("att", b)][:, 1:].half() for b in range(NATT)], 2).reshape(-1, NATT, D))
            EM.append(C.cap[("emb", 0)][:, 1:].half().reshape(-1, D)); XM.append(C.cap[("xin", B)][:, 1:].float().reshape(-1, D)); AB.append(C.cap[("act", B)][:, 1:].half().reshape(-1, DFF))
    S = dict(UP=torch.cat(UP), AT=torch.cat(AT), EM=torch.cat(EM), XM=torch.cat(XM), AB=torch.cat(AB), TOK=ids_[:, 1:].reshape(-1), PREV=ids_[:, :-1].reshape(-1)); S["LP"] = torch.cat(LP).reshape(S["TOK"].numel(), -1) if need_lp else None; return S
E = capture(ids, NWIN, True); H = capture(ids_h, NWH, False); L_(f"block {B} of {NB}: {N} evaluation and {NH} held-out positions, {NMLP} upstream MLP units + {NATT} attention outputs + embedding")
Rd = read_w(M, B); RB = read_bias(M, B); RT = {k: read_tilde(M, B, Rd[k]) for k in Rd}; RC = {k: read_const(M, B, Rd[k], RB[k]) for k in Rd}; WUP = torch.cat([write_w(M, b).T for b in range(B)]) if B else torch.zeros(0, D, device=DEV)
BIAS = sum((write_bias(M, b) for b in range(B)), torch.zeros(D, device=DEV)); P_ = ln_params(M, B)
sig = lambda XM: (XM.var(-1, unbiased=False) + P_["eps"]).sqrt() if P_["centre"] else (XM.pow(2).mean(-1) + P_["eps"]).sqrt()
E["SIG"] = sig(E["XM"]); H["SIG"] = sig(H["XM"]); key0 = "h" if M["fam"] != "gated" else "g"
def contrib(S, k, P, key):
    """[len(P), NU]: the exact split at rows P of read key's pre-activation of neuron k into every upstream MLP unit, attention output and the embedding; and the constant"""
    r = RT[key][k]; V = WUP @ r; s = S["SIG"][P]
    C = torch.cat([S["UP"][P].float() * V[None] / s[:, None], (S["AT"][P].float() @ r) / s[:, None], ((S["EM"][P].float() @ r) / s)[:, None]], 1); return C, (BIAS @ r) / s + RC[key][k]
def ledger(S, k, key, n):
    """fp16 contributions [n, NU] (for gathers) and, in fp32, the constant, the total and for each K the K largest inputs' indices and their sum"""
    C16 = torch.empty(n, NU, device=DEV, dtype=torch.half); c0 = torch.empty(n, device=DEV); tot = torch.empty(n, device=DEV); TOP = {K: torch.empty(n, K, device=DEV, dtype=torch.long) for K in KS}; TS = {K: torch.empty(n, device=DEV) for K in KS}
    for s in range(0, n, 2048):
        P = torch.arange(s, min(s + 2048, n), device=DEV); c, cc = contrib(S, k, P, key); C16[P] = c.half(); c0[P] = cc; tot[P] = c.sum(1)
        for K in KS: top = c.abs().topk(K, dim=1).indices; TOP[K][P] = top; TS[K][P] = c.gather(1, top).sum(1)
    return C16, c0, tot, TOP, TS
def pre_true(k, P, key): xn, _ = ln_apply(M, B, E["XM"][P]); return xn @ Rd[key][k] + RB[key][k]
# ---------------- tokens: tables, candidates for the nearest neighbour and the resample
uniq, inv = torch.unique(E["TOK"], return_inverse=True); VE = uniq.numel()
hpos = torch.searchsorted(uniq, H["TOK"]).clamp(max=VE - 1); hval = uniq[hpos] == H["TOK"]; hidx = torch.where(hval, hpos, torch.full_like(hpos, VE)); cnt_tok = torch.bincount(hidx, minlength=VE + 1)[:VE]   # bin VE: held-out tokens absent from the evaluation set
bkey = lambda S: S["PREV"] * VOC + S["TOK"]; uniq2, inv2 = torch.unique(bkey(E), return_inverse=True); VB = uniq2.numel()
hpos2 = torch.searchsorted(uniq2, bkey(H)).clamp(max=VB - 1); hval2 = uniq2[hpos2] == bkey(H); hidx2 = torch.where(hval2, hpos2, torch.full_like(hpos2, VB)); cnt_bi = torch.bincount(hidx2, minlength=VB + 1)[:VB]
rows_h = torch.nonzero(hval)[:, 0]; o_h = torch.argsort(hidx[rows_h]); rows_sorted = rows_h[o_h]; keys_sorted = hidx[rows_h][o_h]; starts = torch.searchsorted(keys_sorted, torch.arange(VE + 1, device=DEV))
gen = torch.Generator(device=DEV).manual_seed(692 + B); m_e = cnt_tok[inv]; r_e = (torch.rand(N, device=DEV, generator=gen) * m_e).long().clamp(max=(m_e - 1).clamp_min(0))
RSROW = torch.where(m_e > 0, rows_sorted[(starts[inv] + r_e).clamp(max=max(rows_sorted.numel() - 1, 0))], torch.full_like(inv, -1)) if rows_sorted.numel() else torch.full_like(inv, -1)
o_e = torch.argsort(inv); starts_e = torch.searchsorted(inv[o_e], torch.arange(VE + 1, device=DEV))
COVER = dict(token=float((cnt_tok[inv] >= 2).float().mean()), bigram=float((cnt_bi[inv2] >= 2).float().mean()), candidates=float((m_e > 0).float().mean()))
def tok_table(y_h, fallback):
    """the per-token mean of a held-out row quantity y_h [NH] (a token with fewer than two held-out occurrences falls back), looked up at the evaluation rows"""
    s = torch.zeros(VE + 1, device=DEV).index_add_(0, hidx, y_h); tab = torch.where(cnt_tok >= 2, s[:VE] / cnt_tok.clamp_min(1), torch.full((VE,), float(fallback), device=DEV)); return tab[inv]
def bi_table(y_h, tok_e):
    s = torch.zeros(VB + 1, device=DEV).index_add_(0, hidx2, y_h); tab = s[:VB] / cnt_bi.clamp_min(1); return torch.where(cnt_bi[inv2] >= 2, tab[inv2], tok_e)
def knn_rows(C16e, C16h, TOPe):
    """for every evaluation row, the held-out row with the same token whose contributions on the row's K largest units are nearest (at most MAXC candidates); -1 without candidates"""
    out = torch.full((N,), -1, device=DEV, dtype=torch.long)
    for v in range(VE):
        n_v = int(starts_e[v + 1] - starts_e[v]); m_v = int(starts[v + 1] - starts[v])
        if n_v == 0 or m_v == 0: continue
        Te = o_e[starts_e[v]:starts_e[v + 1]]; cands = rows_sorted[starts[v]:starts[v + 1]]
        if m_v > MAXC: cands = cands[torch.randperm(m_v, device=DEV, generator=gen)[:MAXC]]
        top = TOPe[Te]; X = C16h[cands][:, top].float(); Ce = C16e[Te].gather(1, top).float(); d = ((X - Ce[None]) ** 2).sum(-1); out[Te] = cands[d.argmin(0)]
    return out
# ---------------- guilds of upstream MLP units: by write row (weights only), by co-activation on held-out text, random of the same sizes; attention outputs and the embedding are guilds of their own
def kmeans_gpu(X, k, iters=20, seed=0):
    """cosine k-means on the rows of X (used in fp32 chunks); labels [n]"""
    n = X.shape[0]; g = torch.Generator(device=DEV).manual_seed(seed); nrm = torch.cat([X[s:s + 4096].float().norm(dim=1) for s in range(0, n, 4096)]).clamp_min(1e-6)
    Cm = X[torch.randperm(n, device=DEV, generator=g)[:k]].float(); Cm = Cm / Cm.norm(dim=1, keepdim=True).clamp_min(1e-6)
    for it in range(iters):
        lab = torch.cat([((X[s:s + 4096].float() / nrm[s:s + 4096, None]) @ Cm.T).argmax(1) for s in range(0, n, 4096)]); Cn = torch.zeros_like(Cm)
        for s in range(0, n, 4096): Cn.index_add_(0, lab[s:s + 4096], X[s:s + 4096].float() / nrm[s:s + 4096, None])
        empty = Cn.norm(dim=1) < 1e-6
        if bool(empty.any()): Cn[empty] = X[torch.randperm(n, device=DEV, generator=g)[:int(empty.sum())]].float()
        Cm = Cn / Cn.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return lab
if NMLP:
    lab_w = kmeans_gpu(WUP, NG, seed=1)
    Xc = H["UP"].T.contiguous(); mu = torch.cat([Xc[s:s + 4096].float().mean(1) for s in range(0, NMLP, 4096)])
    for s in range(0, NMLP, 4096): Xc[s:s + 4096] = (Xc[s:s + 4096].float() - mu[s:s + 4096, None]).half()
    lab_c = kmeans_gpu(Xc, NG, seed=2); del Xc; lab_r = lab_w[torch.randperm(NMLP, device=DEV, generator=gen)]
else: lab_w = lab_c = lab_r = torch.zeros(0, dtype=torch.long, device=DEV)
extra = NG + torch.arange(NATT + 1, device=DEV); LABS = {"guild_write": torch.cat([lab_w, extra]), "guild_coact": torch.cat([lab_c, extra]), "guild_random": torch.cat([lab_r, extra])}; NGU = NG + NATT + 1
LABM = {"guild_write": lab_w, "guild_coact": lab_c, "guild_random": lab_r}; GA = {g: torch.zeros(N, NG, device=DEV) for g in GUILDS}   # each guild's total activation at every evaluation row
for s in range(0, N, 2048):
    for g in GUILDS: GA[g][s:s + 2048].index_add_(1, LABM[g], E["UP"][s:s + 2048].float())
GCNT = {g: torch.bincount(LABM[g], minlength=NG)[:NG].float().clamp_min(1) for g in GUILDS}
Xs = H["XM"] / H["SIG"][:, None]; ev_s, evec_s = torch.linalg.eigh(Xs.T @ Xs / NH); PS = {16: evec_s[:, -16:].contiguous(), 64: evec_s[:, -64:].contiguous()}; del Xs   # the stream's principal directions (second moment, held-out)
gsize = lambda lab: torch.bincount(lab, minlength=NG)[:NG].float(); L_(f"guilds: {NG} write clusters (size median {float(gsize(lab_w).median()):.0f}, max {float(gsize(lab_w).max()):.0f}), {NG} co-activation clusters (median {float(gsize(lab_c).median()):.0f}, max {float(gsize(lab_c).max()):.0f}); table coverage {COVER}")
def guild_summary(C16, TOPe, lab):
    """the sum of the TOPG guilds with the largest crowd sums at each evaluation row (the crowd: every unit outside the row's K largest), and the share of the crowd's absolute guild mass they carry"""
    out = torch.empty(N, device=DEV); share = torch.empty(N, device=DEV)
    for s in range(0, N, 2048):
        P = torch.arange(s, min(s + 2048, N), device=DEV); c = C16[P].float(); c.scatter_(1, TOPe[P], 0.0); GS = torch.zeros(len(P), NGU, device=DEV).index_add_(1, lab, c)
        tg = GS.abs().topk(TOPG, dim=1); out[P] = GS.gather(1, tg.indices).sum(1); share[P] = tg.values.sum(1) / GS.abs().sum(1).clamp_min(1e-9)
    return out, share
def guild_summary_approx(C16, TOPe, g, V):
    """the guild rung at group resolution: each MLP guild's crowd sum approximated by its mean gain times its total activation (one edge per guild), attention outputs and the embedding exact; the TOPG largest"""
    lab = LABM[g]; gbar = torch.zeros(NG, device=DEV).index_add_(0, lab, V) / GCNT[g]; out = torch.empty(N, device=DEV)
    for s in range(0, N, 2048):
        P = torch.arange(s, min(s + 2048, N), device=DEV); c = C16[P].float(); top = TOPe[P]; c.scatter_(1, top, 0.0); GSa = torch.zeros(len(P), NGU, device=DEV); sc = E["SIG"][P][:, None]
        GSa[:, :NG] = GA[g][P] * gbar[None] / sc; GSa[:, NG:] = c[:, NMLP:]
        isM = (top < NMLP).float(); ti = top.clamp(max=max(NMLP - 1, 0)); ai = E["UP"][P].gather(1, ti).float() * isM; GSa.scatter_add_(1, lab[ti], -gbar[lab[ti]] * ai / sc)   # the K largest units leave their guilds' sums
        tg = GSa.abs().topk(TOPG, dim=1); out[P] = GSa.gather(1, tg.indices).sum(1)
    return out
def spca_summary(S_, TOPe, r_, Pr):
    """the crowd's write vector (the stream minus the K largest units' writes, scaled as the read is) projected on the stream's top principal directions and read by the neuron"""
    out = torch.empty(N, device=DEV); pr = Pr @ (Pr.T @ r_)
    for s in range(0, N, 2048):
        P = torch.arange(s, min(s + 2048, N), device=DEV); top = TOPe[P]; isM = (top < NMLP).float(); isA = (top >= NMLP) & (top < NMLP + NATT); isE = (top >= NMLP + NATT).any(1).float()
        ti = top.clamp(max=max(NMLP - 1, 0)); a = S_["UP"][P].gather(1, ti).float() * isM; xm = torch.einsum("nk,nkd->nd", a, WUP[ti]) if NMLP else torch.zeros(len(P), D, device=DEV)
        ai_ = (top - NMLP).clamp(0, NATT - 1); xa = (S_["AT"][P].float().gather(1, ai_[:, :, None].expand(-1, -1, D)) * isA[:, :, None]).sum(1)
        xc = (S_["XM"][P] - BIAS[None] - xm - xa - S_["EM"][P].float() * isE[:, None]) / S_["SIG"][P][:, None]; out[P] = xc @ pr
    return out
# ---------------- species: the per-token mean of every upstream MLP unit's activation on held-out text (lexical part) against the rest (contextual)
AU = torch.zeros(VE + 1, NMLP, device=DEV)
for s in range(0, NMLP, 8192): AU[:, s:s + 8192].index_add_(0, hidx, H["UP"][:, s:s + 8192].float())
AUM = torch.cat([H["UP"][:, s:s + 8192].float().mean(0) for s in range(0, NMLP, 8192)]) if NMLP else torch.zeros(0, device=DEV); AU = torch.where((cnt_tok >= 2)[:, None], AU[:VE] / cnt_tok.clamp_min(1)[:, None], AUM[None].expand(VE, NMLP)) if NMLP else AU[:VE]
# ---------------- the neurons (e685b's sample), the ablation and the simulation
ab = E["AB"].float(); mx = ab.max(0).values; a100 = ab.topk(100, dim=0).values[-1]; floor_act = 1.0 if M["fam"] != "gated" else float(mx.quantile(0.1)); cand = torch.nonzero((mx > floor_act) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(685 + B).sample(cand, min(NNEU, len(cand))))
def act_of(PRE): return M["act"](PRE["h"]) if M["fam"] != "gated" else F_.silu(PRE["g"]) * PRE["u"]
def kl_with(k, repl, strong):
    """downstream KL at the strong positions and over all positions when neuron k's activation is replaced by repl [N] at every position (a firing at activation row i is scored at log-prob row i+1)"""
    rows_ = torch.arange(N, device=DEV).reshape(NWIN, T - 1); tot_s, tot_a = [], []; strong = strong[(strong % (T - 1)) != (T - 2)] + 1
    for s0 in range(0, NWIN, 8):
        x = ids[s0:s0 + 8]; rr = rows_[s0:s0 + 8]
        def hook(m, a):
            h = a[0].clone(); h[:, 1:, k] = repl[rr].to(h.dtype); return (h,)
        hd = write_mod(M, B).register_forward_pre_hook(hook)
        try: o = M["base"](input_ids=x, use_cache=False)
        finally: hd.remove()
        lp = logprobs_at(M, o.last_hidden_state[:, :-1]).reshape(-1, E["LP"].shape[-1]); k_ = kl(E["LP"][rr.reshape(-1)].float(), lp); tot_a.append(k_); tot_s.append(k_[torch.isin(rr.reshape(-1), strong)])
    return float(torch.cat(tot_s).mean()), float(torch.cat(tot_a).mean())
K1 = max(1, N // 100)
def overlap(cols, strong):
    """per unit: the share of the neuron's strong rows inside the unit's top-1% class on the evaluation rows (chance about 0.01)"""
    x = E["UP"][:, cols].float(); x = x if M["fam"] != "gated" else x.abs(); cl = x.topk(K1, dim=0).indices; return torch.isin(cl, strong).float().sum(0) / NSTRONG
rows = {}; err_max = 0.0; r2 = lambda s, u: float(1 - ((s - u) ** 2).sum() / ((s - s.mean()) ** 2).sum().clamp_min(1e-12))
for k in NEU:
    a = ab[:, k]; strong = a.topk(NSTRONG).indices; mean_act = torch.full((N,), float(a.mean()), device=DEV); kl_abl = kl_with(k, mean_act, strong)
    SUMM = {}; TSK = {}; C0 = {}; TOT = {}; R2 = {}; GSH = {}; species = None; coh = None
    for key in Rd:
        C16e, c0e, tote, TOPe, TSe = ledger(E, k, key, N); C16h, c0h, toth, TOPh, TSh = ledger(H, k, key, NH); C0[key] = c0e; TOT[key] = tote
        if key == key0: err_max = max(err_max, float(((tote + c0e)[strong] - pre_true(k, strong, key)).abs().max() / pre_true(k, strong, key).abs().max().clamp_min(1e-9)))
        for K in KS:
            crowd_e = tote - TSe[K]; crowd_h = toth - TSh[K]; S = {"none": torch.zeros(N, device=DEV), "mean": torch.full((N,), float(crowd_h.mean()), device=DEV)}
            S["token"] = tok_table(crowd_h, crowd_h.mean()); S["bigram"] = bi_table(crowd_h, S["token"]); nn = knn_rows(C16e, C16h, TOPe[K])
            S["knn"] = torch.where(nn >= 0, crowd_h[nn.clamp(min=0)], S["token"]); S["resample"] = torch.where(RSROW >= 0, crowd_h[RSROW.clamp(min=0)], S["token"])
            for g in GUILDS: S[g], sh = guild_summary(C16e, TOPe[K], LABS[g]); GSH[(key, K, g)] = float(sh.median()); S[g + "_a"] = guild_summary_approx(C16e, TOPe[K], g, WUP @ RT[key][k])
            for r_ in (16, 64): S[f"spca{r_}"] = spca_summary(E, TOPe[K], RT[key][k], PS[r_])
            SUMM[(key, K)] = S; TSK[(key, K)] = TSe[K]
            if key == key0: R2[K] = {r: dict(all=r2(crowd_e, S[r]), strong=r2(crowd_e[strong], S[r][strong])) for r in S}
        if key == key0:   # species and coherence at the strong firings, the crowd beyond the 16 largest inputs
            P = strong; c, _ = contrib(E, k, P, key); V = WUP @ RT[key][k]; sc = E["SIG"][P][:, None]; top = TOPe[16][P]; c.scatter_(1, top, 0.0)
            const = AUM[None] * V[None] / sc; lex = ((AU[inv[P]] - AUM[None]) * V[None] / sc) * (cnt_tok[inv[P]] >= 2)[:, None]   # each unit's corpus mean; its token-specific mean beyond that (tokens with a table); the rest is contextual
            flat = (torch.arange(len(P), device=DEV)[:, None].expand_as(top) * NMLP + top).view(-1)[(top < NMLP).view(-1)]; const.view(-1)[flat] = 0.0; lex.view(-1)[flat] = 0.0   # the K largest units are not in the crowd
            cm = c[:, :NMLP]; ctx = cm - const - lex; al = (V.abs() >= V.abs().median())[None] if NMLP else torch.zeros(1, 0, dtype=torch.bool, device=DEV); crowd_sum = float(c.sum())
            part = lambda X: float(X.sum() / crowd_sum) if abs(crowd_sum) > 1e-9 else None
            species = dict(const_aligned=part(const * al), const_unaligned=part(const * ~al), lex_aligned=part(lex * al), lex_unaligned=part(lex * ~al), ctx_aligned=part(ctx * al), ctx_unaligned=part(ctx * ~al), attention=part(c[:, NMLP:NMLP + NATT]), embedding=part(c[:, NMLP + NATT:]), crowd_sum=crowd_sum)
            if NMLP:
                mc = cm.abs().mean(0); topc = mc.topk(min(NCOH, NMLP)).indices; rnd = torch.randperm(NMLP, device=DEV, generator=gen)[:NCOH]; topg = V.abs().topk(min(NCOH, NMLP)).indices
                coh = dict(top_contributors=float(overlap(topc, strong).median()), random_units=float(overlap(rnd, strong).median()), top_gain=float(overlap(topg, strong).median()), chance=K1 / N)
        del C16e, C16h
    sims = {}
    def run(K, rung):
        PRE = {key: TSK[(key, K)] + C0[key] + SUMM[(key, K)][rung] for key in Rd}; v = kl_with(k, act_of(PRE), strong); return dict(strong=v[0], all=v[1], ratio_strong=v[0] / max(kl_abl[0], 1e-12), ratio_all=v[1] / max(kl_abl[1], 1e-12))
    for rung in RUNGS16: sims[f"16_{rung}"] = run(16, rung)
    for rung in RUNGS64: sims[f"64_{rung}"] = run(64, rung)
    v = kl_with(k, act_of({key: TOT[key] + C0[key] for key in Rd}), strong); sims["all"] = dict(strong=v[0], all=v[1], ratio_strong=v[0] / max(kl_abl[0], 1e-12), ratio_all=v[1] / max(kl_abl[1], 1e-12))
    rows[k] = dict(kl_ablate_strong=kl_abl[0], kl_ablate_all=kl_abl[1], sim=sims, r2={str(K): R2[K] for K in KS}, guild_share={f"{K}_{g}": GSH[(key0, K, g)] for K in KS for g in GUILDS}, species=species, coherence=coh)
    if len(rows) % 5 == 0: L_(f"{len(rows)} neurons; last: K=16 ratio at strong firings none {sims['16_none']['ratio_strong']:.2f}, mean {sims['16_mean']['ratio_strong']:.2f}, token {sims['16_token']['ratio_strong']:.2f}, knn {sims['16_knn']['ratio_strong']:.2f}, guild_write {sims['16_guild_write']['ratio_strong']:.2f}, coact {sims['16_guild_coact']['ratio_strong']:.2f}")
SPECIES = ("const_aligned", "const_unaligned", "lex_aligned", "lex_unaligned", "ctx_aligned", "ctx_unaligned", "attention", "embedding"); R = list(rows.values()); mr = lambda c, key: med([r["sim"][c][key] for r in R]); mr2 = lambda K, rung, key: med([r["r2"][str(K)][rung][key] for r in R])
S_ = dict(n=len(R), split_max_rel_error=err_max, coverage=COVER, kl_ablate_strong=med([r["kl_ablate_strong"] for r in R]), kl_ablate_all=med([r["kl_ablate_all"] for r in R]),
          ratio_strong={c: mr(c, "ratio_strong") for c in R[0]["sim"]}, ratio_all={c: mr(c, "ratio_all") for c in R[0]["sim"]}, r2_all={f"{K}_{rung}": mr2(K, rung, "all") for K in KS for rung in (RUNGS16 if K == 16 else RUNGS64)}, r2_strong={f"{K}_{rung}": mr2(K, rung, "strong") for K in KS for rung in (RUNGS16 if K == 16 else RUNGS64)},
          guild_share={c: med([r["guild_share"][c] for r in R]) for c in R[0]["guild_share"]}, species={s: med([r["species"][s] for r in R if r["species"]]) for s in SPECIES},
          coherence={s: med([r["coherence"][s] for r in R if r["coherence"]]) for s in ("top_contributors", "random_units", "top_gain", "chance")} if R[0]["coherence"] else None, aligned_share=med([r["species"]["const_aligned"] + r["species"]["lex_aligned"] + r["species"]["ctx_aligned"] for r in R if r["species"] and r["species"]["lex_aligned"] is not None]), lexical_share=med([r["species"]["const_aligned"] + r["species"]["const_unaligned"] + r["species"]["lex_aligned"] + r["species"]["lex_unaligned"] + r["species"]["embedding"] for r in R if r["species"] and r["species"]["lex_aligned"] is not None]), token_specific_share=med([r["species"]["lex_aligned"] + r["species"]["lex_unaligned"] for r in R if r["species"] and r["species"]["lex_aligned"] is not None]), guild_sizes=dict(write_median=float(gsize(lab_w).median()) if NMLP else None, coact_median=float(gsize(lab_c).median()) if NMLP else None))
rs = S_["ratio_strong"]; sp = S_["species"]; co = S_["coherence"]
V = dict(Q0_all_inputs_reproduce=bool(S_["ratio_all"]["all"] < 0.05), Q1_token_halves16=bool(rs["16_token"] < 0.5), Q1b_token_r2_above_half=bool((S_["r2_all"]["16_token"] or 0) > 0.5), Q2_knn_beats_resample=bool(rs["16_knn"] < rs["16_resample"] - 0.1),
         Q3_guild_write_halves16=bool(rs["16_guild_write"] < 0.5), Q3b_guild_coact_halves16=bool(rs["16_guild_coact"] < 0.5), Q3c_guild_write_beats_random=bool(rs["16_guild_write"] < rs["16_guild_random"] - 0.1), Q3d_guild_write_approx_halves16=bool(rs["16_guild_write_a"] < 0.5), Q3e_guild_write_approx_beats_random=bool(rs["16_guild_write_a"] < rs["16_guild_random_a"] - 0.1), Q3f_guild_coact_approx_halves16=bool(rs["16_guild_coact_a"] < 0.5), Q7_stream16_halves16=bool(rs["16_spca16"] < 0.5), Q7b_stream64_halves16=bool(rs["16_spca64"] < 0.5),
         Q4_aligned_majority=bool(S_["aligned_share"] is not None and S_["aligned_share"] > 0.5), Q5_coherent_crowd=bool(co is not None and co["top_contributors"] > 5 * co["chance"]), Q6_lexical_majority=bool(S_["lexical_share"] is not None and S_["lexical_share"] > 0.5), Q6b_token_specific_above_quarter=bool(S_["token_specific_share"] is not None and S_["token_specific_share"] > 0.25))
res = dict(model=M["hf"], block=B, n_blocks=NB, positions=N, heldout_positions=NH, upstream_units=NU, guilds=NG, ks=KS, rungs16=RUNGS16, rungs64=RUNGS64, summary=S_, verdicts=V, neurons={str(k): v for k, v in rows.items()})
summ = (f"[{name} block {B}] {len(R)} neurons, split error {err_max:.1e}: KL ratio of the 16-input simulation over the ablation's at the strong firings, by crowd summary: " + ", ".join(f"{r} {rs['16_' + r]:.2f}" for r in RUNGS16) + "; 64 inputs: " + ", ".join(f"{r} {rs['64_' + r]:.2f}" for r in RUNGS64) + f"; all inputs {S_['ratio_all']['all']:.2f}; "
        f"variance of the crowd explained (all rows): " + ", ".join(f"{r} {S_['r2_all']['16_' + r]:.2f}" for r in RUNGS16[1:]) + "; top-16 guilds' share of the crowd's guild mass write/coact/random " + "/".join(f"{S_['guild_share'][f'16_{g}']:.2f}" for g in GUILDS) +
        f"; species of the crowd at strong firings (shares of its sum): " + ", ".join(f"{s} {sp[s]:.2f}" if sp[s] is not None else f"{s} n/a" for s in SPECIES) + f" (aligned {S_['aligned_share']:.2f}, lexical {S_['lexical_share']:.2f}, token-specific {S_['token_specific_share']:.2f}); "
        + (f"class overlap of the 200 largest contributors {co['top_contributors']:.3f} vs random units {co['random_units']:.3f}, largest gains {co['top_gain']:.3f} (chance {co['chance']:.3f}); " if co else "") + f"table coverage {COVER}; verdicts {V}")
save(TAG, res, summ)
