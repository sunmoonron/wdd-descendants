"""e584 (session 106): watching words being born without forgetting their parameter ancestry, against a learned dictionary
that must. Pythia-410m block 12, steps 1000-16000 (e582's cache, 24 sequences), a TopK autoencoder trained at every
checkpoint (e583). The eventual words are the 256 most used rows at 16000 with a class (over-the-floor positions) of
ten or more; the tracked ones are those not yet words at 1000. For every tracked word and checkpoint:
 - the row's S and whether it is a word (recruitment);
 - the class's existence in the cloud without the row: the mean pairwise cosine of the class's unit states against
   random sets of the same size (present when above the random sets' 95th percentile);
 - the checkpoint's autoencoder: the feature whose active set best matches the class (Jaccard), and that feature's
   decoder cosine with the row at that checkpoint (does the learned dictionary point at the parameter?);
 - the coalition's coherence along the row's direction at the class's eight largest projections (the checkpoint model
   run with hooks, the MLP producers of blocks 0-12);
 - the row's direction against its final direction.
Events: the first checkpoint at which each holds and keeps holding (class present, autoencoder feature at Jaccard 0.25,
recruitment, coalition coherence 0.3, the feature's decoder at cosine 0.3 with the row). The orderings and leads.
Pre-registered (probabilities are honest guesses):
 L1 (0.6) the class is present in the cloud before the row is recruited, by two checkpoints or more at the median;
 L2 (0.5) the autoencoder has a feature for the class at the checkpoint the class becomes present (it dates the class
    as well as the row-free measure);
 L3 (0.6) the autoencoder's feature does not point at the row at recruitment (cosine under 0.3 for most words): it
    dates the class, not the parameter;
 L4 (0.5) the coalition coheres before recruitment (session 95)."""
from s101_common import *
t0 = time.time(); CD = "/workspace/wdd/cache/e582_pythia410"; steps = list(range(1000, 16001, 1000)); T = len(steps); B = 12; KX = 8; JT = 0.25; ids = eval_ids("pythia410")[:24, :512].to(DEV)
ck = {n: torch.load(f"{CD}/step{n}.pt", map_location="cpu") for n in steps}; keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0); kidx = torch.nonzero(keepc)[:, 0]; N = int(keepc.sum()); D = ck[steps[0]]["D"]; DFF = ck[steps[0]]["DFF"]
U = {n: unitr((ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV)) for n in steps}; A = {n: ck[n]["rows"].float().to(DEV) for n in steps}; NORM = {n: ck[n]["norms"].to(DEV) for n in steps}; m = A[steps[0]].shape[0]
log(f"{N} common positions, {m} rows")
ST = {}; WORDS = {}
for n in steps:
    st = stats(U[n], A[n], K); ST[n] = dict(S=st["S"], usage=st["usage"]); WORDS[n] = wordset(st["usage"])
    if n == 16000: RATIO16 = st["ratio"].float()
    del st; torch.cuda.empty_cache(); log(f"step {n}: stats ({time.time() - t0:.0f}s)")
w16 = torch.nonzero(WORDS[16000])[:, 0]; csize = (RATIO16[:, w16] > 1).sum(0); ok = (csize >= 10) & ~WORDS[1000][w16]; tracked = w16[ok]; log(f"eventual words {w16.numel()}, with a class of ten or more and not words at 1000: {tracked.numel()}")
CLS = {int(w): torch.nonzero(RATIO16[:, w] > 1)[:, 0] for w in tracked.tolist()}
# autoencoders
class SAE:
    def __init__(s, n):
        d = torch.load(f"{CD}/sae_step{n}.pt", map_location=DEV); s.We, s.be, s.Wd, s.bd, s.mu, s.scale = d["We"].float(), d["be"].float(), d["Wd"].float(), d["bd"].float(), d["mu"].float(), d["scale"]; s.k = d["meta"]["k"]
    def encode(s, X):
        z = ((X - s.mu) / s.scale - s.bd) @ s.We.T + s.be; top = z.topk(s.k, dim=1); out = torch.zeros_like(z); out.scatter_(1, top.indices, torch.relu(top.values)); return out
g = torch.Generator().manual_seed(0)
def class_presence(n):
    """row-free: the mean pairwise cosine of the class's unit states, against random sets of the same size (95th percentile)"""
    Un = U[n]; out = {}
    for w, pos in CLS.items():
        k = pos.numel(); Uc = Un[pos.to(DEV)]; coh = float((Uc @ Uc.T).sum() - k) / (k * (k - 1)); rnd = []
        for _ in range(20):
            r = torch.randperm(N, generator=g)[:k].to(DEV); Ur = Un[r]; rnd.append(float((Ur @ Ur.T).sum() - k) / (k * (k - 1)))
        rt = torch.tensor(rnd); out[w] = dict(coh=coh, null95=float(rt.quantile(0.95)), null_med=float(rt.median()), present=coh > float(rt.quantile(0.95)))
    return out
def sae_match(n):
    sae = SAE(n); Z = sae.encode(ck[n]["X"][keepc].float().to(DEV)); act = Z > 0; fs = act.sum(0); live = torch.nonzero(fs >= 5)[:, 0]; out = {}
    Fl = act[:, live].float(); An = A[n]; Wd = unitr(sae.Wd[live])
    for w, pos in CLS.items():
        cm = torch.zeros(N, device=DEV); cm[pos.to(DEV)] = 1; inter = Fl.T @ cm; jac = inter / (Fl.sum(0) + cm.sum() - inter).clamp_min(1); j, f = jac.max(0); f = int(live[f])
        out[w] = dict(jaccard=float(j), feature=f, decoder_row_cos=float(unitr(sae.Wd[f]) @ An[w]), feature_size=int(fs[f]))
    del Z, act, Fl; torch.cuda.empty_cache(); return out
def coalition(n):
    """the checkpoint model run with hooks; contributions of the MLP producers of blocks 0-12 along each tracked row at its class's eight largest projections at 16000"""
    model, _, fam = load_model("pythia410", revision=f"step{n}"); arch = Arch(model, fam)
    need = sorted(set(int(p) for w, pos in CLS.items() for p in pos[RATIO16[pos, w].topk(min(KX, pos.numel())).indices].tolist()))
    need_full = kidx[torch.tensor(need)]; sel = torch.zeros(ids.shape[0] * (ids.shape[1] - 1), dtype=torch.bool); sel[need_full] = True; ACT = {b: [] for b in range(B + 1)}
    class Stop(Exception): pass
    def stop(mm, i, o): raise Stop
    hstop = arch.layers[B].register_forward_hook(stop)
    try:
        with torch.no_grad():
            for s0 in range(0, ids.shape[0], 8):
                sub = ids[s0:s0 + 8]; ACTs = {}
                hs2 = [arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda mm, a: ACTs.__setitem__(b_, a[0].detach().float()[:, 1:].reshape(-1, a[0].shape[-1]).half()))(b)) for b in range(B + 1)]
                try: model(sub)
                except Stop: pass
                finally: [h.remove() for h in hs2]
                for b in range(B + 1): ACT[b].append(ACTs[b])
    finally: hstop.remove()
    act = torch.cat([torch.cat(ACT[b]) for b in range(B + 1)], 1)   # [all positions 1:, m] fp16
    act = act[sel.to(DEV)]; pmap = {int(p): i for i, p in enumerate(need)}; An, Nn = A[n], NORM[n]; out = {}
    for w, pos in CLS.items():
        top = pos[RATIO16[pos, w].topk(min(KX, pos.numel())).indices]; rows_ = torch.tensor([pmap[int(p)] for p in top.tolist()], device=DEV); C = act[rows_].float() * ((An @ An[w]) * Nn)[None]; Cu = unitr(C); cs = Cu @ Cu.T; kk = C.shape[0]
        out[w] = dict(coherence=float((cs.sum() - kk) / (kk * (kk - 1))), own_rank=float((C.abs() > C.abs()[:, w][:, None]).sum(1).float().mean()))
    del model, act; torch.cuda.empty_cache(); return out
H = {}
for n in steps:
    H[n] = dict(presence=class_presence(n), sae=sae_match(n), coal=coalition(n)); log(f"step {n}: class present {mean([float(v['present']) for v in H[n]['presence'].values()]):.2f}, SAE feature at Jaccard >= 0.25 {mean([float(v['jaccard'] >= JT) for v in H[n]['sae'].values()]):.2f}, decoder-row cos {med([v['decoder_row_cos'] for v in H[n]['sae'].values()]):.2f}, recruited {mean([float(WORDS[n][w] and ST[n]['S'][w] >= 1) for w in CLS]):.2f}, coalition coherence {med([v['coherence'] for v in H[n]['coal'].values()]):.2f} | {time.time() - t0:.0f}s")
def first_hold(flags):
    """first index from which the flag holds at every later checkpoint (or the last two); None if never"""
    for i in range(len(flags)):
        if all(flags[i:]) and flags[i]: return i
    return None
ev = {}
for w in CLS:
    fl = dict(cls=[H[n]["presence"][w]["present"] for n in steps], sae=[H[n]["sae"][w]["jaccard"] >= JT for n in steps], row=[bool(WORDS[n][w]) and float(ST[n]["S"][w]) >= 1 for n in steps], coal=[H[n]["coal"][w]["coherence"] >= 0.3 for n in steps], align=[H[n]["sae"][w]["decoder_row_cos"] >= 0.3 for n in steps])
    ev[w] = {k: first_hold(v) for k, v in fl.items()}; ev[w]["row_cos_final_at_entry"] = float(A[steps[ev[w]["row"]]][w] @ A[16000][w]) if ev[w]["row"] is not None else None
def lead(a, b): return [ev[w][b] - ev[w][a] for w in ev if ev[w][a] is not None and ev[w][b] is not None]
res = dict(n_tracked=len(CLS), n_positions=N, events={str(w): v for w, v in ev.items()}, defined={k: sum(ev[w][k] is not None for w in ev) for k in ("cls", "sae", "row", "coal", "align")},
           leads={f"{a}->{b}": dict(median=med(lead(a, b)), share_positive=mean([float(x > 0) for x in lead(a, b)]), share_zero=mean([float(x == 0) for x in lead(a, b)]), n=len(lead(a, b))) for a, b in (("cls", "row"), ("sae", "row"), ("cls", "sae"), ("coal", "row"), ("row", "align"), ("cls", "coal"))},
           decoder_cos_at_recruitment=med([H[steps[ev[w]["row"]]]["sae"][w]["decoder_row_cos"] for w in ev if ev[w]["row"] is not None]), share_decoder_cos_ge_03_at_recruitment=mean([float(H[steps[ev[w]["row"]]]["sae"][w]["decoder_row_cos"] >= 0.3) for w in ev if ev[w]["row"] is not None]),
           sae_jaccard_at_class_presence=med([H[steps[ev[w]["cls"]]]["sae"][w]["jaccard"] for w in ev if ev[w]["cls"] is not None]), per_step={str(n): dict(present=mean([float(v["present"]) for v in H[n]["presence"].values()]), sae=mean([float(v["jaccard"] >= JT) for v in H[n]["sae"].values()]), recruited=mean([float(WORDS[n][w] and ST[n]["S"][w] >= 1) for w in CLS]), coal=med([v["coherence"] for v in H[n]["coal"].values()]), decoder_cos=med([v["decoder_row_cos"] for v in H[n]["sae"].values()])) for n in steps})
Ld = res["leads"]; Df = res["defined"]
summ = (f"longitudinal ({len(CLS)} tracked eventual words, {N} positions, 16 checkpoints): events defined class/SAE/row/coalition/alignment {Df['cls']}/{Df['sae']}/{Df['row']}/{Df['coal']}/{Df['align']}; leads in checkpoints (thousands of steps): class present -> row recruited {Ld['cls->row']['median']} (positive in {Ld['cls->row']['share_positive']:.2f}, same checkpoint {Ld['cls->row']['share_zero']:.2f}, n {Ld['cls->row']['n']}); SAE feature -> row {Ld['sae->row']['median']} ({Ld['sae->row']['share_positive']:.2f}); class -> SAE feature {Ld['cls->sae']['median']} ({Ld['cls->sae']['share_positive']:.2f}, same {Ld['cls->sae']['share_zero']:.2f}); coalition -> row {Ld['coal->row']['median']} ({Ld['coal->row']['share_positive']:.2f}); row -> SAE decoder aligned with the row {Ld['row->align']['median']} ({Ld['row->align']['share_positive']:.2f}, n {Ld['row->align']['n']}); "
        f"the SAE's decoder cosine with the row at recruitment {res['decoder_cos_at_recruitment']:.2f} (at 0.3 or more for {res['share_decoder_cos_ge_03_at_recruitment']:.2f}); the SAE's best Jaccard when the class first appears {res['sae_jaccard_at_class_presence']:.2f} | {time.time() - t0:.0f}s")
log(summ); record("e584_longitudinal", res, summ)
