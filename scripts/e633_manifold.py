"""e633 (manifold tangent, 2026-10-12): calendar rings and their parameters. For the seven weekdays and the twelve
months, states at the name's last token over templates; per layer, the plane of the class means, the cyclic order of
the means in it (adjacencies of true neighbours, max 7 or 12), a label-shuffle null, the plane's energy share against
the Beta null for a random 2-plane (the block floor in its simplest form); at the best block, the MLP rows' writes
projected on the plane: how many rows carry half of the MLP-written between-class pattern, the MLP share against the
embedding and attention, and phase coding (do the most class-selective rows write at the class's angle); transport of
the ring to later blocks in the fitted plane. Standalone: torch, transformers, numpy. Arguments: model [--smoke]."""
import sys, os, json, math, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = {"pythia160": "EleutherAI/pythia-160m", "pythia410": "EleutherAI/pythia-410m", "gpt2": "gpt2", "qwen05": "Qwen/Qwen2.5-0.5B", "olmo1b": "allenai/OLMo-1B-0724-hf", "smollm135": "HuggingFaceTB/SmolLM2-135M"}
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); dev = "cuda:0" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
T_DAY = ["The meeting is scheduled for {X}, so please be on time.", "She told me she would arrive on {X} afternoon.", "We usually go to the market on {X} mornings.", "My favourite day of the week is {X} because of the weather.", "The report is due on {X}, according to the manager.", "They play football every {X} at the park.", "On {X} the shop closes early.", "I will call you back on {X} with the details.", "The concert takes place on {X} evening.", "He was born on a {X} in the spring.", "Classes resume on {X} after the break.", "The flight departs on {X} at noon.", "We celebrated the win last {X} with friends.", "Every {X} she visits her grandmother.", "The deadline was moved to {X} by the client.", "On {X} nights the restaurant is full.", "The doctor's appointment is on {X} at ten.", "Our team meets on {X} to plan the week.", "The package should arrive by {X}, the courier said.", "He forgot that {X} was a holiday this year.", "The library is open late on {X}.", "The train on {X} was delayed by an hour.", "Rehearsals are held on {X} in the hall.", "Payday falls on {X} this month.", "The game was postponed from {X} to the following week.", "She always runs on {X} before work.", "Dinner with the neighbours is on {X}.", "The exhibition opens on {X} and runs for a month.", "I have a dentist visit on {X}, unfortunately.", "The bus schedule changes on {X} for the holiday."]
T_MONTH = ["The conference will be held in {X} next year.", "She moved to the city in {X} after graduating.", "The harvest usually begins in {X} in this region.", "In {X} the days grow noticeably longer.", "The invoice is dated {X} and remains unpaid.", "They were married in {X} under a clear sky.", "Enrolment for the course opens in {X}.", "The river tends to flood in {X} when the snow melts.", "His birthday is in {X}, near the end of the month.", "The festival takes place every {X} downtown.", "Prices typically rise in {X} before the holidays.", "In {X} the committee publishes its annual report.", "The museum reopened in {X} after renovations.", "We planted the garden in {X} this year.", "The exam results are released in {X}.", "The ship set sail in {X} for the southern ports.", "In {X} the company announced record profits.", "The lease expires in {X}, so we must decide soon.", "Migration peaks in {X} along this coast.", "The elections are scheduled for {X}.", "The new season of the show premieres in {X}.", "Tax returns are due in {X} in that country.", "In {X} the lake freezes over completely.", "The trial is expected to begin in {X}.", "Construction should finish by {X}, the builder promised.", "The first frost usually arrives in {X}.", "Their anniversary falls in {X} every year.", "The orchestra tours Europe in {X}.", "In {X} the valley fills with wildflowers.", "The grant application window closes in {X}."]
if SMOKE: T_DAY, T_MONTH = T_DAY[:5], T_MONTH[:5]
tok = AutoTokenizer.from_pretrained(MODELS[name]); model = AutoModelForCausalLM.from_pretrained(MODELS[name], torch_dtype=torch.float32).to(dev).eval()
cfg = model.config; fam = "neox" if hasattr(model, "gpt_neox") else ("gpt2" if hasattr(model, "transformer") else "llama")
layers = model.gpt_neox.layers if fam == "neox" else (model.transformer.h if fam == "gpt2" else model.model.layers); NB = len(layers); D = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.n_embd
def mlp_out(l): return l.mlp.dense_4h_to_h if fam == "neox" else (l.mlp.c_proj if fam == "gpt2" else l.mlp.down_proj)
def wrows(b):
    """write rows as [DFF, D]"""
    w = mlp_out(layers[b]).weight.detach().float(); return w if fam == "gpt2" else w.T
def prompts_for(names, templates):
    items = []
    for ci, X in enumerate(names):
        for ti, tpl in enumerate(templates):
            pre, post = tpl.split("{X}"); text = pre + X + post; ids_pre = tok(pre, return_tensors="pt").input_ids[0]; ids = tok(text, return_tensors="pt").input_ids[0]; ids_px = tok(pre + X, return_tensors="pt").input_ids[0]
            pos = ids_px.numel() - 1   # last subtoken of the name
            items.append(dict(cls=ci, tpl=ti, ids=ids, pos=pos))
    return items
def run(items):
    """states at the name token for every layer [n, NB+1, D], MLP activations there per block [n, NB, DFF], and the running mean of all positions per layer"""
    S, A = [], []; msum = torch.zeros(NB + 1, D, device=dev); mcnt = 0; cap = {}
    hs = [mlp_out(layers[b]).register_forward_pre_hook((lambda b_: lambda m, a: cap.__setitem__(b_, a[0].detach()))(b)) for b in range(NB)]
    try:
        for it in items:
            out = model(it["ids"][None].to(dev), output_hidden_states=True); H = torch.stack(out.hidden_states)[:, 0]   # [NB+1, T, D]
            msum += H[:, 1:].sum(1); mcnt += H.shape[1] - 1; S.append(H[:, it["pos"]].cpu()); A.append(torch.stack([cap[b][0, it["pos"]] for b in range(NB)]).cpu())
    finally: [h.remove() for h in hs]
    return torch.stack(S), torch.stack(A), (msum / mcnt).cpu()
def adjacency(angles, k):
    """how many true cyclic neighbours (i, i+1 mod k) are adjacent in the circular order of the angles"""
    order = np.argsort(angles); pos = np.empty(k, dtype=int); pos[order] = np.arange(k); return int(sum(1 for i in range(k) if (pos[i] - pos[(i + 1) % k]) % k in (1, k - 1)))
def plane_of(means):
    M = means - means.mean(0, keepdims=True); U, s, Vt = np.linalg.svd(M, full_matrices=False); share = float((s[:2] ** 2).sum() / (s ** 2).sum()); return Vt[0], Vt[1], share
def ring_stats(X, cls, k, L, mu, rng, nshuf=200):
    """per layer: plane of the class means, top-2 share, adjacencies, label-shuffle null, within/between, plane energy share"""
    Z = (X[:, L] - mu[L]).numpy(); means = np.stack([Z[cls == c].mean(0) for c in range(k)]); u, v, share = plane_of(means); ang = np.arctan2(means @ v, means @ u); adj = adjacency(ang, k)
    rad = float(np.sqrt(((means - means.mean(0)) @ np.stack([u, v]).T) ** 2).sum(1).mean()); within = float(np.mean([np.sqrt(((Z[cls == c] - means[c]) @ np.stack([u, v]).T) ** 2).sum(1).mean() for c in range(k)]))
    null_adj, null_share = [], []
    for _ in range(nshuf):
        pc = rng.permutation(cls); m2 = np.stack([Z[pc == c].mean(0) for c in range(k)]); u2, v2, s2 = plane_of(m2); null_adj.append(adjacency(np.arctan2(m2 @ v2, m2 @ u2), k)); null_share.append(s2)
    E = float((((Z @ u) ** 2 + (Z @ v) ** 2) / (Z ** 2).sum(1)).mean()); d_ = Z.shape[1]
    return dict(share=share, adjacency=adj, radius=rad, within=within, snr=rad / max(within, 1e-9), null_adj_mean=float(np.mean(null_adj)), null_adj_p=float(np.mean(np.array(null_adj) >= adj)), null_share_mean=float(np.mean(null_share)), plane_energy=E, random_plane_energy=2 / d_, angles=ang.tolist(), u=u, v=v, means=means)
def analyse(kind, names, templates, rng):
    k = len(names); items = prompts_for(names, templates); X, A, mu = run(items); cls = np.array([it["cls"] for it in items]); n = len(items); norms = [float((X[:, L] - mu[L]).norm(dim=1).mean()) for L in range(NB + 1)]
    per = {L: ring_stats(X, cls, k, L, mu, rng) for L in range(NB + 1)}
    full = [L for L in range(NB + 1) if per[L]["adjacency"] == k]; best = max(range(1, NB), key=lambda L: (per[L]["adjacency"], per[L]["share"])); Lb = best   # layer NB is after the final norm in HF outputs, so the ledger does not hold there and it is excluded from the choice
    u, v = per[Lb]["u"], per[Lb]["v"]; P = np.stack([u, v])   # [2, D]
    # the ledger at the name token: MLP writes of blocks 0..Lb-1 projected on the plane
    contrib = []; Wu, Wv = [], []
    for b in range(Lb):
        W = wrows(b).cpu().numpy(); Wu.append(W @ u); Wv.append(W @ v)
    Au = np.concatenate([A[:, b].numpy() * Wu[b][None] for b in range(Lb)], 1); Av = np.concatenate([A[:, b].numpy() * Wv[b][None] for b in range(Lb)], 1)   # [n, m]
    Cu = np.stack([Au[cls == c].mean(0) for c in range(k)]); Cv = np.stack([Av[cls == c].mean(0) for c in range(k)]); Cu -= Cu.mean(0); Cv -= Cv.mean(0)   # [k, m] between-class pattern per row
    Tu, Tv = Cu.sum(1), Cv.sum(1)   # the MLP total pattern
    E0 = (X[:, 0] - mu[0]).numpy(); Em = np.stack([E0[cls == c].mean(0) for c in range(k)]); Em -= Em.mean(0); Eu, Ev = Em @ u, Em @ v
    Zb = (X[:, Lb] - mu[Lb]).numpy(); Mb = np.stack([Zb[cls == c].mean(0) for c in range(k)]); Mb -= Mb.mean(0); Su, Sv = Mb @ u, Mb @ v   # the state's pattern
    tot = float((Su ** 2 + Sv ** 2).sum()); share_mlp = float((Tu * Su + Tv * Sv).sum() / tot); share_emb = float((Eu * Su + Ev * Sv).sum() / tot); share_rest = 1 - share_mlp - share_emb
    attr = (Cu * Tu[:, None] + Cv * Tv[:, None]).sum(0) / max(float((Tu ** 2 + Tv ** 2).sum()), 1e-12)   # per row, sums to 1 over rows
    order = np.argsort(-attr); cum = np.cumsum(attr[order]); n_half = int(np.argmax(cum >= 0.5) + 1) if (cum >= 0.5).any() else None; n_90 = int(np.argmax(cum >= 0.9) + 1) if (cum >= 0.9).any() else None
    top_attr = [float(x) for x in attr[order[:10]]]
    # phase coding: the most class-selective rows per class against the class's angle
    Aall = np.concatenate([A[:, b].numpy() for b in range(Lb)], 1); sd = Aall.std(0) + 1e-6; phases_all = np.arctan2(np.concatenate(Wv), np.concatenate(Wu)); ang = np.array(per[Lb]["angles"]); cos_sel, cos_rand = [], []
    for c in range(k):
        sel = (Aall[cls == c].mean(0) - Aall[cls != c].mean(0)) / sd; top = np.argsort(-sel)[:20]; sgn = np.sign(Aall[cls == c].mean(0)[top] + 1e-12); ph = phases_all[top] + np.where(sgn < 0, np.pi, 0.0)
        cos_sel.append(float(np.cos(ph - ang[c]).mean())); rnd = rng.choice(Aall.shape[1], 20, replace=False); cos_rand.append(float(np.cos(phases_all[rnd] - ang[c]).mean()))
    # transport: later layers projected on the best plane; relative radius per layer
    transport = {}
    for L in range(Lb, NB + 1):
        Z = (X[:, L] - mu[L]).numpy(); m = np.stack([Z[cls == c].mean(0) for c in range(k)]); m -= m.mean(0); a2 = np.arctan2(m @ v, m @ u); transport[L] = dict(adjacency_on_best_plane=adjacency(a2, k), radius_on_best_plane=float(np.sqrt(((m @ P.T) ** 2).sum(1)).mean()), relative_radius=float(np.sqrt(((m @ P.T) ** 2).sum(1)).mean() / norms[L]), adjacency_refit=per[L]["adjacency"])
    # the embedding's own plane: do later rings live in it, or in a new plane?
    u0, v0 = per[0]["u"], per[0]["v"]; P0 = np.stack([u0, v0]); emb_plane = {}
    for L in range(NB + 1):
        Z = (X[:, L] - mu[L]).numpy(); m = np.stack([Z[cls == c].mean(0) for c in range(k)]); m -= m.mean(0); a0 = np.arctan2(m @ v0, m @ u0); PL = np.stack([per[L]["u"], per[L]["v"]]); sv = np.linalg.svd(P0 @ PL.T, compute_uv=False)
        emb_plane[L] = dict(adjacency_on_embedding_plane=adjacency(a0, k), pattern_energy_on_embedding_plane=float(((m @ P0.T) ** 2).sum() / (m ** 2).sum()), principal_cosines_with_embedding_plane=[float(x) for x in sv])
    res = dict(kind=kind, k=k, n=n, best_layer=Lb, layers_full_order=full, embedding_plane={str(L): v_ for L, v_ in emb_plane.items()}, per_layer={str(L): {kk: vv for kk, vv in per[L].items() if kk not in ("u", "v", "means")} for L in range(NB + 1)}, norms=norms, relative_radius={str(L): per[L]["radius"] / norms[L] for L in range(NB + 1)},
               ledger=dict(share_mlp=share_mlp, share_embedding=share_emb, share_rest=share_rest, rows_for_half=n_half, rows_for_90=n_90, n_rows=int(attr.size), top10_attr=top_attr), phase=dict(cos_selective=cos_sel, cos_random=cos_rand, mean_selective=float(np.mean(cos_sel)), mean_random=float(np.mean(cos_rand))), transport={str(L): v_ for L, v_ in transport.items()})
    e0 = per[0]; eb = per[Lb]
    print(f"[{name} {kind}] best layer {Lb}/{NB}: adjacency {eb['adjacency']}/{k} (null {eb['null_adj_mean']:.1f}, p {eb['null_adj_p']:.3f}), top-2 share {eb['share']:.2f} (null {eb['null_share_mean']:.2f}), snr {eb['snr']:.2f}, plane energy {eb['plane_energy']:.3f} vs random {eb['random_plane_energy']:.4f}; embedding layer: adjacency {e0['adjacency']}/{k}, share {e0['share']:.2f}; layers with full order {full}; ledger shares mlp {share_mlp:.2f} emb {share_emb:.2f} rest {share_rest:.2f}; rows for half {n_half} / 90% {n_90} of {attr.size}; phase coding selective {np.mean(cos_sel):.2f} vs random {np.mean(cos_rand):.2f}; transport adjacency on the best plane " + " ".join(f"{L}:{transport[L]['adjacency_on_best_plane']}" for L in sorted(transport)) + f"; relative radius " + " ".join(f"{L}:{per[L]['radius'] / norms[L]:.3f}" for L in range(0, NB + 1, max(1, NB // 6))) + f"; on the embedding's plane: adjacency " + " ".join(f"{L}:{emb_plane[L]['adjacency_on_embedding_plane']}" for L in range(0, NB + 1, max(1, NB // 6))) + f", pattern energy " + " ".join(f"{L}:{emb_plane[L]['pattern_energy_on_embedding_plane']:.2f}" for L in range(0, NB + 1, max(1, NB // 6))) + f", principal cosines at best {[round(x, 2) for x in emb_plane[Lb]['principal_cosines_with_embedding_plane']]} ({time.time() - t0:.0f}s)", flush=True)
    return res
rng = np.random.default_rng(0)
R = dict(model=name, hf=MODELS[name], NB=NB, D=D, days=analyse("days", DAYS, T_DAY, rng), months=analyse("months", MONTHS, T_MONTH, rng))
os.makedirs("results", exist_ok=True); json.dump(R, open(f"results/e633_manifold_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=float); print("done", f"{time.time() - t0:.0f}s")
