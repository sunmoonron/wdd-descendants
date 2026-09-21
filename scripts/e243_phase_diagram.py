"""e243: the training phase diagram (Pythia-410m at one revision per run; tokens from the pythia410 cache). Block-2
dominant writes of that checkpoint. Measures: native dual@64 dominant recall at +4 and +10 with a dictionary built
from the checkpoint's weights (blocks 0..level, all atom types, typical-token centering); the along-direction and
total footprint of the write at +4; angle preservation at +4 (an injected pair with initial cosine 0.75); provenance
capacity (nearest-centroid bits, K = 32) from the state and from the footprint at +4, +10 and NB-2 (final
scrambling); descendant cloud dimension at +4 for real writes and random directions."""
import sys, os, glob, shutil; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
rev = sys.argv[1]; c = Cache("pythia410"); hub = os.path.join(os.environ.get("HF_HOME", "/workspace/.hf_home"), "hub", "models--EleutherAI--pythia-410m"); before = set(glob.glob(os.path.join(hub, "snapshots", "*")))
model, tok, fam = load_model(c.name, revision=None if rev == "final" else rev); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = arch.NB; D = arch.D; b = 2; levels = [b + 4, b + 10, NB - 2]; run = make_runner(model, arch, c, ids_seq, levels, NT)
W = arch.wdir(b).to(DEV); WN = W.norm(dim=1); R = W / WN[:, None].clamp_min(1e-9); st = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st.__setitem__("a", inp[0].detach().float().reshape(-1, arch.DFF))); model(ids_seq); h.remove()
led = st["a"] * WN[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; d = R[tn]; big = tc.abs() >= tc.abs().quantile(0.5); S0 = run(); S1 = run(b, tn); res = {}
def pr(M):
    M = M - M.mean(0, keepdim=True); s = torch.linalg.svdvals(M); e = s ** 2; return ((e.sum() ** 2) / (e ** 2).sum()).item()
for lv in levels:
    typ = typical_mask(S0[lv]); F = S0[lv] - S1[lv]; r = {}
    if lv < NB - 2:
        Alv, lablv = build_dictionary(arch, blocks=range(lv + 1)); Alv = Alv.to(DEV); mu = S0[lv][typ].mean(0); X = S0[lv] - mu; S = Alv.T @ Alv; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sel = oneshot(X, Alv, 64, whiten=Winv)[0]
        rows_b = torch.nonzero((lablv["type"] == T_MLP) & (lablv["block"] == b))[:, 0].to(DEV); order = lablv["index"][(lablv["type"] == T_MLP) & (lablv["block"] == b)].to(DEV); lut = torch.zeros(arch.DFF, dtype=torch.long, device=DEV); lut[order] = rows_b; row = lut[tn]
        r["native_recall"] = (sel == row[:, None]).any(1)[typ & big].float().mean().item(); r["along"] = ((F * d).sum(1) / tc)[typ & big].median().item(); r["total"] = (F.norm(dim=1) / tc.abs())[typ & big].median().item()
    idx0 = torch.nonzero(typ & big)[:, 0]; neur0 = tn[idx0]; u, cnt = torch.unique(neur0, return_counts=True); order2 = cnt.argsort(descending=True); u = u[order2][cnt[order2] >= 8]; K = min(32, len(u)); keep = u[:K]
    if K >= 8:
        m = torch.isin(neur0, keep); idx = idx0[m]; lab_i = (neur0[m][:, None] == keep[None, :]).float().argmax(1); torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; cents = centroids(F[idx[split]], lab_i[split], K); Xc = S0[lv] - S0[lv][typ].mean(0)
        r["K"] = K; r["state_bits"] = mutual_info_bits(Xc[idx[~split]], cents, lab_i[~split], K); r["footprint_bits"] = mutual_info_bits(F[idx[~split]], cents, lab_i[~split], K); r["footprint_acc"] = accuracy(F[idx[~split]], cents, lab_i[~split])
    if lv == b + 4:
        torch.manual_seed(0); k = torch.randint(0, arch.DFF, (NT,), device=DEV); s = tc.abs().median(); w = R[k]; uu = torch.randn_like(w); uu = unit(uu - (uu * w).sum(1, keepdim=True) * w); w2 = 0.75 * w + math.sqrt(1 - 0.75 ** 2) * uu
        Sa = run(inject=s * w, inject_block=b + 1); Sb = run(inject=s * w2, inject_block=b + 1); Fa = Sa[lv] - S0[lv]; Fb = Sb[lv] - S0[lv]; r["pair_cos_0p75"] = ((Fa * Fb).sum(1) / (Fa.norm(dim=1) * Fb.norm(dim=1)).clamp_min(1e-9))[typ].median().item()
        U1 = unit(torch.randn(NT, D, device=DEV)); Sr = run(inject=-tc.abs()[:, None] * U1, inject_block=b + 1); r["dim_real"] = pr(F[typ & big]); r["dim_random"] = pr((Sr[lv] - S0[lv])[typ & big]); r["dim_state"] = pr(S0[lv][typ & big])
    res[lv] = r
    log(f"{rev} level {lv}: " + ", ".join(f"{k_}={v:.2f}" if isinstance(v, float) else f"{k_}={v}" for k_, v in r.items()))
record(f"e243_phase_{rev}", dict(revision=rev, per_level={str(k): v for k, v in res.items()}), f"{rev}: native recall +4 {res[b + 4].get('native_recall', float('nan')):.2f}, +10 {res[b + 10].get('native_recall', float('nan')):.2f} | along +4 {res[b + 4].get('along', float('nan')):.2f}, total {res[b + 4].get('total', float('nan')):.2f} | pair cos (0.75 in) {res[b + 4].get('pair_cos_0p75', float('nan')):.2f} | bits K32 state/footprint: +4 {res[b + 4].get('state_bits', float('nan')):.1f}/{res[b + 4].get('footprint_bits', float('nan')):.1f}, +10 {res[b + 10].get('state_bits', float('nan')):.1f}/{res[b + 10].get('footprint_bits', float('nan')):.1f}, NB-2 {res[NB - 2].get('state_bits', float('nan')):.1f}/{res[NB - 2].get('footprint_bits', float('nan')):.1f} | dim real/random/state at +4 {res[b + 4].get('dim_real', float('nan')):.0f}/{res[b + 4].get('dim_random', float('nan')):.0f}/{res[b + 4].get('dim_state', float('nan')):.0f}")
del model; torch.cuda.empty_cache()
for p in set(glob.glob(os.path.join(hub, "snapshots", "*"))) - before: shutil.rmtree(p, ignore_errors=True)
