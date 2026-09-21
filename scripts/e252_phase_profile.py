"""e252: joint per-block profile of every observable, to see whether the regimes (birth, transport, scrambling) change
at the same block. For every level from the write (block 2) to the last block: along-direction survival, footprint
identity (K-way), state identity via descendant centroids, native argmax on the state, raw and whitened footprint
energy (relative to the write), and from three transplant passes: coherent fraction of transport, pair-cosine
preservation (initial 0.75), and the additivity defect of two injected vectors. Transition levels are reported per
observable (first level below half of its initial value, or where the descendant read overtakes the native read)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; levels = list(range(b, NB)); run = make_runner(model, arch, c, ids_seq, levels, NT); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); S1 = run(b, tn); typL = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typL & big, 20); K = len(keep); torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = idx[split], idx[~split]; ltr, lte = lab_i[split], lab_i[~split]; d_w = R[tn]
pool = torch.nonzero(typL & ~torch.isin(tn, keep))[:, 0]; s_inj = tc.abs().median(); torch.manual_seed(1); ka = torch.randint(0, c.DFF, (len(pool),), device=DEV); kb = torch.randint(0, c.DFF, (len(pool),), device=DEV); va, vb = R[ka], R[kb]; u = torch.randn_like(va); u = unit(u - (u * va).sum(1, keepdim=True) * va); vr = 0.75 * va + math.sqrt(1 - 0.75 ** 2) * u
def inj(v):
    x = torch.zeros(NT, D, device=DEV); x[pool] = s_inj * v; return run(inject=x, inject_block=b + 1)
Sa, Sb, Sab, Sr = inj(va), inj(vb), inj(va + vb), inj(vr)
prof = {}
for lv in levels:
    typ = typical_mask(S0[lv]); m = typ & big; F = S0[lv] - S1[lv]; Xc = S0[lv] - S0[lv][typ].mean(0); Xs = Xc[typ]; Sig = Xs.T @ Xs / len(Xs); Si = torch.linalg.inv(Sig + 1e-3 * Sig.diagonal().mean() * torch.eye(D, device=DEV)); wv = tc[:, None] * d_w
    Fa, Fb, Fab, Fr = [(S[lv] - S0[lv])[pool] for S in (Sa, Sb, Sab, Sr)]
    # coherent fraction from Fa: group by injected neuron index ka (contexts differ) -> mean image energy over mean single energy
    ua, inv = torch.unique(ka, return_inverse=True); cnt = torch.bincount(inv).float(); M = torch.zeros(len(ua), D, device=DEV).index_add_(0, inv, Fa) / cnt[:, None]; multi = cnt >= 2; coh = ((M[multi] ** 2).sum(1).mean() / (Fa[multi[inv]] ** 2).sum(1).mean()).item() if multi.any() else float("nan")
    prof[lv] = dict(along=((F * d_w).sum(1) / tc)[m].median().item(), footprint_id=accuracy(F[te], centroids(F[tr], ltr, K), lte), state_id=accuracy(Xc[te], centroids(F[tr], ltr, K), lte), native_argmax=((unit(Xc[te]) @ unit(R[keep]).T).argmax(1) == lte).float().mean().item(), energy_raw=(F.norm(dim=1) / tc.abs())[m].median().item(), energy_whitened=((F @ Si * F).sum(1) / (wv @ Si * wv).sum(1).clamp_min(1e-9)).sqrt()[m].median().item(), coherent_fraction=coh, pair_cos=((Fa * Fr).sum(1) / (Fa.norm(dim=1) * Fr.norm(dim=1)).clamp_min(1e-9)).median().item(), additivity_defect=((Fab - Fa - Fb).norm(dim=1) / Fab.norm(dim=1).clamp_min(1e-6)).median().item())
    log(f"{tag} level {lv}: " + ", ".join(f"{k}={v:.2f}" for k, v in prof[lv].items()))
def transition(key, mode):
    vals = [(lv, prof[lv][key]) for lv in levels]; first = vals[0][1]
    for lv, v in vals:
        if mode == "half" and v < 0.5 * first: return lv
        if mode == "double" and v > 2 * first: return lv
    return None
tr_ = dict(along_half=transition("along", "half"), footprint_id_half=transition("footprint_id", "half"), state_id_half=transition("state_id", "half"), native_argmax_half=transition("native_argmax", "half"), coherent_half=transition("coherent_fraction", "half"), additivity_double=transition("additivity_defect", "double"), descendant_overtakes_native=next((lv for lv in levels if prof[lv]["state_id"] >= prof[lv]["native_argmax"]), None))
record(f"e252_phase_{tag}", dict(model=tag, b=b, L=L, K=K, profile={str(k): v for k, v in prof.items()}, transitions=tr_), "transition levels: " + " ".join(f"{k}={v}" for k, v in tr_.items()) + " | pair cos (0.75 in) by level " + " ".join(f"{lv}:{prof[lv]['pair_cos']:.2f}" for lv in levels[::max(1, len(levels) // 6)]) + " | whitened energy by level " + " ".join(f"{lv}:{prof[lv]['energy_whitened']:.2f}" for lv in levels[::max(1, len(levels) // 6)]))
