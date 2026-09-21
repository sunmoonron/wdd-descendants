"""e298: does the causal core predict attention routing? For the candidates' natural ablation, the change of every
attention head's output (the input of the output projection, split by head) at the source tokens in the three blocks
after L is the routing profile r_t; it is predicted on held-out tokens by ridge from the 16-dimensional function
core z_t, from the full descendant (top-64 PCs), and from a random 16-dimensional projection. Where the model
returns attention probabilities, the same is done for the change of the source token's attention rows."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; NH = arch.NH; blocks = [lv for lv in (L + 1, L + 2, L + 3) if lv < NB]
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); F = (S0i[L] - S1i[L])[idx]; torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]; Fc = F - F[tr].mean(0, keepdim=True)
G = dl[tr] @ dl[tr].T; evg, Vg = torch.linalg.eigh(G); Zt = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Zt = Zt - Zt.mean(0, keepdim=True); U_fn = torch.linalg.svd(Fc[tr].T @ Zt, full_matrices=False)[0][:, :16]; U_pca = torch.linalg.svd(Fc[tr], full_matrices=False)[2][:64].T; U_rnd = torch.linalg.qr(torch.randn(D, 16, device=DEV))[0]
heads = {}
def capture(ablate, want_attn):
    hs = [arch.attn_lin(lv).register_forward_pre_hook((lambda lv_: lambda m, inp: heads.__setitem__(lv_, inp[0].detach().float().reshape(NT, NH, -1).norm(dim=2)))(lv)) for lv in blocks]
    if ablate:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), tn] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    try: out = model(ids_seq, output_attentions=want_attn)
    except Exception: out = model(ids_seq)
    [h.remove() for h in hs]; att = getattr(out, "attentions", None); return {lv: heads[lv].clone() for lv in blocks}, (None if att is None or len(att) == 0 or att[0] is None else [a.detach().float() for a in att])
h0, a0 = capture(False, True); h1, a1 = capture(True, True); r = torch.cat([(h1[lv] - h0[lv])[idx] for lv in blocks], 1)
def ridge_r2(X, Y):
    Xtr, Ytr = X[tr], Y[tr]; mx, my = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True); Xc, Yc = Xtr - mx, Ytr - my; Gm = Xc.T @ Xc; Wm = torch.linalg.solve(Gm + 1e-1 * Gm.diagonal().mean() * torch.eye(Gm.shape[0], device=DEV), Xc.T @ Yc); pred = (X[te] - mx) @ Wm + my; return 1 - ((Y[te] - pred) ** 2).sum().item() / ((Y[te] - my) ** 2).sum().item()
res = dict(head_profile_from_core=ridge_r2(Fc @ U_fn, r), head_profile_from_pca64=ridge_r2(Fc @ U_pca, r), head_profile_from_random16=ridge_r2(Fc @ U_rnd, r), head_profile_from_norm_only=ridge_r2(F.norm(dim=1, keepdim=True), r), n_heads=NH, blocks=blocks)
if a0 is not None and a1 is not None:
    seq = idx // CTX; pos = idx % CTX; rows = torch.cat([torch.stack([(a1[lv][s, :, p, :] - a0[lv][s, :, p, :]).norm(dim=1) for s, p in zip(seq.tolist(), pos.tolist())]) for lv in blocks], 1)
    res.update(attention_rows_from_core=ridge_r2(Fc @ U_fn, rows), attention_rows_from_pca64=ridge_r2(Fc @ U_pca, rows), attention_rows_from_random16=ridge_r2(Fc @ U_rnd, rows), attention_rows_available=True)
else: res["attention_rows_available"] = False
log(f"{tag} (K {K}, heads {NH}, blocks {blocks}): held-out R2 for the per-head attention-output change at the source token: from the 16-dim core {res['head_profile_from_core']:+.2f}, from the descendant's top-64 PCs {res['head_profile_from_pca64']:+.2f}, from a random 16-dim projection {res['head_profile_from_random16']:+.2f}, from the descendant norm alone {res['head_profile_from_norm_only']:+.2f}" + (f" | attention rows: core {res['attention_rows_from_core']:+.2f}, PCs {res['attention_rows_from_pca64']:+.2f}, random {res['attention_rows_from_random16']:+.2f}" if res["attention_rows_available"] else " | attention probabilities not returned by this model"))
record(f"e298_routing_{tag}", dict(model=tag, b=b, L=L, K=K, **res), f"head-output change R2: core {res['head_profile_from_core']:+.2f} pca64 {res['head_profile_from_pca64']:+.2f} random16 {res['head_profile_from_random16']:+.2f} norm {res['head_profile_from_norm_only']:+.2f}" + (f" | attention rows: core {res['attention_rows_from_core']:+.2f} pca64 {res['attention_rows_from_pca64']:+.2f} random16 {res['attention_rows_from_random16']:+.2f}" if res["attention_rows_available"] else " | no attention probabilities"))
