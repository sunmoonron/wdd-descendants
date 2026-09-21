"""e289: is the causal coordinate manipulable? The function subspace is the PLS pairing between descendants and logit
footprints: each descendant-space direction u_k has a paired logit-space direction l_k. Injecting the natural
footprint norm along u_k at the block-(L+1) input of foreign tokens should move the logits along l_k and not along
the other l_k' (a diagonal response matrix), while random directions should move them along none. Reported: the
8x8 matrix of median cosines between the response to u_k and l_k', its diagonal and off-diagonal means, and the
same for eight random directions."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); F = (S0i[L] - S1i[L])[idx]; Fc = F - F.mean(0, keepdim=True); fnorm = F.norm(dim=1).median()
Ud, Sd, Vd = torch.linalg.svd(dl, full_matrices=False); Zs = Ud[:, :256] * Sd[:256][None]; B = Vd[:256].T; Zs = Zs - Zs.mean(0, keepdim=True); Up, Sp, Wt = torch.linalg.svd(Fc.T @ Zs, full_matrices=False); nk = 8; u = Up[:, :nk].T; l = unit((B @ Wt[:nk].T).T)
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0][:1536]; NF = len(foreign); base = run(positions=foreign); lg0 = base["lg"]; torch.manual_seed(0); rnd = unit(torch.randn(nk, D, device=DEV))
def responses(dirs):
    out = []
    for k in range(dirs.shape[0]):
        inj = torch.zeros(NT, D, device=DEV); inj[foreign] = fnorm * dirs[k][None]; r = run(positions=foreign, inject=inj, inject_block=L + 1); dlr = r["lg"] - lg0; dlr = dlr - dlr.mean(1, keepdim=True); out.append((unit(dlr) @ l.T).median(0).values)
    return torch.stack(out)
M = responses(u); Mr = responses(rnd); diag = M.diagonal().mean().item(); off = (M.sum() - M.diagonal().sum()).item() / (nk * nk - nk); rd = Mr.abs().mean().item(); sign_match = (M.diagonal() > 0).float().mean().item(); argmax_hit = (M.abs().argmax(1) == torch.arange(nk, device=DEV)).float().mean().item()
log(f"{tag} (K {K}): response to the eight causal coordinates read against their paired logit directions: diagonal mean {diag:+.2f} (sign match {sign_match:.2f}, argmax hit {argmax_hit:.2f}), off-diagonal mean {off:+.2f}, |cos| for random directions {rd:.2f} | diagonal " + "/".join(f"{v:+.2f}" for v in M.diagonal().tolist()))
record(f"e289_steering_{tag}", dict(model=tag, b=b, L=L, K=K, matrix=M.tolist(), random_matrix=Mr.tolist(), diagonal_mean=diag, offdiag_mean=off, random_abs_mean=rd, sign_match=sign_match, argmax_hit=argmax_hit), f"diagonal {diag:+.2f} (sign match {sign_match:.2f}, argmax hit {argmax_hit:.2f}), off-diagonal {off:+.2f}, random |cos| {rd:.2f}; diagonal " + "/".join(f"{v:+.2f}" for v in M.diagonal().tolist()))
