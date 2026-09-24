"""e397: false friends or an impoverished vocabulary? e396 found that the step-4000 vocabulary describes the final model's
middle-depth states worse than a randomly rotated copy of the final vocabulary (k 16: 0.13 against 0.32 loss recovered),
while the older step-1000 vocabulary does better than random (0.44). Two readings:
 (a) false friends: the step-4000 words are partly aligned with the final model's words (median MLP row cosine 0.35), so
     the greedy description picks them for their overlap and carries their stale remainder;
 (b) impoverished vocabulary: the step-4000 atoms have a degenerate geometry (low effective rank, a shared common
     direction), so any 16 of them span a poor subspace whatever their orientation.
Control: a random rotation of each vocabulary keeps its geometry and discards its orientation.
Pre-registered: (a) predicts the rotated step-4000 vocabulary recovers at least the random level (about 0.3) at k 16 on
the final states; (b) predicts it stays near 0.13. A float64 refit of the same supports guards against a numerical
artifact (near-collinear picks). Also reported: variance left (FVU) for each description (does a vocabulary capture
variance without function?), the family mix of the picks, the median condition number of the picked atoms, each
vocabulary's effective rank (participation ratio) and common-direction strength, and for picked MLP rows the cosine to
the same row of the target checkpoint (are the picked words the ones that kept their direction?)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
VOC = ["step1000", "step4000", "step16000", "step63000", "step143000"]; TGT = ["step63000", "step143000"]
c = Cache("pythia410"); ev = c.s["eval_ids"][:3].to(DEV); B, T = ev.shape; KS = [4, 8, 16, 32, 64]
V, LAB = {}, None
for r in VOC:
    m, tok, fam = load_model("pythia410", revision=r); a = Arch(m, fam); L = a.NB // 2; A, lab = build_dictionary(a, blocks=list(range(L + 1)))
    V[r] = A; LAB = lab; del m, a
typ = LAB["type"].to(DEV); FAM = {"tok": T_TOK, "mlp": T_MLP, "att": T_ATT, "bias": T_BIAS}

def geometry(A):
    out = {}
    for nm, t in [("all", None), ("tok", T_TOK), ("mlp", T_MLP), ("att", T_ATT)]:
        X = A if t is None else A[typ == t]; C = (X.T @ X) / X.shape[0]; ev_ = torch.linalg.eigvalsh(C.double()).clamp_min(0)
        out[nm] = dict(pr=(ev_.sum() ** 2 / (ev_ ** 2).sum()).item(), common=X.mean(0).norm().item())
    return out

res = dict(voc=VOC, tgt=TGT, level=L, k=KS, geometry={r: geometry(V[r]) for r in VOC}, cells={})
for r in VOC: log(f"geometry {r}: " + " ".join(f"{nm} PR {g['pr']:.0f} common {g['common']:.2f}" for nm, g in res["geometry"][r].items()))
for j in TGT:
    model, tok, fam = load_model("pythia410", revision=j); arch = Arch(model, fam); out = {}
    h = arch.layers[L].register_forward_hook(lambda m, i, o: out.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    with torch.no_grad(): Lc = token_loss(model(ev).logits.float(), ev).mean().item()
    h.remove(); x = out["x"][:, 1:].reshape(-1, out["x"].shape[-1]); mu = x.mean(0, keepdim=True); Xc = x - mu; tot = Xc.pow(2).sum().item()
    def splice(Xh):
        def hk(m, i, o):
            xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, 1:] = Xh.to(xo.dtype).view(B, T - 1, -1)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hh = arch.layers[L].register_forward_hook(hk)
        try:
            with torch.no_grad(): return token_loss(model(ev).logits.float(), ev).mean().item()
        finally: hh.remove()
    Lm = splice(mu.expand(x.shape[0], -1)); gap = Lm - Lc
    for i in VOC:
        for rot in [False, True]:
            A = rotate(V[i], seed=7) if rot else V[i]; name = f"{i}{'_rot' if rot else ''}->{j}"
            sel, _, _ = omp(Xc, A, max(KS), batch=256, record_err=False); cell = {}
            for k in KS:
                s = sel[:, :k]; cof, err = refit(Xc, A, s); Ls = splice(mu + torch.einsum("nk,nkd->nd", cof, A[s]))
                cell[str(k)] = dict(recovered=(Lm - Ls) / max(gap, 1e-9), fvu=err.sum().item() / tot)
                if k == 16:
                    As64, X64 = A[s].double(), Xc.double(); G = As64 @ As64.transpose(1, 2)
                    c64 = torch.linalg.solve(G + 1e-10 * torch.eye(k, device=DEV, dtype=torch.float64), As64 @ X64[:, :, None])[:, :, 0]
                    L64 = splice((mu.double() + torch.einsum("nk,nkd->nd", c64, As64)).float())
                    cell["16"]["recovered_f64"] = (Lm - L64) / max(gap, 1e-9); cell["16"]["cond_median"] = torch.linalg.cond(G).median().item()
                    t = typ[s]; cell["16"]["mix"] = {nm: (t == v).float().mean().item() for nm, v in FAM.items()}
                    if not rot and i != j:
                        mm = t == T_MLP; ids_ = s[mm]
                        if ids_.numel():
                            same = (V[i][ids_] * V[j][ids_]).sum(-1); allm = (V[i][typ == T_MLP] * V[j][typ == T_MLP]).sum(-1)
                            cell["16"]["picked_row_cos_same_index"] = same.median().item(); cell["16"]["all_row_cos_same_index"] = allm.median().item()
                            cell["16"]["picked_abs_row_cos"] = same.abs().median().item()
            res["cells"][name] = cell; del sel
            c16 = cell["16"]
            log(f"{name}: rec k4..64 " + " ".join(f"{cell[str(k)]['recovered']:.2f}" for k in KS) + " | fvu " + " ".join(f"{cell[str(k)]['fvu']:.2f}" for k in KS)
                + f" | k16 f64 {c16['recovered_f64']:.2f} cond {c16['cond_median']:.1f} mix " + " ".join(f"{nm} {v:.2f}" for nm, v in c16["mix"].items())
                + (f" | picked-row cos same idx {c16['picked_row_cos_same_index']:.2f} (all {c16['all_row_cos_same_index']:.2f})" if "picked_row_cos_same_index" in c16 else ""))
    res[f"gap_{j}"] = gap; del model, arch
f = lambda n: res["cells"][n]["16"]["recovered"]
record("e397_falsefriends_pythia410", res, "k16 on final states: " + " ".join(f"{i.replace('step', '')} {f(f'{i}->step143000'):.2f}/rot {f(f'{i}_rot->step143000'):.2f}" for i in VOC)
       + " | (a) false friends if 4000_rot >= ~0.3")
