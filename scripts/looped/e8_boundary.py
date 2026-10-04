"""e8: the thought handed between loops (16 sequences). (A) The stand-in at the loop boundary: the pre-norm state at
the end of loop t (t = 1, 2, 3), which after the between-loop norm is loop t+1's whole input, replaced by its k-atom
reconstruction (native OMP, rotated OMP, one-shot, PCA; k = 16, 64) or by its mean, scored at the next exit and at the
loop-4 exit; partial codes keep only the MLP, the attention or the embedding atoms of the 64-atom native code (no
refit). (B) Counterfactual inputs to loop t+1: every token gets the token-mean of the carried state (no token
content), or another token's carried state (content, but the wrong token's)."""
import sys; sys.path.insert(0, "/data/loopwdd/code")
from lw import *
m = LM("ouro"); C = torch.load(f"{ROOT}/cache/ouro_e1.pt"); d, L, TL, NS = m.d, m.L, 4, 16; Q = rotation(d)
ids = C["ids"][:NS]; fit = C["fit_ids"]
Aall, lab = m.atoms(range(L)); typ = lab["type"].to(DEVD)
fit24 = {t: [] for t in range(3)}
with torch.no_grad():
    for b in range(0, fit.shape[0], 4):
        m.run(fit[b:b + 4].to(DEVM), loops=3, cb=lambda t, l, x: fit24[t].append(x[:, 1:].reshape(-1, d).cpu()) if l == L else None)
fit24 = {t: torch.cat(v) for t, v in fit24.items()}
def exits(x, t, l):
    out = []
    with torch.no_grad():
        for b in range(0, NS, 4):
            hs = m.run_from(x[b:b + 4].to(DEVM), t, l, loops=TL)
            out.append(torch.stack([m.ce(h, ids[b:b + 4].to(DEVM)) for h in hs]).cpu())
    return torch.cat(out, 1).mean((1, 2))          # CE at each exit from loop t+1 (index t) to 4
res = dict(clean={i + 1: C["ce"][i][:NS].mean().item() for i in range(TL)})
for t in range(3):
    Xs = C["states"][(t, L)][:NS]; mu = fit24[t].mean(0).to(DEVD); X = Xs[:, 1:].reshape(-1, d).to(DEVD) - mu
    U = torch.linalg.svd(fit24[t].to(DEVD) - mu, full_matrices=False)[2]
    def score(Xh):
        x = Xs.clone(); x[:, 1:] = (Xh + mu).view(NS, -1, d).cpu(); return exits(x, t, L)
    with torch.no_grad():
        t0 = time.time(); sel, cof, err = omp(X, Aall, 64); selr, cofr, _ = omp(X @ Q.T, Aall, 64)
        so = (X @ Aall.T).abs().topk(64, dim=1).indices
    log(f"boundary {t+1}->{t+2}: codes in {(time.time()-t0)/60:.1f} min")
    Lm = score(torch.zeros_like(X)); row = dict(mean=Lm.tolist())
    for k in (16, 64):
        with torch.no_grad():
            c, _ = refit(X, Aall, sel[:, :k]); rec = {"native": recon(Aall, sel[:, :k], c)}
            c, _ = refit(X @ Q.T, Aall, selr[:, :k]); rec["rotated"] = recon(Aall, selr[:, :k], c) @ Q
            c, _ = refit(X, Aall, so[:, :k]); rec["oneshot"] = recon(Aall, so[:, :k], c)
            P = U[:k].T; rec["pca"] = X @ P @ P.T
        for nm, Xh in rec.items():
            ce = score(Xh); Lc = torch.tensor([res["clean"][i + 1] for i in range(t, TL)])
            row[f"{nm}_k{k}"] = dict(ce=ce.tolist(), recovered=((Lm[-1] - ce[-1]) / (Lm[-1] - Lc[-1])).item(),
                                     recovered_next=((Lm[0] - ce[0]) / (Lm[0] - Lc[0])).item(), fvu=((X - Xh).pow(2).sum() / X.pow(2).sum()).item())
    ty = typ[sel]                                    # partial codes from the 64-atom native code, no refit
    for nm, keep in (("mlp_only", T_MLP), ("head_only", T_ATT), ("emb_only", T_EMB)):
        c = cof * (ty == keep).float(); ce = score(recon(Aall, sel, c)); Lc = res["clean"][TL]
        row[nm] = dict(ce=ce.tolist(), recovered=((Lm[-1] - ce[-1]) / (Lm[-1] - Lc)).item(), atom_share=(ty == keep).float().mean().item())
    # (B) counterfactual inputs to the next loop
    Xn = C["states"][(t + 1, 0)][:NS]
    xm = Xn.clone(); xm[:, 1:] = Xn[:, 1:].reshape(-1, d).mean(0)
    g = torch.Generator().manual_seed(1); perm = torch.randperm(NS * (Xn.shape[1] - 1), generator=g)
    xs = Xn.clone(); xs[:, 1:] = Xn[:, 1:].reshape(-1, d)[perm].view(NS, -1, d)
    for nm, x in (("input_token_mean", xm), ("input_shuffled", xs)):
        ce = exits(x, t + 1, 0); row[nm] = dict(ce=ce.tolist())
    res[f"boundary{t+1}"] = row
    log(f"boundary {t+1}->{t+2} (clean exit-4 CE {res['clean'][TL]:.3f}, mean-ablated {Lm[-1]:.3f}): " +
        " | ".join(f"k{k} " + " ".join(f"{nm[:3]} {row[f'{nm}_k{k}']['recovered']:.2f}" for nm in ("native", "rotated", "oneshot", "pca")) for k in (16, 64)) +
        f" | partial mlp {row['mlp_only']['recovered']:.2f} head {row['head_only']['recovered']:.2f} emb {row['emb_only']['recovered']:.2f}" +
        f" | inputs: token-mean exit-4 CE {row['input_token_mean']['ce'][-1]:.3f}, shuffled {row['input_shuffled']['ce'][-1]:.3f}")
    jdump(res, f"{ROOT}/results/e8_ouro.json")
