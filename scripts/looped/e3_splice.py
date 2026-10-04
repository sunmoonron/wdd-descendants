"""e3: the stand-in test. part 'single': the state before layer 12 of one loop is replaced (every position but the
BOS) by its k-atom reconstruction and the model runs on; loss recovered at the trained exit = (L_mean - L_splice) /
(L_mean - L_clean). Readers: OMP over the native dictionary, OMP over a rotated copy, one-shot ranking (top-k by
|cosine|, refit), PCA-k fitted on other text. Ouro loops 1-4 scored at the loop-4 exit, loops 5-8 at the loop-8 exit.
part 'compound': several states replaced in one forward pass, each coded online from the already-replaced stream,
on 8 sequences: one splice per loop, every fourth layer of every loop, every fourth layer of one loop; SmolLM2 one
splice, every fourth layer, every layer. Reported: excess loss over clean at the exit, and loss recovered."""
import sys; sys.path.insert(0, "/data/loopwdd/code")
from lw import *
kind, part = sys.argv[1], sys.argv[2]
m = LM(kind); C = torch.load(f"{ROOT}/cache/{kind}_e1.pt"); d, L = m.d, m.L; Q = rotation(d)
TL = 4 if kind == "ouro" else 1
ids = C["ids"]; NB = ids.shape[0]
Aall, laball = m.atoms(range(L)); layer_all = laball["layer"].to(DEVD)
def dict_for(t, l):
    return Aall if (t > 0 or l >= L) else Aall[layer_all < l]

def exits_from(x, t, l, loops, bs=4, sl=None):
    """CE [loops, B, T-1] at every exit from loop t on, starting from state x [B, T, d] before layer l of loop t."""
    out = []
    with torch.no_grad():
        for b in range(0, x.shape[0], bs):
            xb = x[b:b + bs].to(DEVM); ib = (ids if sl is None else ids[sl])[b:b + bs].to(DEVM)
            out.append(torch.stack([m.ce(h, ib) for h in m.run_from(xb, t, l, loops=loops)]).cpu())
    return torch.cat(out, 1)

res = dict(kind=kind, part=part)
if part == "single":
    K2 = torch.load(f"{ROOT}/cache/{kind}_e2.pt"); KS = (8, 16, 32, 64)
    for t in range(C["loops"]):
        last = TL if t < TL else C["loops"]      # loops 5-8 are scored at the loop-8 exit
        cod = K2["codes"][t]; mu = cod["mu"].to(DEVD); A = dict_for(t, 12)
        Xs = C["states"][(t, 12)]; X = (Xs[:, 1:].reshape(-1, d).to(DEVD) - mu)
        Pf = C["fit_states"][(t, 12)][:, 1:].reshape(-1, d).to(DEVD) - mu
        U = torch.linalg.svd(Pf - Pf.mean(0, keepdim=True), full_matrices=False)[2]
        def splice(Xh):
            x = Xs.clone(); x[:, 1:] = (Xh + mu).view(NB, -1, d).cpu(); return x
        def score(Xh):
            return exits_from(splice(Xh), t, 12, last)[-1].mean().item()
        Lc = C["ce"][last - 1].mean().item(); Lm = score(torch.zeros_like(X))
        row = dict(clean=Lc, mean=Lm, exit=last)
        meths = ("native", "rotated", "oneshot", "pca") if t < TL else ("native", "rotated")
        for k in KS:
            if t >= TL and k not in (16, 64): continue
            for mt in meths:
                with torch.no_grad():
                    if mt == "native": c, _ = refit(X, A, cod["sel"][:, :k].long().to(DEVD)); Xh = recon(A, cod["sel"][:, :k].long().to(DEVD), c)
                    elif mt == "rotated":
                        s_ = cod["selr"][:, :k].long().to(DEVD); c, _ = refit(X @ Q.T, A, s_); Xh = recon(A, s_, c) @ Q
                    elif mt == "oneshot": s_ = cod["oneshot"][:, :k].long().to(DEVD); c, _ = refit(X, A, s_); Xh = recon(A, s_, c)
                    else: P = U[:k].T; Xh = X @ P @ P.T
                    fvu = ((X - Xh).pow(2).sum() / X.pow(2).sum()).item()
                Ls = score(Xh); row[f"{mt}_k{k}"] = dict(loss=Ls, recovered=(Lm - Ls) / max(Lm - Lc, 1e-9), fvu=fvu)
        res[f"loop{t+1}"] = row
        log(kind, f"loop {t+1} (exit {last}): clean {Lc:.3f} mean-ablated {Lm:.3f} | " + " | ".join(
            f"k{k} " + " ".join(f"{mt[:3]} {row[f'{mt}_k{k}']['recovered']:.2f}" for mt in meths if f"{mt}_k{k}" in row) for k in KS))
        jdump(res, f"{ROOT}/results/e3_{kind}_single.json")
        if kind != "ouro": break
else:
    sl = slice(0, 8); ev = ids[sl]
    PTS4 = (4, 8, 12, 16, 20, 24)
    allpts = sorted({(t, l) for t in range(TL) for l in (range(1, L + 1) if kind != "ouro" else PTS4)})
    mus = {p: torch.zeros(d) for p in allpts}; cnt = 0
    with torch.no_grad():   # means at every splice point, from the fitting sequences
        for b in range(0, C["fit_ids"].shape[0], 4):
            def cb(t, l, x):
                if (t, l) in mus: mus[(t, l)] += x[:, 1:].reshape(-1, d).sum(0).cpu()
            m.run(C["fit_ids"][b:b + 4].to(DEVM), loops=TL, cb=cb); cnt += C["fit_ids"][b:b + 4].shape[0] * (C["fit_ids"].shape[1] - 1)
    mus = {p: (v / cnt).to(DEVD) for p, v in mus.items()}
    Lc = C["ce"][TL - 1][sl].mean().item(); res["clean"] = Lc
    def run_cond(points, k, rot=False, mean_only=False):
        def cb(t, l, x):
            if (t, l) not in points: return None
            X = x[:, 1:].reshape(-1, d).to(DEVD) - mus[(t, l)]
            if mean_only: Xh = torch.zeros_like(X)
            else:
                A = dict_for(t, l)
                with torch.no_grad():
                    if rot: s_, c, _ = omp(X @ Q.T, A, k); Xh = recon(A, s_, c) @ Q
                    else: s_, c, _ = omp(X, A, k); Xh = recon(A, s_, c)
                del A
            y = x.clone(); y[:, 1:] = (Xh + mus[(t, l)]).view(x.shape[0], -1, d).to(x.device); return y
        with torch.no_grad():
            hs = m.run(ev.to(DEVM), loops=TL, cb=cb)
            return [m.ce(h, ev.to(DEVM)).mean().item() for h in hs]
    conds = []
    if kind == "ouro":
        for t in (0, 3):
            for k in (16, 64, 128): conds.append((f"single_loop{t+1}_k{k}", {(t, 12)}, k, False))
        for k in (16, 64, 128): conds.append((f"perloop_k{k}", {(t, 12) for t in range(TL)}, k, False))
        conds.append(("perloop_rot_k64", {(t, 12) for t in range(TL)}, 64, True))
        for t in (0, 3): conds.append((f"every4_loop{t+1}_k64", {(t, l) for l in PTS4}, 64, False))
        for k in (64, 128): conds.append((f"every4_all_k{k}", set(allpts), k, False))
        conds.append(("every4_all_rot_k64", set(allpts), 64, True))
    else:
        for k in (16, 64, 128): conds.append((f"single_k{k}", {(0, 12)}, k, False))
        for k in (64, 128): conds.append((f"every4_k{k}", {(0, l) for l in PTS4}, k, False))
        conds.append(("every4_rot_k64", {(0, l) for l in PTS4}, 64, True))
        conds.append(("every_k64", set(allpts), 64, False))
    # mean-ablation references for the recovered fraction
    res["mean_single"] = run_cond({(0, 12)}, 0, mean_only=True)[-1]
    for nm, pts, k, rot in conds:
        t0 = time.time(); ce = run_cond(pts, k, rot)
        res[nm] = dict(ce_by_exit=ce, excess=ce[-1] - Lc, n_splices=len(pts), k=k, rotated=rot, minutes=(time.time() - t0) / 60)
        log(kind, f"{nm}: {len(pts)} splices, exit CE {ce[-1]:.3f} (clean {Lc:.3f}), excess {ce[-1] - Lc:+.3f} nats [{(time.time()-t0)/60:.1f} min]")
        jdump(res, f"{ROOT}/results/e3_{kind}_compound.json")
