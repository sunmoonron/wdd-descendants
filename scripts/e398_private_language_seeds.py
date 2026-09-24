"""e398: private languages across individuals. e396 asked whether one training stage can describe another's states in its
own words; here the question is whether one individual can. PolyPythia (van der Wal et al., ICLR 2025) trained
Pythia-410m again from further seeds on the same data. For the standard run (seed 0) and seeds 1 and 2, each network's
vocabulary (token embeddings, MLP write rows and head output bases up to the middle depth, as e396) describes another
network's middle-depth states (k-sparse OMP, splice, loss recovered in the described network's own forward pass):
 raw: the vocabulary as is, in the other network's coordinates;
 lexicon: translated by an orthogonal map fitted only on the two input token-embedding matrices (the one part of the two
   languages that is shared by construction, the same tokens) - weights only, no activations;
 lexicon_out: the same with the two unembedding matrices;
 state_orth / state_lin: translated by an orthogonal / a ridge-linear map fitted on paired residual states (fit set of 8
   sequences disjoint from the 3 described, levels 3, 6, 9, 12 stacked, each network centred);
 control: the vocabulary rotated at random and then translated by state_orth (translation without meaningful words);
 own and own_rot: the network's own vocabulary and a random rotation of it (the ceiling and the floor).
Stitch: the translated state itself spliced in (the two individuals' thoughts compared directly).
Pre-registered: raw at the random level; state translation recovers most of the own-minus-random gap at k 64;
lexicon translation between random and state translation. If lexicon translation reaches own, one rotation fixed by
the shared tokens relates the two individuals' internal languages; if it stays at random, the internal language is
private to the individual even though the lexicon is shared."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
MODELS["pythia410s1"] = ("EleutherAI/pythia-410m-seed1", "neox"); MODELS["pythia410s2"] = ("EleutherAI/pythia-410m-seed2", "neox")
NETS = ["pythia410", "pythia410s1", "pythia410s2"]; c = Cache("pythia410"); ev = c.s["eval_ids"][:3].to(DEV); fit = c.s["eval_ids"][3:11].to(DEV)
B, T = ev.shape; KS = [16, 64]; FITL = [3, 6, 9, 12]
V, E, U, S, XE = {}, {}, {}, {}, {}
def states(model, arch, ids, levels):
    out = {}; hs = [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: out.__setitem__(l_, (o[0] if isinstance(o, tuple) else o).detach().float()))(l)) for l in levels]
    try:
        with torch.no_grad(): lg = model(ids).logits.float()
    finally: [h.remove() for h in hs]
    return {l: out[l][:, 1:].reshape(-1, out[l].shape[-1]) for l in levels}, lg
for n in NETS:
    m, tok, fam = load_model(n); a = Arch(m, fam); L = a.NB // 2; A, lab = build_dictionary(a, blocks=list(range(L + 1))); V[n] = A
    E[n] = m.get_input_embeddings().weight.detach().float(); U[n] = m.get_output_embeddings().weight.detach().float()
    st, _ = states(m, a, fit, FITL); S[n] = torch.cat([st[l] - st[l].mean(0, keepdim=True) for l in FITL]); XE[n] = states(m, a, ev, [L])[0][L]; del m, a
def procrustes(Xa, Xb):
    Uu, _, Vh = torch.linalg.svd((Xa.T @ Xb).double()); return (Uu @ Vh).float()
def ridge_map(Xa, Xb, lam=1e-3):
    G = (Xa.T @ Xa).double(); lam = lam * G.diagonal().mean(); return torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=DEV, dtype=torch.float64), (Xa.T @ Xb).double()).float()
def r2(Xa_mapped, Xb): return 1 - (Xa_mapped - Xb).pow(2).sum().item() / Xb.pow(2).sum().item()
res = dict(nets=NETS, level=L, k=KS, cells={}, stitch={}, fitq={})
for j in NETS:
    model, tok, fam = load_model(j); arch = Arch(model, fam); stj, lg = states(model, arch, ev, [L]); Lc = token_loss(lg, ev).mean().item()
    x = stj[L]; mu = x.mean(0, keepdim=True); Xc = x - mu
    def splice(Xh):
        def hk(m, i, o):
            xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, 1:] = Xh.to(xo.dtype).view(B, T - 1, -1)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hh = arch.layers[L].register_forward_hook(hk)
        try:
            with torch.no_grad(): return token_loss(model(ev).logits.float(), ev).mean().item()
        finally: hh.remove()
    Lm = splice(mu.expand(x.shape[0], -1)); gap = Lm - Lc; res[f"gap_{j}"] = gap
    def describe(A, name):
        A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-8); sel, _, _ = omp(Xc, A, max(KS), batch=256, record_err=False); cell = {}
        for k in KS:
            cof, err = refit(Xc, A, sel[:, :k]); Ls = splice(mu + torch.einsum("nk,nkd->nd", cof, A[sel[:, :k]])); cell[str(k)] = (Lm - Ls) / max(gap, 1e-9)
        res["cells"][name] = cell; log(f"{name}: k16 {cell['16']:.2f} k64 {cell['64']:.2f}"); del sel
    describe(V[j], f"own->{j}"); describe(rotate(V[j], seed=7), f"own_rot->{j}")
    for i in NETS:
        if i == j: continue
        maps = dict(lexicon=procrustes(E[i], E[j]), lexicon_out=procrustes(U[i], U[j]), state_orth=procrustes(S[i], S[j]), state_lin=ridge_map(S[i], S[j]))
        xic = XE[i] - XE[i].mean(0, keepdim=True)
        res["fitq"][f"{i}->{j}"] = {nm: dict(state_r2=r2(xic @ Q, Xc), emb_r2=r2(E[i] @ Q, E[j]), unemb_r2=r2(U[i] @ Q, U[j])) for nm, Q in maps.items()}
        describe(V[i], f"raw:{i}->{j}")
        for nm, Q in maps.items(): describe(V[i] @ Q, f"{nm}:{i}->{j}")
        describe(rotate(V[i], seed=7) @ maps["state_orth"], f"control:{i}->{j}")
        for nm, Q in maps.items():
            Ls = splice(mu + xic @ Q); res["stitch"][f"{nm}:{i}->{j}"] = (Lm - Ls) / max(gap, 1e-9)
        log(f"fit {i}->{j}: " + " | ".join(f"{nm} stateR2 {q['state_r2']:.2f} embR2 {q['emb_r2']:.2f} unembR2 {q['unemb_r2']:.2f} stitch {res['stitch'][f'{nm}:{i}->{j}']:.2f}" for nm, q in res["fitq"][f"{i}->{j}"].items()))
    del model, arch
def avg(pfx, k):
    v = [res["cells"][f"{pfx}:{i}->{j}"][str(k)] for j in NETS for i in NETS if i != j]; return sum(v) / len(v)
own = {k: sum(res["cells"][f"own->{j}"][str(k)] for j in NETS) / 3 for k in KS}; rot = {k: sum(res["cells"][f"own_rot->{j}"][str(k)] for j in NETS) / 3 for k in KS}
summ = " | ".join(f"k{k}: own {own[k]:.2f} rot {rot[k]:.2f} " + " ".join(f"{p} {avg(p, k):.2f}" for p in ["raw", "lexicon", "lexicon_out", "state_orth", "state_lin", "control"]) for k in KS)
log("summary " + summ); record("e398_seeds_pythia410", res, summ)
