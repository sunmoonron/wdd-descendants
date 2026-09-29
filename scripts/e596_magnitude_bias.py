"""e596 (session 110): the instrument's magnitude bias, and whether a different solver removes it. The ICLR paper found
OMP recovers the sign of an identified write in 99.7% of cases and its magnitude with a median relative error of
0.25-0.54, biased upward for small writes because a 64-atom support absorbs the writes it leaves out. Here the MLP
writes of GPT-2 small at block 6 and of Pythia-410m at block 12 have exact ground truth (the activation times the row's
norm at every position), the state is decomposed over the full native dictionary (embeddings, position embeddings where
present, head bases, MLP rows, biases) and the identified MLP atoms' coefficients are scored against the truth by decile
of true magnitude, for five solvers: OMP with least-squares refit at K = 64 (the paper's); OMP at K = 128 truncated to
the 64 largest then refit; OMP at K = 64 with a non-negative refit; OMP at K = 64 with the refit shrunk by the
residual's share of the state's energy; and the oracle support (least squares on the true 64 largest MLP writers), the
bias floor given the support. Pre-registered (probabilities are honest guesses):
 B1 (0.6) the upward bias for small writes is at least halved by the K = 128 truncation (the absorbed writes are
    represented before truncation);
 B2 (0.5) the oracle support still shows an upward bias for the smallest decile (the bias is partly the omitted
    attention writes, not only the omitted MLP writes);
 B3 (0.6) the non-negative refit does not change the bias (the sign is already right)."""
from s101_common import *
from wdd_common import build_dictionary, T_MLP
name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"; B = {"gpt2": 6, "pythia410": 12}[name]; t0 = time.time(); KS = 64
model, tok, fam = load_model(name); arch = Arch(model, fam); ids = pile_ids(name if name != "pythia410" else "pythia410", nseq=8, T=256)
ACT = {}; hs = [arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: ACT.__setitem__(b_, a[0].detach().float()[:, 1:].reshape(-1, a[0].shape[-1])))(b)) for b in range(B + 1)]
X = block_states(model, arch, ids, [B])[B].reshape(-1, arch.D); [h.remove() for h in hs]; keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]
A, lab = build_dictionary(arch, blocks=list(range(B + 1))); typ, blk, idx = lab["type"], lab["block"], lab["index"]; A = unitr(A.to(DEV).float()); m = A.shape[0]
mlp = torch.nonzero(typ == T_MLP)[:, 0]; DFF = arch.DFF; norms = torch.cat([arch.wdir(b).norm(dim=1) for b in range(B + 1)]).to(DEV)
truth = torch.cat([ACT[b][keep] for b in range(B + 1)], 1) * norms[None]   # [N, (B+1)*DFF] true coefficients of the unit MLP atoms
gidx = {int(a): int(b * DFF + j) for a, b, j in zip(mlp.tolist(), blk[mlp].tolist(), idx[mlp].tolist())}; mlp_of_atom = torch.full((m,), -1, dtype=torch.long); mlp_of_atom[mlp] = torch.tensor([gidx[int(a)] for a in mlp.tolist()])
log(f"{name} block {B}: {N} states, dictionary {m} atoms ({mlp.numel()} MLP), truth from hooks; sum of true MLP writes explains {float(((truth @ A[mlp]) * Xk).sum(1).div(Xk.norm(dim=1) ** 2).median()):.2f} of the state's energy at the median")
TOPK_TRUE = truth.abs().topk(KS, dim=1).indices.cpu(); TOP16_TRUE = truth.abs().topk(16, dim=1).indices.cpu()
def score(sel, cof, tag, restrict=None):
    """relative error and sign of the selected MLP atoms' coefficients against the truth, by decile of true magnitude; restrict='top64' or 'top16' keeps only the selected atoms that are among the position's true largest MLP writes (the paper's identified writes)"""
    rows = []
    for i in range(N):
        allowed = set(TOPK_TRUE[i].tolist()) if restrict == "top64" else (set(TOP16_TRUE[i].tolist()) if restrict == "top16" else None)
        for a, c in zip(sel[i].tolist(), cof[i].tolist()):
            j = int(mlp_of_atom[a])
            if j < 0 or (allowed is not None and j not in allowed): continue
            t = float(truth[i, j]); rows.append((abs(t), c, t))
    if not rows: return dict(n=0)
    T_ = torch.tensor([r[0] for r in rows]); Cc = torch.tensor([r[1] for r in rows]); Tt = torch.tensor([r[2] for r in rows]); rel = (Cc - Tt).abs() / T_.clamp_min(1e-6); ratio = (Cc.abs() / T_.clamp_min(1e-6)); sign = (torch.sign(Cc) == torch.sign(Tt)).float()
    q = T_.quantile(torch.linspace(0, 1, 11)); dec = torch.bucketize(T_, q[1:-1]); out = dict(n=len(rows), rel_median=float(rel.median()), sign_acc=float(sign.mean()), ratio_median=float(ratio.median()), by_decile=[dict(rel=float(rel[dec == k].median()), ratio=float(ratio[dec == k].median()), n=int((dec == k).sum())) for k in range(10)])
    top = truth.abs().topk(KS, dim=1).indices; ident = [len(set(int(mlp_of_atom[a]) for a in sel[i].tolist() if int(mlp_of_atom[a]) >= 0) & set(top[i].tolist())) / KS for i in range(N)]; out["identification_of_true_top64"] = mean(ident)
    log(f"{tag}: {out['n']} identified MLP writes, relative error median {out['rel_median']:.3f}, sign {out['sign_acc']:.4f}, |estimate|/|truth| median {out['ratio_median']:.2f}; by decile of true magnitude (smallest first) ratio " + " ".join(f"{d['ratio']:.2f}" for d in out["by_decile"]) + f"; identification of the true top-64 {out['identification_of_true_top64']:.2f}"); return out
res = dict(model=name, block=B, n_states=N, n_atoms=m, mlp_energy_share_median=float(((truth @ A[mlp]) * Xk).sum(1).div(Xk.norm(dim=1) ** 2).median()), solvers={}, identified={})
sel64, cof64, err64 = omp(Xk, A, KS, batch=512, record_err=True); res["solvers"]["omp64_ls"] = score(sel64, cof64, "OMP K=64, least-squares refit")
res["identified"]["omp64_ls_top64"] = score(sel64, cof64, "OMP K=64 LS, identified writes (selected atoms among the true top-64)", restrict="top64"); res["identified"]["omp64_ls_top16"] = score(sel64, cof64, "OMP K=64 LS, identified writes (true top-16)", restrict="top16")
sel128, cof128, _ = omp(Xk, A, 128, batch=256, record_err=True); keep64 = cof128.abs().topk(KS, dim=1).indices; sel_t = torch.gather(sel128, 1, keep64); cof_t, _ = refit(Xk, A, sel_t); res["solvers"]["omp128_trunc64"] = score(sel_t, cof_t, "OMP K=128, truncated to 64, refit")
def nnls_refit(X_, A_, sel_, iters=60):
    """projected-gradient non-negative refit on the support, per position; signs taken from the least-squares refit, magnitudes constrained to that sign"""
    cof, _ = refit(X_, A_, sel_); sgn = torch.sign(cof); sgn[sgn == 0] = 1; As = A_[sel_] * sgn[:, :, None]; c = cof.abs().clone(); G = As @ As.transpose(1, 2); b = (As @ X_[:, :, None])[:, :, 0]; L = torch.linalg.matrix_norm(G, ord=2).clamp_min(1e-6)
    for _ in range(iters): c = (c - ((G @ c[:, :, None])[:, :, 0] - b) / L[:, None]).clamp_min(0)
    return c * sgn
cof_nn = nnls_refit(Xk, A, sel64); res["solvers"]["omp64_nonneg"] = score(sel64, cof_nn, "OMP K=64, sign-constrained refit")
shrink = (1 - err64[:, -1] / Xk.pow(2).sum(1).clamp_min(1e-9)).clamp(0, 1).sqrt(); res["solvers"]["omp64_shrunk"] = score(sel64, cof64 * shrink[:, None], "OMP K=64, refit shrunk by the explained share")
top = truth.abs().topk(KS, dim=1).indices; sel_or = mlp[torch.tensor([[int((mlp_of_atom[mlp] == j).nonzero()[0]) for j in top[i].tolist()] for i in range(N)])]; cof_or, _ = refit(Xk, A, sel_or); res["solvers"]["oracle_support_ls"] = score(sel_or, cof_or, "oracle support (true top-64 MLP writers), least-squares refit")
S_ = res["solvers"]; d0 = lambda k: S_[k]["by_decile"][0]["ratio"]; d9 = lambda k: S_[k]["by_decile"][9]["ratio"]; I_ = res["identified"]
summ = (f"{name} block {B} ({N} states; the MLP writes carry {res['mlp_energy_share_median']:.2f} of the state's energy at the median): identified writes (selected atoms among the true top-64 / top-16 MLP writers): n {I_['omp64_ls_top64']['n']} / {I_['omp64_ls_top16']['n']}, relative error median {I_['omp64_ls_top64']['rel_median']:.3f} / {I_['omp64_ls_top16']['rel_median']:.3f}, sign {I_['omp64_ls_top64']['sign_acc']:.4f} / {I_['omp64_ls_top16']['sign_acc']:.4f}, |estimate|/|truth| smallest / largest decile {I_['omp64_ls_top64']['by_decile'][0]['ratio']:.2f} / {I_['omp64_ls_top64']['by_decile'][9]['ratio']:.2f}; all selected MLP atoms: |estimate|/|truth| at the smallest / largest decile of true magnitude: OMP-64 LS {d0('omp64_ls'):.2f} / {d9('omp64_ls'):.2f} (median relative error {S_['omp64_ls']['rel_median']:.3f}, sign {S_['omp64_ls']['sign_acc']:.4f}); K=128 truncated {d0('omp128_trunc64'):.2f} / {d9('omp128_trunc64'):.2f} ({S_['omp128_trunc64']['rel_median']:.3f}); sign-constrained {d0('omp64_nonneg'):.2f} / {d9('omp64_nonneg'):.2f} ({S_['omp64_nonneg']['rel_median']:.3f}); shrunk {d0('omp64_shrunk'):.2f} / {d9('omp64_shrunk'):.2f} ({S_['omp64_shrunk']['rel_median']:.3f}); oracle support {d0('oracle_support_ls'):.2f} / {d9('oracle_support_ls'):.2f} ({S_['oracle_support_ls']['rel_median']:.3f}); identification of the true top-64: OMP-64 {S_['omp64_ls']['identification_of_true_top64']:.2f}, K=128 truncated {S_['omp128_trunc64']['identification_of_true_top64']:.2f} | {time.time() - t0:.0f}s")
log(summ); record(f"e596_magnitude_{name}", res, summ)
