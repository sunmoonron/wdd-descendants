"""e561b (session 101): e561's timings were taken while six other jobs shared the GPU. Here, alone: full OMP against the
screen (M = 256 and 1024 candidates per position by |projection|, OMP inside them with batches of 512 positions) at the
present size (2032 positions, 53,248 rows) and at four times the positions, four times the rows (the rows replicated
under three rotations), and both (sixteen times the work). Also the agreement at the present size as a check on e561.
Pre-registered (probabilities are honest guesses):
 S4 (0.6) at the present size the screen does not beat full OMP (the K matmuls are cheap; gathers are not);
 S5 (0.6) at four times the rows the screen at M=1024 is faster than full OMP."""
from s101_common import *
t0 = time.time(); S = lm_states("pythia410"); U, A = S["U"], S["A"]; N, m = U.shape
def timed(f):
    torch.cuda.synchronize(); t = time.time(); r = f(); torch.cuda.synchronize(); return r, time.time() - t
def omp_cand(U, A, cand, Kk, batch=512, ridge=1e-5):
    N, M = cand.shape; sel = torch.zeros(N, Kk, dtype=torch.long, device=U.device); eye = torch.eye(Kk, device=U.device)
    for s in range(0, N, batch):
        x = U[s:s + batch]; c = cand[s:s + batch]; n = x.shape[0]; Ac = A[c]; r = x.clone(); taken = torch.zeros(n, M, dtype=torch.bool, device=U.device); Sl = torch.zeros(n, 0, dtype=torch.long, device=U.device)
        for step in range(Kk):
            corr = torch.bmm(Ac, r[:, :, None])[:, :, 0].abs().masked_fill_(taken, -1.0); pick = corr.argmax(1, keepdim=True); taken.scatter_(1, pick, True); Sl = torch.cat([Sl, pick], 1)
            As = torch.gather(Ac, 1, Sl[:, :, None].expand(-1, -1, Ac.shape[-1])); G = As @ As.transpose(1, 2) + ridge * eye[:step + 1, :step + 1]
            cf = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (As.transpose(1, 2) @ cf)[:, :, 0]
        sel[s:s + n] = torch.gather(c, 1, Sl)
    return sel
def screened(U, A, M):
    cand = torch.cat([(U[s:s + 256] @ A.T).abs().topk(M, dim=1).indices for s in range(0, U.shape[0], 256)]); return omp_cand(U, A, cand, K)
A4 = torch.cat([A] + [unitr(rotate(A, seed=s)) for s in (21, 22, 23)]); U4 = torch.cat([U, unitr(U + 0.3 * torch.randn_like(U)), unitr(U + 0.3 * torch.randn_like(U)), unitr(U + 0.3 * torch.randn_like(U))])
res = dict(sizes={})
for tag, Uu, Aa in (("N x1, m x1", U, A), ("N x4, m x1", U4, A), ("N x1, m x4", U, A4), ("N x4, m x4", U4, A4)):
    _, tw = timed(lambda: omp(Uu[:256], Aa, K, batch=1024, record_err=False))   # warm-up
    (sel_f, _, _), tf = timed(lambda: omp(Uu, Aa, K, batch=1024, record_err=False)); r = dict(N=int(Uu.shape[0]), m=int(Aa.shape[0]), full=tf)
    for M in (256, 1024):
        sel_s, ts = timed(lambda: screened(Uu, Aa, M)); inter = (sel_f[:, :, None] == sel_s[:, None, :]).any(2).sum(1).float(); r[f"screen_{M}"] = ts; r[f"jaccard_{M}"] = float((inter / (2 * K - inter)).median()); r[f"speedup_{M}"] = tf / ts
    res["sizes"][tag] = r; log(f"{tag} ({r['N']} positions, {r['m']} rows): full {tf:.2f}s; screen M=256 {r['screen_256']:.2f}s (x{r['speedup_256']:.1f}, Jaccard {r['jaccard_256']:.2f}), M=1024 {r['screen_1024']:.2f}s (x{r['speedup_1024']:.1f}, Jaccard {r['jaccard_1024']:.2f})")
    del sel_f; torch.cuda.empty_cache()
s = res["sizes"]
summ = ("alone on the GPU: " + "; ".join(f"{tag}: full {r['full']:.2f}s, screen M=256/1024 {r['screen_256']:.2f}/{r['screen_1024']:.2f}s (x{r['speedup_256']:.1f}/x{r['speedup_1024']:.1f})" for tag, r in s.items()) + f"; M=1024 median Jaccard with full OMP at the present size {s['N x1, m x1']['jaccard_1024']:.2f} | {time.time() - t0:.0f}s")
log(summ); record("e561b_screening_scale", res, summ)
