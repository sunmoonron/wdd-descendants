"""e561 (session 101): scaling the instrument. OMP over all 53,248 rows of Pythia-410m's blocks 0-12 at 2040 positions is
cheap; at ten times the width and depth it is not. Screening: keep, per position, the M rows with the largest
|projection| and run OMP inside the candidates (M = 64, 256, 1024, 4096). Against full OMP: the per-position Jaccard
of the selected sets, the coverage (share of full OMP's selections inside the candidates, by selection order), the
residual energy, the word set (256 most used) Jaccard, and the wall time of each. Also a block screen: rows of the last
four blocks only. Pre-registered (probabilities are honest guesses):
 S1 (0.6) M=1024 gives a median per-position Jaccard >= 0.8 with full OMP;
 S2 (0.7) the word set agrees at Jaccard >= 0.9 from M=1024 up;
 S3 (0.7) the selections the screen misses are the later ones (order >= 8) — OMP's later picks are residual-driven
    and need not correlate with the state."""
from s101_common import *
t0 = time.time(); S = lm_states("pythia410"); U, A = S["U"], S["A"]; N, m = U.shape[0], A.shape[0]
def timed(f):
    torch.cuda.synchronize(); t = time.time(); r = f(); torch.cuda.synchronize(); return r, time.time() - t
def omp_cand(U, A, cand, Kk, batch=128, ridge=1e-5):
    """OMP restricted to per-position candidate atoms cand [N, M]; returns global selections [N, Kk] and residual energy [N]"""
    N, M = cand.shape; sel = torch.zeros(N, Kk, dtype=torch.long, device=U.device); err = torch.zeros(N, device=U.device); eye = torch.eye(Kk, device=U.device)
    for s in range(0, N, batch):
        x = U[s:s + batch]; c = cand[s:s + batch]; n = x.shape[0]; Ac = A[c]; r = x.clone(); taken = torch.zeros(n, M, dtype=torch.bool, device=U.device); Sl = torch.zeros(n, 0, dtype=torch.long, device=U.device)
        for step in range(Kk):
            corr = torch.bmm(Ac, r[:, :, None])[:, :, 0].abs().masked_fill_(taken, -1.0); pick = corr.argmax(1, keepdim=True); taken.scatter_(1, pick, True); Sl = torch.cat([Sl, pick], 1)
            As = torch.gather(Ac, 1, Sl[:, :, None].expand(-1, -1, Ac.shape[-1])); G = As @ As.transpose(1, 2) + ridge * eye[:step + 1, :step + 1]
            cf = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (As.transpose(1, 2) @ cf)[:, :, 0]
        sel[s:s + n] = torch.gather(c, 1, Sl); err[s:s + n] = r.pow(2).sum(1)
    return sel, err
(full, tfull) = timed(lambda: omp(U, A, K, batch=1024, record_err=True)); sel_f, cof_f, err_f = full; ef = err_f[:, -1]; words_f = wordset(torch.bincount(sel_f.reshape(-1), minlength=m).float())
log(f"full OMP: {tfull:.2f}s, FVU {float(ef.sum() / U.pow(2).sum()):.4f}")
res = dict(N=N, m=m, full_time=tfull, full_fvu=float(ef.sum() / U.pow(2).sum()), screens={})
def evaluate(tag, sel_s, err_s, tscreen, tomp):
    inter = torch.tensor([len(set(sel_f[i].tolist()) & set(sel_s[i].tolist())) for i in range(N)], dtype=torch.float); jac = inter / (2 * K - inter)
    words_s = wordset(torch.bincount(sel_s.reshape(-1), minlength=m).float()); wj = float((words_s & words_f).sum() / (words_s | words_f).sum())
    r = dict(time_screen=tscreen, time_omp=tomp, time_total=tscreen + tomp, speedup=tfull / (tscreen + tomp), jaccard_median=float(jac.median()), jaccard_mean=float(jac.mean()), share_identical=float((jac == 1).float().mean()),
             fvu=float(err_s.sum() / U.pow(2).sum()), fvu_ratio=float(err_s.sum() / ef.sum()), word_set_jaccard=wj, words_usage_corr=None); return r, jac
for M in (64, 256, 1024, 4096):
    (cand, tscreen) = timed(lambda: torch.cat([(U[s:s + 256] @ A.T).abs().topk(M, dim=1).indices for s in range(0, N, 256)]))
    ((sel_s, err_s), tomp) = timed(lambda: omp_cand(U, A, cand, K))
    r, jac = evaluate(f"M={M}", sel_s, err_s, tscreen, tomp)
    inside = torch.zeros(N, K, dtype=torch.bool)
    cs = cand.cpu(); sf = sel_f.cpu()
    for i in range(N): inside[i] = torch.isin(sf[i], cs[i])
    r["coverage_by_order"] = inside.float().mean(0).tolist(); r["coverage_first8"] = float(inside[:, :8].float().mean()); r["coverage_last8"] = float(inside[:, 8:].float().mean())
    res["screens"][M] = r; log(f"M={M}: screen {tscreen:.2f}s + OMP {tomp:.2f}s (full {tfull:.2f}s, speedup {r['speedup']:.1f}x); Jaccard median {r['jaccard_median']:.2f} (mean {r['jaccard_mean']:.2f}, identical {r['share_identical']:.2f}); FVU ratio {r['fvu_ratio']:.3f}; word set Jaccard {r['word_set_jaccard']:.2f}; coverage first8/last8 {r['coverage_first8']:.2f}/{r['coverage_last8']:.2f}")
# block screen: rows of blocks 9-12 only
blk = torch.arange(m, device=DEV) // S["arch"].DFF; late = torch.nonzero(blk >= 9)[:, 0]
((sel_l, cof_l, err_l), tl) = timed(lambda: omp(U, A[late], K, batch=1024, record_err=True)); sel_l = late[sel_l]
r, _ = evaluate("blocks 9-12", sel_l, err_l[:, -1], 0.0, tl); r["rows"] = int(late.numel()); res["block_screen_9_12"] = r
log(f"block screen (rows of blocks 9-12, {late.numel()} rows): {tl:.2f}s; Jaccard median {r['jaccard_median']:.2f}; FVU ratio {r['fvu_ratio']:.3f}; word set Jaccard {r['word_set_jaccard']:.2f}; full OMP's share of selections in blocks 9-12: {float((blk[sel_f] >= 9).float().mean()):.2f}")
res["full_share_late_blocks"] = float((blk[sel_f] >= 9).float().mean())
s = res["screens"]
summ = (f"screened OMP on Pythia-410m block 12 ({N} positions, {m} rows; full OMP {tfull:.2f}s): M=64/256/1024/4096 per-position Jaccard with full OMP {s[64]['jaccard_median']:.2f}/{s[256]['jaccard_median']:.2f}/{s[1024]['jaccard_median']:.2f}/{s[4096]['jaccard_median']:.2f} (median), "
        f"word set Jaccard {s[64]['word_set_jaccard']:.2f}/{s[256]['word_set_jaccard']:.2f}/{s[1024]['word_set_jaccard']:.2f}/{s[4096]['word_set_jaccard']:.2f}, FVU ratio {s[64]['fvu_ratio']:.2f}/{s[256]['fvu_ratio']:.2f}/{s[1024]['fvu_ratio']:.2f}/{s[4096]['fvu_ratio']:.2f}, "
        f"time {s[64]['time_total']:.2f}/{s[256]['time_total']:.2f}/{s[1024]['time_total']:.2f}/{s[4096]['time_total']:.2f}s; coverage of full OMP's first/last eight picks at M=256 {s[256]['coverage_first8']:.2f}/{s[256]['coverage_last8']:.2f}, at M=1024 {s[1024]['coverage_first8']:.2f}/{s[1024]['coverage_last8']:.2f}; "
        f"blocks 9-12 only: Jaccard {res['block_screen_9_12']['jaccard_median']:.2f}, word set {res['block_screen_9_12']['word_set_jaccard']:.2f} (full OMP picks {res['full_share_late_blocks']:.2f} of its atoms there) | {time.time() - t0:.0f}s")
log(summ); record("e561_screening", res, summ)
