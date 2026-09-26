"""e515: word tracks, the per-checkpoint half. e514 found that the rows that end as native words are used less than
their controls before step 2000: the early vocabulary is other rows, and it turns over. Two accounts compete. In
replacement the network abandons one vocabulary and builds another; in threshold crossing every row evolves
continuously and rows cross the wordhood threshold at different times, so that "turnover" is only the boundary
being crossed in both directions. They differ in the retention of the word set from checkpoint to checkpoint against
a null that keeps the rows' magnitudes, and in whether rows enter the word set in the same state as rows leave it.
This run records, for every MLP row of blocks up to the block, at one checkpoint: its usage as a native word (16-word
OMP over the centred typical states), its mean absolute write, its activity, and the cosine of its write direction
with its direction at the end of training (the run at the end writes the directions; the others wait for them).
e515b reads the checkpoints together and builds the retention curves, the transition matrix and the hysteresis test.
Setup: Pythia-410m, blocks 6 and 12; 8 x 256 evaluation tokens, typical positions.
Arguments: name [revision]."""
import sys, os, json as _json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
BL = [6, 12]; LB = max(BL); K = 16; CDIR = f"/workspace/wdd/cache/e515_{name}"; REF = CDIR + "/main_dirs.pt"; os.makedirs(CDIR, exist_ok=True)
if rev is not None:
    t0 = time.time()
    while not os.path.exists(REF):
        if time.time() - t0 > 1800: raise SystemExit("reference not found")
        time.sleep(20)
    time.sleep(5)
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; ids = eval_ids(name)[:8, :256].to(DEV)
acts = {b: [] for b in range(LB + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)]
try: S_ = block_states(model, arch, ids, BL, chunk=4)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(LB + 1)}; del acts; rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}
dirs_now = {b: unitr(arch.wdir(b).float()) for b in range(LB + 1)}
if rev is None: torch.save({b: v.cpu() for b, v in dirs_now.items()}, REF)
dirs_end = {b: v.to(DEV) for b, v in torch.load(REF).items()}
out = {}; res = dict(model=name, revision=rev, blocks=BL, by_block={})
for b in BL:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0)
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); del A; typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); m = Au.shape[0]
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    sel, _, _ = omp(Xc, Au, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m)
    u = usage[gidx].float(); mag = torch.stack([(A_all[bb][keep].abs() * rown[bb][None]).mean(0) for bb in range(b + 1)]); act = torch.stack([(A_all[bb][keep] > 0).float().mean(0) for bb in range(b + 1)]); cos = torch.stack([(dirs_now[bb] * dirs_end[bb]).sum(1) for bb in range(b + 1)])
    share_words_mlp = float((typ[sel] == T_MLP).float().mean()); n_used = int((u > 0).sum())
    top_u = u.reshape(-1).topk(256).indices; top_m = mag.reshape(-1).topk(256).indices; ov = float(torch.isin(top_u, top_m).float().mean())
    out[b] = dict(usage=u.cpu(), magnitude=mag.cpu(), activity=act.cpu(), cos_end=cos.cpu())
    res["by_block"][b] = dict(n_rows=int(u.numel()), n_used=n_used, share_of_words_that_are_mlp_rows=share_words_mlp, usage_threshold_top256=float(u.reshape(-1).topk(256).values.min()), overlap_top256_usage_with_top256_magnitude=ov)
    log(f"{name}{' ' + rev if rev else ''} block {b}: {n_used} of {u.numel()} rows used as words (MLP rows are {share_words_mlp:.2f} of the words); top-256 usage threshold {res['by_block'][b]['usage_threshold_top256']:.0f}; overlap of the 256 most used with the 256 largest {ov:.2f}")
    del Au; torch.cuda.empty_cache()
torch.save(out, f"{CDIR}/{rev or 'main'}.pt")
summ = f"{name}{' ' + rev if rev else ''}: " + " | ".join(f"block {b}: {o['n_used']} rows used, MLP share of words {o['share_of_words_that_are_mlp_rows']:.2f}, top-256 usage threshold {o['usage_threshold_top256']:.0f}, overlap with the 256 largest {o['overlap_top256_usage_with_top256_magnitude']:.2f}" for b, o in res["by_block"].items())
log(summ); record(f"e515_tracks_{name}{'_' + rev if rev else ''}", res, summ)
