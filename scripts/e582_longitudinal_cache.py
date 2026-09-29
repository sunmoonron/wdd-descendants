"""e582 (session 106): the longitudinal cache. Pythia-410m at steps 1000-16000 (every thousand) on 24 Pile sequences of
512 tokens (12,264 positions): the block-12 states (fp16, all positions), the sink mask, the MLP rows of blocks 0-12
(unit, fp16) with their norms. Twenty-four sequences rather than eight so that the classes are larger and a small
sparse autoencoder can be trained per checkpoint (e583). About twenty seconds a checkpoint."""
from s101_common import *
t0 = time.time(); B = 12; CD = "/workspace/wdd/cache/e582_pythia410"; os.makedirs(CD, exist_ok=True); ids = eval_ids("pythia410")[:24, :512].to(DEV)
for n in list(range(1000, 16001, 1000)):
    out = f"{CD}/step{n}.pt"
    if os.path.exists(out): continue
    model, _, fam = load_model("pythia410", revision=f"step{n}"); arch = Arch(model, fam); X = block_states(model, arch, ids, [B], chunk=4)[B].reshape(-1, arch.D); keep = ~sinkmask(X); A, norms = rows_of(arch, B)
    torch.save(dict(step=n, X=X.half().cpu(), keep=keep.cpu(), rows=A.half().cpu(), norms=norms.cpu(), D=arch.D, DFF=arch.DFF, B=B, n_seq=24, T=512), out); del model; torch.cuda.empty_cache(); log(f"step {n}: {int(keep.sum())} kept of {keep.numel()} | {time.time() - t0:.0f}s")
summ = f"cached 16 checkpoints of Pythia-410m block 12 on 24 x 512 tokens | {time.time() - t0:.0f}s"; log(summ); record("e582_longitudinal_cache", dict(steps=list(range(1000, 16001, 1000)), n_positions=int(keep.numel())), summ)
