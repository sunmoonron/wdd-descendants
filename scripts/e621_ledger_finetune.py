"""e621 (session 118): self-describing by construction. Pythia-160m is fine-tuned on Pile windows with the next-token loss
plus a ledger penalty, the effective number of MLP writes at the middle block (the sum of absolute write coefficients
squared over the sum of their squares, averaged over positions; the write coefficient is the activation times the row's
norm, both live), against a control fine-tune with no penalty at the same learning rate and steps. Before and after:
the effective number of writes, the validation loss, and e388's faithfulness at the middle block (the next-token loss
recovered when the state is replaced by its k-word reconstruction over the model's own MLP rows of the blocks up to it,
against the rotated copy of the same rows; recovered = (mean-ablated loss minus reconstruction loss) over (mean-ablated
loss minus clean loss)). Pre-registered in e619_prereg.json: F1 (0.5) the penalty lowers the effective number of writes
by 30% or more at a validation-loss cost of 0.05 nats or less over the control; F2 (0.5) the native loss recovered at
k = 8 rises by 0.1 or more over the control while the rotated dictionary's does not; F3 (0.4) the gain persists at k = 16
and 32. Arguments: model lambda [--smoke]; lambda 0 is the control."""
import sys, os, time, copy, math, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
from ma_common import Stop
from datasets import load_dataset
import wdd_common
from wdd_common import omp, refit, rotate
name, LAM = sys.argv[1], float(sys.argv[2]); SMOKE = "--smoke" in sys.argv; t0 = time.time(); B = MID[name]; T = 256; STEPS = 6 if SMOKE else 1500; NTR, NVA = (32, 8) if SMOKE else (2400, 24); LR = 1e-5; KS = (8, 16, 32); torch.set_grad_enabled(False)
model, tok, fam = load_model(name); arch = Arch(model, fam); DFF = arch.DFF; D = arch.D; assert fam == "neox"
ds = load_dataset("NeelNanda/pile-10k", split="train")
def windows(nwin, skipdocs=0):
    wins, buf = [], []
    for k, ex in enumerate(ds):
        if k < skipdocs: continue
        buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
        while len(buf) >= T + 1 and len(wins) < nwin: wins.append(buf[:T + 1]); buf = buf[T + 1:]
        if len(wins) >= nwin: break
    return torch.tensor(wins, device=DEV)
tr = windows(NTR); va = windows(NVA, skipdocs=6000); log(f"{name} lambda {LAM}: {tr.shape[0]} training windows, {va.shape[0]} validation windows, block {B} ({time.time() - t0:.0f}s)")
def ce(lg, x): return torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
def val_loss(m):
    tot = 0.0
    for s0 in range(0, va.shape[0], 8): x = va[s0:s0 + 8]; tot += float(ce(m(x).logits.float(), x)) * x.shape[0]
    return tot / va.shape[0]
def down(m): return Arch(m, fam).layers[B].mlp.dense_4h_to_h
def neff_penalty(m, x):
    """the effective number of writes at block B per position, with the activations and the row norms live"""
    cap = {}
    def hk(mod, inp): cap["a"] = inp[0]
    h = down(m).register_forward_pre_hook(hk)
    try: lg = m(x).logits.float()
    finally: h.remove()
    c = cap["a"][:, 1:, :].float() * down(m).weight.norm(dim=0)[None, None, :]; neff = c.abs().sum(-1).pow(2) / c.pow(2).sum(-1).clamp_min(1e-9)
    return lg, neff.mean()
def neff_of(m):
    tot = 0.0
    for s0 in range(0, va.shape[0], 8): x = va[s0:s0 + 8]; _, ne = neff_penalty(m, x); tot += float(ne) * x.shape[0]
    return tot / va.shape[0]
def states_at(m, ids):
    out = {}
    h = Arch(m, fam).layers[B].register_forward_hook(lambda mod, i, o: out.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    try: lg = m(ids).logits.float()
    finally: h.remove()
    return out["x"], lg
def loss_with(m, ids, Xnew):
    def hk(mod, i, o):
        x = o[0] if isinstance(o, tuple) else o; y = x.clone(); y[:, 1:] = Xnew.to(x.dtype).view(x.shape[0], x.shape[1] - 1, -1)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = Arch(m, fam).layers[B].register_forward_hook(hk)
    try: return float(ce(m(ids).logits.float(), ids))
    finally: h.remove()
def faithfulness(m):
    """e388's loss recovered at block B by k own MLP rows (blocks up to B), against the rotated rows; on the validation windows"""
    A, _ = rows_of(Arch(m, fam), B); Ar = rotate(A); Xe, lg = states_at(m, va); Lc = float(ce(lg, va)); Xp = Xe[:, 1:].reshape(-1, D); mu = Xp.mean(0); Xc = Xp - mu[None]
    Lm = loss_with(m, va, mu[None].expand(Xp.shape[0], -1)); out = dict(clean=Lc, mean_ablated=Lm, n_atoms=int(A.shape[0]))
    for nm, Dct in (("native", A), ("rotated", Ar)):
        sel, _, _ = omp(Xc, Dct, max(KS), batch=256, record_err=False)
        for k in KS:
            cof, err = refit(Xc, Dct, sel[:, :k]); Xh = mu[None] + torch.einsum("nk,nkd->nd", cof, Dct[sel[:, :k]]); Ls = loss_with(m, va, Xh); out[f"{nm}_{k}"] = dict(fvu=float(err.sum() / Xc.pow(2).sum()), loss=Ls, recovered=(Lm - Ls) / max(Lm - Lc, 1e-9))
        del sel
    return out
f0 = faithfulness(model); n0 = neff_of(model); v0 = val_loss(model); log(f"before: val {v0:.3f}, N_eff {n0:.0f}, recovered native/rotated " + ", ".join(f"k{k} {f0[f'native_{k}']['recovered']:.2f}/{f0[f'rotated_{k}']['recovered']:.2f}" for k in KS) + f" ({time.time() - t0:.0f}s)")
m = model
for p_ in m.parameters(): p_.requires_grad_(True)
m.train(); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(0); trace = []
with torch.enable_grad():
    for s in range(STEPS):
        x = tr[torch.randint(0, tr.shape[0], (8,), generator=g)]; lg, ne = neff_penalty(m, x); loss_ce = ce(lg, x); loss = loss_ce + LAM * ne / 1000.0
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % (2 if SMOKE else 100) == 0: trace.append(dict(step=s + 1, ce=float(loss_ce), neff=float(ne))); log(f"step {s + 1}: ce {float(loss_ce):.3f}, N_eff {float(ne):.0f} ({time.time() - t0:.0f}s)")
m.eval()
for p_ in m.parameters(): p_.requires_grad_(False)
f1 = faithfulness(m); n1 = neff_of(m); v1 = val_loss(m)
res = dict(model=name, lam=LAM, steps=STEPS, block=B, before=dict(val=v0, neff=n0, faith=f0), after=dict(val=v1, neff=n1, faith=f1), trace=trace)
summ = (f"ledger fine-tune ({name}, lambda {LAM}, {STEPS} steps): val {v0:.3f} -> {v1:.3f}, N_eff {n0:.0f} -> {n1:.0f}; loss recovered native/rotated before " + ", ".join(f"k{k} {f0[f'native_{k}']['recovered']:.2f}/{f0[f'rotated_{k}']['recovered']:.2f}" for k in KS) + "; after " + ", ".join(f"k{k} {f1[f'native_{k}']['recovered']:.2f}/{f1[f'rotated_{k}']['recovered']:.2f}" for k in KS))
log(summ); record(f"e621_ledger_{name}_lam{LAM:g}" + ("_smoke" if SMOKE else ""), res, summ)
