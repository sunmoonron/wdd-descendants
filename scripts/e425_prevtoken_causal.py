"""e425: is the context that native words surface used? Induction needs, at each position of a repeated sequence, the
identity of the previous token at the key positions it attends back to. Repeated random tokens (8 sequences, a random
half of 128 tokens then the same half), at the middle block: every first-copy position (the keys) is described by 16
native words; the word whose own lens readout ranks the previous token in its top 5 is the "previous-token word". Three
forward passes: clean; with the previous-token word removed from every key position that has one (x - c a, all at
once); with a random other of the 16 words removed at the same positions instead. Scored: the induction loss (mean
next-token loss over the second copy, where the prediction is copying) and its increase under each removal.
Pre-registered: removing the previous-token words raises the induction loss more than removing random other words,
in all five models."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lr_common import *
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); where = sys.argv[2] if len(sys.argv) > 2 else "quarter"
L = arch.NB // 4 if where == "quarter" else arch.NB // 2; half = 128
E = eval_ids(name); pool = torch.unique(E.flatten()); g = torch.Generator().manual_seed(0)
r1 = pool[torch.randint(0, len(pool), (8, half), generator=g)]; bos = tok.bos_token_id if tok.bos_token_id is not None else int(E[0, 0])
ids = torch.cat([torch.full((8, 1), bos), r1, r1], 1).to(DEV); T = ids.shape[1]
rd = Reader(model, arch, L); ref = E[:4].to(DEV)
st_ref, _, _, _ = capture(model, arch, ref, [L], 0); mu = st_ref[L][:, 1:].reshape(-1, arch.D).mean(0)
st, _, _, lg = capture(model, arch, ids, [L], 0)
pos = torch.arange(2, half + 1, device=DEV)                      # first-copy (key) positions: their previous token is what induction matches
X = st[L][:, pos].reshape(-1, arch.D); rms = X.pow(2).mean(-1).sqrt(); P = pos.numel()
parts, sel, cof = rd.native_parts(X - mu, L)                     # [8*P, 16, D]
prev = ids[:, pos - 1].reshape(-1)
top5 = torch.stack([rd.lens(parts[:, i], rms).topk(5, dim=-1).indices for i in range(parts.shape[1])], 1)   # [N, 16, 5]
hit = (top5 == prev[:, None, None]).any(-1)                       # [N, 16]
has = hit.any(-1); iw = torch.where(has, (hit.float() * cof.abs()).argmax(-1), torch.zeros_like(prev))
gen = torch.Generator(device=DEV).manual_seed(1); rnd = torch.randint(0, 16, (X.shape[0],), device=DEV, generator=gen)
rnd = torch.where(rnd == iw, (rnd + 1) % 16, rnd)                 # a different word from the previous-token word
ar = torch.arange(X.shape[0], device=DEV)
def run(remove_idx):
    Xn = X.clone()
    if remove_idx is not None: Xn[has] = X[has] - parts[ar[has], remove_idx[has]]
    newst = Xn.view(8, P, arch.D)
    def hk(m, i, o):
        xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, pos] = newst.to(xo.dtype)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): lgt = model(ids).logits.float()
    finally: h.remove()
    q = torch.arange(half + 1, T - 1, device=DEV)                                                   # second-copy predictions (copying)
    lp = torch.log_softmax(lgt[:, q], -1)
    return -lp.gather(2, ids[:, q + 1][..., None])[..., 0].mean().item()
clean = run(None); a = run(iw); b = run(rnd)
res = dict(model=name, level=L, n_positions=int(X.shape[0]), share_with_prev_word=has.float().mean().item(), induction_loss_clean=clean,
           remove_prev_word=a, remove_random_word=b, delta_prev=a - clean, delta_random=b - clean)
summ = (f"{name} L{L}: previous-token word found at {res['share_with_prev_word']:.2f} of key positions; induction loss clean {clean:.3f}, "
        f"removing those words {a:.3f} (+{a - clean:.3f}), removing random other words {b:.3f} (+{b - clean:.3f})")
log(summ); record(f"e425_prevcausal_{name}_{where}", res, summ)
