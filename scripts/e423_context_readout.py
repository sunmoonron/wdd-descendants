"""e423: does the state carry its context and its future in readable form? Natural text, ground truth free: at position t
the state after block L is read by four readers and scored on the identity of the previous token (t-1), the one before
(t-2), the current token (t, a sanity check: its embedding is a native word) and the token after next (t+2, not the
model's prediction but the one after it). Readers: the logit lens (the whole state); the native-word lens (16 OMP words
of the model's own vocabulary on the centred state, each read by the lens, max-pooled); the 16 largest actual writes at
that position (MLP neuron writes, head outputs, embeddings), read and pooled the same way; and a rotated-vocabulary
control. A frequency baseline (the target among the N most frequent tokens of the text) calibrates each target.
Five models, blocks at a quarter, half and three quarters of the depth, 3 sequences x 480 positions.
Pre-registered: at middle depth the native words surface t-1 (recall at 10) more often than the logit lens and than the
actual writes, in all five models; the rotated control does not. No prediction for t+2."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lr_common import *
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam)
Ls = [arch.NB // 4, arch.NB // 2, (3 * arch.NB) // 4]; maxL = max(Ls); E = eval_ids(name)
ev, ref = E[:3].to(DEV), E[3:7].to(DEV); pos = torch.arange(24, 504, device=DEV)            # positions with t-2 and t+2 inside
rd = Reader(model, arch, maxL); act = Actual(arch, maxL)
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
st_ref, _, _, _ = capture(model, arch, ref, Ls, 0); mu = {L: st_ref[L][:, 1:].reshape(-1, arch.D).mean(0) for L in Ls}
st, H, Z, _ = capture(model, arch, ev, Ls, maxL)
B = ev.shape[0]; bi = torch.arange(B, device=DEV)[:, None].expand(-1, len(pos)).reshape(-1); pi = pos[None].expand(B, -1).reshape(-1)
tgt = dict(prev=ev[bi, pi - 1], prev2=ev[bi, pi - 2], cur=ev[bi, pi], next2=ev[bi, pi + 2])
freq = torch.bincount(E[:16].flatten().to(DEV), minlength=rd.WU.shape[0]); fr = (-freq.float()).argsort().argsort() + 1   # frequency rank of every token
res = dict(model=name, layers=Ls, n=int(bi.numel()), freq_baseline={t: {f"@{n}": (fr[v] <= n).float().mean().item() for n in (10, 50)} for t, v in tgt.items()}, readers={})
for L in Ls:
    X = st[L][bi, pi]; rms = X.pow(2).mean(-1).sqrt(); Xc = X - mu[L]
    parts_n, sel, cof = rd.native_parts(Xc, L); parts_r, _, _ = rd.native_parts(Xc, L, R=R)
    parts_a = act.top({b: H[b][bi, pi] for b in range(L + 1)}, {b: Z[b][bi, pi] for b in range(L + 1)}, ev[bi, pi], pi, L)
    S = dict(lens=None, native16=parts_n, actual16=parts_a, rotated16=parts_r)
    for r_name, parts in S.items():
        out = {}
        for s0 in range(0, X.shape[0], 256):
            sc = rd.lens(X[s0:s0 + 256], rms[s0:s0 + 256]) if parts is None else rd.pooled(parts[s0:s0 + 256], rms[s0:s0 + 256])
            for t, v in tgt.items(): out.setdefault(t, []).append(rank_of(sc, v[s0:s0 + 256]))
        out = {t: torch.cat(v) for t, v in out.items()}
        res["readers"][f"{r_name}@{L}"] = {t: {f"@{n}": (rk <= n).float().mean().item() for n in (1, 10, 50)} for t, rk in out.items()}
    q = lambda r_, t, n="@10": res["readers"][f"{r_}@{L}"][t][n]
    log(f"{name} L{L}: recall@10 prev lens {q('lens', 'prev'):.2f} native {q('native16', 'prev'):.2f} actual {q('actual16', 'prev'):.2f} rot {q('rotated16', 'prev'):.2f} | "
        f"prev2 lens {q('lens', 'prev2'):.2f} native {q('native16', 'prev2'):.2f} actual {q('actual16', 'prev2'):.2f} | next2 lens {q('lens', 'next2'):.2f} native {q('native16', 'next2'):.2f} actual {q('actual16', 'next2'):.2f} | "
        f"cur lens {q('lens', 'cur'):.2f} native {q('native16', 'cur'):.2f}")
    del parts_n, parts_r, parts_a
fb = res["freq_baseline"]; M = Ls[1]; q = lambda r_, t: res["readers"][f"{r_}@{M}"][t]["@10"]
summ = (f"{name} middle block {M}, recall@10: prev lens {q('lens', 'prev'):.2f} native {q('native16', 'prev'):.2f} actual {q('actual16', 'prev'):.2f} rotated {q('rotated16', 'prev'):.2f} (freq {fb['prev']['@10']:.2f}) | "
        f"prev2 {q('lens', 'prev2'):.2f}/{q('native16', 'prev2'):.2f}/{q('actual16', 'prev2'):.2f} | next2 {q('lens', 'next2'):.2f}/{q('native16', 'next2'):.2f}/{q('actual16', 'next2'):.2f} (freq {fb['next2']['@10']:.2f})")
log(summ); record(f"e423_context_{name}", res, summ)
