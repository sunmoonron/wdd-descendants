"""e1: clean runs and caches. Ouro for 8 loops (4 trained, 4 extrapolated): the residual state before layers 0, 12 and
after layer 23 (pre-norm end of the loop) of every loop on 32 evaluation sequences, the state before layer 12 on 16
fitting sequences, the per-token cross-entropy of every loop's exit, the exit gate, the between-loop RMS scale, and the
exit-state change between loops. SmolLM2: the same at layers 0, 4, ..., 24."""
import sys; sys.path.insert(0, "/data/loopwdd/code")
from lw import *
kind = sys.argv[1]
m = LM(kind); loops = 8 if kind == "ouro" else 1
POS = (0, 12, 24) if kind == "ouro" else (0, 4, 8, 12, 16, 20, 24)
ev = corpus(m.tok, 32, T=256, start=0); fit = corpus(m.tok, 16, T=256, start=40)
out = dict(ids=ev, fit_ids=fit, pos=POS, loops=loops)
def collect(ids, pos):
    S = {(t, l): torch.empty(ids.shape[0], ids.shape[1], m.d) for t in range(loops) for l in pos}
    ce = torch.empty(loops, ids.shape[0], ids.shape[1] - 1); gate = torch.empty(loops, ids.shape[0], ids.shape[1])
    scale = torch.empty(loops, ids.shape[0], ids.shape[1]); dh = torch.empty(loops, ids.shape[0], ids.shape[1])
    for b in range(0, ids.shape[0], 4):
        x = ids[b:b + 4].to(DEVM)
        def cb(t, l, s):
            if l in pos: S[(t, l)][b:b + 4] = s.cpu()
            if l == m.L: scale[t, b:b + 4] = torch.rsqrt(s.pow(2).mean(-1) + m.eps).cpu()
        with torch.no_grad():
            hs = m.run(x, loops=loops, cb=cb)
            for t, h in enumerate(hs):
                ce[t, b:b + 4] = m.ce(h, x).cpu()
                if kind == "ouro": gate[t, b:b + 4] = m.gate(h).cpu()
                if t: dh[t, b:b + 4] = ((h - hs[t - 1]).norm(dim=-1) / h.norm(dim=-1)).cpu()
    return S, ce, gate, scale, dh
with torch.no_grad():
    S, ce, gate, scale, dh = collect(ev, POS)
    out.update(states=S, ce=ce, gate=gate, scale=scale, dh=dh)
    Sf, cef, _, _, _ = collect(fit, (12,))
    out.update(fit_states=Sf, fit_ce=cef)
torch.save(out, f"{ROOT}/cache/{kind}_e1.pt")
summ = dict(ce_by_loop=ce.mean((1, 2)).tolist(), gate_by_loop=gate[:, :, 1:].mean((1, 2)).tolist(),
            dh_by_loop=dh[:, :, 1:].median(-1).values.median(-1).values.tolist() if loops > 1 else None,
            state_norm={f"{t},{l}": S[(t, l)][:, 1:].norm(dim=-1).median().item() for (t, l) in S})
log(kind, summ); jdump(summ, f"{ROOT}/results/e1_{kind}.json")
