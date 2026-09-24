"""Shared helpers for e433 onward (after e432's anatomy of the huge directions):
- block outputs at positions 1: for many sequences;
- the sink mask (state norm above 10x the median, phase 1's convention);
- a splice at block L that returns per-position loss, entropy and the change of the centred logits against the clean
  run, optionally tracking later blocks;
- an early-exit pass that returns block b's output after a change at block L without running the rest of the model."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *

def out_of(o): return o[0] if isinstance(o, tuple) else o

class Stop(Exception): pass

def block_states(model, arch, ids, blocks, chunk=8):
    """{block: [B, T-1, D]} outputs of the given blocks at positions 1: (the pass stops after the last of them)"""
    out = {b: [] for b in blocks}; last = max(blocks)
    for s0 in range(0, ids.shape[0], chunk):
        cap = {}
        def mk(b):
            def hk(m, i, o):
                cap[b] = out_of(o)[:, 1:].detach().float()
                if b == last: raise Stop
            return hk
        hs = [arch.layers[b].register_forward_hook(mk(b)) for b in blocks]
        try:
            with torch.no_grad(): model(ids[s0:s0 + chunk])
        except Stop: pass
        finally: [h.remove() for h in hs]
        for b in blocks: out[b].append(cap[b])
    return {b: torch.cat(v) for b, v in out.items()}

def sinkmask(x, f=10.0):
    n = x.norm(dim=-1); return n > f * n.median()

def replace_hook(Xs):
    def hk(m, i, o):
        xo = out_of(o); y = xo.clone(); y[:, 1:] = Xs.to(xo.dtype)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    return hk

def early(model, arch, ids, L, Xnew, b):
    """block b's output at positions 1: when block L's output at positions 1: is replaced by Xnew (b > L); stops there"""
    cap = {}
    def st(m, i, o): cap["x"] = out_of(o)[:, 1:].detach().float(); raise Stop
    hs = [arch.layers[L].register_forward_hook(replace_hook(Xnew)), arch.layers[b].register_forward_hook(st)]
    try:
        with torch.no_grad(): model(ids)
    except Stop: pass
    finally: [h.remove() for h in hs]
    return cap["x"]

class Splicer:
    """Replace block L's output at positions 1: of ids and score the model: per-position loss and entropy [B, T-1]
    (position j predicts token j+1), the ratio of centred-logit norms and their cosine to the clean run; optionally the
    outputs of the track blocks. The clean run is stored at construction."""
    def __init__(self, model, arch, ids, L, track=(), chunk=2):
        self.model, self.arch, self.ids, self.L, self.track, self.chunk = model, arch, ids, L, list(track), chunk
        self.B, self.T = ids.shape; self._z = {}; self.c = self.run(None, track=bool(self.track))
    def run(self, Xnew=None, track=False):
        loss, ent, lnr, lcos, tr = [], [], [], [], {b: [] for b in self.track}
        for s0 in range(0, self.B, self.chunk):
            ids = self.ids[s0:s0 + self.chunk]; hs = []; cap = {}
            if Xnew is not None: hs.append(self.arch.layers[self.L].register_forward_hook(replace_hook(Xnew[s0:s0 + self.chunk])))
            if track:
                for b in self.track: hs.append(self.arch.layers[b].register_forward_hook(lambda m, i, o, b=b: cap.__setitem__(b, out_of(o)[:, 1:].detach().float())))
            try:
                with torch.no_grad(): lg = self.model(ids).logits.float()
            finally: [h.remove() for h in hs]
            lp = torch.log_softmax(lg[:, :-1], -1); loss.append(-lp.gather(2, ids[:, 1:, None])[..., 0]); ent.append(-(lp.exp() * lp).sum(-1))
            z = lg[:, :-1] - lg[:, :-1].mean(-1, keepdim=True)
            if Xnew is None: self._z[s0] = z.to(torch.bfloat16)
            zc = self._z[s0].float(); lnr.append(z.norm(dim=-1) / zc.norm(dim=-1)); lcos.append((z * zc).sum(-1) / (z.norm(dim=-1) * zc.norm(dim=-1)).clamp_min(1e-9))
            if track:
                for b in self.track: tr[b].append(cap[b])
            del lg, lp, z, zc
        out = dict(loss=torch.cat(loss), ent=torch.cat(ent), lnr=torch.cat(lnr), lcos=torch.cat(lcos))
        if track: out["track"] = {b: torch.cat(v) for b, v in tr.items()}
        return out
    def lossmask(self, statemask):
        """per-position loss mask [B, T-1] for the positions (1..T-2) whose state is in statemask [B, T-1]"""
        m = torch.zeros(self.B, self.T - 1, dtype=torch.bool, device=statemask.device); m[:, 1:] = statemask[:, :-1]; return m
    def stats(self, r, lm):
        d = r["loss"] - self.c["loss"]
        return dict(dL=d[lm].mean().item(), dH=(r["ent"] - self.c["ent"])[lm].mean().item(), logit_norm_ratio=r["lnr"][lm].mean().item(), logit_cos=r["lcos"][lm].mean().item())

def pcs(X, n):
    """top principal directions [D, n] of the rows of X (descending), and the eigenvalues"""
    ev, U = torch.linalg.eigh(torch.cov((X - X.mean(0)).T.double(), correction=0)); return U.flip(-1)[:, :n].float().contiguous(), ev.flip(-1)

class NSLevel(Level):
    """Level with the sink positions (norm above 10x the median) kept exact: centring, descriptions and the mean
    baseline use the typical positions only (e437)"""
    def __init__(self, model, arch, ids, L):
        self.model, self.arch, self.ids, self.L, self.B, self.T, self.P = model, arch, ids, L, ids.shape[0], ids.shape[1], None
        out = {}; h = arch.layers[L].register_forward_hook(lambda m, i, o: out.__setitem__("x", out_of(o).detach().float()))
        try:
            with torch.no_grad(): self.lc = token_loss(model(ids).logits.float(), ids).view(self.B, -1)
        finally: h.remove()
        x = out["x"][:, 1:]; self.keep = ~sinkmask(x).reshape(-1); self.full = x.reshape(-1, x.shape[-1])
        self.mu = self.full[self.keep].mean(0, keepdim=True); self.Xc = self.full[self.keep] - self.mu
        self.lm = self.splice(self.mu.expand(self.Xc.shape[0], -1)); self.gap = (self.lm.mean() - self.lc.mean()).item()
    def splice(self, Xh):
        f = self.full.clone(); f[self.keep] = Xh.to(f.dtype); return Level.splice(self, f)
