"""The exact ledger of a neuron's inputs and the K-input simulation (the record's e685 to e706 machinery).
A neuron's pre-activation is split exactly into one contribution per upstream unit: the MLP neurons of the blocks before it,
the attention outputs that feed its block, and the embedding, each folded through the block's normalisation. A subset of
inputs is scored by replacing the neuron's activation with the activation that subset alone would give, and measuring the
downstream KL against the unmodified model, relative to mean-ablating the neuron."""
import random, torch
from .model import DEV, Capture, run_base, logprobs_at, kl, read_w, read_bias, read_tilde, read_const, write_w, write_bias, write_mod, ln_params, ln_apply, n_att_feeding, med
F_ = torch.nn.functional
class Ledger:
    def __init__(self, M, block, ids, T=256, batch=8, need_lp=True):
        """M: a loaded model (wdd.model.load); block: the neuron's block; ids: evaluation windows [nwin, T]"""
        self.M, self.B, self.ids, self.T = M, block, ids, T; NB, D, DFF = M["NB"], M["D"], M["DFF"]; B = block; self.NATT = n_att_feeding(M, B); self.NMLP = B * DFF; self.NU = self.NMLP + self.NATT + 1; self.NWIN = ids.shape[0]; self.N = self.NWIN * (T - 1)
        UP, AT, EM, XM, AB, LP = [], [], [], [], [], []
        with Capture(M, list(range(B + 1)), ["act", "att", "xin", "emb"]) as C:
            for s0 in range(0, self.NWIN, batch):
                o = M["base"](input_ids=ids[s0:s0 + batch], use_cache=False)
                if need_lp: LP.append(logprobs_at(M, o.last_hidden_state[:, :-1]))
                UP.append(torch.cat([C.cap[("act", b)][:, 1:].half() for b in range(B)], -1).reshape(-1, self.NMLP) if B else torch.zeros(batch * (T - 1), 0, device=DEV, dtype=torch.half)); AT.append(torch.stack([C.cap[("att", b)][:, 1:].half() for b in range(self.NATT)], 2).reshape(-1, self.NATT, D))
                EM.append(C.cap[("emb", 0)][:, 1:].half().reshape(-1, D)); XM.append(C.cap[("xin", B)][:, 1:].float().reshape(-1, D)); AB.append(C.cap[("act", B)][:, 1:].half().reshape(-1, DFF))
        self.UP, self.AT, self.EM, self.XM, self.AB = torch.cat(UP), torch.cat(AT), torch.cat(EM), torch.cat(XM), torch.cat(AB); self.LP = torch.cat(LP).reshape(self.N, -1) if need_lp else None
        self.Rd = read_w(M, B); self.RB = read_bias(M, B); self.RT = {k: read_tilde(M, B, self.Rd[k]) for k in self.Rd}; self.RC = {k: read_const(M, B, self.Rd[k], self.RB[k]) for k in self.Rd}; self.key0 = "h" if "h" in self.Rd else "g"
        self.WUP = torch.cat([write_w(M, b).T for b in range(B)]) if B else torch.zeros(0, D, device=DEV); self.BIAS = sum((write_bias(M, b) for b in range(B)), torch.zeros(D, device=DEV)); P_ = ln_params(M, B)
        self.SIG = (self.XM.var(-1, unbiased=False) + P_["eps"]).sqrt() if P_["centre"] else (self.XM.pow(2).mean(-1) + P_["eps"]).sqrt()
    def contributions(self, k, P, key=None):
        """the exact split of neuron k's pre-activation (read key: h, or g and u for gated blocks) at positions P [n] into [n, NU] contributions, and the constant [n]"""
        key = key or self.key0; r = self.RT[key][k]; V = self.WUP @ r; s = self.SIG[P]
        C = torch.cat([self.UP[P].float() * V[None] / s[:, None], (self.AT[P].float() @ r) / s[:, None], ((self.EM[P].float() @ r) / s)[:, None]], 1); return C, (self.BIAS @ r) / s + self.RC[key][k]
    def check(self, k, P=None, key=None):
        """the split's relative error against the model's own pre-activation at positions P (median)"""
        P = torch.arange(min(2048, self.N), device=DEV) if P is None else P; C, c0 = self.contributions(k, P, key); key = key or self.key0; pre = ln_apply(self.M, self.B, self.XM[P])[0] @ self.Rd[key][k] + self.RB[key][k]
        return float(((C.sum(1) + c0 - pre).abs() / pre.abs().clamp_min(1e-6)).median())
    def net_ledger(self, k, P, key=None, kw=16):
        """at positions P: the sixteen largest inputs' share of the net drive (signed) and the crowd's net sum over theirs (signed, negative is a brake)"""
        C, c0 = self.contributions(k, P, key); tp = C.abs().topk(kw, dim=1); t16 = C.gather(1, tp.indices).sum(1); tot = C.sum(1)
        return dict(wires_net_share=float((t16 / tot.abs().clamp_min(1e-9) * torch.sign(tot)).median()), crowd_over_wires=float(((tot - t16) * torch.sign(t16) / t16.abs().clamp_min(1e-9)).median()), n_eff=float((C.abs().sum(1).pow(2) / C.pow(2).sum(1).clamp_min(1e-12)).median()))
    def act_of(self, PRE): return self.M["act"](PRE["h"]) if self.M["fam"] != "gated" else F_.silu(PRE["g"]) * PRE["u"]
    def kl_with(self, k, repl, strong, batch=8):
        """replace neuron k's activation by repl [N] at every position; the mean KL against the unmodified model at the strong positions and over all positions"""
        assert self.LP is not None, "build the ledger with need_lp=True"
        M, B, T, ids = self.M, self.B, self.T, self.ids; rows_ = torch.arange(self.N, device=DEV).reshape(self.NWIN, T - 1); tot_s, tot_a = [], []; strong = strong[(strong % (T - 1)) != (T - 2)] + 1
        for s0 in range(0, self.NWIN, batch):
            x = ids[s0:s0 + batch]; rr = rows_[s0:s0 + batch]
            def hook(m, a):
                h = a[0].clone(); h[:, 1:, k] = repl[rr].to(h.dtype); return (h,)
            hd = write_mod(M, B).register_forward_pre_hook(hook)
            try: o = M["base"](input_ids=x, use_cache=False)
            finally: hd.remove()
            lp = logprobs_at(M, o.last_hidden_state[:, :-1]).reshape(-1, self.LP.shape[-1]); k_ = kl(self.LP[rr.reshape(-1)].float(), lp); tot_a.append(k_); tot_s.append(k_[torch.isin(rr.reshape(-1), strong)])
        return float(torch.cat(tot_s).mean()), float(torch.cat(tot_a).mean())
    def simulate(self, k, kw=16, nstrong=160, extra=None):
        """the K-input simulation of neuron k: ratios (KL over the ablation's) at the firings and over all positions for the sixteen largest inputs by magnitude (abs16), every input (all), and any extra named pre-activation terms added to abs16 (for instance a field summary, see wdd.fields)"""
        a = self.AB[:, k].float(); strong = a.topk(nstrong).indices; mean_act = torch.full((self.N,), float(a.mean()), device=DEV); kl_abl = self.kl_with(k, mean_act, strong); PRE = {}
        for key in self.Rd:
            c0 = torch.empty(self.N, device=DEV); tot = torch.empty(self.N, device=DEV); ts = torch.empty(self.N, device=DEV)
            for s in range(0, self.N, 2048):
                Pp = torch.arange(s, min(s + 2048, self.N), device=DEV); c, cc = self.contributions(k, Pp, key); c0[Pp] = cc; tot[Pp] = c.sum(1); top = c.abs().topk(kw, dim=1).indices; ts[Pp] = c.gather(1, top).sum(1)
            PRE[key] = {"abs16": ts + c0, "all": tot + c0}
            for nm, fn in (extra or {}).items(): PRE[key][nm] = ts + c0 + fn(k, key)
        out = {"kl_ablate_strong": kl_abl[0], "kl_ablate_all": kl_abl[1]}
        for cnd in PRE[self.key0]:
            v = self.kl_with(k, self.act_of({key: PRE[key][cnd] for key in self.Rd}), strong); out[cnd] = dict(ratio_strong=v[0] / max(kl_abl[0], 1e-12), ratio_all=v[1] / max(kl_abl[1], 1e-12))
        return out
    def active_neurons(self, n, seed=685):
        """a sample of n active neurons of the block, as the record chose them (maximum over the floor and a hundredth-largest activation within a quarter of it)"""
        ab = self.AB.float(); mx = ab.max(0).values; a100 = ab.topk(100, dim=0).values[-1]; floor_act = 1.0 if self.M["fam"] != "gated" else float(mx.quantile(0.1)); cand = torch.nonzero((mx > floor_act) & (a100 >= 0.25 * mx))[:, 0].tolist()
        return sorted(random.Random(seed + self.B).sample(cand, min(n, len(cand))))
