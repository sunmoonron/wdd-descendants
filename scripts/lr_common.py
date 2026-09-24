"""Shared helpers for the latent-readout chain (e423 onward): the claim under test is that WDD, by re-describing a state in
the model's own write directions, reads content that is present in the state but hidden both in its whole-state
projection (the logit lens) and in the model's largest actual writes (the value-vector / sub-update reading).
For the five original models (fp32): the native vocabulary up to a block (a prefix of the block-ordered dictionary),
the logit lens with the final norm's centring for LayerNorm models, per-part readouts max-pooled over parts, and the
actual component writes at chosen positions (every MLP neuron's activation times its write row, every head's output,
the token and position embeddings)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *

def eval_ids(name):
    return torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]["eval_ids"]

class Reader:
    def __init__(self, model, arch, maxL):
        self.model, self.arch = model, arch
        self.A, lab = build_dictionary(arch, blocks=list(range(maxL + 1)))
        self.blk, self.typ = lab["block"].to(DEV), lab["type"].to(DEV)
        self.ends = {L: int((self.blk <= L).sum()) for L in range(maxL + 1)}     # block-ordered: a prefix per block
        fl = model.transformer.ln_f if arch.fam == "gpt2" else (model.gpt_neox.final_layer_norm if arch.fam == "neox" else model.model.norm)
        w = getattr(fl, "weight", None); self.gain = w.detach().float() if w is not None else torch.ones(arch.D, device=DEV)   # OLMo: non-parametric norm
        self.centre = "rms" not in type(fl).__name__.lower()
        self.WU = model.get_output_embeddings().weight.detach().float()
    def lens(self, V, rms):
        """token scores of residual vectors V [..., D], each divided by its state's rms [...] (the final norm's scale)"""
        if self.centre: V = V - V.mean(-1, keepdim=True)
        return ((V * self.gain) / rms[..., None]) @ self.WU.T
    def pooled(self, parts, rms, chunk=128):
        """parts [P, k, D]: token score = the largest score any single part gives the token; returns [P, V]"""
        out = []
        for s in range(0, parts.shape[0], chunk):
            p = parts[s:s + chunk]; r = rms[s:s + chunk]
            out.append(self.lens(p, r[:, None].expand(-1, p.shape[1])).amax(1))
        return torch.cat(out)
    def native_parts(self, Xc, L, k=16, R=None):
        """OMP over the native vocabulary up to block L; with R, the vocabulary rotated (emulated by rotating the targets)"""
        An = self.A[:self.ends[L]]; Xt = Xc if R is None else Xc @ R.T
        sel, _, _ = omp(Xt, An, k, batch=256, record_err=False); cof, _ = refit(Xt, An, sel)
        W = An[sel] if R is None else An[sel] @ R
        return W * cof[..., None], sel, cof

def capture(model, arch, ids, L_list, maxL):
    """one forward pass: states after each block in L_list, MLP hidden activations and attention head inputs for blocks
    0..maxL, all positions (fp32, batch of sequences)"""
    st, H, Z = {}, {}, {}
    hs = [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: st.__setitem__(l_, (o[0] if isinstance(o, tuple) else o).detach().float()))(l)) for l in L_list]
    for b in range(maxL + 1):
        hs.append(arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: H.__setitem__(b_, a[0].detach().float()))(b)))
        hs.append(arch.attn_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: Z.__setitem__(b_, a[0].detach().float()))(b)))
    try:
        with torch.no_grad(): lg = model(ids).logits.float()
    finally: [h.remove() for h in hs]
    return st, H, Z, lg

class Actual:
    """the model's actual component writes at chosen positions"""
    def __init__(self, arch, maxL):
        self.arch = arch; self.W = [arch.wdir(b) for b in range(maxL + 1)]                        # [DFF, D] each
        self.cn = [w.norm(dim=-1) for w in self.W]; self.Wo = [arch.wo(b).view(arch.NH, arch.HD, arch.D) for b in range(maxL + 1)]
        self.emb = [e.detach().float() for e in arch.emb]
    def top(self, H, Z, ids_flat, pos_flat, L, k=16, cand=64):
        """H[b], Z[b]: [P, DFF], [P, NH*HD] at the chosen positions; returns the k largest actual writes [P, k, D]"""
        P = ids_flat.shape[0]; DFF = self.W[0].shape[0]
        mags = torch.cat([H[b].abs() * self.cn[b] for b in range(L + 1)], 1); top = mags.topk(cand, dim=1).indices
        Wall = torch.cat(self.W[:L + 1]); Hall = torch.cat([H[b] for b in range(L + 1)], 1)
        neur = Wall[top] * Hall.gather(1, top)[..., None]                                              # [P, cand, D]
        heads = torch.cat([torch.einsum("phd,hdD->phD", Z[b].view(P, self.arch.NH, self.arch.HD), self.Wo[b]) for b in range(L + 1)], 1)
        embs = [self.emb[0][ids_flat][:, None]] + ([self.emb[1][pos_flat][:, None]] if len(self.emb) > 1 else [])
        allv = torch.cat([neur, heads] + embs, 1); idx = allv.norm(dim=-1).topk(k, dim=1).indices
        return allv.gather(1, idx[..., None].expand(-1, -1, allv.shape[-1]))

def rank_of(scores, target):
    """rank (1 = top) of target token ids [P] in scores [P, V]"""
    return (scores > scores.gather(1, target[:, None])).sum(1) + 1
