"""Shared helpers for the quotient program (e317 onward): one setup object holding the model, the block-2 candidates,
their natural per-token descendants at requested levels and the logit footprints; PLS / scatter bases; kNN decoders;
subspace overlaps; data-free transport operators; family injections."""
from func_common import *
import math
def spearman(a_, b_):
    ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
def inside(A, B): return ((B.T @ A) ** 2).sum().item() / A.shape[1]
def prank(X):
    Xc = X - X.mean(0, keepdim=True); ev = torch.linalg.eigvalsh(Xc.T @ Xc).clamp_min(0); return (ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12)).item()
def pls(Fc, Y, d):
    Yc = Y - Y.mean(0, keepdim=True); return torch.linalg.svd(Fc.T @ Yc, full_matrices=False)[0][:, :d]
def scatter_basis(Fc, lab, K, d):
    cents = torch.stack([Fc[lab == k].mean(0) if (lab == k).any() else torch.zeros(Fc.shape[1], device=DEV) for k in range(K)]); w = torch.bincount(lab, minlength=K).float(); Sb = (cents * w[:, None]).T @ cents / w.sum(); return torch.linalg.eigh(Sb)[1].flip(1)[:, :d]
def logit_scores(dl, tr, n=256):
    G = dl[tr] @ dl[tr].T; evg, Vg = torch.linalg.eigh(G); Bv = dl[tr].T @ (Vg.flip(1)[:, :n] / evg.flip(0)[:n].clamp_min(1e-6).sqrt()[None]); Z = dl @ Bv; return Z - Z[tr].mean(0, keepdim=True), Bv
def knn_pred(Ptr, Pte, target_tr, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return target_tr[nn].mean(1)
def knn_cos(P, target, tr, te, k=5): return ((unit(knn_pred(P[tr], P[te], target[tr], k)) * unit(target[te])).sum(1)).median().item()
def knn_spear(P, target, tr, te, k=5): return spearman(knn_pred(P[tr], P[te], target[tr], k), target[te])
def ridge(X, Y, lam=1e-1):
    G = X.T @ X; return torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(X.shape[1], device=DEV), X.T @ Y)
class Setup:
    def __init__(self, tag, levels, NS=12, b=2, min_tokens=20, revision=None, random_init=False):
        self.tag = tag; self.c = Cache(tag); c = self.c; self.L = mid(c); self.model, self.tok, self.fam = load_model(c.name, revision=revision, random_init=random_init); self.arch = Arch(self.model, self.fam); self.NS = NS; self.ids_seq = c.s["eval_ids"][:NS].to(DEV); self.NT = NS * CTX; self.NB = self.arch.NB; self.D = self.arch.D; self.DFF = self.arch.DFF; self.b = b
        self.levels = sorted(set([lv for lv in levels if lv < self.NB])); self.led, self.tn, self.tc = dominant(self.model, self.arch, c, self.ids_seq, b); self.run = make_runner_logits(self.model, self.arch, c, self.ids_seq, self.levels, self.NT, b); self.S0 = self.run(); self.typ = typical_mask(self.S0[self.L] if self.L in self.S0 else self.S0[self.levels[-1]]); big = self.tc.abs() >= self.tc.abs().quantile(0.5); self.idx, self.lab_i, self.keep = classes(self.tn, self.typ & big, min_tokens); self.K = len(self.keep); self.pool = torch.nonzero(self.typ)[:, 0]; self.s_inj = self.tc.abs().median(); self.foreign = torch.nonzero(self.typ & ~torch.isin(self.tn, self.keep))[:, 0]
        self.W2 = unit(self.arch.wdir(b).to(DEV)); self.split = None
    def natural(self, positions=None):
        pos = self.idx if positions is None else positions; r0 = self.run(positions=pos); r1 = self.run(self.tn, positions=pos); lp0, lp1 = torch.log_softmax(r0["lg"], -1), torch.log_softmax(r1["lg"], -1); dl = r0["lg"] - r1["lg"]; dl = dl - dl.mean(1, keepdim=True)
        return dict(F={lv: (r0[lv] - r1[lv])[pos] for lv in self.levels}, dl=dl, kl=(lp0.exp() * (lp0 - lp1)).sum(1), entropy=(-(lp0.exp() * lp0).sum(1)) - (-(lp1.exp() * lp1).sum(1)), top1=lp0.exp().max(1).values - lp1.exp().max(1).values, lp0=lp0, lp1=lp1, r0=r0, r1=r1)
    def halves(self, n, seed=0):
        torch.manual_seed(seed); s = torch.rand(n, device=DEV) < 0.5; return torch.nonzero(s)[:, 0], torch.nonzero(~s)[:, 0]
    def fit_T(self, blk, lv, seed=0, n_dirs=1024, passes=2):
        torch.manual_seed(seed); Vr = unit(torch.randn(n_dirs, self.D, device=DEV)); X = []; Y = []
        for p in range(passes):
            a = torch.randint(0, n_dirs, (len(self.pool),), device=DEV); inj = torch.zeros(self.NT, self.D, device=DEV); inj[self.pool] = self.s_inj * Vr[a]; S2 = self.run(inject=inj, inject_block=blk); X.append(Vr[a]); Y.append((S2[lv] - self.S0[lv])[self.pool] / self.s_inj)
        X, Y = torch.cat(X), torch.cat(Y); return ridge(X, Y, 1e-2)
    def inject_family(self, V, blk, amp=None, positions=None, seed=0):
        """inject rows of V (assigned randomly over the pool) at the input of block blk; returns per-token images at all levels, logit change, assignment"""
        torch.manual_seed(seed); pos = self.pool if positions is None else positions; amp = self.s_inj if amp is None else amp; a = torch.randint(0, len(V), (len(pos),), device=DEV); inj = torch.zeros(self.NT, self.D, device=DEV); inj[pos] = amp * V[a]; base = self.run(positions=pos); r = self.run(positions=pos, inject=inj, inject_block=blk); dl = r["lg"] - base["lg"]; dl = dl - dl.mean(1, keepdim=True)
        return dict(F={lv: (r[lv] - self.S0[lv])[pos] for lv in self.levels}, dl=dl, a=a, pos=pos, lg_base=base["lg"], lg=r["lg"])
