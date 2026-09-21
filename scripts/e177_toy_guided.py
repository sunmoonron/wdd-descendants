"""e177: can the swamping be undone in the toy? Six-block toy as in e176. State-level readings of the block-0
feature: (a) OMP@16 / dual@16 over the full dictionary (baseline), (b) restricted to increment-nominated atoms
(block-0 increment top-8 + each later block's increment top-8), (c) OMP on the state with later-block atoms
removed (block-0 atoms only), (d) the raw projection on block-0 atoms only. Plus what fraction of the state's
energy is later-block writes."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
torch.manual_seed(0); n, d, s, m, B = 400, 64, 0.05, 256, 6; imp = 0.9 ** torch.arange(n, device=DEV)
W1 = torch.nn.Parameter(torch.randn(d, n, device=DEV) * 0.1); Wr = torch.nn.Parameter(torch.randn(n, d, device=DEV) * 0.1); br = torch.nn.Parameter(torch.zeros(n, device=DEV)); params = [W1, Wr, br]; blocks = []
for b in range(B):
    Win = torch.nn.Parameter(torch.randn(m, d, device=DEV) * 0.1); bi = torch.nn.Parameter(torch.zeros(m, device=DEV)); Wout = torch.nn.Parameter(torch.randn(m, d, device=DEV) * (0.1 / math.sqrt(B))); blocks.append((Win, bi, Wout)); params += [Win, bi, Wout]
opt = torch.optim.Adam(params, lr=2e-3)
def fwd(x):
    h = x @ W1.T; hs = [h]; ws = []
    for Win, bi, Wout in blocks:
        w = torch.relu(h @ Win.T + bi) @ Wout; h = h + w; hs.append(h); ws.append(w)
    return hs, ws, torch.relu(h @ Wr.T + br)
with torch.enable_grad():
    for step in range(4000):
        x = torch.rand(1024, n, device=DEV) * (torch.rand(1024, n, device=DEV) < s); _, _, y = fwd(x); loss = (imp * (y - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
x = torch.rand(4096, n, device=DEV) * (torch.rand(4096, n, device=DEV) < s); hs, ws, y = fwd(x); hL = hs[-1].detach(); h0 = hs[0].detach()
norms1 = W1.detach().norm(dim=0); A0 = (W1.detach() / norms1.clamp_min(1e-6)).T; atoms = [A0]
for Win, bi, Wout in blocks: atoms.append(Wout.detach() / Wout.detach().norm(dim=1, keepdim=True).clamp_min(1e-6))
A = torch.cat(atoms); C0 = x * norms1[None]; active = (C0.abs() > 0).any(1); top = C0.abs().argmax(1); hc = hL - hL.mean(0); k = 16
def dual(Xc, Ad, kk, allowed=None):
    S_ = Ad.T @ Ad; ev, V = torch.linalg.eigh(S_); Wi = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sc = ((Xc @ Wi.T) @ Ad.T).abs()
    if allowed is not None: sc = sc.masked_fill(~allowed, -1)
    return sc.topk(kk, dim=1).indices
def omp_masked(X, Ad, kk, allowed, batch=1024):
    N_ = X.shape[0]; sel = torch.zeros(N_, kk, dtype=torch.long, device=DEV); eye = torch.eye(kk, device=DEV)
    for s_ in range(0, N_, batch):
        xx = X[s_:s_ + batch]; nn_ = xx.shape[0]; r = xx.clone(); S = torch.zeros(nn_, 0, dtype=torch.long, device=DEV); taken = ~allowed[s_:s_ + batch].clone()
        for step in range(kk):
            pick = (r @ Ad.T).abs_().masked_fill_(taken, -1.0).argmax(-1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
            As = Ad[S]; G = As @ As.transpose(1, 2) + 1e-5 * eye[:step + 1, :step + 1]; cc = torch.cholesky_solve(As @ xx[:, :, None], torch.linalg.cholesky(G)); r = xx - (cc.transpose(1, 2) @ As)[:, 0]
        sel[s_:s_ + nn_] = S
    return sel
allowed = torch.zeros(x.shape[0], A.shape[0], dtype=torch.bool, device=DEV); off = 0
for b in range(B + 1):
    inc = (hs[b].detach() - (hs[b - 1].detach() if b > 0 else 0)) if b > 0 else h0; inc = inc - inc.mean(0); Ab = atoms[b]; nb = 8
    sb, _, _ = omp(inc, Ab, nb); allowed[torch.arange(x.shape[0], device=DEV)[:, None], sb + off] = True; off += Ab.shape[0]
a_ = active; so, _, _ = omp(hc, A, k); sd = dual(hc, A, k); sg = omp_masked(hc, A, k, allowed); sgd = dual(hc, A, k, allowed); s0, _, _ = omp(hc, A0, k); sp = (hc @ A0.T).abs().topk(k, dim=1).indices
later = sum((w.detach() ** 2).sum(1) for w in ws); res = dict(B=B, later_block_energy_share=(later / (hL ** 2).sum(1))[a_].median().item(),
           recall=dict(full_omp=(so == top[:, None]).any(1)[a_].float().mean().item(), full_dual=(sd == top[:, None]).any(1)[a_].float().mean().item(), guided_omp=(sg == top[:, None]).any(1)[a_].float().mean().item(), guided_dual=(sgd == top[:, None]).any(1)[a_].float().mean().item(),
                       block0_atoms_only_omp=(s0 == top[:, None]).any(1)[a_].float().mean().item(), block0_atoms_only_projection=(sp == top[:, None]).any(1)[a_].float().mean().item()))
record("e177_toy_guided", res, f"later-block share of state energy {res['later_block_energy_share']:.2f} | state-level recall of the block-0 feature: full omp {res['recall']['full_omp']:.2f} dual {res['recall']['full_dual']:.2f} | increment-guided omp {res['recall']['guided_omp']:.2f} dual {res['recall']['guided_dual']:.2f} | block-0 atoms only: omp {res['recall']['block0_atoms_only_omp']:.2f} projection {res['recall']['block0_atoms_only_projection']:.2f}")
