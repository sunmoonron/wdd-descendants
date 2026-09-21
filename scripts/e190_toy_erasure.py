"""e190: the erasure matrix and the hydra test in the B=6 residual toy of e176. After training, for the dominant
block-b neuron write of each block b (and the block-0 feature atoms), the direct contribution of each later block's
write to raw survival at the final stream (erasure matrix), and then the total effect of skipping block b+1 (does
survival recover, or does block b+2 take over the cancellation?). Also the same at initialization (untrained)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
torch.manual_seed(0); n, d, s, m, B = 400, 64, 0.05, 256, 6; imp = 0.9 ** torch.arange(n, device=DEV)
W1 = torch.nn.Parameter(torch.randn(d, n, device=DEV) * 0.1); Wr = torch.nn.Parameter(torch.randn(n, d, device=DEV) * 0.1); br = torch.nn.Parameter(torch.zeros(n, device=DEV)); params = [W1, Wr, br]; blocks = []
for b in range(B):
    Win = torch.nn.Parameter(torch.randn(m, d, device=DEV) * 0.1); bi = torch.nn.Parameter(torch.zeros(m, device=DEV)); Wout = torch.nn.Parameter(torch.randn(m, d, device=DEV) * (0.1 / math.sqrt(B))); blocks.append((Win, bi, Wout)); params += [Win, bi, Wout]
opt = torch.optim.Adam(params, lr=2e-3)
def fwd(x, skip=None):
    h = x @ W1.T; hs = [h]; ws = []; acts = []
    for b, (Win, bi, Wout) in enumerate(blocks):
        a = torch.relu(h @ Win.T + bi); w = a @ Wout
        if b == skip: w = torch.zeros_like(w)
        h = h + w; hs.append(h); ws.append(w); acts.append(a)
    return hs, ws, acts, torch.relu(h @ Wr.T + br)
def analyse(label):
    x = torch.rand(4096, n, device=DEV) * (torch.rand(4096, n, device=DEV) < s); hs, ws, acts, _ = fwd(x); res = {}
    for b in range(B - 2):
        Wout = blocks[b][2].detach(); wn = Wout.norm(dim=1); led = acts[b].detach() * wn[None]; top = led.abs().argmax(1); coef = led.gather(1, top[:, None])[:, 0]; ok = coef.abs() > 1e-6
        if ok.sum() < 50: continue
        dtop = Wout[top] / wn[top].clamp_min(1e-9)[:, None]
        direct = [((ws[bp].detach() * dtop).sum(1) / coef)[ok].mean().item() - (1.0 if bp == b else 0.0) for bp in range(B)]
        tot_clean = ((hs[-1].detach() * dtop).sum(1) / coef)[ok].mean().item()
        hs2, ws2, _, _ = fwd(x, skip=b + 1); tot_abl = ((hs2[-1].detach() * dtop).sum(1) / coef)[ok].mean().item(); direct_abl = [((ws2[bp].detach() * dtop).sum(1) / coef)[ok].mean().item() - (1.0 if bp == b else 0.0) for bp in range(B)]
        res[b] = dict(direct=direct, total_clean=tot_clean, total_skip_next=tot_abl, direct_skip_next=direct_abl)
        log(f"toy {label} born b{b}: direct by block " + " ".join(f"b{bp}:{v:+.2f}" for bp, v in enumerate(direct)) + f" | total {tot_clean:.2f} -> skip b{b + 1}: {tot_abl:.2f} | b+2 direct {direct[b + 2]:+.2f}->{direct_abl[b + 2]:+.2f}")
    return res
init = analyse("init")
with torch.enable_grad():
    for step in range(4000):
        x = torch.rand(1024, n, device=DEV) * (torch.rand(1024, n, device=DEV) < s); _, _, _, y = fwd(x); loss = (imp * (y - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
trained = analyse("trained")
nxt = [trained[b]["direct"][b + 1] for b in trained]; p2 = [trained[b]["direct"][b + 2] for b in trained]; rec = [trained[b]["total_skip_next"] - trained[b]["total_clean"] for b in trained]; take = [trained[b]["direct_skip_next"][b + 2] - trained[b]["direct"][b + 2] for b in trained]
record("e190_toy_erasure", dict(init=init, trained=trained), f"trained toy: next-block direct contribution mean {sum(nxt) / len(nxt):+.2f} (b+2: {sum(p2) / len(p2):+.2f}) | skipping the next block changes total survival by {sum(rec) / len(rec):+.2f} while block b+2's direct contribution changes by {sum(take) / len(take):+.2f} | init next-block direct {sum(init[b]['direct'][b + 1] for b in init) / len(init):+.2f}")
