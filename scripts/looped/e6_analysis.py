"""e6: analyses from the caches, CPU only. (1) The vocabulary across loops: usage of each native atom in the 16-atom
codes of the state before layer 12, overlap of the 256 most used words between loops, per-token support overlap
between successive loops, against the same for the rotated dictionary (states that stay alike keep any code alike).
(2) Halting: does the exit gate, the change of the exit state, or the change of the WDD code tell whether the next
loop still lowers the token's loss? (3) Carried against fresh content: how much of a loop's input is still present at
its middle and end."""
import sys; sys.path.insert(0, "/data/loopwdd/code")
import torch, math, json
from lw import log, jdump, ROOT
torch.set_grad_enabled(False)
C = torch.load(f"{ROOT}/cache/ouro_e1.pt"); K2 = torch.load(f"{ROOT}/cache/ouro_e2.pt"); cods = K2["codes"]; TLX = len(cods)
res = {}
def auc(score, y):
    y = y.bool(); n1, n0 = y.sum().item(), (~y).sum().item()
    if n1 == 0 or n0 == 0: return float("nan")
    r = score.argsort().argsort().float() + 1
    return ((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()
def spearman(a, b):
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean(); return ((ra * rb).sum() / (ra.norm() * rb.norm())).item()
# (1) vocabulary
K = 16; NA = int(max(c["sel"].max().item() for c in cods.values())) + 1
for key in ("sel", "selr"):
    U = torch.stack([torch.bincount(cods[t][key][:, :K].long().flatten(), minlength=NA) for t in range(TLX)]).float()
    W = [set(U[t].topk(256).indices.tolist()) for t in range(TLX)]
    jac = [[len(W[a] & W[b]) / len(W[a] | W[b]) for b in range(TLX)] for a in range(TLX)]
    used = U.sum(0) > 0
    sp = [[spearman(U[a, used], U[b, used]) for b in range(TLX)] for a in range(TLX)]
    tok = []
    for t in range(TLX - 1):
        a, b = cods[t][key][:, :K].long(), cods[t + 1][key][:, :K].long()
        inter = (a[:, :, None] == b[:, None, :]).any(-1).sum(-1).float(); tok.append((inter / (2 * K - inter)).mean().item())
    res[f"vocab_{'native' if key == 'sel' else 'rotated'}"] = dict(top256_jaccard=jac, usage_spearman=sp, token_support_jaccard_next=tok)
    log(key, "top-256 Jaccard with loop 4:", [f"{jac[3][b]:.2f}" for b in range(TLX)], "| successive loops, per-token support Jaccard:", [f"{x:.2f}" for x in tok])
# states that stay alike: cosine of the centred states before layer 12 in successive loops
cs = []
for t in range(TLX - 1):
    a = C["states"][(t, 12)][:, 1:].reshape(-1, C["states"][(t, 12)].shape[-1]) - cods[t]["mu"]
    b = C["states"][(t + 1, 12)][:, 1:].reshape(-1, a.shape[-1]) - cods[t + 1]["mu"]
    cs.append(torch.nn.functional.cosine_similarity(a, b, dim=-1).median().item())
# the 64-atom WDD code as a coefficient vector over atoms: cosine between successive loops, per token
cc64 = []
for t in range(TLX - 1):
    a, b = cods[t]["sel"].long(), cods[t + 1]["sel"].long(); ca, cb = cods[t]["cof"].float(), cods[t + 1]["cof"].float()
    eq = (a[:, :, None] == b[:, None, :]).float()
    dot = torch.einsum("nij,ni,nj->n", eq, ca, cb)
    cc64.append((dot / (ca.norm(dim=-1) * cb.norm(dim=-1)).clamp_min(1e-9)).median().item())
res["code_vector_cos_next"] = cc64; log("64-atom code vector cosine, successive loops:", [f"{x:.2f}" for x in cc64])
res["centred_state_cos_next"] = cs; log("centred state cosine, successive loops:", [f"{x:.2f}" for x in cs])
# (2) halting
ce, gate, dh = C["ce"], C["gate"], C["dh"]       # ce [loops, B, T-1]; gate, dh [loops, B, T]
hal = {}
for t in range(1, TLX - 1):
    gain = (ce[t] - ce[t + 1])[:, 1:].flatten()                       # does loop t+2 (index t+1) still help position p?
    g = gate[t][:, 1:-1].flatten(); dd = dh[t][:, 1:-1].flatten()
    a, b = cods[t - 1]["sel"][:, :K].long(), cods[t]["sel"][:, :K].long()
    inter = (a[:, :, None] == b[:, None, :]).any(-1).sum(-1).float(); cc = (1 - inter / (2 * K - inter)).view(gate.shape[1], -1)[:, :-1].flatten()
    ar, br = cods[t - 1]["selr"][:, :K].long(), cods[t]["selr"][:, :K].long()
    ir = (ar[:, :, None] == br[:, None, :]).any(-1).sum(-1).float(); ccr = (1 - ir / (2 * K - ir)).view(gate.shape[1], -1)[:, :-1].flatten()
    row = {}
    for thr in (0.05, 0.25):
        y = gain > thr
        row[f"gain>{thr}"] = dict(base_rate=y.float().mean().item(), auc_continue_gate=auc(-g, y), auc_state_change=auc(dd, y),
                                  auc_code_change=auc(cc, y), auc_code_change_rotated=auc(ccr, y))
    row["spearman_gate_vs_state_change"] = spearman(g, dd); row["spearman_gate_vs_code_change"] = spearman(g, cc)
    row["mean_gain"] = gain.mean().item()
    hal[t + 1] = row
    r5 = row["gain>0.05"]
    log(f"after loop {t+1}: next loop helps >0.05 nats for {r5['base_rate']:.2f}; AUC gate {r5['auc_continue_gate']:.3f}, state change {r5['auc_state_change']:.3f}, "
        f"WDD code change {r5['auc_code_change']:.3f} (rotated {r5['auc_code_change_rotated']:.3f}); rho(gate, state change) {row['spearman_gate_vs_state_change']:.2f}")
res["halting"] = hal
# (3) carried vs fresh
car = {}
for t in range(TLX):
    x0 = C["states"][(t, 0)][:, 1:].reshape(-1, 2048); row = {}
    for l in (12, 24):
        x = C["states"][(t, l)][:, 1:].reshape(-1, 2048)
        beta = (x * x0).sum(-1) / x0.pow(2).sum(-1); cosv = torch.nn.functional.cosine_similarity(x, x0, dim=-1)
        row[l] = dict(beta=beta.median().item(), cos=cosv.median().item(), norm_ratio=(x.norm(dim=-1) / x0.norm(dim=-1)).median().item())
    car[t + 1] = row
    log(f"loop {t+1}: input still present at layer 12 beta {row[12]['beta']:.2f} (cos {row[12]['cos']:.2f}, norm ratio {row[12]['norm_ratio']:.2f}); "
        f"at loop end beta {row[24]['beta']:.2f} (cos {row[24]['cos']:.2f})")
res["carried"] = car
# (4) the BOS sink across loops: its norm against the median position, at the input, middle and end of each loop
res["bos_norm_ratio"] = {f"{t+1},{l}": (C["states"][(t, l)][:, 0].norm(dim=-1).median() / C["states"][(t, l)][:, 1:].norm(dim=-1).median()).item()
                         for t in range(TLX) for l in (0, 12, 24)}
log("BOS norm / median norm:", {k: round(v, 1) for k, v in res["bos_norm_ratio"].items()})
jdump(res, f"{ROOT}/results/e6_ouro.json")
