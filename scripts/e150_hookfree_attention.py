"""e150 (GPT-2): hook-free attention-pattern reading. From hidden states alone, the block increment minus the
MLP part is not available without a hook, so read the WHOLE increment jointly: NNLS over [all heads' value atoms
for all context positions] + the block's MLP atoms, per token (64 tokens, blocks 2-5). Recovered per-head
attention weights (normalized) vs the true patterns: correlation per head, and the fraction of heads whose top
attended position is recovered. Two conditions: NNLS on the true attention write (upper bound) and on the full
increment (hook-free)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
c = Cache("gpt2"); model, tok, fam = load_model("gpt2"); arch = Arch(model, fam); NS, NH, HD, D = c.n_eval, c.NH, c.HD, c.D
g = torch.Generator().manual_seed(1); pick_seq = torch.randint(0, NS, (64,), generator=g); pick_pos = torch.randint(16, CTX, (64,), generator=g); rows = []
def pg_nnls(B, y, iters=400):
    Lc = torch.linalg.eigvalsh(B @ B.T)[-1].clamp_min(1e-6); a = torch.zeros(B.shape[0], device=DEV)
    for _ in range(iters): a = (a - (B @ (B.T @ a - y)) / Lc).clamp_min(0)
    return a
for b in (2, 3, 4, 5):
    l = arch.layers[b]; W = l.attn.c_attn.weight.detach().float(); bq = l.attn.c_attn.bias.detach().float(); Wo = arch.wo(b).view(NH, HD, D)
    Hb = c.s["H"][b].view(NS, CTX, D).float().to(DEV); ln = torch.nn.functional.layer_norm(Hb, (D,), l.ln_1.weight.float(), l.ln_1.bias.float(), l.ln_1.eps)
    q = (ln @ W[:, :D] + bq[:D]).view(NS, CTX, NH, HD); k = (ln @ W[:, D:2 * D] + bq[D:2 * D]).view(NS, CTX, NH, HD); v = (ln @ W[:, 2 * D:] + bq[2 * D:]).view(NS, CTX, NH, HD)
    Am = c.wdir_cpu(b).to(DEV); Am = Am / Am.norm(dim=1, keepdim=True)
    corr_ub, corr_hf, top_ub, top_hf = [], [], [], []
    for i in range(64):
        s, t = pick_seq[i].item(), pick_pos[i].item(); tid = s * CTX + t
        al = torch.softmax(torch.einsum("jhd,hd->hj", k[s, :t + 1], q[s, t]) / math.sqrt(HD), 1)                        # [H, t+1] true
        VA = torch.einsum("jhd,hde->hje", v[s, :t + 1], Wo).reshape(NH * (t + 1), D)                                       # value atoms (unnormalized: coefficients are then attention weights)
        att_true = c.s["ATT"][b][tid].float().to(DEV) - arch.attn_bias(b).to(DEV); inc = (c.s["H"][b + 1][tid].float() - c.s["H"][b][tid].float()).to(DEV)
        a_ub = pg_nnls(VA, att_true).view(NH, t + 1); B_hf = torch.cat([VA, Am * 1.0]); a_hf = pg_nnls(B_hf, inc)[:NH * (t + 1)].view(NH, t + 1)   # MLP atoms unconstrained sign would need LS; keep NNLS as a crude joint fit
        for h in range(NH):
            for a_, cc, tp in ((a_ub, corr_ub, top_ub), (a_hf, corr_hf, top_hf)):
                x = a_[h] / a_[h].sum().clamp_min(1e-9); cc.append(torch.corrcoef(torch.stack([x, al[h]]))[0, 1].nan_to_num(0).item()); tp.append((x.argmax() == al[h].argmax()).item())
    rows.append(dict(b=b, corr_upper_bound_med=float(np.median(corr_ub)), corr_hookfree_med=float(np.median(corr_hf)), top_pos_upper=float(np.mean(top_ub)), top_pos_hookfree=float(np.mean(top_hf))))
    log(f"b{b}: attention pattern from the write (upper bound): corr med {rows[-1]['corr_upper_bound_med']:.2f}, top position {rows[-1]['top_pos_upper']:.2f} | from the raw increment (hook-free): corr {rows[-1]['corr_hookfree_med']:.2f}, top position {rows[-1]['top_pos_hookfree']:.2f}")
record("e150_hookfree_gpt2", dict(rows=rows), " | ".join(f"b{r['b']}: write-only corr {r['corr_upper_bound_med']:.2f} top {r['top_pos_upper']:.2f}; hook-free corr {r['corr_hookfree_med']:.2f} top {r['top_pos_hookfree']:.2f}" for r in rows))
