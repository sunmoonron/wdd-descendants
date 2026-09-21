"""e17: cheap diagnostics on the cached mid-layer OMP: recall and FVU by position bucket, by token frequency,
by sign of the dominant write, by the block (age) of the dominant write, and the block histogram of the support
versus the block histogram of true write energy (recency bias?)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); hit = (sel == row[:, None]).any(1)
pos = (torch.arange(c.NT) % CTX).to(DEV); tok = c.s["eval_ids"].reshape(-1).to(DEV)
cnt = torch.bincount(c.s["cen_ids"].reshape(-1), minlength=c.n_tok_emb).to(DEV); freq = cnt[tok]
def by(mask_fn, keys):
    out = {}
    for k, m in ((k, mask_fn(k) & typ) for k in keys):
        if m.sum() > 50: out[str(k)] = dict(n=int(m.sum()), recall=hit[m].float().mean().item(), fvu64=fvu(err[:, 63], X, m))
    return out
res = dict(model=tag, L=L)
res["by_position"] = by(lambda k: (pos >= k[0]) & (pos < k[1]), [(0, 1), (1, 8), (8, 32), (32, 128), (128, 512)])
res["by_token_freq"] = by(lambda k: (freq >= k[0]) & (freq < k[1]), [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10 ** 9)])
res["by_sign"] = by(lambda k: (tc[:, 0].to(DEV) > 0) == k, [True, False])
res["by_block_of_dominant"] = by(lambda k: tb[:, 0].to(DEV) == k, list(range(L + 1)))
led = c.ledger(L); E = torch.stack([(led[b] ** 2).sum(1) for b in range(L + 1)]).to(DEV)   # [L+1, NT]
res["energy_by_block_share"] = (E[:, typ].sum(1) / E[:, typ].sum()).tolist()
sb = lab["block"].to(DEV)[sel]; st = lab["type"].to(DEV)[sel]; msel = (st == T_MLP)
res["support_mlp_by_block_share"] = [((sb == b) & msel)[typ].float().sum().item() / msel[typ].float().sum().item() for b in range(L + 1)]
res["support_type_share"] = {int(t): (st == t)[typ].float().mean().item() for t in range(5)}
record(f"e17_diag_{tag}", res, "pos " + " ".join(f"{k}:{v['recall']:.2f}" for k, v in res["by_position"].items()) + " | freq " + " ".join(f"{k}:{v['recall']:.2f}" for k, v in res["by_token_freq"].items()) + " | sign " + " ".join(f"{k}:{v['recall']:.2f}" for k, v in res["by_sign"].items()) + " | block " + " ".join(f"{k}:{v['recall']:.2f}" for k, v in res["by_block_of_dominant"].items()) + " | energy share by block " + " ".join(f"{v:.2f}" for v in res["energy_by_block_share"]) + " vs support " + " ".join(f"{v:.2f}" for v in res["support_mlp_by_block_share"]))
