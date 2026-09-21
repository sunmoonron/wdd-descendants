"""e183: is the next-block eraser chain causal for readability? For b in a few early blocks: ablate the MLP of block
b+1 (zero its down-projection input) in the forward pass, re-hook the level-L state and the block-b ledger, and
measure (i) the raw survival of block-b-born dominant writes at level L and (ii) their OMP@64 / dual@64
identification, vs the clean run. Also the control: ablate the MLP of block b+3 instead."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX
A, lab = c.dictionary(L); mu = c.s["mu"][L + 1].to(DEV); S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
def run(ablate_block=None):
    store = {"acts": {}, "H": None}; hs = []
    for b in range(L + 1): hs.append(arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, inp: store["acts"].__setitem__(b_, inp[0].detach().float().reshape(-1, c.DFF)))(b)))
    hs.append(arch.layers[L].register_forward_hook(lambda m, i, o: store.__setitem__("H", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D))))
    if ablate_block is not None: hs.append(arch.mlp_lin(ablate_block).register_forward_pre_hook(lambda m, inp: (torch.zeros_like(inp[0]),)))
    model(ids_seq); [h.remove() for h in hs]; return store["H"], store["acts"]
H0, acts0 = run(); X0 = H0 - mu; typ = typical_mask(H0); WN = torch.stack([c.d["WN"][b] for b in range(L + 1)]).to(DEV); rows = []
for b in [bb for bb in (1, 2, 3, 4) if bb + 3 <= L]:
    led_b = acts0[b] * WN[b][None]; tcv, tnv = led_b.abs().max(1); tcv = torch.gather(led_b, 1, tnv[:, None])[:, 0]; row = c.atom_index(L, torch.full_like(tnv.cpu(), b), tnv.cpu()).to(DEV); d = A[row]; big = typ & (tcv.abs() >= tcv.abs().quantile(0.5))
    out = {}
    for nm, ab in (("clean", None), ("ablate_next", b + 1), ("ablate_plus3", b + 3)):
        H1, _ = run(ab); X1 = H1 - mu; surv = (H1 * d).sum(1) / tcv; so, _, _ = omp(X1, A, 64); sd, _, _ = oneshot(X1, A, 64, whiten=Winv)
        out[nm] = dict(survival_med=surv[big].median().item(), frac_erased=(surv[big] < 0.5).float().mean().item(), recall_omp=(so == row[:, None]).any(1)[big].float().mean().item(), recall_dual=(sd == row[:, None]).any(1)[big].float().mean().item(), prom_med=((X1 * d).sum(1).abs() / X1.norm(dim=1))[big].median().item())
    rows.append(dict(birth=b, **{f"{k}_{m}": v for k, dd in out.items() for m, v in dd.items()}))
    log(f"{tag} born b{b}: clean surv {out['clean']['survival_med']:.2f} erased {out['clean']['frac_erased']:.2f} recall {out['clean']['recall_omp']:.2f}/{out['clean']['recall_dual']:.2f} | ablate block {b + 1} MLP: surv {out['ablate_next']['survival_med']:.2f} erased {out['ablate_next']['frac_erased']:.2f} recall {out['ablate_next']['recall_omp']:.2f}/{out['ablate_next']['recall_dual']:.2f} | ablate block {b + 3}: surv {out['ablate_plus3']['survival_med']:.2f} recall {out['ablate_plus3']['recall_omp']:.2f}/{out['ablate_plus3']['recall_dual']:.2f}")
record(f"e183_eraserabl_{tag}", dict(model=tag, L=L, rows=rows), " | ".join(f"b{r['birth']}: surv {r['clean_survival_med']:.2f}->{r['ablate_next_survival_med']:.2f} (ctrl {r['ablate_plus3_survival_med']:.2f}), recall omp {r['clean_recall_omp']:.2f}->{r['ablate_next_recall_omp']:.2f} (ctrl {r['ablate_plus3_recall_omp']:.2f}), dual {r['clean_recall_dual']:.2f}->{r['ablate_next_recall_dual']:.2f} (ctrl {r['ablate_plus3_recall_dual']:.2f})" for r in rows))
