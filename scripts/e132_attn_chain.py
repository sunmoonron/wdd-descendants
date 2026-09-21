"""e132 (GPT-2 pilot): the attention -> MLP -> WDD chain. Ablate one attention head (zero its slice of the output
projection input) in block 4 or 5, re-run the model with hooks, and for the level-6 dominant MLP writes: the true
change of the write's coefficient (ledger), the change of the state's projection on the write's direction
(visibility), and the change in WDD's recovered coefficient under OMP@64 and the dual projection@64. If WDD tracks
attention-mediated changes of a downstream write, the chain is recoverable even though the head's own write is
not represented by the static atoms."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = "gpt2"; c = Cache(tag); L = mid(c); model, tok, fam = load_model("gpt2"); arch = Arch(model, fam)
NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; ids = torch.arange(NT); A, lab = c.dictionary(L); mu = c.s["mu"][L + 1].to(DEV)
S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
def run(ablate=None):
    store = {"acts": {}, "H": None}; hs = []
    for b in range(L + 1):
        hs.append(arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, inp: store["acts"].__setitem__(b_, inp[0].detach().float().reshape(-1, c.DFF)))(b)))
    hs.append(arch.layers[L].register_forward_hook(lambda m, i, o: store.__setitem__("H", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D))))
    if ablate is not None:
        b, h = ablate
        def f(m, inp):
            x = inp[0].clone(); x[..., h * c.HD:(h + 1) * c.HD] = 0; return (x,)
        hs.append(arch.attn_lin(b).register_forward_pre_hook(f))
    model(ids_seq); [hh.remove() for hh in hs]; return store["H"], store["acts"]
H0, acts0 = run(); X0 = H0 - mu; typ = typical_mask(H0)
tb, tn, tc = c.top_writes(L, 1); tb, tn = tb[ids, 0], tn[ids, 0]; row = c.atom_index(L, tb, tn).to(DEV); d = A[row]; WN = torch.stack([c.d["WN"][b] for b in range(L + 1)]).to(DEV)
c0 = torch.stack([acts0[b] for b in range(L + 1)], 1)[torch.arange(NT), tb.to(DEV), tn.to(DEV)] * WN[tb.to(DEV), tn.to(DEV)]
so0, co0, _ = omp(X0, A, 64); sd0, cd0, _ = oneshot(X0, A, 64, whiten=Winv)
rec = lambda s, cf: (cf * (s == row[:, None])).sum(1)
r0_omp, r0_dual = rec(so0, co0), rec(sd0, cd0); p0 = (X0 * d).sum(1); rows = []
for b in (4, 5):
    for h in range(c.NH):
        H1, acts1 = run((b, h)); X1 = H1 - mu
        c1 = torch.stack([acts1[bb] for bb in range(L + 1)], 1)[torch.arange(NT), tb.to(DEV), tn.to(DEV)] * WN[tb.to(DEV), tn.to(DEV)]
        dc = c1 - c0; big = typ & (dc.abs() > 0.05 * c0.abs()) & (c0.abs() > 0)
        if big.sum() < 50: continue
        so1, co1, _ = omp(X1, A, 64); sd1, cd1, _ = oneshot(X1, A, 64, whiten=Winv)
        dr_omp, dr_dual = rec(so1, co1) - r0_omp, rec(sd1, cd1) - r0_dual; dp = (X1 * d).sum(1) - p0
        both_omp = big & (so0 == row[:, None]).any(1) & (so1 == row[:, None]).any(1); both_dual = big & (sd0 == row[:, None]).any(1) & (sd1 == row[:, None]).any(1)
        cor = lambda a, b_, m: torch.corrcoef(torch.stack([a[m], b_[m]]))[0, 1].item() if m.sum() > 20 else None
        slope = lambda a, b_, m: ((a[m] * b_[m]).sum() / (b_[m] ** 2).sum()).item() if m.sum() > 20 else None
        rows.append(dict(block=b, head=h, n_affected=int(big.sum()), frac_affected=big[typ].float().mean().item() / max(typ.float().mean().item(), 1e-9), mean_rel_dc=(dc.abs() / c0.abs())[big].median().item(),
                         corr_visibility=cor(dp, dc, big), slope_visibility=slope(dp, dc, big), corr_omp=cor(dr_omp, dc, both_omp), slope_omp=slope(dr_omp, dc, both_omp), corr_dual=cor(dr_dual, dc, both_dual), slope_dual=slope(dr_dual, dc, both_dual),
                         identified_both_omp=both_omp[big].float().mean().item(), identified_both_dual=both_dual[big].float().mean().item(), dominant_changes=((c1.abs() < torch.stack([acts1[bb] for bb in range(L + 1)], 1).mul(WN[None]).abs().flatten(1).max(1).values * 0.999))[big].float().mean().item()))
        log(f"b{b}h{h}: affected {rows[-1]['frac_affected']:.2f} of typical (|dc|/|c| med {rows[-1]['mean_rel_dc']:.2f}) | visibility corr {rows[-1]['corr_visibility']} slope {rows[-1]['slope_visibility']} | omp corr {rows[-1]['corr_omp']} slope {rows[-1]['slope_omp']} (both identified {rows[-1]['identified_both_omp']:.2f}) | dual corr {rows[-1]['corr_dual']} slope {rows[-1]['slope_dual']} ({rows[-1]['identified_both_dual']:.2f})")
def med(k): v = [r[k] for r in rows if r[k] is not None]; return float(np.median(v)) if v else None
res = dict(model=tag, L=L, heads=rows, summary=dict(n_heads=len(rows), frac_affected_median=med("frac_affected"), corr_visibility_median=med("corr_visibility"), slope_visibility_median=med("slope_visibility"), corr_omp_median=med("corr_omp"), slope_omp_median=med("slope_omp"), corr_dual_median=med("corr_dual"), slope_dual_median=med("slope_dual"), identified_both_omp_median=med("identified_both_omp"), identified_both_dual_median=med("identified_both_dual")))
record("e132_attnchain_gpt2", res, f"{len(rows)} heads | affected tokens median {res['summary']['frac_affected_median']:.2f} | state visibility of the write change: corr {res['summary']['corr_visibility_median']:.2f} slope {res['summary']['slope_visibility_median']:.2f} | WDD tracks the change: OMP corr {res['summary']['corr_omp_median']:.2f} slope {res['summary']['slope_omp_median']:.2f} (identified before+after {res['summary']['identified_both_omp_median']:.2f}); dual corr {res['summary']['corr_dual_median']:.2f} slope {res['summary']['slope_dual_median']:.2f} ({res['summary']['identified_both_dual_median']:.2f})")
