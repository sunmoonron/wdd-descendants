"""e390: is WDD's small-k advantage provenance or generic trained geometry? e389 found the model's own MLP write rows carry
it. Here the final Pythia-410m's states at the middle depth are reconstructed and spliced (e388's test) with MLP rows
(blocks up to that depth) taken from other checkpoints of the same run: steps 0, 1000, 4000, 16000, 33000, 63000 and the
final one (own). If only the rows that actually wrote the state work, the advantage is provenance; if rows from
another trained checkpoint work as well, it is the geometry trained MLPs write in."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
from transformers import AutoModelForCausalLM
c = Cache("pythia410"); model, tok, fam = load_model("pythia410"); arch = Arch(model, fam); NB = arch.NB; L = NB // 2; KS = [8, 16, 32, 64]
ev = c.s["eval_ids"][:3].to(DEV)
def states(ids):
    out = {}
    h = arch.layers[L].register_forward_hook(lambda m, i, o: out.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    try:
        with torch.no_grad(): lg = model(ids).logits.float()
    finally: h.remove()
    return out["x"], lg
def loss_with(Xnew):
    def hk(m, i, o):
        x = o[0] if isinstance(o, tuple) else o; y = x.clone(); y[:, 1:] = Xnew.to(x.dtype).view(x.shape[0], x.shape[1] - 1, -1)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): return token_loss(model(ev).logits.float(), ev).mean().item()
    finally: h.remove()
Xe, lg = states(ev); Lc = token_loss(lg, ev).mean().item(); D = Xe.shape[-1]; Xp = Xe[:, 1:].reshape(-1, D); mu = c.s["mu"][L + 1].to(DEV).float(); Xc = Xp - mu[None]; Lm = loss_with(mu[None].expand(Xp.shape[0], -1))
res = dict(level=L, clean=Lc, mean_ablate=Lm, by_source={})
for src in ["step0", "step1000", "step4000", "step16000", "step33000", "step63000", "final"]:
    if src == "final": rows_ = torch.cat([arch.wdir(b) for b in range(L + 1)]).to(DEV)
    else:
        m2 = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", revision=src, dtype=torch.float32); a2 = Arch(m2, "neox"); rows_ = torch.cat([a2.wdir(b) for b in range(L + 1)]).to(DEV); del m2
    Dct = rows_ / rows_.norm(dim=-1, keepdim=True).clamp_min(1e-8); sel, _, _ = omp(Xc, Dct, max(KS), batch=256, record_err=False); row = {}
    own = torch.cat([arch.wdir(b) for b in range(L + 1)]).to(DEV); own = own / own.norm(dim=-1, keepdim=True).clamp_min(1e-8); row["row_cos_to_final_median"] = (Dct * own).sum(-1).median().item()
    for k in KS:
        cof, err = refit(Xc, Dct, sel[:, :k]); Xh = mu[None] + torch.einsum("nk,nkd->nd", cof, Dct[sel[:, :k]]); Ls = loss_with(Xh)
        row[str(k)] = dict(fvu=(err.sum() / Xc.pow(2).sum()).item(), recovered=(Lm - Ls) / max(Lm - Lc, 1e-9))
    res["by_source"][src] = row; del sel
log(f"pythia410 final states at L{L}, MLP rows from: " + " | ".join(f"{s} (row cos to final {r['row_cos_to_final_median']:.2f}): " + " ".join(f"k{k} {r[str(k)]['recovered']:.2f}" for k in KS) for s, r in res["by_source"].items()))
record("e390_provenance_pythia410", res, " | ".join(f"{s}: k8 {r['8']['recovered']:.2f} k32 {r['32']['recovered']:.2f}" for s, r in res["by_source"].items()))
