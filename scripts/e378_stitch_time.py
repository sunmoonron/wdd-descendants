"""e378: stitching across training time, the causal test of "implementation settles before function". Pythia-410m at
steps 1000, 4000, 16000, 33000, 63000 and 143000: models built from the embedding and blocks below a cut L of one
checkpoint (lower) and the blocks from L up, the final norm and the unembedding of another (upper), for cuts at blocks
4, 8, 12, 16, 20; eval loss on held-out wikitext. Raw stitching (the two parts meet in their own residual bases) and
linear stitching (an affine map at the cut fitted by least squares on other sequences, which forgives any change of
basis). If lower-layer implementation settles first and later learning is downstream, the stitched loss is set by the
upper part's age; if the residual basis merely rotates, linear stitching repairs raw stitching."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
from transformers import AutoModelForCausalLM
revs = ["step1000", "step4000", "step16000", "step33000", "step63000", "step143000"]; cuts = [4, 8, 12, 16, 20]; c = Cache("pythia410")
SD = {r: {k: v.to(DEV) for k, v in AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", revision=r, dtype=torch.float32).state_dict().items()} for r in revs}
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", dtype=torch.float32).to(DEV).eval(); layers = model.gpt_neox.layers
ev = c.s["eval_ids"][:12].to(DEV); fit = c.s["eval_ids"][12:24].to(DEV)
def assemble(lo, up, L):
    sd = {}
    for k in SD[up]:
        if k.startswith("gpt_neox.embed_in"): sd[k] = SD[lo][k]
        elif k.startswith("gpt_neox.layers."): sd[k] = SD[lo][k] if int(k.split(".")[2]) < L else SD[up][k]
        else: sd[k] = SD[up][k]
    model.load_state_dict(sd, strict=False)
def loss_of(ids, hook=None):
    h = layers[hook[0]].register_forward_pre_hook(hook[1], with_kwargs=True) if hook else None
    try:
        with torch.no_grad(): return token_loss(model(ids).logits.float(), ids).mean().item()
    finally:
        if h: h.remove()
def resid_at(rev, L, ids):
    assemble(rev, rev, 0); out = {}
    def pre(m, args, kwargs):
        x = args[0] if len(args) > 0 else kwargs["hidden_states"]; out["x"] = x.detach().float().reshape(-1, x.shape[-1]); return None
    h = layers[L].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        with torch.no_grad(): model(ids)
    finally: h.remove()
    return out["x"]
def mapper(W):
    def pre(m, args, kwargs):
        if len(args) > 0: x = args[0]; y = (torch.cat([x, torch.ones_like(x[..., :1])], -1) @ W).to(x.dtype); return (y,) + tuple(args[1:]), kwargs
        kw = dict(kwargs); x = kw["hidden_states"]; kw["hidden_states"] = (torch.cat([x, torch.ones_like(x[..., :1])], -1) @ W).to(x.dtype); return args, kw
    return pre
full = {}
for r in revs: assemble(r, r, 0); full[r] = loss_of(ev)
R = {(r, L): resid_at(r, L, fit) for r in revs for L in cuts}
raw, lin = {}, {}
for L in cuts:
    for lo in revs:
        for up in revs:
            if lo == up: raw[(L, lo, up)] = lin[(L, lo, up)] = full[lo]; continue
            assemble(lo, up, L); raw[(L, lo, up)] = loss_of(ev)
            Xa = R[(lo, L)]; Xb = R[(up, L)]; Xa1 = torch.cat([Xa, torch.ones_like(Xa[:, :1])], 1); W = torch.linalg.lstsq(Xa1, Xb).solution
            lin[(L, lo, up)] = loss_of(ev, (L, mapper(W)))
def decompose(tab, L):
    M = torch.tensor([[tab[(L, lo, up)] for up in revs] for lo in revs]); g = M.mean(); a = M.mean(1, keepdim=True) - g; b = M.mean(0, keepdim=True) - g; tot = ((M - g) ** 2).sum()
    return dict(share_lower=((a ** 2).sum() * len(revs) / tot).item(), share_upper=((b ** 2).sum() * len(revs) / tot).item(), share_interaction=(((M - g - a - b) ** 2).sum() / tot).item(), matrix=M.tolist())
res = dict(revisions=revs, cuts=cuts, full=full, raw={str(L): decompose(raw, L) for L in cuts}, linear={str(L): decompose(lin, L) for L in cuts})
fin = revs[-1]; res["old_lower_final_upper"] = {str(L): {lo: raw[(L, lo, fin)] - full[fin] for lo in revs[:-1]} for L in cuts}; res["final_lower_old_upper"] = {str(L): {up: raw[(L, fin, up)] - full[up] for up in revs[:-1]} for L in cuts}
res["old_lower_final_upper_linear"] = {str(L): {lo: lin[(L, lo, fin)] - full[fin] for lo in revs[:-1]} for L in cuts}; res["final_lower_old_upper_linear"] = {str(L): {up: lin[(L, fin, up)] - full[up] for up in revs[:-1]} for L in cuts}
log("full losses " + " ".join(f"{r} {v:.3f}" for r, v in full.items()))
for L in cuts:
    rw, ln = res["raw"][str(L)], res["linear"][str(L)]
    log(f"cut {L}: RAW variance share lower {rw['share_lower']:.2f} upper {rw['share_upper']:.2f} interaction {rw['share_interaction']:.2f} | LINEAR lower {ln['share_lower']:.2f} upper {ln['share_upper']:.2f} interaction {ln['share_interaction']:.2f} | old lower + final upper, excess over final: " + " ".join(f"{lo} {v:+.3f}" for lo, v in res["old_lower_final_upper"][str(L)].items()) + " | final lower + old upper, excess over that upper's own model: " + " ".join(f"{up} {v:+.3f}" for up, v in res["final_lower_old_upper"][str(L)].items()) + " | linear-stitched old lower + final upper: " + " ".join(f"{lo} {v:+.3f}" for lo, v in res["old_lower_final_upper_linear"][str(L)].items()))
record("e378_stitchtime_pythia410", res, " | ".join(f"cut {L}: raw upper/lower share {res['raw'][str(L)]['share_upper']:.2f}/{res['raw'][str(L)]['share_lower']:.2f}, linear {res['linear'][str(L)]['share_upper']:.2f}/{res['linear'][str(L)]['share_lower']:.2f}" for L in cuts))
