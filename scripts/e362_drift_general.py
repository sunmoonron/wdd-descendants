"""e362: the e360 drift measurement for any model with public checkpoints (OLMo-1B-0724 here); blocks 2, NB/3 and 2NB/3; per-neuron drift across training, measured without aligning residual bases. For one checkpoint,
for every MLP neuron of blocks 2, 8 and 16: its write and read vectors, its mean activation magnitude and its loss
attribution (mean |act x dL/dact|) on 12 eval sequences, and its transplant logit signature: the unit write injected
after its block at 5% of the median residual norm at every token (1024 neurons per pass, about 36 tokens per neuron),
the centred logit change averaged per neuron and projected onto a fixed random 256-dimensional sketch of the
vocabulary shared by all checkpoints. Also the eval loss and the top-1 predictions. Saves results/e360/<rev>.pt;
e360b compares checkpoints."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True  # drift signatures are forward-pass sketches; TF32 error (~1e-3 relative) is far below the injected effect
tag, rev = sys.argv[1], sys.argv[2]; SEED = sys.argv[3] if len(sys.argv) > 3 else None; SUF = f"_{SEED}" if SEED else ""; ODIR = f"e362_{tag}" + ("_ceil" if SEED else "")
if SEED: torch.manual_seed(int(SEED)); torch.cuda.manual_seed(int(SEED))
model, tok, fam = load_model(tag, revision=rev); arch = Arch(model, fam); c = Cache(tag); NB, D, DFF = arch.NB, arch.D, arch.DFF; blocks = [2, NB // 3, 2 * NB // 3]
ids = c.s["eval_ids"][:12].to(DEV); NS, T = ids.shape; NT = NS * T; acts, rn = {}, {}
def mk(l):
    def pre(m, a): a[0].retain_grad(); acts[l] = a[0]; return None
    return pre
hs = [arch.mlp_lin(l).register_forward_pre_hook(mk(l)) for l in blocks] + [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: rn.__setitem__(l_, (o[0] if isinstance(o, tuple) else o).detach().float().norm(dim=-1).median()))(l)) for l in blocks]
with torch.enable_grad():
    lg = model(ids).logits.float(); loss = token_loss(lg, ids).mean(); loss.backward()
[h.remove() for h in hs]
attr = {l: (acts[l].detach().float() * acts[l].grad.float()).abs().mean((0, 1)).cpu() for l in blocks}; actmag = {l: acts[l].detach().float().abs().mean((0, 1)).cpu() for l in blocks}; model.zero_grad(set_to_none=True); del acts
lg0 = lg.detach(); top1 = lg0[:, :-1].argmax(-1).cpu(); V = lg0.shape[-1]; g = torch.Generator().manual_seed(1234); R = (torch.randn(V, 256, generator=g) / 16).to(DEV)
P = torch.nonzero((torch.arange(NT, device=DEV) % T) != 0)[:, 0]; lg0f = lg0.reshape(NT, V); sig, cnt = {}, {}
def run_inj(inj, blk):
    def pre(m, args, kwargs):
        if len(args) > 0: return (args[0] + inj.reshape(args[0].shape),) + tuple(args[1:]), kwargs
        kw = dict(kwargs); kw["hidden_states"] = kw["hidden_states"] + inj.reshape(kw["hidden_states"].shape); return args, kw
    h = arch.layers[blk].register_forward_pre_hook(pre, with_kwargs=True)
    try: return model(ids).logits.float().reshape(NT, V)
    finally: h.remove()
for l in blocks:
    Wn = unit(arch.wdir(l).to(DEV)); amp = 0.05 * rn[l]; S = torch.zeros(DFF, 256, device=DEV); Cn = torch.zeros(DFF, device=DEV)
    for ch in range(DFF // 1024):
        for p in range(6):
            a = ch * 1024 + torch.randint(0, 1024, (len(P),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[P] = amp * Wn[a]; d = (run_inj(inj, l + 1) - lg0f)[P]; d = d - d.mean(-1, keepdim=True)
            S.index_add_(0, a, (d @ R) / amp); Cn.index_add_(0, a, torch.ones(len(P), device=DEV))
    sig[l] = (S / Cn[:, None].clamp_min(1)).half().cpu(); cnt[l] = Cn.cpu()
read = {l: (arch.layers[l].mlp.dense_h_to_4h.weight if fam == "neox" else (arch.layers[l].mlp.c_fc.weight.T if fam == "gpt2" else arch.layers[l].mlp.up_proj.weight)).detach().float().half().cpu() for l in blocks}; write = {l: arch.wdir(l).half().cpu() for l in blocks}
os.makedirs(os.path.join(RESULTS, ODIR), exist_ok=True); torch.save(dict(rev=rev, blocks=blocks, sig=sig, cnt=cnt, write=write, read=read, attr=attr, actmag=actmag, top1=top1, eval_loss=loss.item(), resnorm={l: rn[l].item() for l in blocks}), os.path.join(RESULTS, ODIR, f"{rev}{SUF}.pt"))
log(f"{rev}: eval loss {loss.item():.3f}; tokens per neuron (median) " + ", ".join(f"block {l} {cnt[l].median().item():.0f}" for l in blocks))
record(f"e362_drift_{tag}_{rev}{SUF}", dict(rev=rev, eval_loss=loss.item(), resnorm={str(l): rn[l].item() for l in blocks}), f"eval loss {loss.item():.3f}")
