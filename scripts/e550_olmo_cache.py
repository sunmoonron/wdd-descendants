"""e550: the cache for a second model. Everything from session 72 on is Pythia-410m; before any intervention or write-up
the chain of sessions 89-97 needs a second family, and OLMo-1B (a different architecture, tokenizer and corpus) has
checkpoints every thousand steps at about two billion tokens a step, so its steps 1000-16000 cover the same two to
thirty-three billion tokens as Pythia's. At each of those sixteen checkpoints, block 8 of 16 (the middle, as block 12
of 24 was), the 8 x 256 evaluation tokens: the block's output states at every position, the sink mask, the unit centred
states; the MLP write rows of blocks 0-8 (unit, with norms); the activations of every MLP writer of blocks 0-8 at every
position, the summed attention outputs and the embedding, so that the exits, the coalitions, the precedence and the
necessity test can be run from the cache. Each checkpoint's weights are removed from the download cache after use.
Arguments: none."""
import sys, os, json as _json, re, shutil; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
from huggingface_hub import list_repo_refs, scan_cache_dir
name = "olmo1b"; B = 8; STEPS = list(range(1000, 16001, 1000)); CDIR = "/workspace/wdd/cache/e550_olmo1b"; os.makedirs(CDIR, exist_ok=True); REPO = "allenai/OLMo-1B-0724-hf"
refs = {int(re.search(r"step(\d+)", b.name).group(1)): b.name for b in list_repo_refs(REPO).branches if re.search(r"step(\d+)-tokens", b.name)}
idsB = eval_ids(name)[:8, :256].to(DEV)
class Stop(Exception): pass
def capture(model, arch, fam):
    cap = {"act": {}, "attn": {}, "out": None, "emb": None}; layers = arch.layers; attn_mod = lambda bb: layers[bb].attention if fam == "neox" else (layers[bb].attn if fam == "gpt2" else layers[bb].self_attn); emb_mod = model.gpt_neox.embed_in if fam == "neox" else (model.transformer.wte if fam == "gpt2" else model.model.embed_tokens)
    def h_attn(bb):
        def f(m, i, o): cap["attn"][bb] = (o[0] if isinstance(o, tuple) else o).detach().float()
        return f
    def h_act(bb):
        def f(m, a): cap["act"][bb] = a[0].detach().float(); return None
        return f
    def h_out(m, i, o): cap["out"] = (o[0] if isinstance(o, tuple) else o).detach().float(); raise Stop
    def h_emb(m, i, o): cap["emb"] = o.detach().float()
    hs = [attn_mod(bb).register_forward_hook(h_attn(bb)) for bb in range(B + 1)] + [arch.mlp_lin(bb).register_forward_pre_hook(h_act(bb)) for bb in range(B + 1)] + [emb_mod.register_forward_hook(h_emb), layers[B].register_forward_hook(h_out)]
    try: model(idsB)
    except Stop: pass
    finally: [h.remove() for h in hs]
    return cap
def flat(t): return t.reshape(8, 256, -1)[:, 1:].reshape(2040, -1)
for n in STEPS:
    out = f"{CDIR}/step{n}.pt"
    if os.path.exists(out): log(f"step{n} cached"); continue
    rev = refs[n]; model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF
    for p_ in model.parameters(): p_.requires_grad_(False)
    cap = capture(model, arch, fam); Wv = torch.cat([arch.wdir(bb).float() for bb in range(B + 1)]); X = flat(cap["out"]); keep = ~sinkmask(X); mu = X[keep].mean(0)
    ACT = torch.cat([flat(cap["act"][bb]) for bb in range(B + 1)], 1); ATT = torch.stack([flat(cap["attn"][bb]) for bb in range(B + 1)], 1).sum(1); EMB = flat(cap["emb"])
    rec_ok = float(((EMB + ATT + torch.cat([ACT[:, bb * DFF:(bb + 1) * DFF] @ Wv[bb * DFF:(bb + 1) * DFF] for bb in range(B + 1)], 0).reshape(B + 1, 2040, D).sum(0)) * X).sum(1).div((X.norm(dim=1) ** 2).clamp_min(1e-9)).median())
    torch.save(dict(step=n, revision=rev, block=B, D=D, DFF=DFF, rows=unitr(Wv).half().cpu(), norms=Wv.norm(dim=1).cpu(), X=X.cpu(), keep=keep.cpu(), U=unitr(X - mu).half().cpu(), act=ACT.half().cpu(), att=ATT.cpu(), emb=EMB.cpu()), out)
    log(f"{name} step{n} ({rev}): {int(keep.sum())} kept positions of 2040, {Wv.shape[0]} rows of dimension {D}; state from its writes (relative projection, biases apart) {rec_ok:.3f}")
    del model, cap, Wv, ACT, ATT, EMB, X; torch.cuda.empty_cache()
    try:
        info = scan_cache_dir(); rmv = [r.commit_hash for repo in info.repos if repo.repo_id == REPO for r in repo.revisions if rev in r.refs]
        if rmv: info.delete_revisions(*rmv).execute(); log(f"removed the download of {rev}")
    except Exception as e: log(f"could not remove the download of {rev}: {e}")
summ = f"{name}: cached steps {STEPS[0]}-{STEPS[-1]} at block {B} in {CDIR}"; log(summ); record(f"e550_olmo_cache", dict(model=name, steps=STEPS, revisions={n: refs[n] for n in STEPS}, cache=CDIR, block=B), summ)
