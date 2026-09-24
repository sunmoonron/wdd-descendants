"""e447b: e447 without the overfitting. The e447 control (lambda 0) lost 0.41 nats of held-out loss by fitting 64 sequences for 7.5 epochs,
which swamps any cost of the self-description term. Here the fine-tuning text is 512 sequences of 256 tokens sampled from GPT-2 itself (temperature 1),
so the LM term stays near its optimum and the self-description term drives the change; held-out loss is still measured on real text.
Original description follows.
e447: can self-describability be trained in, and at what cost? (vision: interpretable by design with a free
dictionary)
In a world that had WDD for a decade, models could be trained so that their states are sparse in their own words, just
as SAEs are trained on frozen models today, but with no second network: the dictionary is the model's own write rows.
Relevant earlier results:
- self-description emerges on its own in pretraining (e395, e401) and tracks capability (e427);
- instruction tuning barely moves the words (e431).
So the LM objective and self-describability may be aligned, and a small push might be cheap.
GPT-2 small fine-tuned for 120 steps (4 x 512 tokens per step, AdamW lr 3e-5) on 64 sequences, with loss =
LM + lambda * (fraction of the middle-depth state left unexplained by its 16-word own description). The support is
chosen by OMP without gradient; the least-squares fit on it is differentiable in both the states and the words.
Words used for training: token and position embeddings, attention output rows and MLP write rows of blocks 0..L.
lambda (argument): 0 (fine-tuning control), 0.3, 1, 3.
Evaluated on 6 held-out sequences with the standard dictionary (head bases by SVD):
- LM loss;
- loss recovered at k 4 and 16 by own words, rotation and covA;
- where the change lives: new states with the old words, and old states with the new words;
- how far the write rows moved.
Pre-registered (honest guesses):
 S1 (0.6) at lambda 1 the own-over-rotation gap at k 16 grows by at least 0.05 over lambda 0, for at most 0.05 nats
          of held-out LM loss;
 S2 (0.5) the gain lives mostly in the states (new states with old words gain more than old states with new words);
 S3 (0.5) at lambda 3 the LM cost exceeds 0.1 nats."""
import sys, os, copy; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
import torch.nn.functional as F
lam = float(sys.argv[1]); name = "gpt2"
model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
base = copy.deepcopy(model).eval(); [q.requires_grad_(False) for q in base.parameters()]; barch = Arch(base, fam)
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; te = EA["eval_ids"][:6].to(DEV); prompts = EA["cen_ids"][:, :8].to(DEV); del EA
torch.manual_seed(0); gen = []
with torch.no_grad():
    for s0 in range(0, 512, 64):
        pr = prompts[torch.arange(s0, s0 + 64) % prompts.shape[0]]
        gen.append(base.generate(pr, max_new_tokens=248, min_new_tokens=248, do_sample=True, temperature=1.0, top_k=0, pad_token_id=tok.eos_token_id))
tr = torch.cat(gen)[:, :256]
def train_words(m):
    h = m.transformer.h
    return torch.cat([m.transformer.wte.weight, m.transformer.wpe.weight] + [h[b].attn.c_proj.weight for b in range(L + 1)] + [h[b].mlp.c_proj.weight for b in range(L + 1)])
opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.0); g = torch.Generator().manual_seed(0); order = torch.randperm(tr.shape[0], generator=g)
torch.set_grad_enabled(True); hist = []                                   # eval mode: no dropout, deterministic states
for step in range(120):
    ids = tr[order[(step * 4) % tr.shape[0]:(step * 4) % tr.shape[0] + 4]]; cap = {}
    h = arch.layers[L].register_forward_hook(lambda m, i, o: cap.__setitem__("x", out_of(o)))
    try: lg = model(ids).logits
    finally: h.remove()
    lm = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), ids[:, 1:].reshape(-1))
    x = cap["x"][:, 1:].reshape(-1, D); keep = ~sinkmask(x.detach().view(ids.shape[0], -1, D)).reshape(-1); xc = x[keep] - x[keep].mean(0)
    Wd = train_words(model); Dm = Wd / Wd.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    with torch.no_grad(): sel, _, _ = omp(xc.detach(), Dm.detach(), 16, batch=512, record_err=False)
    As = Dm[sel]; G = As @ As.transpose(1, 2) + 1e-4 * torch.eye(16, device=DEV); c = torch.linalg.solve(G, As @ xc[..., None]); r = xc - (c.transpose(1, 2) @ As)[:, 0]
    lsd = (r.pow(2).sum(-1) / xc.pow(2).sum(-1).clamp_min(1e-6)).mean(); loss = lm + lam * lsd
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 20 == 0 or step == 119: hist.append(dict(step=step, lm=lm.item(), unexplained=lsd.item())); log(f"lambda {lam} step {step}: LM {lm.item():.3f}, unexplained {lsd.item():.3f}")
    del lg, cap, x, xc, As, G, c, r
torch.set_grad_enabled(False); model.eval()
res = dict(lam=lam, train=hist)
A_new, _ = build_dictionary(arch, blocks=list(range(L + 1))); A_old, _ = build_dictionary(barch, blocks=list(range(L + 1)))
lv_new, lv_old = NSLevel(model, arch, te, L), NSLevel(base, barch, te, L)
cells = dict(own=describe(lv_new, A_new, [4, 16]), rot=describe(lv_new, rotate(A_new, seed=7), [4, 16]), covA=describe(lv_new, gauss_like(A_new.shape[0], (A_new.T @ A_new) / A_new.shape[0], seed=1), [4, 16]),
             new_states_old_words=describe(lv_new, A_old, [4, 16]), old_states_new_words=describe(lv_old, A_new, [4, 16]), old_own=describe(lv_old, A_old, [4, 16]), old_rot=describe(lv_old, rotate(A_old, seed=7), [4, 16]))
r_ = lambda c, k: cells[c][str(k)]["rec"]
mlp_cos = torch.cat([(unitr(arch.wdir(b)) * unitr(barch.wdir(b))).sum(-1) for b in range(L + 1)])
res.update(test_lm_new=lv_new.lc.mean().item(), test_lm_old=lv_old.lc.mean().item(), cells={c: {k: v["rec"] for k, v in d_.items()} for c, d_ in cells.items()},
           fvu={c: {k: v["fvu"] for k, v in d_.items()} for c, d_ in cells.items()}, gap_new={k: r_("own", k) - r_("rot", k) for k in (4, 16)}, gap_old={k: r_("old_own", k) - r_("old_rot", k) for k in (4, 16)},
           covA_share_new=(r_("covA", 16) - r_("rot", 16)) / max(r_("own", 16) - r_("rot", 16), 1e-9), mlp_row_cos_median=mlp_cos.median().item(), mlp_row_cos_p01=torch.quantile(mlp_cos, 0.01).item())
summ = (f"gpt2 (self-generated text) lambda {lam}: held-out LM {res['test_lm_old']:.3f} -> {res['test_lm_new']:.3f} | k4/k16 own {r_('own', 4):.2f}/{r_('own', 16):.2f} rot {r_('rot', 4):.2f}/{r_('rot', 16):.2f} covA {r_('covA', 4):.2f}/{r_('covA', 16):.2f} "
        f"(before: own {r_('old_own', 4):.2f}/{r_('old_own', 16):.2f} rot {r_('old_rot', 4):.2f}/{r_('old_rot', 16):.2f}) | gap k16 {res['gap_old'][16]:+.3f} -> {res['gap_new'][16]:+.3f}, k4 {res['gap_old'][4]:+.3f} -> {res['gap_new'][4]:+.3f} | "
        f"new states + old words {r_('new_states_old_words', 4):.2f}/{r_('new_states_old_words', 16):.2f}, old states + new words {r_('old_states_new_words', 4):.2f}/{r_('old_states_new_words', 16):.2f} | "
        f"FVU k16 own {res['fvu']['own']['16']:.3f} (before {res['fvu']['old_own']['16']:.3f}) | covA share {res['covA_share_new']:.2f} | MLP rows cos median {res['mlp_row_cos_median']:.4f}, 1st pct {res['mlp_row_cos_p01']:.3f} | train unexplained {hist[0]['unexplained']:.3f} -> {hist[-1]['unexplained']:.3f}")
log(summ); record(f"e447b_trainsd_selfgen_gpt2_lam{lam}", res, summ)
