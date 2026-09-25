"""e462: how does a gradient update reshape a native word? This is the microscopic side of "the vocabulary is written",
suggested by an external review.
By the chain rule, the loss gradient on writer j's row (its MLP output column) is sum_t a_j(t) delta_t: its activations
times the gradient at the residual stream. So inactive writers receive almost none. The open questions are where the
update points, and how that changes over training:
- along its own row (the word grows or shrinks), or orthogonal to it (the word rotates)?
- does one batch's gradient predict the row's actual change between two checkpoints?
- do the words the model uses to describe its states (forward, WDD) get the most learning pressure (backward)?
Per checkpoint, on 4 x 512 tokens of evaluation text (next-token loss, fp32), for every MLP writer of every block:
- the gradient on its row, g_j;
- the along-row share of |g_j|^2, compared with chance 1/D;
- whether descent lengthens the row (-g_j . w_j > 0);
- its mean absolute activation.
With a later checkpoint (second argument), the net change of each row, dw_j, gives the along-row share of |dw_j|^2
and cos(-g_j, dw_j).
Also: each writer's usage in the 16-word native descriptions of the batch's states at the middle depth (blocks up to L)
against its gradient norm and along-row share (Spearman).
Runs (arguments):
- pythia410 at step1000 (next step4000), step4000 (next step16000), step16000 (next step64000), step64000 (next main);
- gpt2 (no next checkpoint).
Pre-registered (honest guesses):
- the gradient on an active writer's row is close to orthogonal to it (along-row share under 10 times chance) at every
  checkpoint (0.5);
- the net change between checkpoints is more along the row late than early (0.5);
- one batch's gradient predicts the net change's direction early (median cosine above 0.1) but not late (0.4);
- native usage and gradient norm correlate across writers (Spearman above 0.3) (0.6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
from transformers import AutoModelForCausalLM
name = sys.argv[1]; arg = lambda i: (sys.argv[i] if len(sys.argv) > i and sys.argv[i] not in ("none", "final") else None); rev, nxt = arg(2), arg(3)
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
ids = eval_ids(name)[:4, :512].to(DEV)
for p in model.parameters(): p.requires_grad_(False)
W = [arch.mlp_lin(b).weight for b in range(arch.NB)]
for w in W: w.requires_grad_(True)
act = {b: torch.zeros(arch.DFF, device=DEV) for b in range(arch.NB)}; hs = []
def mk_pre(b):
    def pre(m, a):
        act[b].add_(a[0].detach().abs().float().reshape(-1, a[0].shape[-1]).sum(0)); return None     # a pre-hook must return None, or it replaces the input
    return pre
for b in range(arch.NB):
    hs.append(arch.mlp_lin(b).register_forward_pre_hook(mk_pre(b)))
cap = {}
hs.append(arch.layers[L].register_forward_hook(lambda m, i, o: cap.__setitem__("x", out_of(o).detach().float())))
torch.set_grad_enabled(True)
lg = model(ids).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)); loss.backward()
torch.set_grad_enabled(False); [h.remove() for h in hs]; ntok = ids.numel()
def rows_of(t):                                                                    # the weight's layout -> [DFF, D] rows
    return t.T if t.shape[0] == D else t
G = [rows_of(w.grad.detach().float()) for w in W]; R = [rows_of(w.detach().float()) for w in W]
chance = 1.0 / D; out = dict(model=name, revision=rev, next_revision=nxt, loss=loss.item(), chance=chance, blocks={})
allf, allgrow, alla, allg = [], [], [], []
for b in range(arch.NB):
    g, r = G[b], R[b]; rn = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12); along = (g * rn).sum(-1)
    f = along.pow(2) / g.pow(2).sum(-1).clamp_min(1e-30); a = act[b] / ntok
    allf.append(f); allgrow.append(-along > 0); alla.append(a); allg.append(g.norm(dim=-1))
F_, GR, A_, GN = torch.cat(allf), torch.cat(allgrow), torch.cat(alla), torch.cat(allg)
top = A_ >= torch.quantile(A_, 0.9)
def spearman(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()))
out["along_share_over_chance"] = dict(active_median=float(F_[top].median() / chance), others_median=float(F_[~top].median() / chance), active_mean=float(F_[top].mean() / chance))
out["grow_share_active"] = float(GR[top].float().mean()); out["spearman_activity_gradnorm"] = spearman(A_, GN)
# forward usage at the middle depth (writers of blocks 0..L) against backward pressure
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); isM = (lab["type"] == T_MLP).to(DEV)
X = cap["x"][:, 1:].reshape(-1, D); keep = ~sinkmask(X); Xc = X[keep] - X[keep].mean(0)
sel, _, _ = omp(Xc, unitr(A), 16, batch=512, record_err=False)
use = torch.bincount(sel.flatten(), minlength=A.shape[0]).float()[isM] / Xc.shape[0]      # blocks 0..L in order, as G
nM = use.numel(); GNL, FL, AL = GN[:nM], F_[:nM], A_[:nM]
out["spearman_usage_gradnorm"] = spearman(use, GNL); out["spearman_usage_along"] = spearman(use, FL); out["spearman_usage_activity"] = spearman(use, AL)
used = use > 0; out["gradnorm_used_over_unused"] = float(GNL[used].median() / GNL[~used].median().clamp_min(1e-30))
del A, lab, sel
if nxt:
    m2 = AutoModelForCausalLM.from_pretrained(MODELS[name][0], revision=nxt, dtype=torch.float32, device_map=DEV).eval(); a2 = Arch(m2, fam)
    R2 = [rows_of(a2.mlp_lin(b).weight.detach().float()) for b in range(arch.NB)]; del m2
    dW = torch.cat([R2[b] - R[b] for b in range(arch.NB)]); Rall = torch.cat(R); Gall = torch.cat(G)
    rn = Rall / Rall.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    fnet = (dW * rn).sum(-1).pow(2) / dW.pow(2).sum(-1).clamp_min(1e-30)
    cosg = (-Gall * dW).sum(-1) / (Gall.norm(dim=-1) * dW.norm(dim=-1)).clamp_min(1e-30)
    rel = dW.norm(dim=-1) / Rall.norm(dim=-1).clamp_min(1e-12)
    out["net"] = dict(along_share_over_chance_active=float(fnet[top].median() / chance), along_share_over_chance_all=float(fnet.median() / chance),
                      cos_grad_net_active=float(cosg[top].median()), cos_grad_net_all=float(cosg.median()), rel_change_active=float(rel[top].median()),
                      grow_share_active=float(((dW * Rall).sum(-1) > 0)[top].float().mean()))
o = out; n_ = o.get("net")
o["checks"] = dict(orthogonal_under_10x=o["along_share_over_chance"]["active_median"] < 10, usage_gradnorm_over_0_3=o["spearman_usage_gradnorm"] > 0.3)
summ = (f"{name} {rev or 'final'} (loss {o['loss']:.3f}): gradient on a writer's row, along-row share over chance (1/{D}): active median {o['along_share_over_chance']['active_median']:.1f}, others {o['along_share_over_chance']['others_median']:.1f}; "
        f"descent lengthens {o['grow_share_active']:.2f} of active rows; Spearman activity-gradnorm {o['spearman_activity_gradnorm']:+.2f} | forward usage vs backward pressure (blocks 0..{L}): "
        f"Spearman usage-gradnorm {o['spearman_usage_gradnorm']:+.2f}, usage-along {o['spearman_usage_along']:+.2f}, usage-activity {o['spearman_usage_activity']:+.2f}, used/unused gradnorm {o['gradnorm_used_over_unused']:.1f}"
        + (f" | net change to {nxt}: along-row share over chance, active {n_['along_share_over_chance_active']:.1f} (all {n_['along_share_over_chance_all']:.1f}), cos(-grad, net) active {n_['cos_grad_net_active']:+.3f} (all {n_['cos_grad_net_all']:+.3f}), relative change {n_['rel_change_active']:.2f}, lengthened {n_['grow_share_active']:.2f}" if n_ else "")
        + f" | checks {_json.dumps(o['checks'])}")
log(summ); record(f"e462_gradwriters_{name}_{rev or 'final'}", out, summ)
