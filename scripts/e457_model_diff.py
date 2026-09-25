"""e457: can WDD describe what fine-tuning changed, write by write? This tests WDD as a model-diff tool, suggested by an
external review. e431 compared the dictionaries and found the write rows almost unchanged (cosine 0.997). This
experiment looks at the states instead.
Setup: base and instruct pairs (Qwen2.5-0.5B, SmolLM2-135M) on the same inputs (e431's chat-formatted conversations and
natural text), at the middle depth. For every token, d = x_instruct - x_base is the fine-tune's change of the state
(sinks excluded in either model).
- Is the change sparse in the model's own words? The centred per-token differences are described with k = 4 and 16 base
  words (OMP). The baselines are rotated words and the top-k principal directions of the differences themselves (fitted
  on the same differences, an optimistic dense baseline).
- Is the change concentrated in few writers? Each MLP word's selection frequency in the 16-word descriptions of the two
  models' states is compared, each model with its own words. Reported: the share of the total absolute usage change
  carried by the top 1% and top 10% of words, and the words that change most.
- Is it chat-specific? The mean norm of d relative to the state norm, chat against natural text.
Pre-registered (honest guesses):
- native words describe the differences better than rotated words at k 16 (0.6);
- the top 1% of words carry at least 30% of the usage change (0.5);
- the change is larger on chat than on natural text (0.7)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
from transformers import AutoModelForCausalLM, AutoTokenizer
base = sys.argv[1]; INST = {"qwen05": "Qwen/Qwen2.5-0.5B-Instruct", "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct"}[base]
mb, tok, fam = load_model(base); ab = Arch(mb, fam); L = ab.NB // 2
ti = AutoTokenizer.from_pretrained(INST); mi = AutoModelForCausalLM.from_pretrained(INST, dtype=torch.float32, device_map=DEV).eval(); ai = Arch(mi, fam)
QA = [("What is the capital of France?", "The capital of France is Paris."), ("How many legs does a spider have?", "A spider has eight legs."),
      ("Can you suggest a name for a cat?", "Sure, how about Whiskers?"), ("What is 12 times 7?", "12 times 7 is 84."),
      ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."), ("Why is the sky blue?", "Because air scatters blue light more than red light."),
      ("Give me a synonym for happy.", "Joyful."), ("What is the boiling point of water?", "Water boils at 100 degrees Celsius at sea level."),
      ("Translate 'thank you' into Spanish.", "Gracias."), ("What is the largest planet?", "Jupiter is the largest planet in our solar system.")]
rows = []
for s in range(8):
    msgs = []
    for q, a in QA[s % 10:] + QA[:s % 10]: msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    r = ti.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False); rows.append(list(r["input_ids"]) if not isinstance(r, list) else r)
T = min(256, min(len(r) for r in rows)); chat = torch.tensor([r[:T] for r in rows]).to(DEV); nat = eval_ids(base)[:8, :256].to(DEV)
Vb, lab = build_dictionary(ab, blocks=list(range(L + 1))); Vi, _ = build_dictionary(ai, blocks=list(range(L + 1))); isM = (lab["type"] == T_MLP).to(DEV)
blkv, idxv = lab["block"], lab["index"]
Q = torch.linalg.qr(torch.randn(ab.D, ab.D, generator=torch.Generator().manual_seed(11)))[0].to(DEV)
def fvu(X, Dct, ks=(4, 16)):
    sel, _, _ = omp(X, Dct, max(ks), batch=512, record_err=False); tot = X.pow(2).sum(); out = {}
    for k in ks: _, err = refit(X, Dct, sel[:, :k]); out[k] = (err.sum() / tot).item()
    return out
def pca_fvu(X, ks=(4, 16)):
    ev_, U_ = torch.linalg.eigh((X.T @ X).double()); U_ = U_.flip(-1).float(); tot = X.pow(2).sum(); out = {}
    for k in ks: P = U_[:, :k]; out[k] = ((X - X @ P @ P.T).pow(2).sum() / tot).item()
    return out
res = dict(base=base, instruct=INST, level=L, texts={})
for tn, ids in (("chat", chat), ("natural", nat)):
    xb = block_states(mb, ab, ids, [L], chunk=2)[L].reshape(-1, ab.D); xi = block_states(mi, ai, ids, [L], chunk=2)[L].reshape(-1, ab.D)
    keep = ~sinkmask(xb) & ~sinkmask(xi); xb, xi = xb[keep], xi[keep]; d = xi - xb
    rel = (d.norm(dim=-1) / (xb - xb.mean(0)).norm(dim=-1).clamp_min(1e-9)).mean().item(); mean_shift = (d.mean(0).norm() / (xb - xb.mean(0)).norm(dim=-1).mean()).item()
    dc = d - d.mean(0)
    f_own, f_rot, f_pca = fvu(dc, unitr(Vb)), fvu(dc, unitr(Vb) @ Q), pca_fvu(dc)
    ub = torch.bincount(omp(xb - xb.mean(0), unitr(Vb), 16, batch=512, record_err=False)[0].flatten(), minlength=Vb.shape[0]).float() / xb.shape[0]
    ui = torch.bincount(omp(xi - xi.mean(0), unitr(Vi), 16, batch=512, record_err=False)[0].flatten(), minlength=Vi.shape[0]).float() / xi.shape[0]
    du = (ui - ub).abs()[isM]; srt = du.sort(descending=True).values; n = srt.numel()
    top = torch.nonzero(isM)[:, 0][(ui - ub).abs()[isM].argsort(descending=True)[:5]].tolist()
    out = dict(n_tokens=int(keep.sum()), rel_change_norm=rel, mean_shift_rel=mean_shift, fvu_own=f_own, fvu_rot=f_rot, fvu_pca_of_diffs=f_pca,
               usage_change_total=float(du.sum() / 2), top1pct_share=float(srt[:max(1, n // 100)].sum() / srt.sum().clamp_min(1e-12)),
               top10pct_share=float(srt[:max(1, n // 10)].sum() / srt.sum().clamp_min(1e-12)),
               most_changed=[dict(block=int(blkv[a]), index=int(idxv[a]), base_use=float(ub[a]), instruct_use=float(ui[a])) for a in top])
    res["texts"][tn] = out
    log(f"{base} {tn}: |d|/|x| {rel:.3f} (mean shift {mean_shift:.3f}) | diffs unexplained k4/k16: own {f_own[4]:.2f}/{f_own[16]:.2f} rot {f_rot[4]:.2f}/{f_rot[16]:.2f} pca-of-diffs {f_pca[4]:.2f}/{f_pca[16]:.2f} | "
        f"usage change {out['usage_change_total']:.2f} of 16, top 1% of words carry {out['top1pct_share']:.2f}, top 10% {out['top10pct_share']:.2f} | most changed {[(m_['block'], m_['index'], round(m_['base_use'], 3), round(m_['instruct_use'], 3)) for m_ in out['most_changed'][:3]]}")
C_, N_ = res["texts"]["chat"], res["texts"]["natural"]
res["checks"] = dict(own_beats_rot_k16=C_["fvu_own"][16] < C_["fvu_rot"][16], top1pct_over_0_3=C_["top1pct_share"] >= 0.3, chat_changes_more=C_["rel_change_norm"] > N_["rel_change_norm"])
f_ = lambda o: (f"|d|/|x| {o['rel_change_norm']:.3f}, diffs unexplained at k16 own {o['fvu_own'][16]:.2f} rot {o['fvu_rot'][16]:.2f} pca {o['fvu_pca_of_diffs'][16]:.2f} (k4 {o['fvu_own'][4]:.2f}/{o['fvu_rot'][4]:.2f}/{o['fvu_pca_of_diffs'][4]:.2f}), "
                f"usage change {o['usage_change_total']:.2f}/16, top 1% of words {o['top1pct_share']:.2f}, top 10% {o['top10pct_share']:.2f}")
summ = f"{base} vs instruct, L{L}: chat: " + f_(C_) + " | natural: " + f_(N_) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e457_modeldiff_{base}", res, summ)
