"""e605 (session 112): the merging audit. Two fine-tunes of Pythia-160m, one on Github and one on PubMed Abstracts
(one epoch over up to 640 documents each at lr 1e-5, as e600), then merged by averaging their weights (task
arithmetic with equal weights), against sequential fine-tuning (Github then PubMed). For each domain the audit of
e600: the words on the domain's held-out states in the original, the entrants each fine-tune recruits, and in the
merged and sequential models whether those entrants still write at their classes (the entrants' still-writing
share) and whether the two fine-tunes' entrants collide (the same rows recruited by both, and rows whose classes
are mixed). Losses on both domains and on a control domain. Controls: each fine-tune alone, and a merge of one
fine-tune with the original (which should preserve that fine-tune's entrants at half strength). Argument: --smoke.
Pre-registered (honest guesses):
 M1 (0.6) the two fine-tunes' entrants overlap by Jaccard 0.2 or less (different rows for different domains);
 M2 (0.5) in the merged model at least 0.6 of each fine-tune's entrants still write at their classes, against
    0.3 or less of the Github entrants after sequential fine-tuning on PubMed (merging keeps what sequence loses);
 M3 (0.6) the merged model's loss on each domain is within 0.1 nats of the corresponding fine-tune's."""
from s101_common import *
from datasets import load_dataset
import copy
SMOKE = "--smoke" in sys.argv; name = "pythia160"; t0 = time.time(); B = MID[name]; T = 256; NTR, NEV = (16, 8) if SMOKE else (640, 16); LR, STEPS = 1e-5, (3 if SMOKE else 80); torch.set_grad_enabled(False)
model, tok, fam = load_model(name); arch = Arch(model, fam); orig = copy.deepcopy(model).eval()
for p_ in orig.parameters(): p_.requires_grad_(False)
ds = load_dataset("NeelNanda/pile-10k", split="train")
DOMS = {"github": lambda s: s == "Github", "pubmed": lambda s: s == "PubMed Abstracts"}; CTL = lambda s: s not in ("Github", "PubMed Abstracts", "PubMed Central")
def windows(pred, nwin, skipdocs=0):
    wins, seen = [], 0
    for ex in ds:
        if not pred(ex["meta"]["pile_set_name"]): continue
        seen += 1
        if seen <= skipdocs: continue
        ids = tok(ex["text"])["input_ids"]
        if len(ids) >= T + 1: wins.append(ids[:T + 1])
        if len(wins) >= nwin: break
    return torch.tensor(wins, device=DEV)
TR = {d: windows(p, NTR) for d, p in DOMS.items()}; EVW = {d: windows(p, NEV, skipdocs=760) for d, p in DOMS.items()}; EVW["control"] = windows(CTL, NEV, skipdocs=1000)
log("windows: " + ", ".join(f"{d} {TR[d].shape[0]}/{EVW[d].shape[0]}" for d in DOMS) + f", control {EVW['control'].shape[0]} ({time.time() - t0:.0f}s)")
def lossof(m, ids, chunk=8):
    tot = 0.0
    for s0 in range(0, ids.shape[0], chunk):
        x = ids[s0:s0 + chunk]; lg = m(x[:, :T]).logits.float(); tot += float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:T].reshape(-1))) * x.shape[0]; del lg
    return tot / ids.shape[0]
def analyse(m, ids):
    a2 = Arch(m, fam); X = block_states(m, a2, ids[:, :T], [B])[B].reshape(-1, arch.D); keep = ~sinkmask(X); A, _ = rows_of(a2, B); U = unitr(X[keep] - X[keep].mean(0)); st = stats(U, A, K)
    return dict(words=set(torch.nonzero(wordset(st["usage"]))[:, 0].tolist()), R=st["ratio"].float(), keep=keep.cpu(), A=A)
def still_writing(base, cur, wsel):
    kb = torch.nonzero(base["keep"])[:, 0]; kc = torch.nonzero(cur["keep"])[:, 0]; pmap = {int(p): i for i, p in enumerate(kc.tolist())}; out = []
    for w in wsel:
        c = torch.nonzero(base["R"][:, w] > 1)[:, 0]
        if c.numel() < 5: continue
        idx = torch.tensor([pmap[int(p)] for p in kb[c].tolist() if int(p) in pmap]); out.append(float((cur["R"][idx, w] > 1).float().mean() >= 0.5) if idx.numel() else 0.0)
    return (mean(out) if out else None), len(out)
def finetune(m0, ids_tr):
    m = copy.deepcopy(m0); params = list(m.parameters())
    for p_ in params: p_.requires_grad_(True)
    m.train(); opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(1)
    with torch.enable_grad():
        for s in range(STEPS):
            b_ = ids_tr[torch.randint(0, ids_tr.shape[0], (8,), generator=g)]; lg = m(b_[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), b_[:, 1:T].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    m.eval()
    for p_ in params: p_.requires_grad_(False)
    return m
def merge(models, weights):
    m = copy.deepcopy(orig); po = dict(orig.named_parameters()); pms = [dict(mm.named_parameters()) for mm in models]
    with torch.no_grad():
        for n, p_ in m.named_parameters(): p_.copy_(po[n] + sum(wt * (pm[n] - po[n]) for wt, pm in zip(weights, pms)))
    return m
base = {d: analyse(orig, EVW[d]) for d in DOMS}; L0 = {d: lossof(orig, EVW[d]) for d in EVW}; log("original losses: " + ", ".join(f"{d} {v:.3f}" for d, v in L0.items()))
FT = {d: finetune(orig, TR[d]) for d in DOMS}; cur = {d: analyse(FT[d], EVW[d]) for d in DOMS}; ENT = {d: sorted(cur[d]["words"] - base[d]["words"]) for d in DOMS}
jac = lambda a_, b_: len(set(a_) & set(b_)) / max(len(set(a_) | set(b_)), 1)
log("entrants: " + ", ".join(f"{d} {len(ENT[d])}" for d in DOMS) + f"; Jaccard between the two fine-tunes' entrants {jac(ENT['github'], ENT['pubmed']):.2f}; between their word sets {jac(cur['github']['words'], cur['pubmed']['words']):.2f} (original word sets {jac(base['github']['words'], base['pubmed']['words']):.2f}) ({time.time() - t0:.0f}s)")
MODELS_ = {"github_ft": FT["github"], "pubmed_ft": FT["pubmed"], "merged": merge([FT["github"], FT["pubmed"]], [0.5, 0.5]), "merged_full": merge([FT["github"], FT["pubmed"]], [1.0, 1.0]), "github_half": merge([FT["github"]], [0.5]), "sequential_github_then_pubmed": finetune(FT["github"], TR["pubmed"])}
res = dict(n_entrants={d: len(ENT[d]) for d in DOMS}, entrant_jaccard=jac(ENT["github"], ENT["pubmed"]), wordset_jaccard_ft=jac(cur["github"]["words"], cur["pubmed"]["words"]), wordset_jaccard_orig=jac(base["github"]["words"], base["pubmed"]["words"]), loss0=L0, conditions={})
for k, m in MODELS_.items():
    an = {d: analyse(m, EVW[d]) for d in DOMS}; c = dict(losses={d: lossof(m, EVW[d]) for d in EVW})
    for d in DOMS:
        sw, n_ = still_writing(cur[d], an[d], ENT[d]); c[f"{d}_entrants_still_writing"] = sw; c[f"{d}_entrants_now_words"] = mean([float(w in an[d]["words"]) for w in ENT[d]]) if ENT[d] else None; c[f"{d}_original_words_still_writing"] = still_writing(base[d], an[d], sorted(base[d]["words"]))[0]
    res["conditions"][k] = c; log(f"{k}: losses " + ", ".join(f"{d} {v:.3f}" for d, v in c["losses"].items()) + "; " + "; ".join(f"{d} entrants still writing {c[f'{d}_entrants_still_writing']} (words {c[f'{d}_entrants_now_words']}), original words {c[f'{d}_original_words_still_writing']}" for d in DOMS) + f" | {time.time() - t0:.0f}s")
    del m, an; torch.cuda.empty_cache()
C = res["conditions"]; summ = (f"merging audit: entrants github {len(ENT['github'])} / pubmed {len(ENT['pubmed'])}, Jaccard {res['entrant_jaccard']:.2f}; " + "; ".join(f"{k}: github loss {c['losses']['github']:.3f}, pubmed {c['losses']['pubmed']:.3f}, control {c['losses']['control']:.3f}, github entrants writing {c['github_entrants_still_writing']}, pubmed entrants writing {c['pubmed_entrants_still_writing']}" for k, c in C.items()) + f" | {time.time() - t0:.0f}s")
log(summ); record("e605_merging_audit" + ("_smoke" if SMOKE else ""), res, summ)
