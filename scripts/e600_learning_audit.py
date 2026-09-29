"""e600 (session 111): the learning audit, the mirror of the unlearning audit. When a model is fine-tuned on a domain,
where does the learning live by provenance: do new write rows become words on that domain's contexts (recruitment),
or do the rows that already spoke there speak more (re-use), and do the old words keep writing elsewhere? Pythia-160m,
block-6 words; the domain is Github or PubMed Abstracts from pile-10k (up to 640 training windows, one per document, 610 on
Github, 16 held out from later documents), the rest of the corpus the control domain (16 held-out windows). Three fine-tunings
of one epoch (80 steps of 8 windows at lr 1e-5; two earlier runs with 17-25 epochs over 96 windows memorised them,
batch loss 1.3 to 0.07, and raised the held-out domain loss by two nats, so the audit needs one pass over many
documents):
all parameters; attention only (the write rows cannot change, so any new word there is re-use); MLP only. Before and
after: the 256 words on the domain's held-out states and on the control's, the entrants and leavers among them, and
for each entrant whether its row moved (cosine with its original) or was recruited as it stood; the share of the
domain's held-out positions at which some entrant writes over the floor; the old domain words' and the control words'
still-writing shares; the domain and control losses. Pre-registered (honest guesses):
 L1 (0.6) full fine-tuning makes twenty or more entrants on the domain's states, most with rows at cosine 0.99 or
    more with their originals (recruitment of rows as they stand, not rewriting);
 L2 (0.6) attention-only fine-tuning reaches at least half of the full fine-tuning's loss gain with fewer entrants;
 L3 (0.5) the control's words keep writing (still-writing share 0.9 or more) under all three."""
from s101_common import *
from datasets import load_dataset
import copy
name, DOMAIN = "pythia160", (sys.argv[1] if len(sys.argv) > 1 else "github"); t0 = time.time(); B = MID[name]; T = 256; NTR, NEV = 640, 16; LR, STEPS = 1e-5, 80; torch.set_grad_enabled(False)
model, tok, fam = load_model(name); arch = Arch(model, fam); orig = copy.deepcopy(model).eval()
for p_ in orig.parameters(): p_.requires_grad_(False)
ds = load_dataset("NeelNanda/pile-10k", split="train")
FSET = {"pubmed": ("PubMed Abstracts", ("PubMed Abstracts", "PubMed Central")), "github": ("Github", ("Github",))}[DOMAIN]
DOM = lambda s: s == FSET[0]; CTL = lambda s: s not in FSET[1]
def windows(pred, nwin, skipdocs=0):
    """one window per document (the first T+1 tokens of each document long enough), so that a training set spans many documents"""
    wins, seen = [], 0
    for ex in ds:
        if not pred(ex["meta"]["pile_set_name"]): continue
        seen += 1
        if seen <= skipdocs: continue
        ids = tok(ex["text"])["input_ids"]
        if len(ids) >= T + 1: wins.append(ids[:T + 1])
        if len(wins) >= nwin: break
    return torch.tensor(wins, device=DEV)
d_tr = windows(DOM, NTR); d_ev = windows(DOM, NEV, skipdocs=760); c_ev = windows(CTL, NEV, skipdocs=1000); log(f"{DOMAIN}: windows domain {d_tr.shape[0]}/{d_ev.shape[0]}, control {c_ev.shape[0]} ({time.time() - t0:.0f}s)")
def lossof(m, ids):
    lg = m(ids).logits.float(); return float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)))
def analyse(m, ids):
    """words on the states of ids: the word set, the class of each word, the ratio matrix, the rows"""
    a2 = Arch(m, fam); X = block_states(m, a2, ids[:, :T], [B])[B].reshape(-1, arch.D); keep = ~sinkmask(X); A, _ = rows_of(a2, B); U = unitr(X[keep] - X[keep].mean(0)); st = stats(U, A, K)
    words = torch.nonzero(wordset(st["usage"]))[:, 0]; R = st["ratio"].float(); return dict(words=set(words.tolist()), R=R, keep=keep.cpu(), A=A, S=st["S"])
base_d, base_c = analyse(orig, d_ev), analyse(orig, c_ev); L0d, L0c = lossof(orig, d_ev[:, :T]), lossof(orig, c_ev[:, :T]); log(f"original: domain loss {L0d:.3f}, control loss {L0c:.3f}; {len(base_d['words'])} domain words, {len(base_c['words'])} control words, shared {len(base_d['words'] & base_c['words'])}")
def still_writing(base, cur, words):
    """share of the given words (classes on the base states) still over the floor at half their class positions on the current states"""
    kb = torch.nonzero(base["keep"])[:, 0]; kc = torch.nonzero(cur["keep"])[:, 0]; pmap = {int(p): i for i, p in enumerate(kc.tolist())}; out = []
    for w in words:
        c = torch.nonzero(base["R"][:, w] > 1)[:, 0]
        if c.numel() < 5: continue
        idx = torch.tensor([pmap[int(p)] for p in kb[c].tolist() if int(p) in pmap]); out.append(float((cur["R"][idx, w] > 1).float().mean() >= 0.5) if idx.numel() else 0.0)
    return mean(out) if out else None, len(out)
def finetune(which):
    m = copy.deepcopy(orig); params = [p_ for n, p_ in m.named_parameters() if which == "all" or (which == "attention" and "attention" in n) or (which == "mlp" and "mlp" in n)]
    for p_ in m.parameters(): p_.requires_grad_(False)
    for p_ in params: p_.requires_grad_(True)
    m.train(); opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0); g = torch.Generator().manual_seed(1)
    with torch.enable_grad():
        for s in range(STEPS):
            b_ = d_tr[torch.randint(0, d_tr.shape[0], (8,), generator=g)]; lg = m(b_[:, :T]).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), b_[:, 1:T].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if s % 20 == 0 or s == STEPS - 1:
                m.eval()
                with torch.no_grad(): log(f"  {which} step {s}: batch loss {float(loss):.3f}, held-out domain {lossof(m, d_ev[:, :T]):.3f}, control {lossof(m, c_ev[:, :T]):.3f}")
                m.train()
    m.eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    return m, len(params)
res = dict(domain=DOMAIN, loss0=dict(domain=L0d, control=L0c), n_domain_words=len(base_d["words"]), n_control_words=len(base_c["words"]), conditions={})
for which in ("all", "attention", "mlp"):
    m, npar = finetune(which); Ld, Lc = lossof(m, d_ev[:, :T]), lossof(m, c_ev[:, :T]); cur_d, cur_c = analyse(m, d_ev), analyse(m, c_ev)
    ent = sorted(cur_d["words"] - base_d["words"]); left = sorted(base_d["words"] - cur_d["words"]); kept = sorted(base_d["words"] & cur_d["words"])
    ent_cos = [float(base_d["A"][w] @ cur_d["A"][w]) for w in ent]; kept_cos = [float(base_d["A"][w] @ cur_d["A"][w]) for w in kept]
    ent_S_before = [float(base_d["S"][w]) for w in ent]; ent_S_after = [float(cur_d["S"][w]) for w in ent]
    cover = float((cur_d["R"][:, ent] > 1).any(1).float().mean()) if ent else 0.0; cover_base = float((base_d["R"][:, ent] > 1).any(1).float().mean()) if ent else 0.0
    sw_d, nd = still_writing(base_d, cur_d, sorted(base_d["words"])); sw_c, nc = still_writing(base_c, cur_c, sorted(base_c["words"])); ent_c = sorted(cur_c["words"] - base_c["words"])
    cnd = dict(trained=which, n_params=npar, loss_domain=Ld, loss_control=Lc, gain_domain=L0d - Ld, change_control=Lc - L0c, entrants=len(ent), leavers=len(left), kept=len(kept), entrant_row_cos=med(ent_cos) if ent_cos else None, entrant_rows_unchanged=mean([float(c >= 0.99) for c in ent_cos]) if ent_cos else None, kept_row_cos=med(kept_cos) if kept_cos else None,
               entrant_S_before=med(ent_S_before) if ent else None, entrant_S_after=med(ent_S_after) if ent else None, entrants_cover_domain_positions=cover, entrants_covered_before=cover_base, old_domain_words_still_writing=sw_d, control_words_still_writing=sw_c, control_entrants=len(ent_c))
    res["conditions"][which] = cnd
    log(f"{which} ({npar} tensors): domain loss {L0d:.3f} -> {Ld:.3f} (control {Lc - L0c:+.3f}); domain words: {len(ent)} entrants, {len(left)} leavers, {len(kept)} kept; entrant rows' cosine {cnd['entrant_row_cos']} (unchanged for {cnd['entrant_rows_unchanged']}), kept rows' {cnd['kept_row_cos']}; entrants' S {cnd['entrant_S_before']} -> {cnd['entrant_S_after']}; entrants write at {cover:.2f} of domain positions (before {cover_base:.2f}); old domain words still writing {sw_d}, control words {sw_c} ({len(ent_c)} control entrants) | {time.time() - t0:.0f}s")
    del m; torch.cuda.empty_cache()
C = res["conditions"]; summ = f"learning audit ({DOMAIN}, Pythia-160m, one epoch of 640 windows): " + "; ".join(f"{k}: gain {c['gain_domain']:+.3f} (control {c['change_control']:+.3f}), {c['entrants']} entrants (rows unchanged {c['entrant_rows_unchanged']}), cover {c['entrants_cover_domain_positions']:.2f}, old words writing {c['old_domain_words_still_writing']}, control words {c['control_words_still_writing']}" for k, c in C.items()) + f" | {time.time() - t0:.0f}s"
log(summ); record(f"e600_learning_audit_{DOMAIN}", res, summ)
