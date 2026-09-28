"""e566 (session 102): do independently trained models converge on the same partition of the state space? A word's
context set (its over-the-floor positions) is the unit e557 found conserved across scale. Panel A: Pythia-160m against
its seed variants on the same text (same tokenizer, token positions align): different initialization with the same data
order (weight-seed), the same initialization with a different data order (data-seed), both different (seed), and the
deduped data twin. Twin rate at Jaccard >= 0.25 against the rotated-dictionary null, both directions, and the rows'
cosine by index (near 1 only for a shared initialization that training has not erased). Panel B: across families and
tokenizers by character spans: Pythia-410m (block 12), OLMo-1B (block 8), GPT-2 (block 6) and Mamba-130m (block 12)
on the same eight Pile documents; each position maps to its characters, a context set to the union of its characters
within the range all four tokenizations cover, and the Jaccard is over characters. Mamba shares Pythia's tokenizer
and data, so that pair isolates the architecture; GPT-2 differs in data and tokenizer; OLMo in everything.
Pre-registered (probabilities are honest guesses):
 P1 (0.6) the seed variants twin with the main model at the rate of the deduped twin (about 0.4), whether the
    initialization or the data order or both differ: the partition is the data's, not the seed's;
 P2 (0.6) rows by index between weight-seed variants sit at the noise level (cosine under 0.05) and between data-seed
    variants above it (a shared initialization partly remembered);
 P3 (0.55) Pythia-410m and Mamba-130m twin at the seed-variant rate (architecture does not matter), OLMo and GPT-2 lower
    but above the null."""
from s101_common import *
from transformers import AutoTokenizer, MambaForCausalLM
from datasets import load_dataset
t0 = time.time(); JT = 0.25
def jac(P, Q):
    P = P.float(); Q = Q.float(); inter = P.T @ Q; return inter / (P.sum(0)[:, None] + Q.sum(0)[None] - inter).clamp_min(1)
def sets_of(U, A):
    st = stats(U, A, K); w = torch.nonzero(wordset(st["usage"]))[:, 0]; over = st["ratio"][:, w].float() > 1
    Ar = unitr(rotate(A, seed=7)); sr = stats(U, Ar, K); wr = torch.nonzero(wordset(sr["usage"]))[:, 0]; overr = sr["ratio"][:, wr].float() > 1
    return dict(w=w, over=over, over_r=overr, S=float(st["S"][w].median()))
def twin(a, b, Pa=None, Pb=None):
    """twin rate of a's words among b's words (and against b's rotated words); optional position-to-character maps"""
    A_, B_, Br = a["over"], b["over"], b["over_r"]
    if Pa is not None: A_ = (A_.float().T @ Pa.float()).T > 0; B_ = (B_.float().T @ Pb.float()).T > 0; Br = (Br.float().T @ Pb.float()).T > 0
    best = jac(A_, B_).max(1).values; bestr = jac(A_, Br).max(1).values
    return dict(share=float((best >= JT).float().mean()), median=float(best.median()), null_share=float((bestr >= JT).float().mean()), null_median=float(bestr.median()))
res = dict(seeds={}, families={})
# ---------------- Panel A ----------------
VAR = {"main": "EleutherAI/pythia-160m", "deduped": "EleutherAI/pythia-160m-deduped", "weight-seed1": "EleutherAI/pythia-160m-weight-seed1", "weight-seed2": "EleutherAI/pythia-160m-weight-seed2", "data-seed1": "EleutherAI/pythia-160m-data-seed1", "data-seed2": "EleutherAI/pythia-160m-data-seed2", "seed1": "EleutherAI/pythia-160m-seed1", "seed2": "EleutherAI/pythia-160m-seed2"}
St = {}
for tag, hf in VAR.items():
    try:
        wdd_common.MODELS[f"p160_{tag}"] = (hf, "neox"); S = lm_states(f"p160_{tag}", B=6, ids=pile_ids("pythia410")); St[tag] = dict(X=S["X"].cpu(), keep=S["keep"].cpu(), A=S["A"]); del S["model"]; torch.cuda.empty_cache(); log(f"{tag}: loaded ({time.time() - t0:.0f}s)")
    except Exception as e: log(f"{tag}: unavailable ({str(e)[:120]})")
keepc = torch.stack([v["keep"] for v in St.values()]).all(0); N = int(keepc.sum())
for tag, v in St.items(): X = v["X"][keepc].to(DEV); v["sets"] = sets_of(unitr(X - X.mean(0)), v["A"]); torch.cuda.empty_cache()
tags = list(St); res["seeds"]["n_positions"] = N; res["seeds"]["pairs"] = {}; res["seeds"]["row_cosine_by_index"] = {}
for i, a in enumerate(tags):
    for b in tags:
        if a == b: continue
        r = twin(St[a]["sets"], St[b]["sets"]); res["seeds"]["pairs"][f"{a}->{b}"] = r
        if a < b: c = (St[a]["A"] * St[b]["A"]).sum(1); res["seeds"]["row_cosine_by_index"][f"{a}|{b}"] = dict(median=float(c.median()), q99=float(c.quantile(0.99)), block0=float(c[:3072].median()))
for a in tags: log(f"{a}: twins among " + ", ".join(f"{b} {res['seeds']['pairs'][f'{a}->{b}']['share']:.2f} (null {res['seeds']['pairs'][f'{a}->{b}']['null_share']:.2f})" for b in tags if b != a))
log("rows by index: " + ", ".join(f"{k} {v['median']:.3f} (block 0 {v['block0']:.2f})" for k, v in res["seeds"]["row_cosine_by_index"].items()))
del St; torch.cuda.empty_cache()
# ---------------- Panel B ----------------
FAM = {"pythia410": ("EleutherAI/pythia-410m", 12, "neox"), "mamba130": ("state-spaces/mamba-130m-hf", 12, "mamba"), "gpt2": ("openai-community/gpt2", 6, "gpt2"), "olmo1b": ("allenai/OLMo-1B-0724-hf", 8, "llama")}
ds = load_dataset("NeelNanda/pile-10k", split="train"); texts = [ds[i]["text"] for i in range(200) if len(ds[i]["text"]) >= 4000][:8]; T = 256
toks = {k: AutoTokenizer.from_pretrained(hf) for k, (hf, _, _) in FAM.items()}; enc = {}
for k, tk in toks.items():
    e = [tk(t, return_offsets_mapping=True, add_special_tokens=False) for t in texts]
    assert all(len(x["input_ids"]) >= T for x in e), k
    enc[k] = dict(ids=torch.tensor([x["input_ids"][:T] for x in e]), off=[x["offset_mapping"][:T] for x in e])
Cd = [min(enc[k]["off"][d][T - 1][1] for k in FAM) for d in range(len(texts))]; base = [0] + list(torch.tensor(Cd).cumsum(0).tolist())[:-1]; Ctot = int(sum(Cd))
def charmap(k, keep):
    """[N_kept, Ctot] bool: the characters of each kept position (token index 1..T-1 of each document)"""
    M = torch.zeros(len(texts) * (T - 1), Ctot, dtype=torch.bool)
    for d in range(len(texts)):
        for p in range(1, T):
            s, e = enc[k]["off"][d][p]; e = min(e, Cd[d])
            if e > s: M[d * (T - 1) + p - 1, base[d] + s: base[d] + e] = True
    return M[keep]
FS = {}
for k, (hf, B, kind) in FAM.items():
    ids = enc[k]["ids"].to(DEV)
    if kind == "mamba":
        model = MambaForCausalLM.from_pretrained(hf, dtype=torch.float32).to(DEV).eval(); cap = {}
        class Stop(Exception): pass
        def hk(m, i, o): cap["x"] = (o[0] if isinstance(o, tuple) else o).detach().float(); raise Stop
        h = model.backbone.layers[B].register_forward_hook(hk)
        try:
            with torch.no_grad(): model(ids)
        except Stop: pass
        finally: h.remove()
        X = cap["x"][:, 1:].reshape(-1, cap["x"].shape[-1]); A = unitr(torch.cat([model.backbone.layers[b].mixer.out_proj.weight.detach().float().T for b in range(B + 1)]))
    else:
        wdd_common.MODELS[k] = (hf, kind); model, _, fam = load_model(k); arch = Arch(model, fam); X = block_states(model, arch, ids, [B])[B].reshape(-1, arch.D); A = rows_of(arch, B)[0]
    keep = ~sinkmask(X); U = unitr(X[keep] - X[keep].mean(0)); FS[k] = dict(sets=sets_of(U, A), P=charmap(k, keep.cpu()), n=int(keep.sum())); del model, X, U, A; torch.cuda.empty_cache(); log(f"{k}: {FS[k]['n']} positions, words' median S {FS[k]['sets']['S']:.2f} ({time.time() - t0:.0f}s)")
res["families"]["n_chars"] = Ctot; res["families"]["pairs"] = {}
for a in FAM:
    for b in FAM:
        if a != b: res["families"]["pairs"][f"{a}->{b}"] = twin(FS[a]["sets"], FS[b]["sets"], FS[a]["P"], FS[b]["P"])
for a in FAM: log(f"{a}: character twins among " + ", ".join(f"{b} {res['families']['pairs'][f'{a}->{b}']['share']:.2f} (null {res['families']['pairs'][f'{a}->{b}']['null_share']:.2f})" for b in FAM if b != a))
sp, fp = res["seeds"]["pairs"], res["families"]["pairs"]; g_ = lambda d, k: f"{d[k]['share']:.2f}" if k in d else "n/a"; rc = res["seeds"]["row_cosine_by_index"]
summ = (f"seed variants of Pythia-160m ({N} positions): twins of main among deduped/weight-seed1/data-seed1/seed1 {g_(sp, 'main->deduped')}/{g_(sp, 'main->weight-seed1')}/{g_(sp, 'main->data-seed1')}/{g_(sp, 'main->seed1')} (nulls about {sp['main->deduped']['null_share']:.2f}); weight-seed1->weight-seed2 {g_(sp, 'weight-seed1->weight-seed2')}, data-seed1->data-seed2 {g_(sp, 'data-seed1->data-seed2')}, seed1->seed2 {g_(sp, 'seed1->seed2')}; "
        f"rows by index main|weight-seed1 {rc.get('main|weight-seed1', {}).get('median', float('nan')):.3f}, main|data-seed1 {rc.get('data-seed1|main', rc.get('main|data-seed1', {})).get('median', float('nan')):.3f}, main|deduped {rc.get('deduped|main', {}).get('median', float('nan')):.3f}; "
        f"across families by characters ({Ctot} characters): pythia410->mamba130 {g_(fp, 'pythia410->mamba130')} (null {fp['pythia410->mamba130']['null_share']:.2f}), mamba130->pythia410 {g_(fp, 'mamba130->pythia410')}, pythia410->gpt2 {g_(fp, 'pythia410->gpt2')} (null {fp['pythia410->gpt2']['null_share']:.2f}), pythia410->olmo1b {g_(fp, 'pythia410->olmo1b')} (null {fp['pythia410->olmo1b']['null_share']:.2f}), olmo1b->pythia410 {g_(fp, 'olmo1b->pythia410')}, gpt2->olmo1b {g_(fp, 'gpt2->olmo1b')} | {time.time() - t0:.0f}s")
log(summ); record("e566_partitions", res, summ)
