"""e618e (session 117): the floors of the silencing test. For every result file of e617 and e618b for one model and domain,
and every neuron set it silenced, the original model with that set's inputs zeroed (no unlearning): its forget and retain
losses are the floor a relearning can reach with those neurons dead. The steps-back-within-0.1-nats criterion of the
silencing tests measures distance to the unsilenced original, so a silenced model can be censored by the floor alone; the
floor-corrected recovery (the share of the recoverable rise a 20-step relearning recovers) is the honest measure, scored
in e618d. Arguments: model domain ga 0 (the method and seed are placeholders; the windows must match the full runs, so no
--smoke on the real run).
"""
from s101_common import *
from ma_common import Stop
from datasets import load_dataset
import copy, re, collections
import wdd_common
if len(sys.argv) > 1 and sys.argv[1] == "pythia1b": wdd_common.MODELS["pythia1b"] = ("EleutherAI/pythia-1b", "neox"); MID["pythia1b"] = 8
wdd_common.MODELS.setdefault("qwen05", ("Qwen/Qwen2.5-0.5B", "llama")); MID.setdefault("qwen05", 12)   # e617 held-out architecture
name, DOMAIN, METHOD, SEED = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); SMOKE = "--smoke" in sys.argv
assert METHOD in ("ga", "gd", "npo", "scrub"), METHOD   # e618b gradient methods only
t0 = time.time(); B = MID[name]; T = 256; NTR, NEV = (16, 8) if SMOKE else (96, 16); LR, LAM, MAXS, CHK = 2e-6, 5.0, (6 if SMOKE else 600), (3 if SMOKE else 5); TARGETS = (0.2,) if SMOKE else (2.0,); BETA = 0.1 * 20 / (T - 1); RL, BN = (10, 5) if SMOKE else (100, 50); torch.set_grad_enabled(False)   # npo's beta scaled from the 20-token answers it was tuned on to these 255-token windows
model, tok, fam = load_model(name); arch = Arch(model, fam); orig = copy.deepcopy(model).eval(); DFF = arch.DFF
assert fam in ("neox", "llama"), fam
def mlp_down(m, b): l = Arch(m, fam).layers[b].mlp; return l.dense_4h_to_h if fam == "neox" else l.down_proj
def zero_rows(m, rows_):
    """silence the neurons' inputs (the rows that compute them): dense_h_to_4h rows and bias for neox, gate and up rows for llama"""
    a2 = Arch(m, fam)
    with torch.no_grad():
        for w in rows_:
            b_, j = w // DFF, w % DFF; l = a2.layers[b_].mlp
            if fam == "neox": l.dense_h_to_4h.weight[j, :] = 0; l.dense_h_to_4h.bias[j] = 0
            else:
                l.gate_proj.weight[j, :] = 0; l.up_proj.weight[j, :] = 0
                if l.gate_proj.bias is not None: l.gate_proj.bias[j] = 0; l.up_proj.bias[j] = 0
    return m
for p_ in orig.parameters(): p_.requires_grad_(False)
ds = load_dataset("NeelNanda/pile-10k", split="train")
FSET = {"pubmed": ("PubMed Abstracts", ("PubMed Abstracts", "PubMed Central")), "github": ("Github", ("Github",)), "stackexchange": ("StackExchange", ("StackExchange",)), "wiki": ("Wikipedia (en)", ("Wikipedia (en)",)), "uspto": ("USPTO Backgrounds", ("USPTO Backgrounds",)), "freelaw": ("FreeLaw", ("FreeLaw",)), "dm_math": ("DM Mathematics", ("DM Mathematics",)), "arxiv": ("ArXiv", ("ArXiv",))}[DOMAIN]
FORGET = lambda s: s == FSET[0]; RETAIN = lambda s: s not in FSET[1]
def windows(pred, nwin, skipdocs=0):
    wins, buf, seen = [], [], 0
    for ex in ds:
        if not pred(ex["meta"]["pile_set_name"]): continue
        seen += 1
        if seen <= skipdocs: continue
        buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
        while len(buf) >= T + 1 and len(wins) < nwin: wins.append(buf[:T + 1]); buf = buf[T + 1:]
        if len(wins) >= nwin: break
    return torch.tensor(wins, device=DEV), seen
f_tr, nfd = windows(FORGET, NTR); f_ev, _ = windows(FORGET, NEV, skipdocs=nfd); r_tr, nrd = windows(RETAIN, NTR); r_ev, _ = windows(RETAIN, NEV, skipdocs=nrd); assert f_tr.shape[0] == NTR and f_ev.shape[0] == NEV and r_ev.shape[0] == NEV, "not enough windows"; log(f"{name} {DOMAIN} {METHOD}: windows forget {f_tr.shape[0]}/{f_ev.shape[0]}, retain {r_tr.shape[0]}/{r_ev.shape[0]} ({time.time() - t0:.0f}s)")
def lossof(m, ids, chunk=8):
    tot = 0.0
    for s0 in range(0, ids.shape[0], chunk):
        x = ids[s0:s0 + chunk]; lg = m(x).logits.float(); tot += float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))) * x.shape[0]; del lg
    return tot / ids.shape[0]
EV = torch.cat([f_ev, r_ev])[:, :T]; is_forget_seq = torch.cat([torch.ones(NEV), torch.zeros(NEV)]).bool()
def states(m):
    X = block_states(m, Arch(m, fam), EV, [B])[B].reshape(-1, arch.D); keep = ~sinkmask(X); A, norms = rows_of(Arch(m, fam), B); return X, keep, A
X0, keep0, A0 = states(orig); U0 = unitr(X0[keep0] - X0[keep0].mean(0)); st0 = stats(U0, A0, K); w0 = torch.nonzero(wordset(st0["usage"]))[:, 0]; R0 = st0["ratio"].float(); pos_forget = is_forget_seq.repeat_interleave(T - 1)[keep0.cpu()]
CLS = {int(w): torch.nonzero(R0[:, w] > 1)[:, 0] for w in w0.tolist()}; fshare = {w: float(pos_forget[c].float().mean()) if c.numel() else 0.0 for w, c in CLS.items()}
fw = [w for w, s in fshare.items() if s >= 0.7 and CLS[w].numel() >= 5]; rw = [w for w, s in fshare.items() if s <= 0.3 and CLS[w].numel() >= 5]; kidx0 = torch.nonzero(keep0.cpu())[:, 0]
log(f"{len(w0)} words: {len(fw)} forget words, {len(rw)} retain words (forget positions {float(pos_forget.float().mean()):.2f} of kept)")
L0f, L0r = lossof(orig, f_ev[:, :T]), lossof(orig, r_ev[:, :T]); log(f"original loss forget {L0f:.3f}, retain {L0r:.3f}")
# ---- e618e: floors
import json, os, glob as _glob
out = {}; files = sorted(_glob.glob(f"/workspace/wdd/results/e617_heldout_{name}_{DOMAIN}_*.json") + _glob.glob(f"/workspace/wdd/results/e618b_controls_{name}_{DOMAIN}_*.json"))
for f in files:
    if "smoke" in f: continue
    r = json.load(open(f)); tag = os.path.basename(f)[:-5]; sets = r.get("sets", {}); need = set()
    for k, c in r["conditions"].items():
        for s_ in (c.get("silenced") or {}): need.add(s_)
    fl = {}
    for s_ in sorted(need):
        if s_ not in sets: continue
        ms = zero_rows(copy.deepcopy(orig), sets[s_]); fl[s_] = dict(n_rows=len(sets[s_]), floor_forget=lossof(ms, f_ev[:, :T]), floor_retain=lossof(ms, r_ev[:, :T])); del ms; torch.cuda.empty_cache()
    out[tag] = fl; log(f"{tag}: " + ", ".join(f"{s_} {v['floor_forget']:.3f}" for s_, v in fl.items()))
res = dict(model=name, domain=DOMAIN, loss0=dict(forget=L0f, retain=L0r), n_files=len(out), floors=out)
record(f"e618e_floors_{name}_{DOMAIN}" + ("_smoke" if SMOKE else ""), res, f"floors for {len(out)} result files of {name} {DOMAIN}: " + "; ".join(f"{t}: " + ", ".join(f"{s_} {v['floor_forget'] - L0f:+.2f}" for s_, v in fl.items()) for t, fl in list(out.items())[:3]))
