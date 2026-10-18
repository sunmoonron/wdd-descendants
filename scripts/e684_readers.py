"""e684 (session 155): do readers arrive after the row? The essay's buildability: a network recognises cars before its
car neurons form and consolidates so that specialists have a sheltered place to build (a Schelling point); e591 found
recruitment is not a performance event. If consolidation is for readers, a recruited row should gain downstream readers
after recruitment, and sub-classes of its class should get rows of their own. Pythia-160m (seeds 0, 1, 2), e666's cache
(block 6, 24 Pile sequences of 512, checkpoints 1000 to 16000, the words and their recruitment checkpoints from the e629
tables). For each eventual word recruited at index 2 to 13: at every checkpoint, each later-block neuron's (blocks
7-11) exact mean contribution from the word's write at the class positions, in units of that neuron's pre-activation
standard deviation; a reader has |share| above 1. Controls: three norm-matched non-word rows of the same block and three
random rows, scored on the same positions. The readers' alignment with the row against the class direction, before and
after. Specialists: later-block neurons whose top-1% positions select one k-means sub-class of the word's class.
Arguments: --tag pythia160|pythia160s1|pythia160s2 [--smoke]. Pre-registered in e684_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
tag = A_.tag; TAG = f"e684_{tag}" + ("_smoke" if SMOKE else ""); L_ = lambda s: log(TAG, s); B = 6; NSEQ = 8 if SMOKE else 24; T = 512; LATE = list(range(1000, 16001, 1000)); READ = list(range(B + 1, 12)); NCTRL = 3
ids = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["pythia410"]["eval_ids"][:NSEQ, :T].to(DEV)
tab = torch.load(f"/workspace/wdd/e629_tables/e629_scores_{tag}_B6.pt", weights_only=False); SCw = {int(k): v for k, v in tab["scores"].items()}; WORDS = {n: SCw[n]["words"].bool().cpu() for n in LATE}
def first_hold(flags):
    for i in range(len(flags)):
        if flags[i] and all(flags[i:]): return i
    return None
def forward(M):
    out = collections.defaultdict(list)
    with Capture(M, list(range(12)), ["act", "x", "xin"]) as C:
        for s0 in range(0, NSEQ, 4):
            run_base(M, ids[s0:s0 + 4]); out["x6"].append(C.cap[("x", B)][:, 1:].float().reshape(-1, M["D"])); out["acts"].append(torch.cat([C.cap[("act", b)][:, 1:].half() for b in range(B + 1)], -1).reshape(-1, (B + 1) * M["DFF"]))
            for b in READ: out[("xin", b)].append(C.cap[("xin", b)][:, 1:].float().reshape(-1, M["D"]))
    return {k: torch.cat(v) for k, v in out.items()}
CACHE = {}
for n in LATE:
    M = load(tag, revision=f"step{n}"); S = forward(M); DFF, D = M["DFF"], M["D"]
    W = torch.cat([write_w(M, b).T for b in range(B + 1)]); nr = W.norm(dim=1); A = W / nr[:, None].clamp_min(1e-12)
    RT = {b: read_tilde(M, b, read_w(M, b)["h"]) for b in READ}; PRE = {}; SIG = {}
    for b in READ:
        xn, sig = ln_apply(M, b, S[("xin", b)]); PRE[b] = (xn @ read_w(M, b)["h"].T + read_bias(M, b)["h"]).half(); SIG[b] = sig
    CACHE[n] = dict(x6=S["x6"], acts=S["acts"].cpu(), A=A, nr=nr, RT=RT, SIG=SIG, PRESTD={b: PRE[b].float().std(0).clamp_min(1e-6) for b in READ}, TOP={b: PRE[b].float().topk(max(10, S["x6"].shape[0] // 100), dim=0).indices for b in READ})   # the activations (the large cache) stay in CPU RAM: sixteen checkpoints of them do not fit beside the other jobs
    del M, S, PRE; torch.cuda.empty_cache(); L_(f"cached step {n}")
N = CACHE[LATE[0]]["x6"].shape[0]; keep = ~sinkmask(CACHE[LATE[-1]]["x6"]); kd = keep.to(DEV); m = CACHE[LATE[-1]]["A"].shape[0]; DFF = m // (B + 1)
last = LATE[-1]; Xl = CACHE[last]["x6"]; MU = Xl[keep].mean(0); U = unitr(Xl - MU); cal = floor_calibration(U[keep], CACHE[last]["A"]); Lf = floor_of(U, cal)
R16 = torch.cat([(U[s:s + 1024] @ CACHE[last]["A"].T).abs() / Lf[s:s + 1024, None] for s in range(0, N, 1024)]); wl = torch.nonzero(WORDS[last])[:, 0]; csize = ((R16[:, wl.to(DEV)] > 1) & kd[:, None]).sum(0).cpu(); tracked = []
for w in wl[(csize >= 10) & ~WORDS[LATE[0]][wl]].tolist():
    j = first_hold([bool(WORDS[n][w]) for n in LATE])
    if j is not None and 2 <= j <= 13: tracked.append((w, j))
CLS = {w: torch.nonzero((R16[:, w] > 1) & kd)[:, 0] for w, _ in tracked}; del R16
if SMOKE: tracked = tracked[:12]
L_(f"{len(tracked)} eventual words recruited at index 2..13 (class size median {med([int(CLS[w].numel()) for w, _ in tracked])})")
def reader_shares(n, row, P):
    """[sum over read blocks of DFF]: the row's mean contribution at positions P to each later neuron's pre-activation, in units of that neuron's std"""
    C = CACHE[n]; a = C["acts"][P.cpu(), row].to(DEV).float() * C["nr"][row]; d = C["A"][row]; out = []
    for b in READ: V = C["RT"][b] @ d; out.append((a[:, None] * V[None] / C["SIG"][b][P][:, None]).mean(0) / C["PRESTD"][b])
    return torch.cat(out)
THR_R = 0.2   # a reader: the row's mean contribution at the class is a fifth of the reader's pre-activation spread (one std found nobody; the crowd's typical share is a hundredth)
def n_readers(n, row, P): s = reader_shares(n, row, P).abs(); return int((s > THR_R).sum()), float(s.sum()), int((s > 0.1).sum()), int((s > 0.5).sum()), float(s.topk(10).values.mean())
def specialists(n, P, subs):
    """later-block neurons whose top-1% positions select one sub-class of P: hits in the sub-class over its size minus hits in the rest over theirs, above 0.2, with at least 3 hits"""
    C = CACHE[n]; cnt = 0; Pset = set(P.tolist())
    for b in READ:
        top = C["TOP"][b]   # [K, DFF]
        for s in subs:
            ss = torch.zeros(N, dtype=torch.bool, device=DEV); ss[s] = True; rest = torch.zeros(N, dtype=torch.bool, device=DEV); rest[P] = True; rest &= ~ss
            hs = ss[top].sum(0).float(); hr = rest[top].sum(0).float(); sel = hs / len(s) - hr / max(int(rest.sum()), 1); cnt += int(((sel > 0.2) & (hs >= 3)).sum())
    return cnt
rng = np.random.default_rng(684); PW = {}
for w, j in tracked:
    P = CLS[w]; n_r = LATE[j]; blk = w // DFF; same = torch.nonzero((torch.arange(m, device=DEV) // DFF == blk) & ~WORDS[last].to(DEV))[:, 0]; nr_r = CACHE[n_r]["nr"]
    ctrl = same[(nr_r[same] - nr_r[w]).abs().argsort()[:NCTRL]].tolist(); rand = rng.choice(m, NCTRL, replace=False).tolist()
    curve = {str(n): n_readers(n, w, P) for n in LATE}; curve_c = {str(n): [n_readers(n, r, P) for r in ctrl] for n in [LATE[j - 2], LATE[j + 2]]}; curve_r = {str(n): [n_readers(n, r, P) for r in rand] for n in [LATE[j - 2], LATE[j + 2]]}
    before, after = LATE[j - 2], LATE[j + 2]; d_word = curve[str(after)][0] - curve[str(before)][0]; d_ctrl = mean([a[0] - b_[0] for a, b_ in zip(curve_c[str(after)], curve_c[str(before)])]); d_rand = mean([a[0] - b_[0] for a, b_ in zip(curve_r[str(after)], curve_r[str(before)])])
    s_after = reader_shares(after, w, P); readers = torch.nonzero(s_after.abs() > THR_R)[:, 0]
    if readers.numel() < 5: readers = s_after.abs().topk(10).indices   # the ten strongest readers when few clear the bar
    def align(n, vec):
        C = CACHE[n]; cos = torch.cat([torch.nn.functional.cosine_similarity(C["RT"][b], vec[None], dim=1) for b in READ]).abs(); return float(cos[readers].mean()) if readers.numel() else None
    dC = unitr((CACHE[before]["x6"][P] - CACHE[before]["x6"][keep].mean(0)).mean(0, keepdim=True))[0]
    al = dict(row_before=align(before, CACHE[before]["A"][w]), row_after=align(after, CACHE[after]["A"][w]), class_before=align(before, dC), class_after=align(after, dC))
    Uj = unitr(CACHE[n_r]["x6"][P] - CACHE[n_r]["x6"][keep].mean(0)); lab = kmeans(Uj, 3, seed=w); subs = [P[(lab == c).to(DEV)] for c in range(3) if int((lab == c).sum()) >= 5]
    if subs:
        sp = dict(before=specialists(before, P, subs), after=specialists(after, P, subs)); rsubs = [[torch.tensor(rng.choice(N, int(s.numel()), replace=False), device=DEV) for s in subs] for _ in range(3)]
        sp["null_before"] = mean([specialists(before, torch.cat(rs), rs) for rs in rsubs]); sp["null_after"] = mean([specialists(after, torch.cat(rs), rs) for rs in rsubs])
    else: sp = None   # a class with no cluster of five positions: no specialists test for this word
    d_top10 = curve[str(after)][4] - curve[str(before)][4]; d_top10_c = mean([a[4] - b_[4] for a, b_ in zip(curve_c[str(after)], curve_c[str(before)])]); d_mass = curve[str(after)][1] - curve[str(before)][1]; d_mass_c = mean([a[1] - b_[1] for a, b_ in zip(curve_c[str(after)], curve_c[str(before)])])
    PW[w] = dict(recruit_index=j, block=int(blk), class_size=int(P.numel()), readers_curve=curve, d_word=d_word, d_ctrl=d_ctrl, d_rand=d_rand, d_top10=d_top10, d_top10_ctrl=d_top10_c, d_mass=d_mass, d_mass_ctrl=d_mass_c, mass_before=curve[str(before)][1], mass_after=curve[str(after)][1], align=al, specialists=sp, n_readers_after=int(readers.numel()))
R = list(PW.values()); RS = [r for r in R if r["specialists"] is not None]
S_ = dict(n=len(R), share_word_gain_above_ctrl=mean([float(r["d_word"] > r["d_ctrl"]) for r in R]), share_word_gain_above_rand=mean([float(r["d_word"] > r["d_rand"]) for r in R]), median_d_word=med([r["d_word"] for r in R]), median_d_ctrl=med([r["d_ctrl"] for r in R]), median_d_rand=med([r["d_rand"] for r in R]),
          share_top10_gain_above_ctrl=mean([float(r["d_top10"] > r["d_top10_ctrl"]) for r in R]), share_mass_gain_above_ctrl=mean([float(r["d_mass"] > r["d_mass_ctrl"]) for r in R]), median_d_top10=med([r["d_top10"] for r in R]), median_d_top10_ctrl=med([r["d_top10_ctrl"] for r in R]),
          median_readers_before=med([r["readers_curve"][str(LATE[r["recruit_index"] - 2])][0] for r in R]), median_readers_after=med([r["readers_curve"][str(LATE[r["recruit_index"] + 2])][0] for r in R]), median_mass_ratio=med([r["mass_after"] / max(r["mass_before"], 1e-9) for r in R]),
          align_row_gain=med([r["align"]["row_after"] - r["align"]["row_before"] for r in R if r["align"]["row_after"] is not None]), align_class_gain=med([r["align"]["class_after"] - r["align"]["class_before"] for r in R if r["align"]["class_after"] is not None]),
          share_row_gain_above_class=mean([float((r["align"]["row_after"] - r["align"]["row_before"]) > (r["align"]["class_after"] - r["align"]["class_before"])) for r in R if r["align"]["row_after"] is not None]),
          median_spec_before=med([r["specialists"]["before"] for r in RS]), median_spec_after=med([r["specialists"]["after"] for r in RS]), median_spec_null_after=med([r["specialists"]["null_after"] for r in RS]), share_spec_gain_above_null=mean([float((r["specialists"]["after"] - r["specialists"]["before"]) > (r["specialists"]["null_after"] - r["specialists"]["null_before"])) for r in RS]))
ge = lambda v, t: bool(v is not None and v >= t)
V = dict(R1_readers_gain_above_control_65pct=ge(S_["share_word_gain_above_ctrl"], 0.65), R1b_top10_share_gain_above_control_65pct=ge(S_["share_top10_gain_above_ctrl"], 0.65), R1c_mass_gain_above_control_65pct=ge(S_["share_mass_gain_above_ctrl"], 0.65), R2_row_alignment_gain_above_class=ge(S_["share_row_gain_above_class"], 0.6), R3_specialists_gain_above_null_60pct=ge(S_["share_spec_gain_above_null"], 0.6))
res = dict(tag=tag, block=B, read_blocks=READ, checkpoints=LATE, n_words=len(R), summary=S_, verdicts=V, words={str(w): v for w, v in PW.items()})
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = (f"[{tag}] {len(R)} recruited words: readers (|share| > {THR_R} std) two checkpoints before -> after, median {S_['median_readers_before']} -> {S_['median_readers_after']}; the word's gain beats its norm-matched control rows in {f2(S_['share_word_gain_above_ctrl'])} (random rows {f2(S_['share_word_gain_above_rand'])}; median gains word {S_['median_d_word']}, control {S_['median_d_ctrl']}, random {S_['median_d_rand']}); top-10 reader share gain above control in {f2(S_['share_top10_gain_above_ctrl'])} (medians {S_['median_d_top10']} vs {S_['median_d_top10_ctrl']}), reader mass gain above control in {f2(S_['share_mass_gain_above_ctrl'])}, mass ratio {f2(S_['median_mass_ratio'])}; "
        f"readers' alignment gain with the row {S_['align_row_gain']} vs the class direction {S_['align_class_gain']} (row above class in {f2(S_['share_row_gain_above_class'])}); sub-class specialists {S_['median_spec_before']} -> {S_['median_spec_after']} (null after {S_['median_spec_null_after']}; gain above null in {f2(S_['share_spec_gain_above_null'])}); verdicts {V}")
save(TAG, res, summ)
