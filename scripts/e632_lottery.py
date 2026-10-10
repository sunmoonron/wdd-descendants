"""e632 (session 127): lottery or recruitment. The small-initialisation theory (Maennel 2018; Boursier and Flammarion;
Chen et al. 2025 for autoencoder training) says the neuron that comes to carry a feature is the one already most aligned
with it at initialisation. In real pretraining, is the row that is recruited for a class the one that was nearest at
step 0? Pythia-160m seeds 0, 1 and 2 (PolyPythias), block 6, the e629 caches for steps 1000 to 16000 and the early
checkpoints 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 loaded here. The eventual words are the words at 16000 with a
class of ten or more positions, not words at 1000, recruited at index 2 or later; a class's direction is the unit
mean of its unit states at 16000 (row-free). At every checkpoint, for every eventual word: its rank among all rows by
cosine of the write row with the class direction (the write-side ticket), and by activation selectivity at the class
positions (the read-side ticket); the same ranks for a random row and for the row against another class's direction
(the nulls); the lineage onset (the first checkpoint from which the row stays in the top tenth). Part B (Wang 2026):
two checkpoints before recruitment, the recruit's read-weight norm, write-column norm and their ratio against the ten
nearest competitors by class ratio. Part C: given the class, the recruit's rank among the non-word rows by mean class
ratio, by usage (class-blind) and by the autoencoder decoder cosine, two checkpoints before. Arguments: tag [--smoke].
Pre-registered in e632_prereg.json."""
import sys, os, time, json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
import wdd_common
wdd_common.MODELS.update({"pythia160s1": ("EleutherAI/pythia-160m-seed1", "neox"), "pythia160s2": ("EleutherAI/pythia-160m-seed2", "neox")}); MID.update({"pythia160s1": 6, "pythia160s2": 6})
tag = sys.argv[1]; SMOKE = "--smoke" in sys.argv; REF = next((a.split("=")[1] for a in sys.argv if a.startswith("--ref=")), None); t0 = time.time(); torch.set_grad_enabled(False); B = 6; NSEQ = 24; T = 512
EARLY = [0, 512] if SMOKE else [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]; LATE = [1000, 2000, 3000, 4000] if SMOKE else list(range(1000, 16001, 1000)); ALL = EARLY + LATE; sfx = "_smoke" if SMOKE else ""
CD = f"/workspace/wdd/cache/e629_{tag}_B6"; assert all(os.path.exists(f"{CD}/step{n}.pt") for n in LATE), "e629 cache missing"
ids = pile_ids(tag, NSEQ, T); ck = {n: torch.load(f"{CD}/step{n}.pt", map_location="cpu") for n in LATE}; keepc = torch.stack([ck[n]["keep"] for n in LATE]).all(0); kidx = torch.nonzero(keepc)[:, 0]; N = int(keepc.sum())
U = {n: unitr((ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV)) for n in LATE}; XC = {n: (ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV) for n in (LATE[-1],)}; m = ck[LATE[0]]["rows"].shape[0]; DFF = int(ck[LATE[0]]["DFF"]); D = U[LATE[0]].shape[1]
tab = torch.load(f"/workspace/wdd/results/e629_tables/e629_scores_{tag}_B6.pt"); SCw = {int(n): v for n, v in tab["scores"].items()}; assert all(n in SCw for n in LATE)
WORDS = {n: SCw[n]["words"] for n in LATE}; USAGE = {n: SCw[n]["usage"].float() for n in LATE}; FMAX = {n: SCw[n]["fmax"].float() for n in LATE}; ANYMAX = {n: SCw[n]["anymax"].float() for n in LATE}
last = LATE[-1]; Al = ck[last]["rows"].float().to(DEV); cal = floor_calibration(U[last], Al); Ll = floor_of(U[last], cal); R16 = torch.cat([(U[last][s:s + 1024] @ Al.T).abs() / Ll[s:s + 1024, None] for s in range(0, N, 1024)]).half().cpu()
wl = torch.nonzero(WORDS[last])[:, 0]; csize = (R16[:, wl].float() > 1).sum(0)
def first_hold(flags):
    for i in range(len(flags)):
        if flags[i] and all(flags[i:]): return i
    return None
tracked = []
for w in wl[(csize >= 10) & ~WORDS[LATE[0]][wl]].tolist():
    j = first_hold([bool(WORDS[n][w]) for n in LATE])
    if j is not None and j >= 2: tracked.append((w, j))
CLS = {w: torch.nonzero(R16[:, w].float() > 1)[:, 0] for w, _ in tracked}; DIR = {w: unitr(U[last][CLS[w].to(DEV)].mean(0, keepdim=True))[0] for w, _ in tracked}
g = torch.Generator().manual_seed(0); RND = {w: int(torch.randint(0, m, (1,), generator=g)) for w, _ in tracked}; OTHER = {tracked[i][0]: tracked[(i + 1) % len(tracked)][0] for i in range(len(tracked))}
os.makedirs("/workspace/wdd/results/e632_classes", exist_ok=True); torch.save(dict(classes={w: kidx[CLS[w]] for w, _ in tracked}, steps={w: LATE[j] for w, j in tracked}), f"/workspace/wdd/results/e632_classes/{tag}{sfx}.pt")
REFC = None
if REF:
    rf = f"/workspace/wdd/results/e632_classes/{REF}{sfx}.pt"
    if os.path.exists(rf):
        rc = torch.load(rf)["classes"]; pos_now = torch.full((keepc.numel(),), -1, dtype=torch.long); pos_now[kidx] = torch.arange(N); REFC = {}
        for w, full in rc.items():
            idx = pos_now[full]; idx = idx[idx >= 0]
            if idx.numel() < 10: continue
            over = (R16[idx].float() > 1).sum(0); spk = int(over.argmax())
            if int(over[spk]) >= idx.numel() / 2: REFC[int(w)] = dict(pos=idx, speaker=spk, dir=unitr(U[last][idx.to(DEV)].mean(0, keepdim=True))[0], word_at_first=bool(WORDS[LATE[0]][spk]))
        log(f"reference {REF}: {len(rc)} classes, {len(REFC)} with a speaker here at {last} (not words at {LATE[0]}: {sum(1 for v in REFC.values() if not v['word_at_first'])})")
log(f"{tag}: {N} positions, {m} rows, {len(tracked)} eventual words recruited at index 2 or later; checkpoints {ALL}")
def acts_at(model, arch):
    out = []
    class Stop(Exception): pass
    def stop(mm, i, o): raise Stop
    hstop = arch.layers[B].register_forward_hook(stop)
    try:
        for s0 in range(0, ids.shape[0], 8):
            cap = {}; hs = [arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda mm, a: cap.__setitem__(b_, a[0].detach()[:, 1:].reshape(-1, a[0].shape[-1]).half()))(b)) for b in range(B + 1)]
            try: model(ids[s0:s0 + 8])
            except Stop: pass
            finally: [h.remove() for h in hs]
            out.append(torch.cat([cap[b] for b in range(B + 1)], 1))
    finally: hstop.remove()
    return torch.cat(out)[keepc.to(DEV)]
rk = lambda v, w: int((v > v[w]).sum()) + 1
PER = {}; RN = {}; WN = {}; AROWS = {}; COEF = {}; PERS = {}; XR = {}
for n in ALL:
    model, _, fam = load_model(tag, revision=f"step{n}"); arch = Arch(model, fam); A, norms = rows_of(arch, B); RN[n] = torch.cat([arch.rdir(b).norm(dim=1) for b in range(B + 1)]).cpu(); WN[n] = norms.cpu(); AROWS[n] = A.half().cpu()
    act = acts_at(model, arch).float(); mu = act.mean(0); sd = act.std(0).clamp_min(1e-6); per = {}; COEF.setdefault(n, {})
    for w, j in tracked:
        pos = CLS[w].to(DEV); z = (act[pos].mean(0) - mu) / sd; cos = A @ DIR[w]; cos_o = A @ DIR[OTHER[w]]; COEF[n][w] = (act[pos, w] * norms[w]).cpu()
        per[w] = dict(rank_cos=rk(cos, w), rank_sel=rk(z, w), rank_cos_random=rk(cos, RND[w]), rank_sel_random=rk(z, RND[w]), rank_cos_other=rk(cos_o, w), cos=float(cos[w]), sel=float(z[w]))
    PER[n] = per; PERS[n] = dict(tracked=med([float(A[w] @ AROWS[ALL[0]][w].float().to(DEV)) for w, _ in tracked]) if n != ALL[0] else 1.0, all=float((A * AROWS[ALL[0]].float().to(DEV)).sum(1).median()) if n != ALL[0] else 1.0)
    if REFC is not None:
        XR[n] = {}
        for w, v in REFC.items():
            pos = v["pos"].to(DEV); z = (act[pos].mean(0) - mu) / sd; cos = A @ v["dir"]; XR[n][w] = dict(rank_cos=rk(cos, v["speaker"]), rank_sel=rk(z, v["speaker"]), rank_cos_random=rk(cos, RND[tracked[w % len(tracked)][0]]))
    del model, act; torch.cuda.empty_cache(); log(f"step {n}: median rank by cosine {med([p['rank_cos'] for p in per.values()]):.0f}, by selectivity {med([p['rank_sel'] for p in per.values()]):.0f} of {m}; top 100 {mean([float(p['rank_cos'] <= 100) for p in per.values()]):.2f} / {mean([float(p['rank_sel'] <= 100) for p in per.values()]):.2f} ({time.time() - t0:.0f}s)")
# the class direction with the row's own write subtracted, at the final checkpoint and two before recruitment
A0 = AROWS[ALL[0]].float().to(DEV); DEB = {}
for w, j in tracked:
    pos = CLS[w].to(DEV); out = {}
    for lab, n in (("final", last), ("two_before", LATE[j - 2])):
        Xn = XC[n] if n in XC else (ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV); Xp = Xn[pos] - COEF[n][w].to(DEV)[:, None] * AROWS[n][w].float().to(DEV)[None]; d = unitr(unitr(Xp).mean(0, keepdim=True))[0]; cos = A0 @ d
        out[lab] = dict(rank=rk(cos, w), norm_rank=rk(cos, w) / m, cos=float(cos[w]), cos_dir_with_row_final=float(d @ AROWS[last][w].float().to(DEV)), random_rank=rk(cos, RND[w]) / m)
    DEB[w] = out
DEBs = {lab: dict(median_norm_rank=med([DEB[w][lab]["norm_rank"] for w, _ in tracked]), top100=mean([float(DEB[w][lab]["rank"] <= 100) for w, _ in tracked]), top1000=mean([float(DEB[w][lab]["rank"] <= 1000) for w, _ in tracked]), random_median=med([DEB[w][lab]["random_rank"] for w, _ in tracked]), dir_cos_with_final_row=med([DEB[w][lab]["cos_dir_with_row_final"] for w, _ in tracked])) for lab in ("final", "two_before")}
XRs = None
if REFC is not None and XR:
    XRs = {str(n): dict(n=len(XR[n]), median_norm_rank_cos=med([v["rank_cos"] / m for v in XR[n].values()]), top100_cos=mean([float(v["rank_cos"] <= 100) for v in XR[n].values()]), top1000_cos=mean([float(v["rank_cos"] <= 1000) for v in XR[n].values()]), median_norm_rank_sel=med([v["rank_sel"] / m for v in XR[n].values()]), top100_sel=mean([float(v["rank_sel"] <= 100) for v in XR[n].values()]), random_median=med([v["rank_cos_random"] / m for v in XR[n].values()])) for n in ALL}
    nw = [w for w, v in REFC.items() if not v["word_at_first"]]
    XRs["step0_not_words_at_first"] = dict(n=len(nw), median_norm_rank_cos=med([XR[ALL[0]][w]["rank_cos"] / m for w in nw]), top100_cos=mean([float(XR[ALL[0]][w]["rank_cos"] <= 100) for w in nw])) if nw else None
summ_ck = {}
for n in ALL:
    per = PER[n]; f = lambda key, thr: mean([float(p[key] <= thr) for p in per.values()])
    summ_ck[n] = dict(n=len(per), median_norm_rank_cos=med([p["rank_cos"] / m for p in per.values()]), median_norm_rank_sel=med([p["rank_sel"] / m for p in per.values()]), top10_cos=f("rank_cos", 10), top100_cos=f("rank_cos", 100), top1000_cos=f("rank_cos", 1000), top10_sel=f("rank_sel", 10), top100_sel=f("rank_sel", 100), top1000_sel=f("rank_sel", 1000),
                      random_median_norm_rank_cos=med([p["rank_cos_random"] / m for p in per.values()]), random_top100_cos=f("rank_cos_random", 100), other_median_norm_rank_cos=med([p["rank_cos_other"] / m for p in per.values()]), other_top100_cos=f("rank_cos_other", 100), median_cos=med([p["cos"] for p in per.values()]), median_sel=med([p["sel"] for p in per.values()]))
# lineage onset: first checkpoint from which the row stays in the top tenth by cosine (and by selectivity)
def onset(key):
    out = {}
    for w, j in tracked:
        flags = [PER[n][w][key] <= m // 10 for n in ALL]; i = first_hold(flags); out[w] = ALL[i] if i is not None else None
    return out
ON_c, ON_s = onset("rank_cos"), onset("rank_sel"); step_of = {w: LATE[j] for w, j in tracked}
share_early = lambda on: mean([float(on[w] is not None and on[w] <= 512) for w, _ in tracked]); share_none = lambda on: mean([float(on[w] is None) for w, _ in tracked])
lead_c = [ (step_of[w] - ON_c[w]) / 1000 for w, _ in tracked if ON_c[w] is not None]; lead_s = [(step_of[w] - ON_s[w]) / 1000 for w, _ in tracked if ON_s[w] is not None]
# the step-0 rank against the recruitment checkpoint
def spearman(x, y):
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64); r = lambda v: v.argsort().argsort().double(); return float(torch.corrcoef(torch.stack([r(x), r(y)]))[0, 1]) if x.std() > 0 and y.std() > 0 else None
sp0 = spearman([PER[ALL[0]][w]["rank_cos"] for w, _ in tracked], [j for _, j in tracked]); sp0s = spearman([PER[ALL[0]][w]["rank_sel"] for w, _ in tracked], [j for _, j in tracked])
# ---------------- part B: read and write norms of the recruit against its ten nearest competitors, two checkpoints before
def class_ratio(n, pos):
    A = AROWS[n].float().to(DEV); Un = U[n]; cal_ = floor_calibration(Un, A); L = floor_of(Un[pos], cal_); return ((Un[pos] @ A.T).abs() / L[:, None]).mean(0)
PB = {lead: [] for lead in (2, 1, 0)}
for w, j in tracked:
    for lead in (2, 1, 0):
        i = j - lead; n = LATE[i]; mr = class_ratio(n, CLS[w].to(DEV)); mr[w] = -1; comp = mr.topk(10).indices.cpu(); comp = comp[~WORDS[n][comp]] if int((~WORDS[n][comp]).sum()) >= 3 else comp
        rn, wn = RN[n], WN[n]; ratio = rn / wn.clamp_min(1e-6); nonw = torch.nonzero(~WORDS[n])[:, 0]
        PB[lead].append(dict(read_vs_comp=float(rn[w] > rn[comp].median()), write_vs_comp=float(wn[w] > wn[comp].median()), ratio_vs_comp=float(ratio[w] > ratio[comp].median()), read_vs_all=float(rn[w] > rn[nonw].median()), write_vs_all=float(wn[w] > wn[nonw].median()), ratio_vs_all=float(ratio[w] > ratio[nonw].median()), read_rel=float(rn[w] / rn[comp].median()), write_rel=float(wn[w] / wn[comp].median()), ratio_rel=float(ratio[w] / ratio[comp].median())))
PBs = {lead: {k: mean([x[k] for x in v]) for k in v[0]} for lead, v in PB.items() if v}
# ---------------- part C: identification given the class, two checkpoints before, among the non-word rows
PC = []
for w, j in tracked:
    n = LATE[j - 2]; nonw = ~WORDS[n]; mr = class_ratio(n, CLS[w].to(DEV)).cpu(); rk_in = lambda v: int(((v > v[w]) & nonw).sum()) + 1
    PC.append(dict(rank_ratio=rk_in(mr), rank_usage=rk_in(USAGE[n]), rank_decoder=rk_in(FMAX[n]), rank_anymax=rk_in(ANYMAX[n])))
PCs = {k: dict(median=med([x[k] for x in PC]), top10=mean([float(x[k] <= 10) for x in PC]), top100=mean([float(x[k] <= 100) for x in PC])) for k in PC[0]}
res = dict(model=tag, block=B, N=N, m=m, n_tracked=len(tracked), checkpoints=ALL, per_checkpoint={str(n): v for n, v in summ_ck.items()}, onset=dict(cos_share_by_512=share_early(ON_c), cos_share_never=share_none(ON_c), sel_share_by_512=share_early(ON_s), sel_share_never=share_none(ON_s), cos_median_lead_checkpoints=med(lead_c), sel_median_lead_checkpoints=med(lead_s), cos_onsets={str(w): ON_c[w] for w, _ in tracked}, sel_onsets={str(w): ON_s[w] for w, _ in tracked}, recruitment_steps={str(w): step_of[w] for w, _ in tracked}),
           spearman_step0_rank_vs_recruitment=dict(cos=sp0, sel=sp0s), partB=PBs, partC=PCs, per_word_step0={str(w): PER[ALL[0]][w] for w, _ in tracked}, debiased=DEBs, persistence={str(n): v for n, v in PERS.items()}, cross_seed=dict(reference=REF, summary=XRs, n_reference_classes_with_speaker=(len(REFC) if REFC is not None else None)))
pj = PERS[LATE[-1]]; pr_ = {w: PERS[LATE[j]] for w, j in tracked}; pers_rec = med([float(AROWS[LATE[j]][w].float() @ AROWS[ALL[0]][w].float()) for w, j in tracked]); pers_all_at_rec = med([PERS[LATE[j]]["all"] for w, j in tracked]); res["persistence_at_recruitment"] = dict(tracked=pers_rec, all_rows=pers_all_at_rec)
s0 = summ_ck[ALL[0]]; s512 = summ_ck[512] if 512 in summ_ck else None; s1k = summ_ck[1000]
summ = (f"{tag} ({len(tracked)} eventual words, {m} rows): at step 0 the eventual row's normalised rank by cosine with its final class direction {s0['median_norm_rank_cos']:.2f} (top 100 for {s0['top100_cos']:.2f}; random row {s0['random_median_norm_rank_cos']:.2f}, other class {s0['other_median_norm_rank_cos']:.2f}), by selectivity {s0['median_norm_rank_sel']:.2f} (top 100 {s0['top100_sel']:.2f}); at step 512 {s512['median_norm_rank_cos']:.2f} / {s512['median_norm_rank_sel']:.2f}; at step 1000 {s1k['median_norm_rank_cos']:.2f} (top 100 {s1k['top100_cos']:.2f}) / {s1k['median_norm_rank_sel']:.2f} (top 100 {s1k['top100_sel']:.2f}); lineage onset (top tenth and staying) by step 512 for {res['onset']['cos_share_by_512']:.2f} (cosine) / {res['onset']['sel_share_by_512']:.2f} (selectivity), never for {res['onset']['cos_share_never']:.2f} / {res['onset']['sel_share_never']:.2f}, median lead before recruitment {res['onset']['cos_median_lead_checkpoints']} / {res['onset']['sel_median_lead_checkpoints']} checkpoints; Spearman of the step-0 rank with the recruitment index {sp0} (cosine) / {sp0s} (selectivity); "
        + f"two before recruitment the recruit beats its ten nearest competitors' median on read norm {PBs[2]['read_vs_comp']:.2f}, write norm {PBs[2]['write_vs_comp']:.2f}, read/write ratio {PBs[2]['ratio_vs_comp']:.2f} (relative read {PBs[2]['read_rel']:.2f}, write {PBs[2]['write_rel']:.2f}); against all non-words {PBs[2]['read_vs_all']:.2f} / {PBs[2]['write_vs_all']:.2f}; "
        + f"own write subtracted: step-0 rank by cosine with the final class direction {DEBs['final']['median_norm_rank']:.2f} (top 1000 {DEBs['final']['top1000']:.2f}; that direction's cosine with the final row {DEBs['final']['dir_cos_with_final_row']:.2f}), with the direction two before recruitment {DEBs['two_before']['median_norm_rank']:.2f} (top 1000 {DEBs['two_before']['top1000']:.2f}); persistence of the recruited row from step 0 to its recruitment {res['persistence_at_recruitment']['tracked']:.2f} against {res['persistence_at_recruitment']['all_rows']:.2f} for all rows; "
        + (f"cross-seed ({REF}: {res['cross_seed']['n_reference_classes_with_speaker']} classes with a speaker here): the speaker's step-0 rank by cosine {XRs[str(ALL[0])]['median_norm_rank_cos']:.2f} (top 100 {XRs[str(ALL[0])]['top100_cos']:.2f}, top 1000 {XRs[str(ALL[0])]['top1000_cos']:.2f}; random {XRs[str(ALL[0])]['random_median']:.2f}), by selectivity {XRs[str(ALL[0])]['median_norm_rank_sel']:.2f}; " if XRs else "")
        + f"identification given the class among non-words two before: top 10 by class ratio {PCs['rank_ratio']['top10']:.2f} (median rank {PCs['rank_ratio']['median']:.0f}), by usage {PCs['rank_usage']['top10']:.2f} ({PCs['rank_usage']['median']:.0f}), by decoder cosine {PCs['rank_decoder']['top10']:.2f} ({PCs['rank_decoder']['median']:.0f}), by the class-side score {PCs['rank_anymax']['top10']:.2f} ({PCs['rank_anymax']['median']:.0f}) | {time.time() - t0:.0f}s")
log(summ); record(f"e632_lottery_{tag}{sfx}", res, summ)
