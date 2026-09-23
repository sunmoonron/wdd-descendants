"""e365: when does the induction circuit's unit structure appear in training, relative to the functional funnel?
Pythia-410m at one checkpoint: induction loss on the repeated copy against the first copy; prefix-matching and
previous-token attention scores; the eight heads with the largest ablation effect and their pairwise interactions
against eight random heads (signed: sub-additive = series, super-additive = backup); joint non-additivity of the
eight; and, in the same run, the function dimension of the block-2 quotient at the middle level (e349's measure)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
from quot_common import Setup, pls, logit_scores, knn_cos, inside, unit
rev = sys.argv[1]; model, tok, fam = load_eager("pythia410", revision=rev); arch = Arch(model, fam); c = Cache("pythia410"); NB, NH = arch.NB, arch.NH; rng = random.Random(0)
ind, off, half = induction_batch(tok, c, n=12, half=128, seed=0); m_ind = lambda lg: induction_loss(lg, ind, off, half)
MA, MM, lg = capture_means(model, arch, ind); base = m_ind(lg); first = induction_loss(lg, ind, off, half, second=False).mean().item(); del lg
pref, prev = attention_scores(model, ind, off, half, NB, NH); heads = [(l, h) for l in range(NB) for h in range(NH)]
E1 = {x: m_ind(ablate(model, arch, ind, heads=[x], MA=MA, MM=MM)) - base for x in heads}; eff = torch.tensor([E1[x].mean().item() for x in heads]); order = eff.argsort(descending=True).tolist(); top = [heads[i] for i in order[:8]]; U = top + rng.sample([heads[i] for i in order[8:]], 8)
I, Sg = pairwise(model, arch, ind, m_ind, base, U, "head", MA, MM, E1); st = block_stats(I, Sg, 8); na = set_nonadditivity(model, arch, ind, m_ind, base, top, "head", MA, MM, E1)
res = dict(rev=rev, first_copy_loss=first, second_copy_loss=base.mean().item(), induction_gain=first - base.mean().item(), max_prefix_score=pref.max().item(), n_prefix_above_0p2=int((pref >= 0.2).sum()), max_prev_score=prev.max().item(), n_prev_above_0p3=int((prev >= 0.3).sum()), top_heads=[list(x) for x in top], top_effects=[E1[x].mean().item() for x in top], top_prefix=[pref[l, h].item() for l, h in top], top_prev=[prev[l, h].item() for l, h in top], interactions=st, nonadditivity_top8=na)
del model; torch.cuda.empty_cache()
S = Setup("pythia410", levels=[12], revision=rev); nat = S.natural(); F = nat["F"][12]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); dln = unit(nat["dl"]); Q = pls(Fc[tr], Z[tr], 64)
curve = {q: knn_cos(Fc @ Q[:, :q], dln, tr, te) for q in (1, 2, 4, 8, 16, 32, 64)}; full = knn_cos(Fc, dln, tr, te); res.update(quotient_dim=next((q for q in curve if curve[q] >= 0.9 * full), 64), quotient_full=full, quotient_curve={str(k): v for k, v in curve.items()}, K=S.K)
# basis-free image of the quotient: move the level-12 state along random unit combinations of the top-16 quotient
# directions (and, as the null, along random residual directions) at the natural descendant scale, record the centred
# logit change sketched on the same fixed random 256-dimensional vocabulary projection as e360, and keep (i) the
# top-16 principal subspace of the per-token images, (ii) the least-squares linear response map from quotient
# coordinates to the sketch, (iii) the gain (image norm per unit move) of quotient against random directions.
Q16 = Q[:, :16].contiguous(); amp = F.norm(dim=1).median(); gq = torch.Generator().manual_seed(7); Uq = unit(torch.randn(1024, 16, generator=gq)).to(DEV); Vq = Uq @ Q16.T; Vr = unit(torch.randn(1024, S.D, generator=gq).to(DEV)); Vsz = nat["dl"].shape[1]; Rv = (torch.randn(Vsz, 256, generator=torch.Generator().manual_seed(1234)) / 16).to(DEV); del nat
oq = S.inject_family(Vq, 13, amp=amp, seed=1); Yq = (oq["dl"] @ Rv) / amp; Xq = Uq[oq["a"]]; del oq
orr = S.inject_family(Vr, 13, amp=amp, seed=2); Yr = (orr["dl"] @ Rv) / amp; del orr
Mq = torch.linalg.lstsq(Xq, Yq).solution
def topsub(Y, k=16):
    Yc = Y - Y.mean(0, keepdim=True); U_, S_, Vh = torch.linalg.svd(Yc, full_matrices=False); return Vh[:k].T.contiguous(), S_[:k]
Bq, sq = topsub(Yq); Br, sr = topsub(Yr); gain_q = Yq.norm(dim=1).median().item(); gain_r = Yr.norm(dim=1).median().item()
res.update(image_gain_quotient=gain_q, image_gain_random=gain_r, image_gain_ratio=gain_q / max(gain_r, 1e-12), image_overlap_quotient_vs_random_same_ckpt=inside(Bq, Br), image_linear_r2=1 - ((Yq - Xq @ Mq) ** 2).sum().item() / ((Yq - Yq.mean(0)) ** 2).sum().item(), descendant_scale=amp.item())
os.makedirs(os.path.join(RESULTS, "e365"), exist_ok=True); torch.save(dict(rev=rev, Q16=Q16.cpu(), Mq=Mq.cpu(), Bq=Bq.cpu(), sq=sq.cpu(), Br=Br.cpu(), sr=sr.cpu(), gain_q=gain_q, gain_r=gain_r, amp=amp.item()), os.path.join(RESULTS, "e365", f"{rev}.pt"))
log(f"{rev}: induction gain {res['induction_gain']:.2f} nats (first {first:.2f}, second {res['second_copy_loss']:.2f}); max prefix score {res['max_prefix_score']:.2f} ({res['n_prefix_above_0p2']} heads >= 0.2), max previous-token {res['max_prev_score']:.2f}; top-8 interaction {st['I_top']:.2f} vs random {st['I_random']:.2f}, signed {st['signed_top']:+.2f}; non-additivity {na['nonadditivity']:.2f} | quotient dim {res['quotient_dim']} (full {full:.2f}); quotient image gain {gain_q:.3g} vs random {gain_r:.3g} (ratio {res['image_gain_ratio']:.2f}), image overlap with random-direction image {res['image_overlap_quotient_vs_random_same_ckpt']:.2f}, linear R2 {res['image_linear_r2']:.2f}")
record(f"e365_inddev_{rev}", res, f"gain {res['induction_gain']:.2f} prefix {res['max_prefix_score']:.2f} I {st['I_top']:.2f}/{st['I_random']:.2f} signed {st['signed_top']:+.2f} NA {na['nonadditivity']:.2f} | qdim {res['quotient_dim']} full {full:.2f} | image gain ratio {res['image_gain_ratio']:.2f} overlap-with-random {res['image_overlap_quotient_vs_random_same_ckpt']:.2f}")
