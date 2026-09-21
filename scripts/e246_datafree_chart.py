"""e246: a data-free descendant chart. Transplant images of the K candidate neurons (block 2) built on RANDOM-TOKEN
sequences (uniform token ids) instead of text, then used to identify natural WikiText footprints at b+2 and L; compared
with images built on WikiText contexts (zero-shot as in e242) and with fitted natural centroids. If the random-token
images work, the chart needs the model and nothing else."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; levels = sorted({b + 2, L}); run = make_runner(model, arch, c, ids_seq, levels, NT); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(b, tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); med = torch.stack([tc[idx][lab_i == k].median() for k in range(K)])
torch.manual_seed(0); vocab = arch.emb[0].shape[0]; ids_rand = torch.randint(0, vocab, (NS, CTX), device=DEV); ids_rand[:, 0] = ids_seq[:, 0]; run_r = make_runner(model, arch, c, ids_rand, levels, NT); S0r = run_r(); typr = typical_mask(S0r[L])
def build(runner, base, mask, natural_led=None):
    acc = {lv: torch.zeros(K, c.D, device=DEV) for lv in levels}; cnt = torch.zeros(K, device=DEV); pool = torch.nonzero(mask)[:, 0]; torch.manual_seed(1)
    for p in range(3):
        assign = torch.randint(0, K, (len(pool),), device=DEV); f2, a2 = pool, assign
        if natural_led is not None:
            act = (natural_led[pool].abs() >= 0.05 * natural_led[pool].abs().max(1, keepdim=True).values); ok = ~act[torch.arange(len(pool), device=DEV), keep[assign]]; f2, a2 = pool[ok], assign[ok]
        inj = torch.zeros(NT, c.D, device=DEV); inj[f2] = med[a2][:, None] * R[keep[a2]]; S2 = runner(inject=inj, inject_block=b + 1)
        for lv in levels: acc[lv].index_add_(0, a2, (S2[lv] - base[lv])[f2])
        cnt.index_add_(0, a2, torch.ones(len(a2), device=DEV))
    return {lv: unit(acc[lv] / cnt.clamp_min(1)[:, None]) for lv in levels}
img_text = build(run, S0, typ & ~torch.isin(torch.arange(NT, device=DEV), idx), led); img_rand = build(run_r, S0r, typr); torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; out = {}
for lv in levels:
    F = S0[lv] - S1[lv]; nat = centroids(F[idx[split]], lab_i[split], K); te = idx[~split]; lte = lab_i[~split]
    out[lv] = dict(K=K, chance=1.0 / K, random_token_images=accuracy(F[te], img_rand[lv], lte), text_images=accuracy(F[te], img_text[lv], lte), natural_centroids=accuracy(F[te], nat, lte), cos_rand_vs_text_images=((img_rand[lv] * img_text[lv]).sum(1)).median().item(), cos_rand_images_vs_natural=((img_rand[lv] * nat).sum(1)).median().item())
    log(f"{tag} level {lv} (K {K}, chance {1 / K:.2f}): natural footprints identified by random-token images {out[lv]['random_token_images']:.2f}, by text images {out[lv]['text_images']:.2f}, by fitted centroids {out[lv]['natural_centroids']:.2f} | cos(random-token image, text image) {out[lv]['cos_rand_vs_text_images']:.2f}, cos(random-token image, natural centroid) {out[lv]['cos_rand_images_vs_natural']:.2f}")
record(f"e246_datafree_{tag}", dict(model=tag, b=b, L=L, per_level={str(k): v for k, v in out.items()}), " | ".join(f"lv{lv}: random-token images {v['random_token_images']:.2f}, text images {v['text_images']:.2f}, fitted centroids {v['natural_centroids']:.2f} (chance {v['chance']:.2f}); image cos rand/text {v['cos_rand_vs_text_images']:.2f}" for lv, v in out.items()))
