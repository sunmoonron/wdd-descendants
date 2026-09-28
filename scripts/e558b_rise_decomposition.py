"""e558b (session 101): the entrant's rise between steps 8000 and 16000 split into the row's motion and the cloud's. For
each row, S in four combinations: its 8000 direction in the 8000 cloud, its 16000 direction in the 16000 cloud, its
16000 direction in the 8000 cloud and its 8000 direction in the 16000 cloud (dictionary swaps; the floor of each cloud
is kept). Rise = S(16|16) - S(8|8) = row part [S(16|8) - S(8|8)] + cloud part [S(16|16) - S(16|8)], and the other
order. For entrants (words at 16000 only), leavers (words at 8000 only), stayers and never-words, and over all rows the
correlation of each part with entering. e558's dictionary-only transplant is the S(16|8) column. Pre-registered:
 R1 (0.55) for entrants the cloud part exceeds the row part in the order row-then-cloud (the coalition, not the row,
    lifts the word: sessions 91 and 95);
 R2 (0.6) for leavers the cloud part is also the larger (erosion is the cloud's, session 90)."""
from s101_common import *
t0 = time.time(); B = 12; ids = pile_ids("pythia410")
m8, tok, fam = load_model("pythia410", revision="step8000"); m16, _, _ = load_model("pythia410", revision="step16000")
S8 = lm_states("pythia410", B=B, ids=ids, model=m8); S16 = lm_states("pythia410", B=B, ids=ids, model=m16); keepc = S8["keep"] & S16["keep"]
for S_ in (S8, S16): S_["U"] = unitr(S_["X"][keepc] - S_["X"][keepc].mean(0))
def Sof(U, A): st = stats(U, A, K); return st["S"], wordset(st["usage"])
S88, w8 = Sof(S8["U"], S8["A"]); S1616, w16 = Sof(S16["U"], S16["A"]); S168, _ = Sof(S8["U"], S16["A"]); S816, _ = Sof(S16["U"], S8["A"])
cos = (S8["A"] * S16["A"]).sum(1).cpu()
groups = dict(entrants=w16 & ~w8, leavers=w8 & ~w16, stayers=w8 & w16, never=~w8 & ~w16); res = dict(groups={})
for g, mk in groups.items():
    rise = S1616[mk] - S88[mk]; row_first = S168[mk] - S88[mk]; cloud_after_row = S1616[mk] - S168[mk]; cloud_first = S816[mk] - S88[mk]; row_after_cloud = S1616[mk] - S816[mk]
    r = dict(n=int(mk.sum()), S88=float(S88[mk].median()), S1616=float(S1616[mk].median()), S168=float(S168[mk].median()), S816=float(S816[mk].median()), rise=float(rise.median()), row_then_cloud=(float(row_first.median()), float(cloud_after_row.median())),
             cloud_then_row=(float(cloud_first.median()), float(row_after_cloud.median())), row_cos=float(cos[mk].median()), share_over_floor_168=float((S168[mk] >= 1).float().mean()), share_over_floor_816=float((S816[mk] >= 1).float().mean()),
             share_cloud_part_larger=float((cloud_after_row.abs() > row_first.abs()).float().mean()))
    res["groups"][g] = r; log(f"{g} ({r['n']}): S 8|8 {r['S88']:.2f}, 16|8 {r['S168']:.2f}, 8|16 {r['S816']:.2f}, 16|16 {r['S1616']:.2f}; rise {r['rise']:.2f} = row {r['row_then_cloud'][0]:.2f} + cloud {r['row_then_cloud'][1]:.2f} (row first) or cloud {r['cloud_then_row'][0]:.2f} + row {r['cloud_then_row'][1]:.2f} (cloud first); row cosine {r['row_cos']:.2f}; cloud part larger in {r['share_cloud_part_larger']:.2f}")
ent = (w16 & ~w8).float(); rowp = S168 - S88; cloudp = S1616 - S168
res["all_rows"] = dict(corr_entering_row_part=float(torch.corrcoef(torch.stack([ent, rowp]))[0, 1]), corr_entering_cloud_part=float(torch.corrcoef(torch.stack([ent, cloudp]))[0, 1]), corr_row_cloud_parts=float(torch.corrcoef(torch.stack([rowp, cloudp]))[0, 1]))
E, L = res["groups"]["entrants"], res["groups"]["leavers"]
summ = (f"rise of entrants ({E['n']}) from 8000 to 16000: S {E['S88']:.2f} -> {E['S1616']:.2f}; the 16000 direction in the 8000 cloud {E['S168']:.2f} (over the floor {E['share_over_floor_168']:.2f}), the 8000 direction in the 16000 cloud {E['S816']:.2f}; row-then-cloud split {E['row_then_cloud'][0]:.2f} + {E['row_then_cloud'][1]:.2f}, cloud-then-row {E['cloud_then_row'][0]:.2f} + {E['cloud_then_row'][1]:.2f} (cloud part larger in {E['share_cloud_part_larger']:.2f}); "
        f"leavers ({L['n']}): {L['S88']:.2f} -> {L['S1616']:.2f}, row-then-cloud {L['row_then_cloud'][0]:.2f} + {L['row_then_cloud'][1]:.2f} (cloud part larger in {L['share_cloud_part_larger']:.2f}); stayers row cosine {res['groups']['stayers']['row_cos']:.2f}, entrants {E['row_cos']:.2f}; over all rows entering correlates with the row part {res['all_rows']['corr_entering_row_part']:.2f} and the cloud part {res['all_rows']['corr_entering_cloud_part']:.2f} | {time.time() - t0:.0f}s")
log(summ); record("e558b_rise_decomposition", res, summ)
