"""e560 (session 101): does the ledger see a factual error coming? 170 prompts "The capital of X is" for real countries
and 30 for invented ones. Pythia-410m's next token is right when it is the first token of the capital. At the last
position of the prompt, block 12: S (the largest projection over the floor, calibrated on the Pile states), the
cutoff, the number of words among the 16 atoms selected, the native codes (256), the rotated-dictionary codes (256),
the raw state (1024), and the model's own confidence (top-1 probability, entropy). Right against wrong, for the real
countries (block 12 by default; the block is the script's argument, run also at 6, 18 and 22): the medians of the ledger scalars, and five-fold logistic-regression AUCs of each feature set (with the
confidence alone as the baseline the ledger must beat). Invented countries against real ones: the same scalars.
Pre-registered (probabilities are honest guesses):
 F1 (0.5) wrong answers sit at a lower S than right ones (the state is less spoken when the fact is missing);
 F2 (0.65) AUC raw > native codes > rotated codes;
 F3 (0.55) native codes beat the model's confidence alone;
 F4 (0.5) invented countries are less spoken (lower S, fewer words) than real ones."""
from s101_common import *
CAPS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"), ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Greece", "Athens"),
 ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("India", "New Delhi"), ("Pakistan", "Islamabad"), ("Afghanistan", "Kabul"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Malaysia", "Kuala Lumpur"),
 ("Philippines", "Manila"), ("South Korea", "Seoul"), ("North Korea", "Pyongyang"), ("Australia", "Canberra"), ("New Zealand", "Wellington"), ("Canada", "Ottawa"), ("Mexico", "Mexico City"), ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"),
 ("Peru", "Lima"), ("Colombia", "Bogota"), ("Venezuela", "Caracas"), ("Cuba", "Havana"), ("Jamaica", "Kingston"), ("Ireland", "Dublin"), ("Scotland", "Edinburgh"), ("Wales", "Cardiff"), ("England", "London"), ("Norway", "Oslo"),
 ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Iceland", "Reykjavik"), ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Switzerland", "Bern"), ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"), ("Hungary", "Budapest"),
 ("Romania", "Bucharest"), ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"), ("Croatia", "Zagreb"), ("Ukraine", "Kyiv"), ("Belarus", "Minsk"), ("Lithuania", "Vilnius"), ("Latvia", "Riga"), ("Estonia", "Tallinn"), ("Georgia", "Tbilisi"),
 ("Armenia", "Yerevan"), ("Azerbaijan", "Baku"), ("Kazakhstan", "Astana"), ("Uzbekistan", "Tashkent"), ("Mongolia", "Ulaanbaatar"), ("Nepal", "Kathmandu"), ("Bangladesh", "Dhaka"), ("Sri Lanka", "Colombo"), ("Myanmar", "Naypyidaw"), ("Cambodia", "Phnom Penh"),
 ("Laos", "Vientiane"), ("Saudi Arabia", "Riyadh"), ("Syria", "Damascus"), ("Lebanon", "Beirut"), ("Jordan", "Amman"), ("Israel", "Jerusalem"), ("Kuwait", "Kuwait City"), ("Qatar", "Doha"), ("Oman", "Muscat"), ("Yemen", "Sanaa"),
 ("Morocco", "Rabat"), ("Algeria", "Algiers"), ("Tunisia", "Tunis"), ("Libya", "Tripoli"), ("Sudan", "Khartoum"), ("Ethiopia", "Addis Ababa"), ("Kenya", "Nairobi"), ("Tanzania", "Dodoma"), ("Uganda", "Kampala"), ("Nigeria", "Abuja"),
 ("Ghana", "Accra"), ("Senegal", "Dakar"), ("Mali", "Bamako"), ("Niger", "Niamey"), ("Chad", "N'Djamena"), ("Cameroon", "Yaounde"), ("Angola", "Luanda"), ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"), ("Mozambique", "Maputo"),
 ("Madagascar", "Antananarivo"), ("South Africa", "Pretoria"), ("Namibia", "Windhoek"), ("Botswana", "Gaborone"), ("Rwanda", "Kigali"), ("Somalia", "Mogadishu"), ("Eritrea", "Asmara"), ("Liberia", "Monrovia"), ("Sierra Leone", "Freetown"), ("Ivory Coast", "Yamoussoukro"),
 ("Guinea", "Conakry"), ("Togo", "Lome"), ("Benin", "Porto-Novo"), ("Gabon", "Libreville"), ("Congo", "Brazzaville"), ("Malawi", "Lilongwe"), ("Lesotho", "Maseru"), ("Mauritius", "Port Louis"), ("Cyprus", "Nicosia"), ("Malta", "Valletta"),
 ("Luxembourg", "Luxembourg"), ("Monaco", "Monaco"), ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"), ("Czech Republic", "Prague"), ("Albania", "Tirana"), ("North Macedonia", "Skopje"), ("Bosnia and Herzegovina", "Sarajevo"), ("Montenegro", "Podgorica"), ("Moldova", "Chisinau"),
 ("Bolivia", "La Paz"), ("Ecuador", "Quito"), ("Paraguay", "Asuncion"), ("Uruguay", "Montevideo"), ("Guyana", "Georgetown"), ("Suriname", "Paramaribo"), ("Panama", "Panama City"), ("Costa Rica", "San Jose"), ("Nicaragua", "Managua"), ("Honduras", "Tegucigalpa"),
 ("Guatemala", "Guatemala City"), ("El Salvador", "San Salvador"), ("Haiti", "Port-au-Prince"), ("Dominican Republic", "Santo Domingo"), ("Bahamas", "Nassau"), ("Barbados", "Bridgetown"), ("Trinidad and Tobago", "Port of Spain"), ("Fiji", "Suva"), ("Papua New Guinea", "Port Moresby"), ("Samoa", "Apia"),
 ("Taiwan", "Taipei"), ("Bhutan", "Thimphu"), ("Maldives", "Male"), ("Bahrain", "Manama"), ("United Arab Emirates", "Abu Dhabi"), ("Tajikistan", "Dushanbe"), ("Kyrgyzstan", "Bishkek"), ("Turkmenistan", "Ashgabat"), ("Brunei", "Bandar Seri Begawan"), ("Singapore", "Singapore")]
FAKE = ["Zorbania", "Quellmar", "Vastoria", "Brenwick", "Tolmeria", "Ashkandar", "Velmoria", "Kestrelia", "Orvantia", "Drellmark", "Sulvenia", "Pravonia", "Maldoria", "Threnholm", "Ilvarra", "Corvantis", "Nemoria", "Galdorin", "Ustrakhan", "Fennmark",
 "Lorvath", "Zendaria", "Ombrellia", "Tavistria", "Quorvane", "Halderon", "Vespuria", "Meridonia", "Kalvistan", "Endrakia"]
t0 = time.time(); B = int(sys.argv[1]) if len(sys.argv) > 1 else 12; TAG = "" if B == 12 else f"_b{B}"; model, tok, fam = load_model("pythia410"); arch = Arch(model, fam)
P = lm_states("pythia410", B=B, model=model); cal = floor_calibration(P["U"], P["A"]); A = P["A"]; st = stats(P["U"], A, K); words = torch.nonzero(wordset(st["usage"]))[:, 0]
Ar = unitr(rotate(A, seed=7)); sr = stats(P["U"], Ar, K); wr = torch.nonzero(wordset(sr["usage"]))[:, 0]; mu = P["mu"]
def run(names):
    prompts = [f"The capital of {n} is" for n in names]; enc = [tok(p)["input_ids"] for p in prompts]; X = []; LG = []
    for e in enc:
        ids = torch.tensor([e], device=DEV); cap = {}
        h = arch.layers[B].register_forward_hook(lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[0, -1].detach().float()))
        try:
            with torch.no_grad(): lg = model(ids).logits[0, -1].float()
        finally: h.remove()
        X.append(cap["x"]); LG.append(lg)
    return torch.stack(X), torch.stack(LG)
Xr, LGr = run([c for c, _ in CAPS]); Xf, LGf = run(FAKE)
first = torch.tensor([tok(" " + cap)["input_ids"][0] for _, cap in CAPS], device=DEV); pred = LGr.argmax(1); correct = (pred == first).cpu()
pr = LGr.softmax(1); conf = pr.max(1).values.cpu(); ent = -(pr * pr.clamp_min(1e-12).log()).sum(1).cpu(); prf = LGf.softmax(1); conff = prf.max(1).values.cpu()
log(f"accuracy {float(correct.float().mean()):.2f} on {len(CAPS)} countries; examples wrong: " + ", ".join(f"{CAPS[i][0]}->{tok.decode([int(pred[i])])!r}" for i in torch.nonzero(~correct)[:8, 0].tolist()))
def feats(X, Dd, wsel):
    U = unitr(X - mu); sel, cof, err = omp(U, Dd, K, batch=1024, record_err=True); C = torch.zeros(U.shape[0], wsel.numel(), device=DEV)
    pos = torch.full((Dd.shape[0],), -1, dtype=torch.long, device=DEV); pos[wsel] = torch.arange(wsel.numel(), device=DEV); p = pos[sel]; ok = p >= 0; C[torch.nonzero(ok)[:, 0], p[ok]] = cof[ok]
    L = floor_of(U, cal); ratio = (U @ Dd.T).abs() / L[:, None]; return dict(U=U, C=C, nwords=ok.sum(1).float().cpu(), S=ratio.max(1).values.cpu(), cut=ratio.topk(K, dim=1).values[:, -1].cpu(), fvu=(err[:, -1] / U.pow(2).sum(1)).cpu())
fr = feats(Xr, A, words); ff = feats(Xf, A, words); rr = feats(Xr, Ar, wr)
res = dict(n=len(CAPS), accuracy=float(correct.float().mean()), scalars={}, auc={}, fake={})
for k in ("S", "cut", "nwords", "fvu"):
    res["scalars"][k] = dict(right=float(fr[k][correct].median()), wrong=float(fr[k][~correct].median()), fake=float(ff[k].median()), auc_right_vs_wrong=auc(fr[k], correct))
res["scalars"]["confidence"] = dict(right=float(conf[correct].median()), wrong=float(conf[~correct].median()), fake=float(conff.median()), auc_right_vs_wrong=auc(conf, correct))
res["scalars"]["entropy"] = dict(right=float(ent[correct].median()), wrong=float(ent[~correct].median()), auc_right_vs_wrong=auc(-ent, correct))
for k, v in res["scalars"].items(): log(f"{k}: right {v['right']:.3f}, wrong {v['wrong']:.3f}" + (f", invented {v['fake']:.3f}" if "fake" in v else "") + f"; AUC right vs wrong {v['auc_right_vs_wrong']:.3f}")
def cv_auc(F, y, folds=5, l2=1e-1):
    n = F.shape[0]; perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)); sc = torch.zeros(n)
    for f in range(folds):
        te = perm[f::folds]; tr = torch.tensor([i for i in perm.tolist() if i not in set(te.tolist())]); sc[te] = logreg(F[tr].to(DEV), y[tr].to(DEV), F[te].to(DEV), l2=l2).cpu()
    return auc(sc, y)
y = correct.clone()
FS = {"raw": fr["U"].cpu(), "native_codes": fr["C"].cpu(), "rotated_codes": rr["C"].cpu(), "ledger_scalars": torch.stack([fr["S"], fr["cut"], fr["nwords"], fr["fvu"]], 1), "confidence": torch.stack([conf, ent], 1),
      "native_codes_plus_confidence": torch.cat([fr["C"].cpu(), torch.stack([conf, ent], 1)], 1), "ledger_scalars_plus_confidence": torch.stack([fr["S"], fr["cut"], fr["nwords"], fr["fvu"], conf, ent], 1)}
for k, F in FS.items(): res["auc"][k] = cv_auc(F, y, l2=1e-1 if F.shape[1] > 16 else 1e-3); log(f"AUC right vs wrong from {k} ({F.shape[1]}): {res['auc'][k]:.3f}")
res["fake"] = dict(S_real=float(fr["S"].median()), S_fake=float(ff["S"].median()), nwords_real=float(fr["nwords"].median()), nwords_fake=float(ff["nwords"].median()), auc_real_vs_fake_S=auc(torch.cat([fr["S"], ff["S"]]), torch.cat([torch.ones(len(CAPS)), torch.zeros(len(FAKE))])),
                   auc_real_vs_fake_conf=auc(torch.cat([conf, conff]), torch.cat([torch.ones(len(CAPS)), torch.zeros(len(FAKE))])), fake_top_answers=[tok.decode([int(i)]) for i in LGf.argmax(1)[:10]])
res["wrong_examples"] = [(CAPS[i][0], tok.decode([int(pred[i])])) for i in torch.nonzero(~correct)[:, 0].tolist()][:40]
sc = res["scalars"]; a = res["auc"]
summ = (f"capitals, Pythia-410m: accuracy {res['accuracy']:.2f} of {len(CAPS)}; S right/wrong/invented {sc['S']['right']:.2f}/{sc['S']['wrong']:.2f}/{sc['S']['fake']:.2f} (AUC {sc['S']['auc_right_vs_wrong']:.2f}), words among 16 {sc['nwords']['right']:.0f}/{sc['nwords']['wrong']:.0f}/{sc['nwords']['fake']:.0f}, confidence {sc['confidence']['right']:.2f}/{sc['confidence']['wrong']:.2f}/{sc['confidence']['fake']:.2f} (AUC {sc['confidence']['auc_right_vs_wrong']:.2f}); "
        f"five-fold AUC right vs wrong: raw {a['raw']:.2f}, native codes {a['native_codes']:.2f}, rotated codes {a['rotated_codes']:.2f}, ledger scalars {a['ledger_scalars']:.2f}, confidence {a['confidence']:.2f}, codes+confidence {a['native_codes_plus_confidence']:.2f}, scalars+confidence {a['ledger_scalars_plus_confidence']:.2f}; "
        f"real vs invented by S {res['fake']['auc_real_vs_fake_S']:.2f}, by confidence {res['fake']['auc_real_vs_fake_conf']:.2f} | {time.time() - t0:.0f}s")
res["block"] = B; summ = f"block {B}: " + summ; log(summ); record(f"e560_facts{TAG}", res, summ)
