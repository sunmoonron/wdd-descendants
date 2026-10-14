"""e654 (session 142; Gemini 4's idea 2, after e560 and e568): peeling hedge words off the wrong capital answers. On
e560's 160 prompts "The capital of X is" Pythia-410m is right 0.59 of the time (next token = the capital's first token),
and about half of its wrong answers are function words (" the", " in", " located"). Gemini's proposal: the answer is the
argmax of a running total x_L = x_fact + x_hedge, so decomposing the final-block state into its 16 native words and
subtracting the words that push function words should reveal capitals the model knew all along. Its two kill baselines:
(1) mask the function words in the logits directly; (2) subtract the mean state of the prompts. Function words = the 200
most frequent tokens of wikitext-2 train plus " located", " situated", " home" (declared before the run), minus any
capital's first token. A word pushes the token its signed logit signature W_U g(coef * row) raises most (g: the final
norm's centring and gain). Interventions, each read by the model's own final norm and unembedding, raw and with the
function words masked: none; the leave-one-out prompt mean subtracted (literal, and replaced by the corpus mean); the
function-pushing words peeled (native, and from the rotated dictionary); the words recurring in half the prompts'
supports peeled; a random peel of as many non-function words (control); and e480's native union of the non-function
words' top-5 tokens against the masked logits' top-n at the same set size. Scored on hedge-wrong items (raw answer a
function word), content-wrong items, and right items (kept). Arguments: [model] [--smoke]."""
from bx_common import *
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
args = [x for x in sys.argv[1:] if not x.startswith("--")]; name = args[0] if args else "pythia410"; SMOKE = "--smoke" in sys.argv; t0 = time.time()
M = load(name); tok = M["tok"]; WU = M["WU"]; V = WU.shape[0]; D = M["D"]
if SMOKE: CAPS = CAPS[:48]
NP = len(CAPS); capfirst = torch.tensor([tok(" " + c)["input_ids"][0] for _, c in CAPS], device=DEV)
tid = train_token_ids(M, 3000 if SMOKE else None); uni = np.bincount(np.concatenate([np.array(x) for x in tid]), minlength=V); cf = set(capfirst.tolist())
hedge_words = [w for w in (" located", " situated", " home") if len(tok(w)["input_ids"]) == 1]
STOP = sorted(set(int(i) for i in np.argsort(-uni)[:200] if int(i) not in cf) | set(tok(w)["input_ids"][0] for w in hedge_words) - cf)
isstop_tok = torch.zeros(V, dtype=torch.bool, device=DEV); isstop_tok[torch.tensor(STOP, device=DEV)] = True
print(f"[{name}] {NP} prompts; {len(STOP)} function words (hedge words added: {hedge_words}) ({time.time() - t0:.0f}s)", flush=True)
A, blk = dictionary(M); Ar = unitr(rotate(A))
TEw = wikitext(M, "test", 8 if SMOKE else 24, 128); Hc = []
for s0 in range(0, TEw.shape[0], 4): h, _ = final_states(M, TEw[s0:s0 + 4]); h = h[:, 1:].reshape(-1, D); Hc.append(h[~sinkmask(h)])
mu = torch.cat(Hc).mean(0)
H, LG = [], []
for c, _ in CAPS:
    ids = torch.tensor([tok(f"The capital of {c} is")["input_ids"]], device=DEV); h, lg = final_states(M, ids); H.append(h[0, -1]); LG.append(lg[0, -1])
H = torch.stack(H); LG = torch.stack(LG); assert float((readout(M, H) - LG).abs().max()) < 1e-2 * float(LG.abs().max()), "readout mismatch"
raw = LG.argmax(1); right = raw == capfirst; hedge = ~right & isstop_tok[raw]; content = ~right & ~isstop_tok[raw]
print(f"[{name}] accuracy {float(right.float().mean()):.3f}; wrong {int((~right).sum())}: hedge-wrong {int(hedge.sum())}, content-wrong {int(content.sum())} ({time.time() - t0:.0f}s)", flush=True)
scale = (H - mu).norm(dim=1); U = unitr(H - mu)
def decomp(Adict):
    sel, cof = omp(U, Adict); ga = gain(M, Adict[sel]); push = torch.stack([((cof[i][:, None] * ga[i]) @ WU.T).topk(5, -1).indices for i in range(NP)])  # [NP,16,5]
    return sel, cof, push
seln, cofn, pushn = decomp(A); selr, cofr, pushr = decomp(Ar)
stopn = isstop_tok[pushn[:, :, 0]]; stopr = isstop_tok[pushr[:, :, 0]]
def peel(Adict, sel, cof, mask): return H - scale[:, None] * ((mask.float() * cof)[..., None] * Adict[sel]).sum(1)
def argmax_of(h, masked):
    z = readout(M, h)
    if masked: z = z.masked_fill(isstop_tok[None], -1e9)
    return z.argmax(1)
cnt_atom = torch.bincount(seln.reshape(-1), minlength=A.shape[0]); recur = cnt_atom[seln] >= 0.5 * NP
hbar = (H.sum(0, keepdim=True) - H) / (NP - 1)
g = torch.Generator(device="cpu").manual_seed(654); rand_masks = []
for r_ in range(5):
    m_ = torch.zeros(NP, K, dtype=torch.bool)
    for i in range(NP):
        nonf = torch.nonzero(~stopn[i].cpu())[:, 0]; k_ = min(int(stopn[i].sum()), nonf.numel())
        if k_: m_[i, nonf[torch.randperm(nonf.numel(), generator=g)[:k_]]] = True
    rand_masks.append(m_.to(DEV))
STATES = {"none": H, "prompt_mean_subtracted": H - hbar, "prompt_mean_to_corpus_mean": H - hbar + mu, "function_words_peeled": peel(A, seln, cofn, stopn),
          "rotated_function_words_peeled": peel(Ar, selr, cofr, stopr), "recurring_words_peeled": peel(A, seln, cofn, recur)}
def score(pred): return dict(hedge=float((pred == capfirst)[hedge].float().mean()) if hedge.any() else None, content=float((pred == capfirst)[content].float().mean()) if content.any() else None,
                             wrong=float((pred == capfirst)[~right].float().mean()), n_wrong_recovered=int(((pred == capfirst) & ~right).sum()), kept_right=float((pred == capfirst)[right].float().mean()) if right.any() else None)
res = dict(model=name, n=NP, accuracy=float(right.float().mean()), n_wrong=int((~right).sum()), n_hedge=int(hedge.sum()), n_content=int(content.sum()), n_stop=len(STOP), hedge_words=hedge_words,
           mean_function_words_in_support=float(stopn.float().sum(1).mean()), mean_recurring_in_support=float(recur.float().sum(1).mean()), interventions={})
for sn, h in STATES.items():
    for masked in (False, True):
        k_ = sn + ("+mask" if masked else ""); res["interventions"][k_] = score(argmax_of(h, masked))
for masked in (False, True):
    sc_ = [score(argmax_of(peel(A, seln, cofn, rm), masked)) for rm in rand_masks]
    res["interventions"]["random_peel" + ("+mask" if masked else "")] = {k: (float(np.mean([s[k] for s in sc_])) if sc_[0][k] is not None else None) for k in sc_[0]}
for k_, v in res["interventions"].items(): print(f"[{name}] {k_:38s} recovers hedge {v['hedge']} content {v['content']} (wrong {v['n_wrong_recovered']} of {res['n_wrong']}), keeps right {v['kept_right']}", flush=True)
# e480's native union against the masked logits at the same set size
zm = LG.masked_fill(isstop_tok[None], -1e9); un_hit, ml_hit, usz = [], [], []
for i in range(NP):
    u_ = set(pushn[i][~stopn[i]][:, :5].reshape(-1).tolist()); n_ = max(len(u_), 1); usz.append(len(u_))
    un_hit.append(int(capfirst[i]) in u_); ml_hit.append(int(capfirst[i]) in set(zm[i].topk(n_).indices.tolist()))
un_hit = torch.tensor(un_hit, device=DEV); ml_hit = torch.tensor(ml_hit, device=DEV)
res["union"] = dict(mean_size=float(np.mean(usz)), native_union_wrong=float(un_hit[~right].float().mean()), masked_logits_same_size_wrong=float(ml_hit[~right].float().mean()), native_union_right=float(un_hit[right].float().mean()), masked_logits_same_size_right=float(ml_hit[right].float().mean()))
rank = lambda z, i: int((z[i] > z[i, capfirst[i]]).sum()) + 1
res["items_wrong"] = [dict(country=CAPS[i][0], capital=CAPS[i][1], kind="hedge" if bool(hedge[i]) else "content", raw=tok.decode([int(raw[i])]), masked=tok.decode([int(zm[i].argmax())]),
                           peeled_masked=tok.decode([int(argmax_of(STATES["function_words_peeled"][i:i + 1], True)[0])]), rank_raw=rank(LG, i), rank_masked=rank(zm, i)) for i in torch.nonzero(~right)[:, 0].tolist()]
I = res["interventions"]; m0 = I["none+mask"]["n_wrong_recovered"]
res["verdicts"] = dict(S1=(I["none+mask"]["hedge"] or 0) >= 0.30, S2=I["function_words_peeled+mask"]["n_wrong_recovered"] >= m0 + 3, S3=I["function_words_peeled"]["n_wrong_recovered"] > m0,
                       S4=I["recurring_words_peeled+mask"]["n_wrong_recovered"] > I["prompt_mean_subtracted+mask"]["n_wrong_recovered"],
                       S5=I["function_words_peeled+mask"]["n_wrong_recovered"] > I["rotated_function_words_peeled+mask"]["n_wrong_recovered"], S6=(I["function_words_peeled+mask"]["kept_right"] or 0) >= 0.90)
summ = (f"[{name}] peeling hedge words, {NP} capitals, accuracy {res['accuracy']:.3f}, wrong {res['n_wrong']} (hedge {res['n_hedge']}, content {res['n_content']}): wrong answers recovered by masking alone {m0}, "
        f"by the function-word peel {I['function_words_peeled']['n_wrong_recovered']} (with the mask {I['function_words_peeled+mask']['n_wrong_recovered']}), rotated peel+mask {I['rotated_function_words_peeled+mask']['n_wrong_recovered']}, "
        f"recurring peel+mask {I['recurring_words_peeled+mask']['n_wrong_recovered']}, prompt mean subtracted+mask {I['prompt_mean_subtracted+mask']['n_wrong_recovered']}, random peel+mask {I['random_peel+mask']['n_wrong_recovered']:.1f}; verdicts {res['verdicts']} ({time.time() - t0:.0f}s)")
save(f"e654_peel_{name}" + ("_smoke" if SMOKE else ""), res, summ); print("done", flush=True)
