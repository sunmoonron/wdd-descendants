"""e448: is the native vocabulary language-independent? (vision: words carry meaning, not surface form)
e439 showed descriptions almost never name the current token, and e440 that what context adds is described word-level.
If the words carry meaning, a sentence and its translation should be described with the same native words.
Multilingual models are known to share a language-independent continuous "semantic hub" in middle layers (Wendler et
al. 2024; Wu et al. 2024). Whether the model's own discrete vocabulary is shared has not been tested.
Data: 32 simple sentences written for this test, each in English, French, Spanish and German. Model (argument):
Qwen2.5-0.5B (multilingual) or SmolLM2-135M (mostly English, as a contrast).
At a quarter, half and three quarters of the depth:
- every position's state (after the first token) is described by 16 own words (OMP);
- each sentence becomes a tf-idf weighted bag of the words used;
- translations are retrieved across languages by cosine similarity (top-1 among 32; chance 1/32).
Compared with:
- the same pipeline over the rotated dictionary;
- the mean-pooled raw states (the continuous hub);
- a lexical baseline (bag of the sentence's token ids).
Also: the native words shared by the most translation quadruples, and what their unembedding promotes.
Pre-registered (honest guesses):
 I1 (0.8) native-word bags retrieve translations far above chance at the middle depth in Qwen;
 I2 (0.55) native words beat rotated words;
 I3 (0.3) native words match or beat the pooled raw states, plain or whitened (whitening is the continuous analogue of idf);
 I4 (0.5) retrieval peaks at the middle depth, not at a quarter or three quarters."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
S = {
"en": ["The cat is sleeping on the warm sofa.", "My brother bought a new red car yesterday.", "The children are playing in the park after school.", "She drinks a cup of coffee every morning.",
       "The train to the city leaves at eight o'clock.", "We ate fresh bread and cheese for lunch.", "The old man walks his dog along the river.", "It rained all day, so we stayed at home.",
       "The teacher wrote the answer on the board.", "I am reading a book about the history of Rome.", "The doctor told him to rest for a week.", "Our neighbors have three small children.",
       "The museum is closed on Mondays.", "He forgot his keys in the car.", "The sun rises in the east and sets in the west.", "My sister works in a hospital in the capital.",
       "They are building a new bridge over the river.", "The soup is too hot to eat.", "We will travel to the mountains next summer.", "The baby started crying when the dog barked.",
       "The shop sells fruit, vegetables and flowers.", "She lost her phone at the train station.", "The students must finish their homework before Friday.", "The birds fly south when winter comes.",
       "My grandfather was born in a small village.", "The water in the lake is very cold.", "He plays the guitar in a band.", "The restaurant was full, so we went home.",
       "I need to buy milk and eggs.", "The window is open because it is hot.", "The company hired twenty new workers.", "The moon is bright tonight."],
"fr": ["Le chat dort sur le canapé chaud.", "Mon frère a acheté une nouvelle voiture rouge hier.", "Les enfants jouent dans le parc après l'école.", "Elle boit une tasse de café tous les matins.",
       "Le train pour la ville part à huit heures.", "Nous avons mangé du pain frais et du fromage pour le déjeuner.", "Le vieil homme promène son chien le long de la rivière.", "Il a plu toute la journée, alors nous sommes restés à la maison.",
       "Le professeur a écrit la réponse au tableau.", "Je lis un livre sur l'histoire de Rome.", "Le médecin lui a dit de se reposer pendant une semaine.", "Nos voisins ont trois petits enfants.",
       "Le musée est fermé le lundi.", "Il a oublié ses clés dans la voiture.", "Le soleil se lève à l'est et se couche à l'ouest.", "Ma sœur travaille dans un hôpital de la capitale.",
       "Ils construisent un nouveau pont sur la rivière.", "La soupe est trop chaude pour être mangée.", "Nous voyagerons à la montagne l'été prochain.", "Le bébé a commencé à pleurer quand le chien a aboyé.",
       "Le magasin vend des fruits, des légumes et des fleurs.", "Elle a perdu son téléphone à la gare.", "Les élèves doivent finir leurs devoirs avant vendredi.", "Les oiseaux volent vers le sud quand l'hiver arrive.",
       "Mon grand-père est né dans un petit village.", "L'eau du lac est très froide.", "Il joue de la guitare dans un groupe.", "Le restaurant était plein, alors nous sommes rentrés à la maison.",
       "J'ai besoin d'acheter du lait et des œufs.", "La fenêtre est ouverte parce qu'il fait chaud.", "L'entreprise a embauché vingt nouveaux employés.", "La lune est brillante ce soir."],
"es": ["El gato duerme en el sofá caliente.", "Mi hermano compró un coche rojo nuevo ayer.", "Los niños juegan en el parque después de la escuela.", "Ella bebe una taza de café todas las mañanas.",
       "El tren a la ciudad sale a las ocho.", "Comimos pan fresco y queso para el almuerzo.", "El anciano pasea a su perro a lo largo del río.", "Llovió todo el día, así que nos quedamos en casa.",
       "El profesor escribió la respuesta en la pizarra.", "Estoy leyendo un libro sobre la historia de Roma.", "El médico le dijo que descansara durante una semana.", "Nuestros vecinos tienen tres hijos pequeños.",
       "El museo está cerrado los lunes.", "Él olvidó sus llaves en el coche.", "El sol sale por el este y se pone por el oeste.", "Mi hermana trabaja en un hospital de la capital.",
       "Están construyendo un puente nuevo sobre el río.", "La sopa está demasiado caliente para comerla.", "Viajaremos a las montañas el próximo verano.", "El bebé empezó a llorar cuando el perro ladró.",
       "La tienda vende fruta, verduras y flores.", "Ella perdió su teléfono en la estación de tren.", "Los estudiantes deben terminar sus deberes antes del viernes.", "Los pájaros vuelan hacia el sur cuando llega el invierno.",
       "Mi abuelo nació en un pueblo pequeño.", "El agua del lago está muy fría.", "Él toca la guitarra en una banda.", "El restaurante estaba lleno, así que nos fuimos a casa.",
       "Necesito comprar leche y huevos.", "La ventana está abierta porque hace calor.", "La empresa contrató a veinte trabajadores nuevos.", "La luna está brillante esta noche."],
"de": ["Die Katze schläft auf dem warmen Sofa.", "Mein Bruder hat gestern ein neues rotes Auto gekauft.", "Die Kinder spielen nach der Schule im Park.", "Sie trinkt jeden Morgen eine Tasse Kaffee.",
       "Der Zug in die Stadt fährt um acht Uhr ab.", "Wir haben zum Mittagessen frisches Brot und Käse gegessen.", "Der alte Mann geht mit seinem Hund am Fluss spazieren.", "Es hat den ganzen Tag geregnet, also sind wir zu Hause geblieben.",
       "Der Lehrer hat die Antwort an die Tafel geschrieben.", "Ich lese ein Buch über die Geschichte Roms.", "Der Arzt sagte ihm, er solle sich eine Woche lang ausruhen.", "Unsere Nachbarn haben drei kleine Kinder.",
       "Das Museum ist montags geschlossen.", "Er hat seine Schlüssel im Auto vergessen.", "Die Sonne geht im Osten auf und im Westen unter.", "Meine Schwester arbeitet in einem Krankenhaus in der Hauptstadt.",
       "Sie bauen eine neue Brücke über den Fluss.", "Die Suppe ist zu heiß zum Essen.", "Wir werden nächsten Sommer in die Berge reisen.", "Das Baby fing an zu weinen, als der Hund bellte.",
       "Der Laden verkauft Obst, Gemüse und Blumen.", "Sie hat ihr Handy am Bahnhof verloren.", "Die Schüler müssen ihre Hausaufgaben vor Freitag erledigen.", "Die Vögel fliegen nach Süden, wenn der Winter kommt.",
       "Mein Großvater wurde in einem kleinen Dorf geboren.", "Das Wasser im See ist sehr kalt.", "Er spielt Gitarre in einer Band.", "Das Restaurant war voll, also sind wir nach Hause gegangen.",
       "Ich muss Milch und Eier kaufen.", "Das Fenster ist offen, weil es heiß ist.", "Die Firma hat zwanzig neue Arbeiter eingestellt.", "Der Mond ist heute Nacht hell."]}
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D
LANGS = list(S); NS = len(S["en"]); DEPTHS = sorted({arch.NB // 4, arch.NB // 2, (3 * arch.NB) // 4})
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; cen = EA["cen_ids"][:16].to(DEV); del EA
enc = {lg: [torch.tensor(tok(s, add_special_tokens=False)["input_ids"], device=DEV) for s in S[lg]] for lg in LANGS}
def sent_states(Ls):
    """{depth: list over (lang, sentence) of [n_tokens-1, D] states (first token dropped)}"""
    out = {Lb: [] for Lb in Ls}
    for lg in LANGS:
        for ids in enc[lg]:
            st = block_states(model, arch, ids[None], Ls, chunk=1)
            for Lb in Ls: out[Lb].append(st[Lb][0])
    return out
ST = sent_states(DEPTHS); res = dict(model=name, depths=DEPTHS, n_sentences=NS, langs=LANGS, retrieval={}, shared_words={})
def retrieval(V):
    """V: [4*NS, F] sentence vectors (lang-major). Mean top-1 translation retrieval over the 12 ordered language pairs, and the similarity gap"""
    Vn = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-9); acc, gap = [], []
    for i, l1 in enumerate(LANGS):
        for j, l2 in enumerate(LANGS):
            if i == j: continue
            sim = Vn[i * NS:(i + 1) * NS] @ Vn[j * NS:(j + 1) * NS].T; acc.append((sim.argmax(1) == torch.arange(NS, device=DEV)).float().mean().item())
            off = sim[~torch.eye(NS, dtype=torch.bool, device=DEV)]; gap.append(((sim.diag().mean() - off.mean()) / off.std().clamp_min(1e-9)).item())
    return sum(acc) / len(acc), sum(gap) / len(gap)
def tfidf(C):
    df = (C > 0).float().sum(0); idf = torch.log(C.shape[0] / df.clamp_min(1)); return C * idf[None]
lex = torch.zeros(len(LANGS) * NS, int(max(int(e.max()) for l in LANGS for e in enc[l])) + 1, device=DEV)
for i, e in enumerate([e for l in LANGS for e in enc[l]]): lex[i].index_add_(0, e, torch.ones_like(e, dtype=torch.float))
res["retrieval"]["lexical_tokens"] = retrieval(tfidf(lex))
for Lb in DEPTHS:
    A, lab = build_dictionary(arch, blocks=list(range(Lb + 1))); typ = lab["type"].to(DEV)
    ref = block_states(model, arch, cen, [Lb], chunk=4)[Lb]; mu = ref[~sinkmask(ref)].mean(0)
    allx = torch.cat(ST[Lb]); seg = torch.cat([torch.full((x.shape[0],), i, device=DEV) for i, x in enumerate(ST[Lb])]); Xc = allx - mu
    out = {}
    for vn, V in (("native", A), ("rotated", rotate(A, seed=7))):
        sel, cof, _ = omp(Xc, V, 16, batch=256, record_err=False)
        C = torch.zeros(len(ST[Lb]), V.shape[0], device=DEV); C.index_put_((seg[:, None].expand_as(sel).long().flatten(), sel.flatten()), cof.abs().flatten(), accumulate=True)
        out[vn] = retrieval(tfidf(C))
        if vn == "native": out["native_no_idf"] = retrieval(C)
        if vn == "native":
            # words used in all four versions of the most sentences
            used = (C > 0).view(len(LANGS), NS, -1); allfour = used.all(0).float().sum(0); mlp_any = allfour.clone(); top = allfour.topk(10)
            WU = model.get_output_embeddings().weight.detach().float(); fl = model.transformer.ln_f if fam == "gpt2" else (model.gpt_neox.final_layer_norm if fam == "neox" else model.model.norm)
            gf = getattr(fl, "weight", None); gf = gf.detach().float() if gf is not None else torch.ones(D, device=DEV)
            res["shared_words"][str(Lb)] = [dict(word=int(w), type=int(typ[w]), n_quadruples=int(v), promotes=[tok.decode([int(t)]) for t in ((A[w] * gf) @ WU.T).topk(5).indices]) for v, w in zip(top.values, top.indices)]
            q = allfour[allfour > 0]; out["words_shared_by_all_four"] = dict(n_words=int((allfour > 0).sum()), mean_quadruples=q.mean().item() if q.numel() else 0.0)
    Vm = torch.stack([(x - mu).mean(0) for x in ST[Lb]]); out["mean_pooled_state"] = retrieval(Vm)
    R0 = ref[~sinkmask(ref)] - mu; evw, Uw = torch.linalg.eigh(torch.cov(R0.T.double(), correction=0)); Wh = ((Uw / (evw.clamp_min(0) + 1e-3 * evw.mean()).sqrt()) @ Uw.T).float()
    out["whitened_pooled_state"] = retrieval(torch.stack([((x - mu) @ Wh).mean(0) for x in ST[Lb]]))       # the continuous analogue of idf: rare directions weighted up
    res["retrieval"][str(Lb)] = out
    log(f"{name} L{Lb}: translation retrieval top-1 (similarity z-gap): native words {out['native'][0]:.2f} ({out['native'][1]:.1f}), without idf {out['native_no_idf'][0]:.2f}, rotated words {out['rotated'][0]:.2f} ({out['rotated'][1]:.1f}), "
        f"mean-pooled state {out['mean_pooled_state'][0]:.2f} ({out['mean_pooled_state'][1]:.1f}), whitened pooled state {out['whitened_pooled_state'][0]:.2f} ({out['whitened_pooled_state'][1]:.1f}) | lexical tokens {res['retrieval']['lexical_tokens'][0]:.2f}")
    del A, Xc
mid = str(arch.NB // 2); R = res["retrieval"]
res["checks"] = dict(I1=R[mid]["native"][0] > 0.5, I2=R[mid]["native"][0] > R[mid]["rotated"][0], I3=R[mid]["native"][0] >= max(R[mid]["mean_pooled_state"][0], R[mid]["whitened_pooled_state"][0]),
                     I4=R[mid]["native"][0] >= max(R[str(Lb)]["native"][0] for Lb in DEPTHS))
summ = (f"{name}: translation retrieval top-1 across 12 language pairs (chance 0.03), by depth " + " ; ".join(f"L{Lb}: native {R[str(Lb)]['native'][0]:.2f} rotated {R[str(Lb)]['rotated'][0]:.2f} mean-pooled {R[str(Lb)]['mean_pooled_state'][0]:.2f} whitened {R[str(Lb)]['whitened_pooled_state'][0]:.2f}" for Lb in DEPTHS)
        + f" | lexical tokens {R['lexical_tokens'][0]:.2f} | most shared native words at L{mid}: " + "; ".join(f"{d_['n_quadruples']}x type{d_['type']} -> {''.join(d_['promotes'][:3])!r}" for d_ in res["shared_words"][mid][:5]) + f" | checks {json.dumps(res['checks'])}")
log(summ); record(f"e448_interlingua_{name}", res, summ)
