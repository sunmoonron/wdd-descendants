"""WDD in the weights: a live demo in three acts. Every number on screen is measured when you run it; nothing is
quoted from the record.

Act 1  Qwen2.5-0.5B. Find the MLP rows that write grammatical gender and number with a direct-contribution scan (each
       row's activation change times its write's alignment with the pole difference; no training, no dictionary
       learning), negate their write columns with one line of PyTorch, and measure what changes: held-out
       translations, free text in four languages, the cost on wikitext. Then the standard tools on the same items.
Act 2  Qwen2.5-Coder-0.5B-Instruct (code continued pretraining, then instruction tuning). Apply the base model's row
       addresses and its residual direction zero-shot, against the Coder model's own.
Act 3  Pythia-410m. Tamper with one MLP write row (scale it, negate it, or reflect it about its read axis so its norm
       and read-write cosine are unchanged) and audit the tampered model blind with three detectors.

Usage (a GPU with 8 GB is enough; the three models download once from Hugging Face):
    python wdd_demo.py                 interactive: pauses between scenes; you can type prompts and pick the tamper
    python wdd_demo.py --auto          no pauses, default choices
    python wdd_demo.py --acts 1,3      a subset of acts
    python wdd_demo.py --k 8           how many rows Act 1 and Act 2 negate (default 8)
    python wdd_demo.py --sweep 12      how many rows Act 3 tampers with for its catch-rate table (default 12; 0 skips it)
    python wdd_demo.py --seed 3        which secret row Act 3 picks
    python wdd_demo.py --save run.json write the measured numbers to a JSON file
"""
import sys, time, math, argparse, collections, textwrap, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
ap = argparse.ArgumentParser(); ap.add_argument("--auto", action="store_true"); ap.add_argument("--acts", default="1,2,3"); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sweep", type=int, default=12); ap.add_argument("--save", default=""); ap.add_argument("--no-color", action="store_true"); args = ap.parse_args()
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"; ACTS = {int(a) for a in args.acts.split(",")}; INTER = not args.auto and sys.stdin.isatty(); T0 = time.time()
# ---------------------------------------------------------------- presentation
_C = dict(b="\033[1m", d="\033[2m", r="\033[31m", g="\033[32m", y="\033[33m", c="\033[36m", m="\033[35m", x="\033[0m")
if args.no_color: _C = {k: "" for k in _C}
def c(s, *k): return "".join(_C[x] for x in k) + str(s) + _C["x"]
def act_title(n, s): print("\n" + c("═" * 100, "c") + "\n" + c(f"  ACT {n}  ", "b", "c") + c(s, "b") + "\n" + c("═" * 100, "c"))
def scene(s): print("\n" + c("── " + s + " ", "b", "y") + c("─" * max(0, 96 - len(s)), "y"))
def say(s, ind=2): print(textwrap.fill(s, 100, initial_indent=" " * ind, subsequent_indent=" " * ind))
def code(s): print(c("  >>> ", "d") + c(s, "m"))
def bar(v, vmax, w=28): n = 0 if vmax <= 0 else max(0, min(w, int(round(w * v / vmax)))); return "█" * n + "·" * (w - n)
def pct(x): return f"{100 * x:5.1f}%"
def pause(msg="press enter to continue"):
    if INTER: input(c(f"  [{msg}] ", "d"))
def ask(msg, default):
    if not INTER: return default
    s = input(c(f"  {msg} ", "y")).strip(); return s if s else default
def clock(): return c(f"({time.time() - T0:.0f}s)", "d")
# ---------------------------------------------------------------- word lists (e473, e622, e624, e625)
QUADS = {"en": [["king", "queen", "prince", "princess"], ["father", "mother", "son", "daughter"], ["man", "woman", "boy", "girl"], ["uncle", "aunt", "nephew", "niece"], ["grandfather", "grandmother", "grandson", "granddaughter"]],
         "fr": [["roi", "reine", "prince", "princesse"], ["père", "mère", "fils", "fille"], ["homme", "femme", "garçon", "fille"], ["oncle", "tante", "neveu", "nièce"], ["grand-père", "grand-mère", "petit-fils", "petite-fille"]],
         "es": [["rey", "reina", "príncipe", "princesa"], ["padre", "madre", "hijo", "hija"], ["hombre", "mujer", "niño", "niña"], ["tío", "tía", "sobrino", "sobrina"], ["abuelo", "abuela", "nieto", "nieta"]],
         "de": [["König", "Königin", "Prinz", "Prinzessin"], ["Vater", "Mutter", "Sohn", "Tochter"], ["Mann", "Frau", "Junge", "Mädchen"], ["Onkel", "Tante", "Neffe", "Nichte"], ["Großvater", "Großmutter", "Enkel", "Enkelin"]]}
NUMBER = {"en": [("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("house", "houses"), ("car", "cars"), ("tree", "trees"), ("table", "tables"), ("flower", "flowers")],
          "fr": [("chat", "chats"), ("chien", "chiens"), ("livre", "livres"), ("maison", "maisons"), ("voiture", "voitures"), ("arbre", "arbres"), ("table", "tables"), ("fleur", "fleurs")],
          "es": [("gato", "gatos"), ("perro", "perros"), ("libro", "libros"), ("casa", "casas"), ("coche", "coches"), ("árbol", "árboles"), ("mesa", "mesas"), ("flor", "flores")],
          "de": [("Katze", "Katzen"), ("Hund", "Hunde"), ("Buch", "Bücher"), ("Haus", "Häuser"), ("Auto", "Autos"), ("Baum", "Bäume"), ("Tisch", "Tische"), ("Blume", "Blumen")]}
SHOTS = {"fr": [("chien", "dog"), ("maison", "house"), ("livre", "book"), ("eau", "water"), ("arbre", "tree"), ("voiture", "car"), ("pain", "bread"), ("lait", "milk"), ("soleil", "sun"), ("lune", "moon")],
         "es": [("perro", "dog"), ("casa", "house"), ("libro", "book"), ("agua", "water"), ("árbol", "tree"), ("coche", "car"), ("pan", "bread"), ("leche", "milk"), ("sol", "sun"), ("luna", "moon")],
         "de": [("Hund", "dog"), ("Haus", "house"), ("Buch", "book"), ("Wasser", "water"), ("Baum", "tree"), ("Auto", "car"), ("Brot", "bread"), ("Milch", "milk"), ("Sonne", "sun"), ("Mond", "moon")]}
LN = {"fr": "French", "es": "Spanish", "de": "German"}; LANGS = ("fr", "es", "de"); NT, DISC = 5, (0, 1, 2)   # templates 0-2 find the rows, 3-4 are held out
POLES = {"gender": ("male", "female"), "number": ("singular", "plural")}
EN_SUBJ = [("king", 0), ("queen", 1), ("father", 0), ("mother", 1), ("actor", 0), ("actress", 1), ("boy", 0), ("girl", 1), ("man", 0), ("woman", 1), ("prince", 0), ("princess", 1),
           ("grandfather", 0), ("grandmother", 1), ("nephew", 0), ("niece", 1), ("husband", 0), ("wife", 1), ("brother", 0), ("sister", 1), ("son", 0), ("daughter", 1), ("uncle", 0), ("aunt", 1)]
EN_TEMPL = ["The {} said that", "When the {} came home,", "The {} looked at the letter, and then"]
X_SUBJ = {"fr": [("roi", "reine"), ("prince", "princesse"), ("père", "mère"), ("fils", "fille"), ("homme", "femme"), ("garçon", "fille"), ("oncle", "tante"), ("neveu", "nièce"), ("grand-père", "grand-mère"), ("frère", "sœur"), ("mari", "femme"), ("acteur", "actrice")],
          "es": [("rey", "reina"), ("príncipe", "princesa"), ("padre", "madre"), ("hijo", "hija"), ("hombre", "mujer"), ("niño", "niña"), ("tío", "tía"), ("sobrino", "sobrina"), ("abuelo", "abuela"), ("hermano", "hermana"), ("marido", "esposa"), ("actor", "actriz")],
          "de": [("König", "Königin"), ("Prinz", "Prinzessin"), ("Vater", "Mutter"), ("Sohn", "Tochter"), ("Mann", "Frau"), ("Herr", "Dame"), ("Onkel", "Tante"), ("Neffe", "Nichte"), ("Großvater", "Großmutter"), ("Bruder", "Schwester"), ("Ehemann", "Ehefrau"), ("Schauspieler", "Schauspielerin")]}
X_ART = {"fr": ("Le", "La"), "es": ("El", "La"), "de": ("Der", "Die")}; X_TAIL = {"fr": " est très", "es": " está muy", "de": " sagte, dass"}
X_FORMS = {"fr": [("beau", "belle"), ("grand", "grande"), ("petit", "petite"), ("content", "contente")], "es": [("cansado", "cansada"), ("contento", "contenta"), ("alto", "alta"), ("pequeño", "pequeña")], "de": [("er", "sie")]}
# ---------------------------------------------------------------- a Qwen-family model with the item harness of e622/e659
class LM:
    def __init__(self, hf):
        t = time.time(); self.hf = hf; self.tok = AutoTokenizer.from_pretrained(hf); self.model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).to(DEV).eval()
        self.layers = self.model.model.layers; self.NB = len(self.layers); self.L = self.NB // 2; self.D = self.model.config.hidden_size; self.DFF = self.layers[0].mlp.down_proj.weight.shape[1]
        self.BS = 16 if self.D > 1000 else 32; self.wiki = None; say(f"loaded {hf}: {self.NB} blocks, {self.DFF} MLP rows per block, {self.NB * self.DFF:,} rows in all {clock()}")
    def W(self, b): return self.layers[b].mlp.down_proj.weight
    def ftok(self, w): return self.tok(" " + w, add_special_tokens=False)["input_ids"][0]
    def prompt(self, lg, w, t):
        (w1, e1), (w2, e2) = SHOTS[lg][2 * t], SHOTS[lg][2 * t + 1]; p = f"{LN[lg]}: {w1}\nEnglish: {e1}\n{LN[lg]}: {w2}\nEnglish: {e2}\n{LN[lg]}: {w}\nEnglish:"; a = p.rindex(w); z = a + len(w)
        enc = self.tok(p, add_special_tokens=False, return_offsets_mapping=True); return enc["input_ids"], [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < z][-1]
    def run(self, its, capture=False):
        """next-token prediction over each item's two candidates; with capture, the MLP activations and block outputs at the word"""
        groups = collections.defaultdict(list)
        for n, it in enumerate(its): groups[len(it["ids"])].append(n)
        for idx in groups.values():
            for s0 in range(0, len(idx), self.BS):
                ch = idx[s0:s0 + self.BS]; ids = torch.tensor([its[n]["ids"] for n in ch], device=DEV); cap, hs = {}, []
                if capture:
                    for b in range(self.L + 1):
                        hs.append(self.layers[b].mlp.down_proj.register_forward_pre_hook(lambda m, a, b=b: cap.__setitem__(("a", b), a[0].float())))
                        hs.append(self.layers[b].register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(("x", b), (o[0] if isinstance(o, tuple) else o).float())))
                try: lg = self.model(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                finally: [h.remove() for h in hs]
                for k, n in enumerate(ch):
                    it = its[n]; it["pred"] = int(lg[k, it["cand"]].argmax())
                    if capture: p = it["p"]; it["A"] = torch.stack([cap[("a", b)][k, p] for b in range(self.L + 1)]).cpu(); it["X"] = torch.stack([cap[("x", b)][k, p] for b in range(self.L + 1)]).cpu()
    def items(self, attr):
        its = []
        for lg in LANGS:
            src = [(q[0], q[1]) for q in QUADS[lg]] + [(q[2], q[3]) for q in QUADS[lg]] if attr == "gender" else NUMBER[lg]
            en = [(q[0], q[1]) for q in QUADS["en"]] + [(q[2], q[3]) for q in QUADS["en"]] if attr == "gender" else NUMBER["en"]
            for k, (pr, e) in enumerate(zip(src, en)):
                cand = [self.ftok(w) for w in e]
                if cand[0] == cand[1]: continue
                for t in range(NT):
                    for i in (0, 1): ids, p = self.prompt(lg, pr[i], t); its.append(dict(lg=lg, pair=k, t=t, side=i, word=pr[i], en=e, cand=cand, ids=ids, p=p, disc=t in DISC))
        self.run(its, capture=True); grp = collections.defaultdict(list)
        for it in its: grp[(it["lg"], it["pair"], it["t"])].append(it)
        return [it for g in grp.values() if len(g) == 2 and all(x["pred"] == x["side"] for x in g) for it in g]
    def flips(self, its):
        """held-out items: the share of each starting pole's items now answered with the other pole"""
        held = [it for it in its if not it["disc"]]; self.run(held); out = {}
        for s in (0, 1):
            g = [it for it in held if it["side"] == s]; out[s] = sum(it["pred"] != s for it in g) / max(len(g), 1)
        return out
    def scan(self, its):
        """direct contribution of every MLP row of blocks 0..L to the pole difference: activation change x write cosine"""
        disc = [it for it in its if it["disc"]]; A = torch.stack([it["A"] for it in disc]); X = torch.stack([it["X"] for it in disc]); s = torch.tensor([it["side"] for it in disc])
        da = A[s == 1].mean(0) - A[s == 0].mean(0); md = X[s == 1].mean(0) - X[s == 0].mean(0); dh = md / md.norm(dim=1, keepdim=True).clamp_min(1e-12)
        score = torch.stack([da[b] * ((self.W(b).float().cpu().T @ dh[b]) / self.W(b).float().cpu().norm(dim=0).clamp_min(1e-12)) for b in range(self.L + 1)])
        order = [divmod(int(i), self.DFF) for i in score.reshape(-1).argsort(descending=True)[:64]]
        pairs = collections.defaultdict(dict)
        for it in disc: pairs[(it["lg"], it["pair"], it["t"])][it["side"]] = it["X"]
        diff = torch.stack([g[1] - g[0] for g in pairs.values() if 0 in g and 1 in g]); cons = [float(torch.nn.functional.cosine_similarity(diff[:, b], md[b][None], dim=1).mean()) for b in range(self.L + 1)]
        level = max(range(1, self.L + 1), key=lambda b: cons[b])
        return dict(order=order, score=score, md=md, A=A, s=s, level=level)
    def wiki_setup(self, n=12, T=128):
        import datasets
        wt = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.split()) > 60]; win = []
        for t in wt:
            ids = self.tok(t)["input_ids"]
            if len(ids) >= T: win.append(ids[:T])
            if len(win) >= n: break
        self.wiki = torch.tensor(win, device=DEV); self.wbase = [self.model(self.wiki[s:s + 4], use_cache=False).logits.float().log_softmax(-1).half().cpu() for s in range(0, self.wiki.shape[0], 4)]
    def cost(self):
        """wikitext: next-token loss (nats per token) and KL from the unedited model (per token)"""
        tl, tk, n = 0.0, 0.0, 0
        for k, s in enumerate(range(0, self.wiki.shape[0], 4)):
            x = self.wiki[s:s + 4]; lp = self.model(x, use_cache=False).logits.float().log_softmax(-1); lb = self.wbase[k].to(DEV).float()
            tl += float(-lp[:, :-1].gather(-1, x[:, 1:, None]).sum()); tk += float((lb.exp() * (lb - lp)).sum()); n += x.shape[0]; del lp, lb
        T = self.wiki.shape[1]; return tl / (n * (T - 1)), tk / (n * T)
    def next_probs(self, texts):
        out = []
        for t in texts:
            ids = torch.tensor([self.tok(t)["input_ids"]], device=DEV); out.append(torch.softmax(self.model(ids, use_cache=False, logits_to_keep=1).logits[0, -1].float(), -1))
        return out
    def free_text(self):
        """agreement with the subject's gender at the next token, split by the subject's gender: English pronouns (he/she), French and
        Spanish predicative adjectives, German pronouns (er/sie); each the probability of the agreeing forms over both forms"""
        res = {}
        he, she = self.ftok("he"), self.ftok("she"); texts = [t.format(s) for t in EN_TEMPL for s, _ in EN_SUBJ]; gs = [g for t in EN_TEMPL for _, g in EN_SUBJ]
        ps = self.next_probs(texts); res["en"] = {g: sum(float(p[[he, she][g]] / (p[he] + p[she])) for p, gg in zip(ps, gs) if gg == g) / gs.count(g) for g in (0, 1)}
        for lg in LANGS:
            pairs = [(self.ftok(m), self.ftok(f)) for m, f in X_FORMS[lg] if self.ftok(m) != self.ftok(f)]; mi = [a for a, _ in pairs]; fi = [b for _, b in pairs]
            texts, gs = [], []
            for m, f in X_SUBJ[lg]:
                for g, n in ((0, m), (1, f)):
                    art = X_ART[lg][g]; art, sep = ("L'", "") if (lg == "fr" and n[0].lower() in "aeiouhé") else (art, " "); texts.append(f"{art}{sep}{n}{X_TAIL[lg]}"); gs.append(g)
            ps = self.next_probs(texts); res[lg] = {g: sum(float(p[[mi, fi][g]].sum() / (p[mi].sum() + p[fi].sum())) for p, gg in zip(ps, gs) if gg == g) / gs.count(g) for g in (0, 1)}
        return res
    def generate(self, text, n=14):
        ids = self.tok(text, return_tensors="pt").to(DEV); out = self.model.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).replace("\n", " ⏎ ")
class Negate:
    """negate the write columns of the given (block, row) MLP rows in place: the edit is in the weights, no hooks"""
    def __init__(self, lm, rows): self.lm, self.rows = lm, list(dict.fromkeys(rows))
    def __enter__(self):
        self.saved = [(b, r, self.lm.W(b)[:, r].clone()) for b, r in self.rows]
        for b, r in self.rows: self.lm.W(b).data[:, r] *= -1
        return self
    def __exit__(self, *a):
        for b, r, w in self.saved: self.lm.W(b).data[:, r] = w
class Hook:
    """forward hooks on block outputs: ablate a direction (x <- x - (x.d)d) or add a vector, at every position"""
    def __init__(self, lm, ablate=None, add=None): self.lm, self.ablate, self.add, self.h = lm, ablate or {}, add or {}, []
    def __enter__(self):
        for b in set(self.ablate) | set(self.add):
            d = self.ablate.get(b); d = None if d is None else (d / d.norm().clamp_min(1e-12)).to(DEV); v = self.add.get(b); v = None if v is None else v.to(DEV)
            def hk(m, a, o, d=d, v=v):
                x = o[0] if isinstance(o, tuple) else o
                if d is not None: x = x - (x @ d)[..., None] * d
                if v is not None: x = x + v
                return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
            self.h.append(self.lm.layers[b].register_forward_hook(hk))
        return self
    def __exit__(self, *a): [h.remove() for h in self.h]
def show_free(before, after, title):
    say(c(title, "b")); print(c(f"    {'':22}{'male subjects':>26}{'female subjects':>26}", "d"))
    for lg, nm in (("en", "English pronouns"), ("fr", "French adjectives"), ("es", "Spanish adjectives"), ("de", "German pronouns")):
        cells = []
        for g in (0, 1):
            b0, a0 = before[lg][g], after[lg][g]; col = "g" if abs(a0 - b0) < 0.05 else ("r" if a0 < b0 else "g"); cells.append(f"{pct(b0)} → " + c(pct(a0), col))
        print(f"    {nm:22}{cells[0]:>35}{cells[1]:>35}")
# ================================================================ ACT 1
STATE = {}
def act1():
    act_title(1, "find the gender and number rows in a raw checkpoint, and flip them in the weights")
    say("Qwen2.5-0.5B, straight from Hugging Face: no probe, no autoencoder, no gradient steps. The items are few-shot translations into "
        "English (French, Spanish, German), e.g. 'French: reine  English: ___'. Rows are chosen on templates 0-2; translation effects are measured on held-out templates 3-4, free text on separate sentences.")
    lm = LM("Qwen/Qwen2.5-0.5B"); STATE["base_hf"] = lm.hf
    scene("1. the scan")
    t = time.time(); IT = {a: lm.items(a) for a in POLES}; SC = {a: lm.scan(IT[a]) for a in POLES}; dt = time.time() - t
    say(f"{sum(len(v) for v in IT.values())} translation items the model gets right on both poles ({', '.join(f'{a} {len(v)}' for a, v in IT.items())}); one forward pass collects every MLP activation and residual state at the word; "
        f"the ranking is one matrix product per block over {(lm.L + 1) * lm.DFF:,} rows. Scan time {dt:.1f}s.")
    for a in POLES:
        o = SC[a]["order"][:args.k]; sc = SC[a]["score"]
        print(c(f"    {a}: top {args.k} rows by direct contribution ({POLES[a][1]} minus {POLES[a][0]})", "b"))
        print("    " + "  ".join(c(f"(block {b}, row {r})", "m") for b, r in o[:4])); print("    " + "  ".join(c(f"(block {b}, row {r})", "m") for b, r in o[4:8]))
    pause()
    scene("2. what the rows read: one pole or both?")
    say("The mean |activation| of each row at the word, on items of each pole (discovery items). A row that fires for one pole and stays quieter for the other is a one-pole detector.")
    for a in POLES:
        A, s = SC[a]["A"], SC[a]["s"]; o = SC[a]["order"][:args.k]; m0 = [float(A[s == 0, b, r].abs().mean()) for b, r in o]; m1 = [float(A[s == 1, b, r].abs().mean()) for b, r in o]; vmax = max(m0 + m1)
        print(c(f"    {a}:", "b") + c(f"   {POLES[a][0]} = cyan, {POLES[a][1]} = magenta", "d"))
        for (b, r), x0, x1 in zip(o, m0, m1): print(f"    b{b:<2} r{r:<5} " + c(bar(x0, vmax), "c") + f" {x0:5.2f}   " + c(bar(x1, vmax), "m") + f" {x1:5.2f}")
        ratio = sum(m1) / max(sum(m0), 1e-9); nrev = sum(x0 > x1 for x0, x1 in zip(m0, m1)); SC[a]["mark"] = (ratio, nrev)
        say(f"summed over these rows, {POLES[a][1]} / {POLES[a][0]} = {ratio:.1f}x; {nrev} of {len(o)} rows fire more for {POLES[a][0]}" + ("" if ratio > 1.3 else " (no strong one-pole asymmetry)"), 4)
    pause()
    scene("3. the edit: one line of PyTorch on the weights")
    lm.wiki_setup(); l0, _ = lm.cost(); RES = {}
    for a in POLES:
        rows = SC[a]["order"][:args.k]
        code(f"for b, j in {a}_rows: model.model.layers[b].mlp.down_proj.weight.data[:, j] *= -1   # {len(rows)} columns of {lm.NB * lm.DFF:,}")
        before = lm.flips(IT[a])
        with Negate(lm, rows):
            after = lm.flips(IT[a]); loss, kl = lm.cost(); held = [it for it in IT[a] if not it["disc"] and it["side"] == 1]; lm.run(held)
            ex = [next((it for it in held if it["pred"] == 0 and it["lg"] == lg), None) for lg in LANGS] + [next((it for it in held if it["pred"] == 1), None)]; ex = [it for it in ex if it is not None]
        RES[a] = dict(after=after, kl=kl, dloss=loss - l0)
        print(f"    held-out {POLES[a][1]} items now answered {POLES[a][0]}: " + c(pct(after[1]), "b", "g") + f"   {POLES[a][0]} items now answered {POLES[a][1]}: " + c(pct(after[0]), "b") + c(f"   (unedited {pct(before[1])}, {pct(before[0])})", "d"))
        print(f"    wikitext loss {l0:.4f} → {loss:.4f} nats ({loss - l0:+.4f}), KL from the unedited model {kl:.4f}")
        for it in ex: print(c(f"      {LN[it['lg']]:8} '{it['word']}' → English: ", "d") + f"{it['en'][1]}  ⟶  " + c(it["en"][it["pred"]], "b", "g" if it["pred"] == 0 else "y") + c("   (flipped)" if it["pred"] == 0 else "   (unchanged)", "d"))
    say(c("Note the direction: the edit strips the marked pole (female, plural) far more than it creates it. The reverse flips stay small.", "d"))
    pause()
    scene("4. the dose: how many rows?")
    print(c(f"    {'rows negated':>14}  {'gender: female→male / male→female':>36}  {'number: plural→singular / singular→plural':>44}  {'KL (gender, number)':>22}", "d"))
    DOSE = {}
    for k in (1, 8, 32):
        cells, kls = [], []
        for a in POLES:
            with Negate(lm, SC[a]["order"][:k]): f = lm.flips(IT[a]); lo, kl_ = lm.cost(); kls.append(kl_); DOSE[(a, k)] = dict(f=f, kl=kl_, dloss=lo - l0)
            cells.append(f"{pct(f[1])} / {pct(f[0])}")
        print(f"    {k:>14}  {cells[0]:>36}  {cells[1]:>44}  {kls[0]:>10.4f}, {kls[1]:.4f}")
    pause()
    scene("5. free text in four languages (the rows were found on translations only)")
    a = "gender"; ft0 = lm.free_text()
    with Negate(lm, SC[a]["order"][:args.k]): ft1 = lm.free_text()
    show_free(ft0, ft1, f"agreement with the subject's gender at the next word, before → after negating the {args.k} gender rows")
    STATE["ft"] = (ft0, ft1)
    prompts = ["The queen looked at the letter, and then", "Die Königin sagte, dass"]
    if INTER:
        p = ask("type a prompt to compare before/after (enter for the defaults):", "")
        if p: prompts = [p] + prompts[:1]
    for p in prompts:
        g0 = lm.generate(p)
        with Negate(lm, SC[a]["order"][:args.k]): g1 = lm.generate(p)
        print(c(f"    {p}", "b")); print(c("      unedited: ", "d") + g0); print(c("      edited:   ", "d") + g1)
    pause()
    scene("6. the same items, with the standard tools")
    say("Directional ablation (Arditi et al. 2024): remove the mean female-minus-male direction from every block output 0..L at every token. "
        "A steering vector: add the male-minus-female mean state at the most consistent block, at every token, scaled.")
    md = SC[a]["md"]; lev = SC[a]["level"]; v = -md[lev]
    rows_ = [(f"negate {args.k} rows (in the weights)", RES[a]["after"], RES[a]["kl"])]
    with Hook(lm, ablate={b: md[b] for b in range(lm.L + 1)}): rows_.append(("ablate the direction, every block", lm.flips(IT[a]), lm.cost()[1]))
    for al in (1, 2, 4):
        with Hook(lm, add={lev: al * v}): rows_.append((f"steering vector x{al} at block {lev}", lm.flips(IT[a]), lm.cost()[1]))
    print(c(f"    {'method':40}{'female→male':>14}{'male→female':>14}{'wikitext KL':>14}", "d"))
    for nm, f, kl in rows_: print(f"    {nm:40}{pct(f[1]):>14}{pct(f[0]):>14}{kl:>14.4f}")
    (_, fr, kr), (_, fa, ka) = rows_[0], rows_[1]; st = [(nm, f, kl) for nm, f, kl in rows_[2:] if f[1] >= fa[1] and f[0] < 0.2]
    msg = f"Read it as a trade-off: the row edit moves {pct(fr[1]).strip()} at KL {kr:.4f}; ablating the direction moves {pct(fa[1]).strip()} at {ka / max(kr, 1e-9):.1f}x that KL"
    msg += (f"; the first steering scale that matches the ablation ({st[0][0].split(' at ')[0]}) costs {st[0][2] / max(kr, 1e-9):.0f}x the row edit's KL" if st else "") + "; at the largest scale the steering vector breaks the model."
    say(c(msg, "d")); STATE["steer_vs_rows"] = (st[0][2] / max(kr, 1e-9)) if st else None; STATE["abl"] = (fa, ka)
    STATE.update(dose=DOSE, marked={a: SC[a]["mark"] for a in POLES}, base_rows={a: SC[a]["order"][:32] for a in POLES}, base_md={a: SC[a]["md"] for a in POLES}, base_level={a: SC[a]["level"] for a in POLES},
                 base_cols={b: lm.W(b).detach().float().cpu().clone() for b in range(lm.L + 1)}, act1=RES)
    del lm; torch.cuda.empty_cache(); pause()
# ================================================================ ACT 2
def act2():
    act_title(2, "the same addresses in a different model")
    if "base_rows" not in STATE: say("(running Act 1's scan first, quietly)"); _quiet_base()
    say("Qwen2.5-Coder-0.5B-Instruct: the same architecture, continued-pretrained on code (5.5T tokens per the Qwen2.5-Coder report) and instruction-tuned. "
        "Below, the base model's row addresses and residual direction are applied without running a single Coder prompt to choose them; the Coder model's own scan is shown for comparison.")
    lm = LM("Qwen/Qwen2.5-Coder-0.5B-Instruct"); IT = {a: lm.items(a) for a in POLES}; SC = {a: lm.scan(IT[a]) for a in POLES}; lm.wiki_setup()
    say(f"{sum(len(v) for v in IT.values())} items the Coder model gets right ({', '.join(f'{a} {len(v)}' for a, v in IT.items())}) {clock()}")
    scene("1. how far did the weights move?")
    allc = torch.cat([torch.nn.functional.cosine_similarity(STATE["base_cols"][b], lm.W(b).detach().float().cpu(), dim=0) for b in range(lm.L + 1)])
    for a in POLES:
        rows = STATE["base_rows"][a][:args.k]; rc = [float(torch.nn.functional.cosine_similarity(STATE["base_cols"][b][:, r], lm.W(b).detach().float().cpu()[:, r], dim=0)) for b, r in rows]
        mdc = [float(torch.nn.functional.cosine_similarity(STATE["base_md"][a][b], SC[a]["md"][b], dim=0)) for b in range(1, lm.L + 1)]
        STATE.setdefault("rot", {})[a] = (sorted(rc)[len(rc) // 2], min(mdc), max(mdc))
        print(f"    {a:7} the {args.k} base rows' write columns: cosine with the base model's {min(rc):.2f} to {max(rc):.2f} (median {sorted(rc)[len(rc) // 2]:.2f}); "
              + c(f"the {a} direction in the residual stream: {min(mdc):.2f} to {max(mdc):.2f} across blocks 1..{lm.L}", "b"))
    say(f"every MLP write column of blocks 0..{lm.L}: median cosine with the base model {float(allc.median()):.2f}. The rows rotated; the attribute's direction mostly did not.", 4); STATE["allc"] = float(allc.median())
    pause()
    scene("2. edit with the base model's addresses and direction, zero-shot")
    print(c(f"    {'edit':52}{'gender: female→male':>22}{'number: plural→singular':>26}{'KL (gender)':>13}", "d"))
    for k in sorted({args.k, 32}):
        r_ = []
        for a in POLES:
            with Negate(lm, STATE["base_rows"][a][:k]): f = lm.flips(IT[a]); kl = lm.cost()[1] if a == "gender" else None
            with Negate(lm, SC[a]["order"][:k]): fo = lm.flips(IT[a])
            r_.append((f[1], fo[1], kl))
        print(f"    {'negate the base model rows (k=' + str(k) + ') / Coder own rows':52}{pct(r_[0][0]) + ' / ' + pct(r_[0][1]):>22}{pct(r_[1][0]) + ' / ' + pct(r_[1][1]):>26}{r_[0][2]:>13.4f}"); STATE.setdefault("act2", {})[k] = r_
    r_ = []
    for a in POLES:
        with Hook(lm, ablate={b: STATE["base_md"][a][b] for b in range(lm.L + 1)}): f = lm.flips(IT[a]); kl = lm.cost()[1] if a == "gender" else None
        with Hook(lm, ablate={b: SC[a]["md"][b] for b in range(lm.L + 1)}): fo = lm.flips(IT[a])
        r_.append((f[1], fo[1], kl))
    print(f"    {'ablate the base model direction / Coder own direction':52}{pct(r_[0][0]) + ' / ' + pct(r_[0][1]):>22}{pct(r_[1][0]) + ' / ' + pct(r_[1][1]):>26}{r_[0][2]:>13.4f}"); STATE["act2"]["ablate"] = r_
    if STATE["act2"][32][0][0] < STATE["act2"][args.k][0][0]: say(c("More rows is not always more: gender peaks at a few rows and falls with 32 (e662 saw the same overshoot).", "d"), 4)
    with Negate(lm, STATE["base_rows"]["gender"][:args.k]): g1 = lm.generate("The queen looked at the letter, and then")
    print(c("    Coder-Instruct, base rows negated: ", "d") + "The queen looked at the letter, and then" + c(g1, "b"))
    del lm; torch.cuda.empty_cache(); pause()
def _quiet_base():
    lm = LM("Qwen/Qwen2.5-0.5B"); IT = {a: lm.items(a) for a in POLES}; SC = {a: lm.scan(IT[a]) for a in POLES}
    STATE.update(base_rows={a: SC[a]["order"][:32] for a in POLES}, base_md={a: SC[a]["md"] for a in POLES}, base_cols={b: lm.W(b).detach().float().cpu().clone() for b in range(lm.L + 1)}); del lm; torch.cuda.empty_cache()
# ================================================================ ACT 3
def act3():
    act_title(3, "tamper with one row, then audit the model blind")
    say("Pythia-410m, block 4 (4,096 MLP rows). Someone edits one write row in place. The auditor gets only the tampered weights and 16 ordinary wikitext windows: "
        "no original checkpoint, no diff. Three detectors (e623, e648b, e651): the column's norm (z-score > 3), the read-write cosine (above the block's 99th percentile; "
        "in GELU models most rows sit well below zero), and activation consistency (does the write agree with the states where its neuron fires, minus its own write; below the 1st percentile).")
    import datasets
    t = time.time(); tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m"); model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", dtype=torch.float32).to(DEV).eval()
    B = 4; lay = model.gpt_neox.layers[B]; down = lay.mlp.dense_4h_to_h.weight; readw = lay.mlp.dense_h_to_4h.weight.detach().float(); DFF = down.shape[1]
    lines = [x for x in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(x.split()) > 40]
    PW = torch.stack([tok(x, return_tensors="pt").input_ids[0, :192] for x in lines[:40] if tok(x, return_tensors="pt").input_ids.shape[1] >= 192][:16]).to(DEV)
    say(f"loaded EleutherAI/pythia-410m {clock()}")
    def lps(): return torch.cat([model(PW[s:s + 4]).logits[:, 1:].float().log_softmax(-1).cpu() for s in range(0, PW.shape[0], 4)]).reshape(-1, model.config.vocab_size)
    def states():
        cap = []; h = lay.mlp.dense_4h_to_h.register_forward_pre_hook(lambda m, a: cap.append(a[0][:, 1:].detach().float().reshape(-1, DFF))); Xs = []
        try:
            for s in range(0, PW.shape[0], 4): Xs.append(model(PW[s:s + 4], output_hidden_states=True).hidden_states[B + 1][:, 1:].detach().float().reshape(-1, model.config.hidden_size))
        finally: h.remove()
        return torch.cat(Xs), torch.cat(cap)
    LP0 = lps(); X0, A0 = states(); top = A0.abs().topk(32, dim=0).indices   # [32, DFF]; the activations at block 4 do not depend on block 4's own write
    def consistency(X, A, W):
        out = torch.empty(DFF, device=DEV)
        for j0 in range(0, DFF, 256):
            js = torch.arange(j0, min(j0 + 256, DFF), device=DEV); idx = top[:, js]; w = W[:, js].T; rest = X[idx] - A[idx, js][..., None] * w[None]
            out[js] = torch.nn.functional.cosine_similarity(w, rest.mean(0), dim=1)
        return out
    def audit(clean_states=False):
        """the blind audit: only the (tampered) model in hand; every statistic and every threshold from it"""
        W = down.detach().float(); nz = W.norm(dim=0); z = (nz - nz.mean()) / nz.std()
        rw = ((W / nz.clamp_min(1e-12)) * (readw / readw.norm(dim=1, keepdim=True).clamp_min(1e-12)).T).sum(0)
        X, A = (X0, A0) if clean_states else states(); cons = consistency(X, A, W)
        return dict(z=z, rw=rw, cons=cons, thr_rw=float(rw.kthvalue(int(0.99 * DFF)).values), thr_cons=float(cons.kthvalue(max(int(0.01 * DFF), 1)).values))
    a0 = audit(); use = A0.abs().mean(0); pool = use.topk(40).indices.tolist()
    say(f"block 4 unedited: read-write cosine median {float(a0['rw'].median()):+.2f} ({float((a0['rw'] < -0.1).float().mean()):.0%} of rows below -0.1). The tamper targets come from the 40 most-used rows (the top 1% by mean |activation|) {clock()}")
    rng = random.Random(args.seed); secret = rng.choice(pool); hows = ["scale", "negate", "reflect"]
    if INTER:
        s = ask(f"pick the secret row (0-{DFF - 1}), or enter for a random high-use row:", ""); secret = int(s) if s.strip().isdigit() and 0 <= int(s) < DFF else secret
        s = ask("pick the tamper: 1 scale x2, 2 negate, 3 reflect about its read axis, enter for all three:", ""); hows = {"1": ["scale"], "2": ["negate"], "3": ["reflect"]}.get(s.strip(), hows)
    def tamper(j, how, w):
        u_ = readw[j] / readw[j].norm()
        if how == "scale": down.data[:, j] = 2 * w
        elif how == "negate": down.data[:, j] = -w
        else: down.data[:, j] = -w + 2 * (w @ u_) * u_
    def kl_at(j): pos = top[:, j].cpu(); return float((LP0[pos].exp() * (LP0[pos] - lps()[pos])).sum(-1).mean())
    def flags(j, a): return abs(float(a["z"][j])) > 3, float(a["rw"][j]) > a["thr_rw"], float(a["cons"][j]) < a["thr_cons"]
    w0 = down[:, secret].detach().clone()
    say(f"the secret row: {secret} (mean |activation| {float(use[secret]) / float(use.median()):.1f}x the block median). Its read-write cosine before tampering {float(a0['rw'][secret]):+.2f}, consistency {float(a0['cons'][secret]):+.2f}.")
    rank = lambda s, desc: int((s.argsort(descending=desc) == secret).nonzero()[0, 0]) + 1
    summary = []
    for how in hows:
        scene(f"tamper: {how}")
        code({"scale": f"W[:, {secret}] *= 2", "negate": f"W[:, {secret}] *= -1", "reflect": f"u = read_row[{secret}] / norm;  W[:, {secret}] = -w + 2 (w·u) u      # norm and read-write cosine unchanged"}[how])
        tamper(secret, how, w0); kl = kl_at(secret)
        say(c(f"ground truth (the auditor never sees this): next-token KL at the row's 32 top-firing positions {kl:.4f} nats", "d"))
        a = audit(); e = audit(clean_states=True)
        f_n, f_r, f_c = flags(secret, a); f_e = flags(secret, e)[2]
        flag = lambda f: c(" FLAGGED ", "b", "r") if f else c("  quiet  ", "g")
        print(f"    norm z-score         {float(a['z'][secret]):+7.2f}   rank {rank(a['z'].abs(), True):>4} of {DFF}  {flag(f_n)}")
        print(f"    read-write cosine    {float(a['rw'][secret]):+7.2f}   rank {rank(a['rw'], True):>4} of {DFF}  {flag(f_r)}   (99th percentile {a['thr_rw']:+.2f})")
        print(f"    consistency          {float(a['cons'][secret]):+7.2f}   rank {rank(a['cons'], False):>4} of {DFF}  {flag(f_c)}   (1st percentile {a['thr_cons']:+.2f})")
        print(c(f"    consistency as e651 computed it (with the clean model's states, which a blind auditor does not have): {float(e['cons'][secret]):+.2f}, rank {rank(e['cons'], False)}, ", "d") + ("flagged" if f_e else "quiet"))
        summary.append((how, kl, f_n, f_r, f_c, f_e)); down.data[:, secret] = w0
    scene("the audit, summed up")
    print(c(f"    {'tamper':10}{'KL at its positions':>21}{'norm':>10}{'read-write':>12}{'consistency':>13}{'caught':>9}{'e651-style consistency':>25}", "d"))
    for how, kl, f_n, f_r, f_c, f_e in summary:
        print(f"    {how:10}{kl:>21.4f}{('yes' if f_n else '-'):>10}{('yes' if f_r else '-'):>12}{('yes' if f_c else '-'):>13}{c('YES', 'b', 'r') if (f_n or f_r or f_c) else c('no', 'b', 'g'):>18}{('yes' if f_e else '-'):>25}")
    STATE["act3_single"] = summary
    nsw = args.sweep
    if INTER: s_ = ask(f"run the catch-rate sweep over {args.sweep} random rows of the 40 (about {3 * args.sweep}s)? number of rows, 0 to skip, enter for {args.sweep}:", str(args.sweep)); nsw = int(s_) if s_.strip().isdigit() else args.sweep
    nsw = min(nsw, 40)
    if nsw > 0:
        scene(f"the catch rate: {nsw} random rows of the 40 most-used, each tampered three ways")
        rows_ = random.Random(args.seed + 1).sample(pool, min(nsw, len(pool))); tab = {h: collections.Counter() for h in ("scale", "negate", "reflect")}; kls = {h: [] for h in tab}
        for j in rows_:
            wj = down[:, j].detach().clone()
            for how in tab:
                tamper(j, how, wj); kls[how].append(kl_at(j)); fn, fr, fc = flags(j, audit()); fe = flags(j, audit(clean_states=True))[2]; down.data[:, j] = wj
                tab[how].update(n=1, norm=fn, rw=fr, cons=fc, any=(fn or fr or fc), e651=fe)
        print(c(f"    {'tamper':10}{'mean KL at its positions':>26}{'norm':>8}{'read-write':>12}{'consistency':>13}{'any':>7}{'e651-style consistency':>25}", "d"))
        for how, t_ in tab.items():
            n_ = t_["n"]; f_ = lambda k: f"{t_[k] / n_:.2f}"
            print(f"    {how:10}{sum(kls[how]) / n_:>26.4f}{f_('norm'):>8}{f_('rw'):>12}{f_('cons'):>13}" + c(f"{f_('any'):>7}", "b") + f"{f_('e651'):>25}")
        STATE["act3_sweep"] = {h: {k: v / t_["n"] for k, v in t_.items()} | {"kl": sum(kls[h]) / t_["n"]} for h, t_ in tab.items()}
        say(c(f"Two things to read off: a single-row tamper barely moves this model (mean KL {max(v['kl'] for v in STATE['act3_sweep'].values()):.4f} nats or less at the row's own top positions), and the reflection mostly slips past all three detectors. "
              "e651 reported consistency catching 0.67-0.80 of high-use reflections in the Pythia models; it scored the edited weights against the clean model's states, which a blind auditor does not have. Compare the last two columns.", "d"))
    del model; torch.cuda.empty_cache()

# ================================================================ the pasted script, checked against this run
def claim_check():
    scene("the pasted demo script's claims, checked against this run")
    rows = []; H, P, N, U = c("holds", "g"), c("partly", "y"), c("does not hold", "r"), c("not tested", "d")
    if "dose" in STATE:
        d = STATE["dose"]; n8, n32 = d[("number", 8)]["f"][1], d[("number", 32)]["f"][1]
        rows.append(("Act 1: plurals translate to singulars 96-100% (e662)", f"{pct(n8).strip()} with 8 rows, {pct(n32).strip()} with 32", H if n32 >= 0.96 else P))
        ft = STATE["ft"][1]; fem = {lg: ft[lg][1] for lg in ("en", "fr", "es", "de")}; flipped = [lg for lg, v in fem.items() if v < 0.5]
        rows.append(("Act 1: feminine agreement flips to masculine in EN/FR/ES/DE", "female-subject agreement after 8 rows: " + ", ".join(f"{lg} {pct(v).strip()}" for lg, v in fem.items()), H if len(flipped) == 4 else (P if flipped else N)))
        rows.append(("Act 1: the edit costs +0.0003 nats", f"8 gender rows {d[('gender', 8)]['dloss']:+.4f} nats on wikitext; 1 row {d[('gender', 1)]['dloss']:+.4f} (the +0.0003 was one row on the Pile, e624)", H if d[("gender", 8)]["dloss"] < 0.001 else P))
        (rg, ng), (rn, nn) = STATE["marked"]["gender"], STATE["marked"]["number"]
        rows.append(("Act 1: marked rows blaze 3-6x, unmarked dead quiet", f"female/male {rg:.1f}x, plural/singular {rn:.1f}x over the top 8; {ng} gender and {nn} number rows fire more for the unmarked pole", H if min(rg, rn) >= 3 and ng + nn == 0 else P))
        sv = STATE.get("steer_vs_rows")
        rows.append(("Act 1: the sign clamp is 4,690x cleaner than steering (e660)", "the clamp is not run here; " + (f"a steering vector that matches the direction ablation costs {sv:.0f}x the 8-row edit's KL" if sv else "no steering scale matched"), U))
    if "act2" in STATE:
        a2 = STATE["act2"]; fk, f32, fab = a2[args.k][0][0], a2[32][0][0], a2["ablate"][0][0]; pk, p32 = a2[args.k][1][0], a2[32][1][0]
        rows.append(("Act 2: base addresses flip 100% of plurals and 56-88% of feminine nouns in Coder-Instruct", f"plurals {pct(pk).strip()} ({args.k} rows), {pct(p32).strip()} (32); female {pct(fk).strip()} ({args.k} rows), {pct(f32).strip()} (32), {pct(fab).strip()} by ablating the base direction", H if p32 >= 0.99 and max(fk, fab) >= 0.56 else P))
        r = STATE["rot"]; rows.append(("Act 2: rows rotate to 0.44-0.46 while the axis stays at 0.79-0.89", f"all rows median {STATE['allc']:.2f}, the base rows {r['gender'][0]:.2f} (gender) and {r['number'][0]:.2f} (number); the directions {min(r['gender'][1], r['number'][1]):.2f} to {max(r['gender'][2], r['number'][2]):.2f} across blocks", P))
    if "act3_sweep" in STATE:
        sw = STATE["act3_sweep"]; ok = sw["scale"]["norm"] >= 0.9 and sw["negate"]["rw"] >= 0.9 and sw["reflect"]["cons"] >= 0.9
        rows.append(("Act 3: scale → norm, negate → read-write, reflect → consistency; no single-row flip escapes", f"caught: scale {sw['scale']['any']:.2f}, negate {sw['negate']['any']:.2f}, reflect {sw['reflect']['any']:.2f} (reflect by consistency {sw['reflect']['cons']:.2f}; e651-style {sw['reflect']['e651']:.2f})", H if ok else (N if sw["reflect"]["any"] < 0.5 else P)))
    elif "act3_single" in STATE:
        caught = {how: (fn or fr or fc) for how, _, fn, fr, fc, _ in STATE["act3_single"]}
        rows.append(("Act 3: no single-row flip escapes the three detectors", "this row: " + ", ".join(f"{h} {'caught' if v else 'missed'}" for h, v in caught.items()) + " (run --sweep for rates)", H if all(caught.values()) else N))
    for claim, meas, v in rows:
        print("  " + c("• " + claim, "b")); print("      measured: " + meas); print("      verdict:  " + v)
# ================================================================
if __name__ == "__main__":
    print(c("\n  WDD in the weights: a live demo. Every number below is measured now, on this machine.", "b"))
    say(c(f"device {DEV}; acts {sorted(ACTS)}; rows per edit k={args.k}; {'interactive' if INTER else 'non-interactive'}", "d"))
    if 1 in ACTS: act1()
    if 2 in ACTS: act2()
    if 3 in ACTS: act3()
    claim_check()
    if args.save:
        import json
        keep = {k: v for k, v in STATE.items() if k not in ("base_cols", "base_md", "base_rows", "base_level")} | {"base_rows": {a: [list(x) for x in v[:args.k]] for a, v in STATE.get("base_rows", {}).items()}, "args": vars(args), "seconds": time.time() - T0}
        conv = lambda o: {str(k): conv(v) for k, v in o.items()} if isinstance(o, dict) else [conv(v) for v in o] if isinstance(o, (list, tuple)) else (o.item() if hasattr(o, "item") else o)
        json.dump(conv(keep), open(args.save, "w"), indent=1); print(c(f"  measured numbers written to {args.save}", "d"))
    print(c(f"\n  done in {time.time() - T0:.0f}s", "b"))
