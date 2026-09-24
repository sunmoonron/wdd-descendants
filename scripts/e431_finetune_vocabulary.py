"""e431: what does instruction fine-tuning do to the native vocabulary? Base and instruct pairs (Qwen2.5-0.5B, SmolLM2-135M;
argument: the base name), middle depth, two texts: natural text (4 x 256 tokens) and chat-formatted conversations (the
instruct tokenizer's chat template over simple question-and-answer turns, 4 x up to 256 tokens). For each text: loss
recovered at k 16 when each model's states are described with its own words, the other model's words, and a rotation
(does fine-tuning keep the language mutually intelligible?); and word drift: the cosine between each MLP write row
before and after fine-tuning, and whether the 1% most-changed words are used more to describe the instruct model's chat
states than its natural-text states (enrichment). Pre-registered: median row cosine above 0.95; cross-description
within 0.03 of own description on both texts; the most-changed words enriched in chat descriptions (ratio above 1)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
from lr_common import eval_ids
from transformers import AutoModelForCausalLM, AutoTokenizer
base = sys.argv[1]; INST = {"qwen05": "Qwen/Qwen2.5-0.5B-Instruct", "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct"}[base]
mb, tok, fam = load_model(base); ab = Arch(mb, fam); L = ab.NB // 2
ti = AutoTokenizer.from_pretrained(INST); mi = AutoModelForCausalLM.from_pretrained(INST, dtype=torch.float32, device_map=DEV).eval(); ai = Arch(mi, fam)
E = eval_ids(base); nat = E[:4, :256].to(DEV)
QA = [("What is the capital of France?", "The capital of France is Paris."), ("How many legs does a spider have?", "A spider has eight legs."),
      ("Can you suggest a name for a cat?", "Sure, how about Whiskers?"), ("What is 12 times 7?", "12 times 7 is 84."),
      ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."), ("Why is the sky blue?", "Because air scatters blue light more than red light."),
      ("Give me a synonym for happy.", "Joyful."), ("What is the boiling point of water?", "Water boils at 100 degrees Celsius at sea level."),
      ("Translate 'thank you' into Spanish.", "Gracias."), ("What is the largest planet?", "Jupiter is the largest planet in our solar system.")]
rows = []
for s in range(4):
    msgs = []
    for q, a in QA[s:] + QA[:s]: msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    r = ti.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    rows.append(list(r["input_ids"]) if not isinstance(r, list) else r)
T = min(256, min(len(r) for r in rows)); chat = torch.tensor([r[:T] for r in rows]).to(DEV)
Vb, lab = build_dictionary(ab, blocks=list(range(L + 1))); Vi, _ = build_dictionary(ai, blocks=list(range(L + 1))); typ = lab["type"].to(DEV)
res = dict(base=base, instruct=INST, level=L, chat_len=T, texts={})
for tn, ids in [("natural", nat), ("chat", chat)]:
    out = {}
    for mn, m, a in [("base", mb, ab), ("instruct", mi, ai)]:
        lv = Level(m, a, ids, L)
        for vn, V in [("base_words", Vb), ("instruct_words", Vi), ("rotated", rotate(Vb if mn == "base" else Vi, seed=7))]:
            out[f"{mn}_states:{vn}"] = describe(lv, V, [16])["16"]["rec"]
        if mn == "instruct":
            sel, _, _ = omp(lv.Xc, unitr(Vi), 16, batch=256, record_err=False); out["_usage"] = torch.bincount(sel.flatten(), minlength=Vi.shape[0]).float()
        del lv
    res["texts"][tn] = {k: v for k, v in out.items() if not k.startswith("_")}; res["texts"][tn]["_usage"] = out["_usage"]
    o = res["texts"][tn]
    log(f"{base} {tn}: base states: own {o['base_states:base_words']:.2f} instruct words {o['base_states:instruct_words']:.2f} rot {o['base_states:rotated']:.2f} | "
        f"instruct states: own {o['instruct_states:instruct_words']:.2f} base words {o['instruct_states:base_words']:.2f} rot {o['instruct_states:rotated']:.2f}")
cos = (unitr(Vb) * unitr(Vi)).sum(-1); mlp = typ == T_MLP; cm = cos[mlp]
q = torch.quantile(cm, torch.tensor([0.01, 0.5], device=DEV)); changed = torch.zeros_like(cos, dtype=torch.bool); changed[mlp] = cm <= torch.quantile(cm, 0.01)
un, uc = res["texts"]["natural"].pop("_usage"), res["texts"]["chat"].pop("_usage")
enrich = ((uc[changed].sum() / uc.sum()) / (un[changed].sum() / un.sum()).clamp_min(1e-9)).item()
res["drift"] = dict(median_cos=q[1].item(), p01_cos=q[0].item(), frac_below_099=(cm < 0.99).float().mean().item(), enrichment_changed_in_chat=enrich,
                    share_chat_uses_on_changed=(uc[changed].sum() / uc.sum()).item(), share_nat_uses_on_changed=(un[changed].sum() / un.sum()).item())
d = res["drift"]
summ = (f"{base} vs instruct: MLP row cosine median {d['median_cos']:.4f}, 1st percentile {d['p01_cos']:.3f}, below 0.99 {d['frac_below_099']:.3f} | "
        f"k16 own/cross/rot natural (instruct states) {res['texts']['natural']['instruct_states:instruct_words']:.2f}/{res['texts']['natural']['instruct_states:base_words']:.2f}/{res['texts']['natural']['instruct_states:rotated']:.2f}, "
        f"chat {res['texts']['chat']['instruct_states:instruct_words']:.2f}/{res['texts']['chat']['instruct_states:base_words']:.2f}/{res['texts']['chat']['instruct_states:rotated']:.2f} | "
        f"most-changed 1% of words: {d['share_chat_uses_on_changed']:.3f} of chat uses vs {d['share_nat_uses_on_changed']:.3f} of natural uses (enrichment {enrich:.2f})")
log(summ); record(f"e431_finetune_{base}", res, summ)
