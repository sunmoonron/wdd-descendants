"""e665 (session 146; Gemini 4's 'write once, patch everywhere'): do relay rows found on a base model edit a heavily
continued-pretrained derivative zero-shot? e659 found the base rows edit Qwen2.5-0.5B-Instruct as well as the base (e619:
rows at cosine 0.997 after instruct tuning). Gemini generalised to every derivative. Qwen2.5-Coder-0.5B was continued
from Qwen2.5-0.5B on trillions of code-heavy tokens, a far larger change. Here, on Qwen2.5-Coder-0.5B and -Instruct:
the base model's WDD chain (e661) and e622 chain negated zero-shot, against the derivative's own WDD chain and random
chains; held-out flips by starting pole and wikitext KL; and the write cosine between base and derivative for the chain
rows and for all rows. Reuses e661's setup. Arguments: model [--smoke]."""
import sys
src = open("e661_markedness.py").read().replace('"qwen15": "Qwen/Qwen2.5-1.5B", ', '"qwen15": "Qwen/Qwen2.5-1.5B", "coder05": "Qwen/Qwen2.5-Coder-0.5B", "coder05i": "Qwen/Qwen2.5-Coder-0.5B-Instruct", ')
exec(src[src.index("import sys\n"): src.index("# ==== SETUP END ====")], globals())
B61 = json.load(open("results/e661_markedness_qwen05.json"))["chains"]; C622 = json.load(open("e622_catalogue_qwen05.json"))["catalogue"]
def chain622(a):
    rows = []
    for lev in sorted(C622[a]["vote_by_block"], key=int):
        v = C622[a]["vote_by_block"][lev]; i_ = v["atom"] - NEMB; b, r = divmod(i_, PER)
        if v["type"] == 2 and v["share"] >= 0.25 and i_ >= 0 and r < DFF and (b, r) not in rows: rows.append((b, r))
    return rows
NEMB = model.get_input_embeddings().weight.shape[0]; PER = DFF + D
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32); BW = [l.mlp.down_proj.weight.detach().float() for l in base.model.layers]; del base
def cos_rows(rows): return [float(torch.nn.functional.cosine_similarity(BW[b][:, r], layers[b].mlp.down_proj.weight[:, r].detach().float().cpu(), dim=0)) for b, r in rows]
allcos = []
for b in range(L + 1):
    Wd = layers[b].mlp.down_proj.weight.detach().float().cpu(); allcos.append(torch.nn.functional.cosine_similarity(BW[b], Wd, dim=0))
allcos = torch.cat(allcos)
RES = dict(model=name, n_items={a: len(ITEMS[a]) for a in ATTRS}, all_rows_cos_median=float(allcos.median()), all_rows_cos_p10=float(allcos.quantile(0.1)), attrs={})
log(f"write cosine with the base model over every row of blocks 0..{L}: median {RES['all_rows_cos_median']:.3f}, 10th percentile {RES['all_rows_cos_p10']:.3f}")
for a in ATTRS:
    own, _ = wdd_chain(a); bw = [tuple(x) for x in B61[a]["wdd"]]; b622 = chain622(a)
    out = dict(own=own, base_wdd=bw, base_e622=b622, own_vs_base_jaccard=len(set(own) & set(bw)) / max(len(set(own) | set(bw)), 1), cos_base_wdd=cos_rows(bw), res={"none": by_pole(a)})
    for kind, rows in (("own_wdd", own), ("base_wdd", bw), ("base_e622", b622)):
        with Negate(rows): out["res"][kind] = by_pole(a) | {"kl": kl_off()}
    rr = []
    for _ in range(2 if SMOKE else 3):
        with Negate([(b, int(rng.integers(DFF))) for b, _ in bw]): rr.append(by_pole(a))
    out["res"]["random"] = {k: float(np.mean([x[k] for x in rr])) for k in rr[0]}
    RES["attrs"][a] = out
    log(f"{a}: own chain vs base chain Jaccard {out['own_vs_base_jaccard']:.2f}; base chain rows' cosine with base {np.mean(out['cos_base_wdd']):.3f}; flips from pole0/pole1: " + "; ".join(f"{k} {v['flip_from_0']:.2f}/{v['flip_from_1']:.2f}" for k, v in out["res"].items()))
fl1 = lambda a, k: RES["attrs"][a]["res"][k]["flip_from_1"]
RES["verdicts"] = dict(T1=all(fl1(a, "base_wdd") >= 0.5 * fl1(a, "own_wdd") for a in ("gender", "number")), T2=all(np.mean(RES["attrs"][a]["cos_base_wdd"]) >= 0.9 for a in ("gender", "number")), T3=all(fl1(a, "own_wdd") >= 0.3 for a in ("gender", "number")))
summ = f"[{name}] transfer from Qwen2.5-0.5B: row cosine median {RES['all_rows_cos_median']:.3f}; female/plural flips base chain vs own chain " + ", ".join(f"{a} {fl1(a, 'base_wdd'):.2f}/{fl1(a, 'own_wdd'):.2f}" for a in ("gender", "number")) + f"; verdicts {RES['verdicts']} ({time.time() - t0:.0f}s)"
RES["summary"] = summ; RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"results/e665_transfer_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print(summ, flush=True); print("done", flush=True)
