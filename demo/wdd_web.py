"""A browser front-end for wdd_demo.py: a small HTTP server (Python standard library only) that keeps the models on the
GPU and runs each step of the demo when you click. Every number on the page is computed by the click that shows it.

On the GPU box (binds 127.0.0.1 only; exits after --idle-exit seconds without a request, default two hours):
    python wdd_web.py --port 8765
From your machine:
    ssh -N -L 8765:127.0.0.1:8765 <gpu-box>     then open http://localhost:8765
"""
import sys, os, json, time, threading, random, argparse, collections, traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--idle-exit", type=int, default=7200); WA = ap.parse_args()
sys.argv = [sys.argv[0], "--auto", "--no-color"]   # wdd_demo parses its own arguments at import
import torch
import wdd_demo as wd
LOCK = threading.Lock(); ST = {}; LAST = [time.time()]
def log(s): print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
# ---------------------------------------------------------------- Act 1
def lm1():
    if "lm1" not in ST:
        lm = wd.LM("Qwen/Qwen2.5-0.5B"); t = time.time(); IT = {a: lm.items(a) for a in wd.POLES}; SC = {a: lm.scan(IT[a]) for a in wd.POLES}; ST["scan_s"] = time.time() - t
        lm.wiki_setup(); ST["l0"] = lm.cost()[0]; ST.update(lm1=lm, IT1=IT, SC1=SC)
    return ST["lm1"], ST["IT1"], ST["SC1"]
def a1_scan(q):
    lm, IT, SC = lm1(); out = {}
    for a in wd.POLES:
        A, s = SC[a]["A"], SC[a]["s"]; rows = SC[a]["order"][:8]
        rr = [dict(block=b, row=r, unmarked=float(A[s == 0, b, r].abs().mean()), marked=float(A[s == 1, b, r].abs().mean())) for b, r in rows]
        ratio = sum(x["marked"] for x in rr) / max(sum(x["unmarked"] for x in rr), 1e-9); nrev = sum(x["unmarked"] > x["marked"] for x in rr)
        out[a] = dict(poles=wd.POLES[a], rows=rr, ratio=ratio, n_rev=nrev, n_items=len(IT[a]), n_held=sum(not it["disc"] for it in IT[a])); ST.setdefault("mark", {})[a] = (ratio, nrev)
    return dict(model=lm.hf, attrs=out, scan_seconds=ST["scan_s"], n_scanned=(lm.L + 1) * lm.DFF, n_total=lm.NB * lm.DFF)
def a1_edit(q):
    lm, IT, SC = lm1(); a = q.get("attr", "gender"); k = int(q.get("k", 8)); assert a in wd.POLES and 1 <= k <= 64
    rows = SC[a]["order"][:k]; held = [it for it in IT[a] if not it["disc"]]
    with wd.Negate(lm, rows):
        f = lm.flips(IT[a]); loss, kl = lm.cost(); lm.run(held)
        ex = [next(({"lang": wd.LN[lg], "word": x["word"], "expected": x["en"][1], "got": x["en"][x["pred"]], "flipped": True} for x in held if x["side"] == 1 and x["lg"] == lg and x["pred"] == 0), None) for lg in wd.LANGS]
        ex += [next(({"lang": wd.LN[x["lg"]], "word": x["word"], "expected": x["en"][0], "got": x["en"][x["pred"]], "flipped": True} for x in held if x["side"] == 0 and x["pred"] == 1), None)]
        ex += [next(({"lang": wd.LN[x["lg"]], "word": x["word"], "expected": x["en"][1], "got": x["en"][x["pred"]], "flipped": False} for x in held if x["side"] == 1 and x["pred"] == 1), None)]
    ST.setdefault("dose", {})[(a, k)] = dict(f=f, kl=kl, dloss=loss - ST["l0"])
    return dict(attr=a, poles=wd.POLES[a], k=k, rows=[list(x) for x in rows], flip_marked=f[1], flip_unmarked=f[0], n_held=len(held), loss0=ST["l0"], loss=loss, kl=kl, examples=[e for e in ex if e],
                code=f"for b, j in {a}_rows: model.model.layers[b].mlp.down_proj.weight.data[:, j] *= -1   # {len(rows)} columns of {lm.NB * lm.DFF:,}")
def a1_free(q):
    lm, IT, SC = lm1(); k = int(q.get("k", 8)); assert 1 <= k <= 64
    if "ft0" not in ST: ST["ft0"] = lm.free_text()
    with wd.Negate(lm, SC["gender"]["order"][:k]): ft1 = lm.free_text()
    ST["ft"] = (ST["ft0"], ft1, k); return dict(k=k, before=ST["ft0"], after=ft1)
def a1_gen(q):
    lm, IT, SC = lm1(); p = str(q.get("prompt", "")).strip()[:300] or "The queen looked at the letter, and then"; a = q.get("attr", "gender"); k = int(q.get("k", 8)); assert a in wd.POLES and 1 <= k <= 64
    g0 = lm.generate(p, n=24)
    with wd.Negate(lm, SC[a]["order"][:k]): g1 = lm.generate(p, n=24)
    return dict(prompt=p, before=g0, after=g1, attr=a, k=k)
def a1_tools(q):
    lm, IT, SC = lm1(); k = int(q.get("k", 8)); a = "gender"; md = SC[a]["md"]; lev = SC[a]["level"]; v = -md[lev]; rows = []
    with wd.Negate(lm, SC[a]["order"][:k]): f = lm.flips(IT[a]); rows.append(dict(method=f"negate the top {k} rows (an edit in the weights)", f1=f[1], f0=f[0], kl=lm.cost()[1]))
    with wd.Hook(lm, ablate={b: md[b] for b in range(lm.L + 1)}): f = lm.flips(IT[a]); rows.append(dict(method="ablate the female-minus-male direction at every block output (Arditi et al. 2024)", f1=f[1], f0=f[0], kl=lm.cost()[1]))
    for al in (1, 2, 4):
        with wd.Hook(lm, add={lev: al * v}): f = lm.flips(IT[a]); rows.append(dict(method=f"steering vector (male minus female) x{al} at block {lev}, every token", f1=f[1], f0=f[0], kl=lm.cost()[1]))
    ST["tools"] = rows; return dict(rows=rows, k=k)
# ---------------------------------------------------------------- Act 2
def a2_run(q):
    lm, IT, SC = lm1(); k = int(q.get("k", 8)); assert 1 <= k <= 32
    if "lm2" not in ST:
        base_cols = {b: lm.W(b).detach().float().cpu().clone() for b in range(lm.L + 1)}
        c = wd.LM("Qwen/Qwen2.5-Coder-0.5B-Instruct"); IT2 = {a: c.items(a) for a in wd.POLES}; SC2 = {a: c.scan(IT2[a]) for a in wd.POLES}; c.wiki_setup(); ST.update(lm2=c, IT2=IT2, SC2=SC2, base_cols=base_cols)
    c, IT2, SC2, bc = ST["lm2"], ST["IT2"], ST["SC2"], ST["base_cols"]; cs = torch.nn.functional.cosine_similarity
    allc = float(torch.cat([cs(bc[b], c.W(b).detach().float().cpu(), dim=0) for b in range(c.L + 1)]).median()); rot = {}
    for a in wd.POLES:
        rc = [float(cs(bc[b][:, r], c.W(b).detach().float().cpu()[:, r], dim=0)) for b, r in SC[a]["order"][:k]]; mdc = [float(cs(SC[a]["md"][b], SC2[a]["md"][b], dim=0)) for b in range(1, c.L + 1)]
        rot[a] = dict(row_cos=rc, row_cos_median=sorted(rc)[len(rc) // 2], dir_cos=mdc, dir_min=min(mdc), dir_max=max(mdc))
    tab = []
    for kk in sorted({k, 32}):
        r_ = dict(edit=f"negate {kk} rows", k=kk)
        for a in wd.POLES:
            with wd.Negate(c, SC[a]["order"][:kk]): fb = c.flips(IT2[a]); klb = c.cost()[1]
            with wd.Negate(c, SC2[a]["order"][:kk]): fo = c.flips(IT2[a])
            r_[a] = dict(base=fb[1], own=fo[1], kl_base=klb)
        tab.append(r_)
    r_ = dict(edit="ablate the direction at every block", k=None)
    for a in wd.POLES:
        with wd.Hook(c, ablate={b: SC[a]["md"][b] for b in range(c.L + 1)}): fb = c.flips(IT2[a]); klb = c.cost()[1]
        with wd.Hook(c, ablate={b: SC2[a]["md"][b] for b in range(c.L + 1)}): fo = c.flips(IT2[a])
        r_[a] = dict(base=fb[1], own=fo[1], kl_base=klb)
    tab.append(r_); p = "The queen looked at the letter, and then"; g0 = c.generate(p, n=24)
    with wd.Negate(c, SC["gender"]["order"][:k]): g1 = c.generate(p, n=24)
    ST["act2"] = dict(tab=tab, rot=rot, allc=allc, k=k)
    return dict(model=c.hf, n_items={a: len(IT2[a]) for a in wd.POLES}, all_cos_median=allc, rot=rot, table=tab, base_rows={a: [list(x) for x in SC[a]["order"][:k]] for a in wd.POLES}, gen=dict(prompt=p, before=g0, after=g1), k=k)
# ---------------------------------------------------------------- Act 3
class Audit:
    def __init__(self):
        import datasets
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m"); self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", dtype=torch.float32).to(wd.DEV).eval()
        self.B = 4; self.lay = self.model.gpt_neox.layers[self.B]; self.down = self.lay.mlp.dense_4h_to_h.weight; self.readw = self.lay.mlp.dense_h_to_4h.weight.detach().float(); self.DFF = self.down.shape[1]
        lines = [x for x in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(x.split()) > 40]
        self.PW = torch.stack([tok(x, return_tensors="pt").input_ids[0, :192] for x in lines[:40] if tok(x, return_tensors="pt").input_ids.shape[1] >= 192][:16]).to(wd.DEV)
        self.LP0 = self.lps(); self.X0, self.A0 = self.states(); self.top = self.A0.abs().topk(32, dim=0).indices; self.base = self.audit(); use = self.A0.abs().mean(0); self.use = use; self.pool = use.topk(40).indices.tolist()
    def lps(self): return torch.cat([self.model(self.PW[s:s + 4]).logits[:, 1:].float().log_softmax(-1).cpu() for s in range(0, self.PW.shape[0], 4)]).reshape(-1, self.model.config.vocab_size)
    def states(self):
        cap = []; h = self.lay.mlp.dense_4h_to_h.register_forward_pre_hook(lambda m, a: cap.append(a[0][:, 1:].detach().float().reshape(-1, self.DFF))); Xs = []
        try:
            for s in range(0, self.PW.shape[0], 4): Xs.append(self.model(self.PW[s:s + 4], output_hidden_states=True).hidden_states[self.B + 1][:, 1:].detach().float().reshape(-1, self.model.config.hidden_size))
        finally: h.remove()
        return torch.cat(Xs), torch.cat(cap)
    def consistency(self, X, A, W):
        out = torch.empty(self.DFF, device=wd.DEV)
        for j0 in range(0, self.DFF, 256):
            js = torch.arange(j0, min(j0 + 256, self.DFF), device=wd.DEV); idx = self.top[:, js]; w = W[:, js].T; rest = X[idx] - A[idx, js][..., None] * w[None]
            out[js] = torch.nn.functional.cosine_similarity(w, rest.mean(0), dim=1)
        return out
    def audit(self, clean_states=False):
        W = self.down.detach().float(); nz = W.norm(dim=0); z = (nz - nz.mean()) / nz.std()
        rw = ((W / nz.clamp_min(1e-12)) * (self.readw / self.readw.norm(dim=1, keepdim=True).clamp_min(1e-12)).T).sum(0)
        X, A = (self.X0, self.A0) if clean_states else self.states(); cons = self.consistency(X, A, W)
        return dict(z=z, rw=rw, cons=cons, thr_rw=float(rw.kthvalue(int(0.99 * self.DFF)).values), thr_cons=float(cons.kthvalue(max(int(0.01 * self.DFF), 1)).values))
    def tamper(self, j, how, w):
        u = self.readw[j] / self.readw[j].norm()
        if how == "scale": self.down.data[:, j] = 2 * w
        elif how == "negate": self.down.data[:, j] = -w
        else: self.down.data[:, j] = -w + 2 * (w @ u) * u
    def kl_at(self, j): pos = self.top[:, j].cpu(); return float((self.LP0[pos].exp() * (self.LP0[pos] - self.lps()[pos])).sum(-1).mean())
    def run(self, j, how):
        w = self.down[:, j].detach().clone()
        try:
            self.tamper(j, how, w); kl = self.kl_at(j); a = self.audit(); e = self.audit(clean_states=True)
        finally: self.down.data[:, j] = w
        rank = lambda s, desc: int((s.argsort(descending=desc) == j).nonzero()[0, 0]) + 1
        return dict(row=j, how=how, kl=kl, z=float(a["z"][j]), rank_z=rank(a["z"].abs(), True), flag_norm=abs(float(a["z"][j])) > 3, rw=float(a["rw"][j]), rank_rw=rank(a["rw"], True), thr_rw=a["thr_rw"], flag_rw=float(a["rw"][j]) > a["thr_rw"],
                    cons=float(a["cons"][j]), rank_cons=rank(a["cons"], False), thr_cons=a["thr_cons"], flag_cons=float(a["cons"][j]) < a["thr_cons"], cons_clean=float(e["cons"][j]), rank_cons_clean=rank(e["cons"], False), flag_cons_clean=float(e["cons"][j]) < e["thr_cons"],
                    before=dict(z=float(self.base["z"][j]), rw=float(self.base["rw"][j]), cons=float(self.base["cons"][j])), dff=self.DFF)
def a3():
    if "a3" not in ST: ST["a3"] = Audit()
    return ST["a3"]
def a3_info(q):
    P = a3(); rw = P.base["rw"]
    return dict(model="EleutherAI/pythia-410m", block=P.B, dff=P.DFF, rw_median=float(rw.median()), rw_below=float((rw < -0.1).float().mean()), pool=P.pool, use_ratio={str(j): float(P.use[j] / P.use.median()) for j in P.pool})
def a3_tamper(q):
    P = a3(); how = q.get("how", "reflect"); assert how in ("scale", "negate", "reflect")
    j = q.get("row"); j = random.Random(int(q.get("seed", 0))).choice(P.pool) if j in (None, "", "random") else int(j); assert 0 <= j < P.DFF
    r = P.run(j, how); r["use_ratio"] = float(P.use[j] / P.use.median()); ST.setdefault("a3_single", []).append(r); return r
def a3_sweep(q):
    P = a3(); n = max(1, min(int(q.get("n", 12)), 40)); rows = random.Random(int(q.get("seed", 1))).sample(P.pool, n); tab = {h: collections.Counter() for h in ("scale", "negate", "reflect")}; kls = {h: [] for h in tab}
    for j in rows:
        for how in tab:
            r = P.run(j, how); kls[how].append(r["kl"]); tab[how].update(n=1, norm=r["flag_norm"], rw=r["flag_rw"], cons=r["flag_cons"], any=(r["flag_norm"] or r["flag_rw"] or r["flag_cons"]), clean=r["flag_cons_clean"])
    out = {h: {k_: v / t["n"] for k_, v in t.items() if k_ != "n"} | {"kl": sum(kls[h]) / t["n"], "n": t["n"]} for h, t in tab.items()}; ST["a3_sweep"] = out; return dict(n=n, rows=rows, table=out)
# ---------------------------------------------------------------- the pasted script's claims, against what has been run so far
def claims(q):
    pc = lambda x: f"{100 * x:.1f}%"; out = []
    d = ST.get("dose", {})
    if ("number", 8) in d or ("number", 32) in d:
        meas = ", ".join(f"{pc(d[('number', kk)]['f'][1])} with {kk} rows" for kk in (1, 8, 32) if ("number", kk) in d); best = max(d[("number", kk)]["f"][1] for kk in (1, 8, 32) if ("number", kk) in d)
        out.append(dict(claim="Act 1: plurals translate to singulars 96-100% (e662)", measured=meas, verdict="holds" if best >= 0.96 else "partly"))
    if "ft" in ST:
        ft, kk = ST["ft"][1], ST["ft"][2]; fem = {lg: ft[lg][1] for lg in ("en", "fr", "es", "de")}; n_ = sum(v < 0.5 for v in fem.values())
        out.append(dict(claim="Act 1: feminine agreement flips to masculine in English, French, Spanish and German", measured=f"female-subject agreement after {kk} rows: " + ", ".join(f"{lg} {pc(v)}" for lg, v in fem.items()), verdict="holds" if n_ == 4 else ("partly" if n_ else "does not hold")))
    if any(x[0] == "gender" for x in d):
        kk = max(x[1] for x in d if x[0] == "gender"); out.append(dict(claim="Act 1: the edit costs +0.0003 nats", measured=f"{kk} gender rows: {d[('gender', kk)]['dloss']:+.4f} nats on wikitext" + (f"; 1 row {d[('gender', 1)]['dloss']:+.4f}" if ("gender", 1) in d else ""), verdict="holds" if d[("gender", kk)]["dloss"] < 0.001 else "partly"))
    if "mark" in ST:
        (rg, ng), (rn, nn) = ST["mark"]["gender"], ST["mark"]["number"]
        out.append(dict(claim="Act 1: the marked rows blaze 3-6x and the unmarked pole is dead quiet", measured=f"female/male {rg:.1f}x, plural/singular {rn:.1f}x over the top 8 rows; {ng + nn} of the 16 rows fire more for the unmarked pole", verdict="holds" if min(rg, rn) >= 3 and ng + nn == 0 else "partly"))
    if "tools" in ST:
        t = ST["tools"]; st = [r for r in t[2:] if r["f1"] >= t[1]["f1"] and r["f0"] < 0.2]
        out.append(dict(claim="Act 1: a sign clamp is 4,690x cleaner than a steering vector (e660)", measured="the clamp is not run here; " + (f"a steering vector matching the direction ablation costs {st[0]['kl'] / max(t[0]['kl'], 1e-9):.0f}x the row edit's KL" if st else "no steering scale matched the ablation"), verdict="not tested"))
    if "act2" in ST:
        A2 = ST["act2"]; tab = A2["tab"]; nk = [r for r in tab if r["k"] is not None]; ab = tab[-1]
        pl = max(r["number"]["base"] for r in nk); fe = max([r["gender"]["base"] for r in nk] + [ab["gender"]["base"]])
        out.append(dict(claim="Act 2: the base addresses flip 100% of plurals and 56-88% of feminine nouns in Coder-Instruct", measured="; ".join(f"{r['edit']}: plurals {pc(r['number']['base'])}, female {pc(r['gender']['base'])}" for r in tab), verdict="holds" if pl >= 0.99 and fe >= 0.56 else "partly"))
        rot = A2["rot"]; out.append(dict(claim="Act 2: the rows rotate to cosine 0.44-0.46 while the attribute axis stays at 0.79-0.89", measured=f"all write columns median {A2['allc']:.2f}; the base rows {rot['gender']['row_cos_median']:.2f} (gender), {rot['number']['row_cos_median']:.2f} (number); the directions {min(rot['gender']['dir_min'], rot['number']['dir_min']):.2f} to {max(rot['gender']['dir_max'], rot['number']['dir_max']):.2f} across blocks", verdict="partly"))
    if "a3_sweep" in ST:
        sw = ST["a3_sweep"]; ok = sw["scale"]["norm"] >= 0.9 and sw["negate"]["rw"] >= 0.9 and sw["reflect"]["cons"] >= 0.9
        out.append(dict(claim="Act 3: scale trips the norm, negate the read-write cosine, reflect the consistency check; no single-row flip escapes", measured=f"caught by some detector: scale {pc(sw['scale']['any'])}, negate {pc(sw['negate']['any'])}, reflect {pc(sw['reflect']['any'])} (consistency with clean states, as e651 scored it: {pc(sw['reflect']['clean'])})", verdict="holds" if ok else ("does not hold" if sw["reflect"]["any"] < 0.5 else "partly")))
    elif ST.get("a3_single"):
        r = ST["a3_single"][-1]; caught = r["flag_norm"] or r["flag_rw"] or r["flag_cons"]
        out.append(dict(claim="Act 3: no single-row flip escapes the three detectors", measured=f"last tamper ({r['how']}, row {r['row']}): " + ("caught" if caught else "missed by all three") + " (run the sweep for rates)", verdict="holds" if caught else "does not hold"))
    return dict(claims=out)
def status(q): return dict(act1="lm1" in ST, act2="lm2" in ST, act3="a3" in ST, device=wd.DEV, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
API = dict(a1_scan=a1_scan, a1_edit=a1_edit, a1_free=a1_free, a1_gen=a1_gen, a1_tools=a1_tools, a2_run=a2_run, a3_info=a3_info, a3_tamper=a3_tamper, a3_sweep=a3_sweep, claims=claims, status=status)
# ---------------------------------------------------------------- HTTP
HERE = os.path.dirname(os.path.abspath(__file__))
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode(); self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        LAST[0] = time.time()
        if self.path in ("/", "/index.html"): return self._send(200, open(os.path.join(HERE, "wdd_web.html"), "rb").read(), "text/html; charset=utf-8")
        if self.path == "/api/status": return self._send(200, json.dumps(status({})))
        self._send(404, json.dumps(dict(error="not found")))
    def do_POST(self):
        LAST[0] = time.time(); name = self.path.rsplit("/", 1)[-1]
        if not self.path.startswith("/api/") or name not in API: return self._send(404, json.dumps(dict(error="not found")))
        n = int(self.headers.get("Content-Length") or 0)
        if n > 10000: return self._send(413, json.dumps(dict(error="request too large")))
        try: q = json.loads(self.rfile.read(n) or b"{}")
        except Exception: return self._send(400, json.dumps(dict(error="bad json")))
        t = time.time()
        with LOCK:
            try:
                with torch.no_grad(): res = API[name](q)
                res["_seconds"] = time.time() - t; log(f"{name} {q} {res['_seconds']:.1f}s"); body = json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            except Exception as e: log(f"{name} failed: {traceback.format_exc()}"); return self._send(500, json.dumps(dict(error=f"{type(e).__name__}: {e}")))
        LAST[0] = time.time(); self._send(200, body)
def watchdog():
    while True:
        time.sleep(60)
        if time.time() - LAST[0] > WA.idle_exit: log("idle, exiting"); os._exit(0)
if __name__ == "__main__":
    threading.Thread(target=watchdog, daemon=True).start(); srv = ThreadingHTTPServer(("127.0.0.1", WA.port), H); log(f"serving on http://127.0.0.1:{WA.port} ({wd.DEV})"); srv.serve_forever()
