"""GPU/CPU job scheduler, version 2. Same queue files as version 1 (append-only, one shell command per line). Adds: live
configuration re-read every loop from logs/sched_conf.json (max GPU/CPU jobs, safety margin, memory scale, per-model
needs), priorities (a '#prio=N' tag on the line, else the first matching regex in logs/sched_prio.json, else 5; lower
runs first), de-duplication by the command without its comment, a persistent done-list, and adoption of jobs that a
previous scheduler left running (found in /proc, zombies counted as finished, success judged by the result file)."""
import subprocess, time, os, re, json
W = "/workspace/wdd"; QG, QC = f"{W}/queue_gpu.txt", f"{W}/queue_cpu.txt"; LOG = f"{W}/logs"; RES = f"{W}/results"
CONF = f"{LOG}/sched_conf.json"; PRIO = f"{LOG}/sched_prio.json"; DONE = f"{LOG}/sched_done.txt"
DEF = dict(MAXG=10, MAXC=24, SAFE=5, SCALE=0.7, NEED={"olmo1b": 26, "qwen05": 18, "pythia410": 12, "step": 12, "gpt2": 8, "smollm2": 7})
ENV = "source /venv/main/bin/activate >/dev/null 2>&1; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WDD_CACHE=/workspace/wdd/cache WDD_RESULTS=/workspace/wdd/results; cd /workspace/wdd/scripts; "
def ev(msg):
    with open(f"{LOG}/sched.log", "a") as f: f.write(time.strftime("%H:%M:%S ") + msg + "\n")
def conf():
    c = dict(DEF)
    try: c.update(json.load(open(CONF)))
    except Exception: pass
    return c
def prio_rules():
    try: return json.load(open(PRIO))
    except Exception: return []
def key(cmd): return " ".join(cmd.split("#")[0].split())
def prio_of(cmd, rules):
    m = re.search(r"#prio=(-?\d+)", cmd)
    if m: return int(m.group(1))
    for rx, p in rules:
        if re.search(rx, cmd): return p
    return 5
def gpu():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip().split(",")
        return float(o[0]) / 1024, float(o[1]) / 1024
    except Exception: return 0.0, 80.0
def need_of(cmd, extra, c):
    n = 8
    for k, v in c["NEED"].items():
        if k in cmd: n = max(n, v)
    m = re.search(r"#need=(\d+)", cmd); n = float(m.group(1)) if m else n
    for rx, v in c.get("NEED_RX", []):
        if re.search(rx, cmd): n = float(v); break
    return n * c["SCALE"] * extra
def done_already(cmd):
    m = re.search(r"python (e\d+[a-z]?)_\S*\.py\s*(.*?)\s*(#|$)", cmd)
    if not m: return False
    pre, args = m.group(1), "_".join(m.group(2).split())
    if not args: return False
    return any(f.startswith(pre + "_") and f.endswith("_" + args + ".json") for f in os.listdir(RES))
def logname(cmd): return f"{LOG}/" + re.sub(r"[^A-Za-z0-9_.]", "_", cmd.split("#")[0].strip())[:150] + ".log"
def alive_pid(pid):
    try: st = open(f"/proc/{pid}/stat").read(); return st[st.rfind(")") + 2] not in "ZX"
    except Exception: return False
def adopt():
    found = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid(): continue
        try:
            argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0"); argv = [a.decode() for a in argv if a]
            if len(argv) >= 2 and os.path.basename(argv[0]).startswith("python") and re.match(r"e\d+[a-z]?_\S*\.py$", argv[1]) and os.readlink(f"/proc/{d}/cwd") == f"{W}/scripts":
                env = open(f"/proc/{d}/environ", "rb").read().split(b"\0"); gpuj = b"CUDA_VISIBLE_DEVICES=" not in env
                found.append((int(d), "python " + " ".join(argv[1:]), "g" if gpuj else "c"))
        except Exception: pass
    return found
done = set()
if os.path.exists(DONE): done = set(l.strip() for l in open(DONE) if l.strip())
if os.path.exists(f"{LOG}/sched.log"):
    for l in open(f"{LOG}/sched.log"):
        m = re.match(r"\S+ DONE \d+s (.*)$", l.strip())
        if m: done.add(key(m.group(1)))
seen = {"g": 0, "c": 0}; pend = {"g": [], "c": []}; run = []; retries = {}; arrival = 0
c0 = conf()
for pid, cmd, k in adopt():
    run.append(dict(p=None, pid=pid, cmd=cmd, k=k, need=need_of(cmd, 1, c0), log=logname(cmd), t0=time.time())); ev(f"ADOPT pid={pid} {k} {cmd}")
ev(f"scheduler v2 start; done-list {len(done)}; adopted {len(run)}")
while True:
    c = conf(); rules = prio_rules()
    for k, q in (("g", QG), ("c", QC)):
        if os.path.exists(q):
            lines = [l.strip() for l in open(q) if l.strip() and not l.startswith("//")]
            for l in lines[seen[k]:]: pend[k].append((arrival, l)); arrival += 1
            seen[k] = len(lines)
    alive = []
    for j in run:
        if j["p"] is None:
            if alive_pid(j["pid"]): alive.append(j); continue
            ok = done_already(j["cmd"]); ev(f"{'DONE' if ok else 'FAIL adopted'} {time.time() - j['t0']:.0f}s {j['cmd']}")
            if ok:
                done.add(key(j["cmd"])); open(DONE, "a").write(key(j["cmd"]) + "\n")
            continue
        rc = j["p"].poll()
        if rc is None: alive.append(j); continue
        txt = open(j["log"], errors="ignore").read()[-6000:] if os.path.exists(j["log"]) else ""
        if rc != 0 and ("out of memory" in txt.lower() or "cuda error" in txt.lower()) and retries.get(key(j["cmd"]), 0) < 2:
            retries[key(j["cmd"])] = retries.get(key(j["cmd"]), 0) + 1; pend[j["k"]].append((-1, j["cmd"])); ev(f"OOM requeue ({retries[key(j['cmd'])]}) {j['cmd']}")
        else:
            ev(f"{'DONE' if rc == 0 else 'FAIL rc=' + str(rc)} {time.time() - j['t0']:.0f}s {j['cmd']}")
            if rc == 0: done.add(key(j["cmd"])); open(DONE, "a").write(key(j["cmd"]) + "\n")
    run = alive
    used, tot = gpu(); declared = sum(j["need"] for j in run if j["k"] == "g"); now = time.time()
    young = sum(j["need"] for j in run if j["k"] == "g" and now - j["t0"] < c.get("YOUNG", 150))
    effective = max(used + young, c.get("FLOOR", 0.6) * declared)
    ng = sum(1 for j in run if j["k"] == "g"); nc = sum(1 for j in run if j["k"] == "c"); running_keys = set(key(j["cmd"]) for j in run)
    for k in ("g", "c"):
        keep = []
        for arr, cmd in sorted(pend[k], key=lambda x: (prio_of(x[1], rules), x[0])):
            kk = key(cmd)
            if kk in done or done_already(cmd): ev(f"SKIP {cmd}"); continue
            if kk in running_keys: keep.append((arr, cmd)); continue
            if k == "g":
                need = need_of(cmd, 1.5 ** retries.get(kk, 0), c); free = tot - effective - c["SAFE"]
                if ng < c["MAXG"] and need <= free and (retries.get(kk, 0) < 2 or ng == 0):
                    p = subprocess.Popen(["bash", "-c", ENV + cmd.split("#")[0]], stdout=open(logname(cmd), "w"), stderr=subprocess.STDOUT); run.append(dict(p=p, pid=p.pid, cmd=cmd, k="g", need=need, log=logname(cmd), t0=time.time())); declared += need; effective += need; ng += 1; running_keys.add(kk); ev(f"START gpu need={need:.0f} free={free:.0f} {cmd}"); continue
            else:
                if nc < c["MAXC"]:
                    p = subprocess.Popen(["bash", "-c", ENV + "export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; " + cmd.split("#")[0]], stdout=open(logname(cmd), "w"), stderr=subprocess.STDOUT); run.append(dict(p=p, pid=p.pid, cmd=cmd, k="c", need=0, log=logname(cmd), t0=time.time())); nc += 1; running_keys.add(kk); ev(f"START cpu {cmd}"); continue
            keep.append((arr, cmd))
        pend[k] = keep
    with open(f"{LOG}/sched_state.json", "w") as f: json.dump(dict(time=time.strftime("%H:%M:%S"), gpu_used=round(used, 1), declared=round(declared, 1), effective=round(effective, 1), running=[j["cmd"] for j in run], pending_gpu=len(pend["g"]), pending_cpu=len(pend["c"]), next_gpu=[x[1] for x in sorted(pend["g"], key=lambda x: (prio_of(x[1], rules), x[0]))[:6]], pending_all=[x[1] for x in pend["g"]] + [x[1] for x in pend["c"]]), f)
    time.sleep(4)
