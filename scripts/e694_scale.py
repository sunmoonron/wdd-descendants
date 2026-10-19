"""e694 (session 156): the scale point. Three of session 155's measurements rerun on Qwen2.5-14B in bf16 on one 80 GB GPU,
each alone: e689's crowd share of the resultant at the middle block (e689_width.py --model qwen14b), e693's weak-activation
ladder in lite mode (e693_weak_ladder.py --model qwen14b --lite: 48 calibration, 48 held-out and 24 evaluation windows), and
e685's readers' ledger in lite mode at block 24 of 48 (e685_readers_ledger.py --model qwen14b --block 24 --lite: 20 neurons,
32 windows, K = 16, 64, 256 and all inputs). The results are e689_qwen14b.json, e693_qwen14b.json and e685_qwen14b_b24.json;
the verdicts S1-S5 of e692_e694_prereg.json are read from them by the session's record script. Usage: python e694_scale.py"""
import subprocess, sys
PY = sys.executable
for cmd in (["scripts/e689_width.py", "--model", "qwen14b"], ["scripts/e693_weak_ladder.py", "--model", "qwen14b", "--lite"], ["scripts/e685_readers_ledger.py", "--model", "qwen14b", "--block", "24", "--lite"]):
    print("running", " ".join(cmd), flush=True); subprocess.run([PY] + cmd, check=True)
