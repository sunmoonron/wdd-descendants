"""e700 (session 157): the ledger at 7B and 14B with the model in fp32. e694's bf16 run left the all-inputs check at
0.95 of the ablation damage, and a bf16 rerun with the activations stored in fp32 left it at 0.86: the residual stream
itself is rounded to bf16 block by block, so the exact split can only be checked against an fp32 model. Here
e685_readers_ledger.py runs with --fp32model (the weights in fp32), --fp32acts (captured activations in fp32), --nofloor
(the floor calibration skipped, its dictionaries not fitting beside a 14B model) and --select (the 20 neurons with the
largest ablation KL among 100 candidates), on Qwen2.5-7B at block 14 with 32 windows and on Qwen2.5-14B at block 24 with
16 windows; it also records the noise floor (the neuron replaced by its own captured activation). The results are
e685_qwen7b_b14.json and e685_qwen14b_b24.json; verdicts N1-N3 of e696_e701_prereg.json are read from them by the record
script. Usage: python e700_ledger_fp32.py"""
import subprocess, sys
PY = sys.executable
for cmd in (["scripts/e685_readers_ledger.py", "--model", "qwen7b", "--block", "14", "--lite", "--fp32model", "--fp32acts", "--nofloor", "--select"], ["scripts/e685_readers_ledger.py", "--model", "qwen14b", "--block", "24", "--lite", "--nwin", "16", "--fp32model", "--fp32acts", "--nofloor", "--select"]):
    print("running", " ".join(cmd), flush=True); subprocess.run([PY] + cmd, check=True)
