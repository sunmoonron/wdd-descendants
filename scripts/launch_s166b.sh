#!/bin/bash
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
SAFETY=12 RAMP=90 bash scripts/gpuq.sh scripts/smokes_s166b.txt 2 > logs/gpuq_smk166b.log 2>&1
ok=1; for f in logs/smk_e727_tiny33_v2.log logs/smk_e726_qwen3_4b_b24_v2.log; do grep -qE "Traceback|error:|No such file" $f && ok=0; done; echo "$(date +%H:%M:%S) smokes b ok=$ok" > logs/auto166b.log
if [ $ok = 1 ]; then SAFETY=12 RAMP=120 bash scripts/gpuq.sh scripts/jobs_s166b.txt 1 > logs/gpuq_s166b.log 2>&1; bash scripts/retry.sh scripts/jobs_s166b.txt >> logs/gpuq_s166b.log 2>&1; echo "$(date +%H:%M:%S) all session 166b jobs done" >> logs/auto166b.log; else echo SMOKES FAILED >> logs/auto166b.log; fi
