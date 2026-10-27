#!/bin/bash
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
SAFETY=12 RAMP=60 bash scripts/gpuq.sh scripts/smokes_s166c.txt 1 > logs/gpuq_smk166c.log 2>&1
ok=1; grep -qE "Traceback|error:|No such file" logs/smk_e728_qwen35.log && ok=0; [ -f results/e728_qwen35_smoke.json ] || ok=0
echo "$(date +%H:%M:%S) smoke c ok=$ok" > logs/auto166c.log
if [ $ok = 1 ]; then SAFETY=12 RAMP=120 bash scripts/gpuq.sh scripts/jobs_s166c.txt 1 > logs/gpuq_s166c.log 2>&1; bash scripts/retry.sh scripts/jobs_s166c.txt >> logs/gpuq_s166c.log 2>&1; echo "$(date +%H:%M:%S) all session 166c jobs done" >> logs/auto166c.log; else echo SMOKE C FAILED >> logs/auto166c.log; fi
