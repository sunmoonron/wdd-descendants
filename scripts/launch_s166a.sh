#!/bin/bash
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
until grep -q "queue done" logs/gpuq_smk166.log 2>/dev/null; do sleep 30; done
ok=1; for f in logs/smk_*_v2.log; do grep -qE "Traceback|error:|No such file" $f && ok=0; done; echo "$(date +%H:%M:%S) smokes ok=$ok" > logs/auto166.log
if [ $ok = 1 ]; then SAFETY=8 RAMP=120 bash scripts/gpuq.sh scripts/jobs_s166a.txt 3 > logs/gpuq_s166a.log 2>&1; bash scripts/retry.sh scripts/jobs_s166a.txt >> logs/gpuq_s166a.log 2>&1; echo "$(date +%H:%M:%S) all session 166a jobs done" >> logs/auto166.log; else echo SMOKES FAILED >> logs/auto166.log; fi
