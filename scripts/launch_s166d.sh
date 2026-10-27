#!/bin/bash
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
SAFETY=10 RAMP=60 bash scripts/gpuq.sh scripts/jobs_s166d.txt 1 > logs/gpuq_s166d.log 2>&1; bash scripts/retry.sh scripts/jobs_s166d.txt >> logs/gpuq_s166d.log 2>&1; echo "$(date +%H:%M:%S) all session 166d jobs done" >> logs/auto166d.log
