#!/bin/bash
# session 158 chain: the scheduler, then a retry pass for any job whose result is missing
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
until grep -q "queue done" logs/gpuq_s158.log 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) scheduler done; retries"; bash scripts/retry.sh scripts/jobs_s158.txt; echo "$(date +%H:%M:%S) all session 158 jobs done"
