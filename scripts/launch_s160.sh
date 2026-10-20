#!/bin/bash
# session 160 chain: the schedulers d (everything but the elasticity runs) and e4 (all thirty-six elasticity runs), then a retry pass over the job files
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
until grep -q "queue done" logs/gpuq_s160d.log 2>/dev/null && grep -q "queue done" logs/gpuq_s160e4.log 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) schedulers done; retries"; bash scripts/retry.sh scripts/jobs_s160d.txt; bash scripts/retry.sh scripts/jobs_s160e.txt; echo "$(date +%H:%M:%S) all session 160 jobs done"
