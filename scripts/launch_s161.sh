#!/bin/bash
# session 161 chain: the scheduler for the Fisher-dictionary runs and the scheduler for the 7B runs, then a retry pass over both job files
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
until grep -q "queue done" logs/gpuq_s161.log 2>/dev/null && grep -q "queue done" logs/gpuq_s161b.log 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) schedulers done; retries"; bash scripts/retry.sh scripts/jobs_s161b.txt; bash scripts/retry.sh scripts/jobs_s161.txt; echo "$(date +%H:%M:%S) all session 161 jobs done"
