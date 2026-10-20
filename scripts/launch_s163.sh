#!/bin/bash
# session 163 chain: wait for the three queues (a, b and the retry queue r), then rerun alone whatever still lacks a result
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
for q in a b r; do until grep -q "queue done" logs/gpuq_s163$q.log 2>/dev/null; do sleep 60; done; echo "$(date +%H:%M:%S) queue $q done"; done
echo "$(date +%H:%M:%S) all queues done; retries"; bash scripts/retry.sh scripts/jobs_s163.txt; echo "$(date +%H:%M:%S) all session 163 jobs done"
