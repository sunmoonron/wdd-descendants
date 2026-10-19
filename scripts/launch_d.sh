#!/bin/bash
# scheduler d (the Pythia-1B rerun and the shells test) starts once scheduler b is done
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
until grep -q "queue done" logs/gpuq_s156b.log 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) scheduler b done; scheduler d starts"; SAFETY=12 RAMP=90 bash scripts/gpuq.sh scripts/jobs_s156d.txt 4
