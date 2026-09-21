#!/bin/bash
# usage: runjob.sh "<cmd>"  -> skips when a matching results/eNN_*_<args>.json exists, else runs and logs
cmd="$1"; cd /workspace/wdd/scripts; source /venv/main/bin/activate >/dev/null 2>&1
export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
script=$(echo "$cmd" | grep -o 'e[0-9][0-9]_[a-z0-9_]*\.py'); pre=${script%%_*}; args=$(echo "$cmd" | sed "s/.*$script//" | xargs echo | tr ' ' '_')
if [ -n "$pre" ] && [ -n "$args" ] && ls /workspace/wdd/results/${pre}_*_${args}.json >/dev/null 2>&1; then echo "SKIP $cmd" >> /workspace/wdd/logs/runjob.log; exit 0; fi
L=/workspace/wdd/logs/$(echo "$cmd" | tr -c 'A-Za-z0-9_.\n' '_').log
eval "$cmd" > $L 2>&1; rc=$?; echo "DONE $cmd (exit $rc)" >> /workspace/wdd/logs/runjob.done; exit $rc
