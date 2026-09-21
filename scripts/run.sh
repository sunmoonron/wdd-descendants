#!/bin/bash
# usage: run.sh <script.py> [args...]  -> nohup python in /workspace/wdd/scripts, log to logs/<script>_<args>.log
cd /workspace/wdd/scripts
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
s=$1; shift
tagargs=$(echo "$@" | tr ' /' '__')
nohup python $s "$@" > /workspace/wdd/logs/$(basename $s .py)${tagargs:+_$tagargs}.log 2>&1 &
echo "launched $s $@ pid $!"
