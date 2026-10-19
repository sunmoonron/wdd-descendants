#!/bin/bash
# retry.sh JOBFILE: rerun, one at a time, every job of JOBFILE whose result file is missing (a crash or an out-of-memory in the queue)
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
while IFS='|' read -r mem tag cmd; do
  [ -z "$tag" ] && continue; [ "${mem:0:1}" = "#" ] && continue; [ -f "results/$tag.json" ] && continue
  echo "$(date +%H:%M:%S) retrying $tag alone"; mv "logs/$tag.log" "logs/${tag}_fail1.log" 2>/dev/null; bash -c "$cmd" > "logs/$tag.log" 2>&1; echo "$(date +%H:%M:%S) $tag retry exit $?"
done < "$1"
