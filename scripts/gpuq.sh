#!/bin/bash
# gpuq.sh JOBFILE [MAXJOBS]: lines "MEM_GB|tag|command". Launches the first pending job that fits (MEM_GB + SAFETY free GB, fewer than MAXJOBS running). Touch logs/gpuq.stop to stop launching.
JOBS=$1; MAXJ=${2:-5}; SAFETY=${SAFETY:-6}; RAMP=${RAMP:-100}; cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
mapfile -t LINES < "$JOBS"; N=${#LINES[@]}; declare -a DONE; PIDS=(); n=0; TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
for ((i = 0; i < N; i++)); do line="${LINES[i]}"; { [ -z "$line" ] || [ "${line:0:1}" = "#" ]; } && DONE[i]=1; done
pending() { local c=0; for ((i = 0; i < N; i++)); do [ -z "${DONE[i]}" ] && c=$((c + 1)); done; echo $c; }
while [ "$(pending)" -gt 0 ]; do
  [ -f logs/gpuq.stop ] && { echo "$(date +%H:%M:%S) stop file seen, no more launches"; break; }
  alive=(); for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done; PIDS=("${alive[@]}")
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); free=$((TOTAL - used)); launched=0
  if [ "${#PIDS[@]}" -lt "$MAXJ" ]; then
    for ((i = 0; i < N; i++)); do
      [ -n "${DONE[i]}" ] && continue; line="${LINES[i]}"; MEM="${line%%|*}"; rest="${line#*|}"; TAG="${rest%%|*}"; CMD="${rest#*|}"
      if [ "$free" -ge $(( (MEM + SAFETY) * 1024 )) ]; then
        nohup bash -c "$CMD" < /dev/null > "logs/$TAG.log" 2>&1 &
        PIDS+=($!); DONE[i]=1; n=$((n + 1)); echo "$(date +%H:%M:%S) launched #$n $TAG (needs ${MEM}G, free ${free}M, running ${#PIDS[@]}, pending $(pending))"; launched=1; break
      fi
    done
  fi
  if [ "$launched" = 1 ]; then sleep "$RAMP"; else sleep 30; fi
done
wait; echo "$(date +%H:%M:%S) queue done ($n launched)"
