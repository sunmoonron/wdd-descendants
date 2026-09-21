#!/bin/bash
# usage: wave.sh <parallel> <jobfile>   (each line = a shell command run from /workspace/wdd/scripts)
cd /workspace/wdd/scripts; source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$1; J=$2; B=$(basename $J)
nohup sh -c "cat $J | xargs -P $P -I{} sh -c 'L=/workspace/wdd/logs/\$(echo \"{}\" | tr -c \"A-Za-z0-9_.\\n\" \"_\").log; {} > \$L 2>&1; echo \"DONE {} (exit \$?)\" >> /workspace/wdd/logs/$B.done'" > /workspace/wdd/logs/$B.log 2>&1 &
echo "wave $B launched P=$P pid $!"
