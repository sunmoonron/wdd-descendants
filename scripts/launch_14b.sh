#!/bin/bash
# waits for scheduler b to finish, then runs the three 14B jobs one at a time (each needs most of the GPU)
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1 C11_THREADS=4
until grep -q "queue done" logs/gpuq_s156d.log 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) scheduler d done; 14B jobs start"
for j in "e689_qwen14b|scripts/e689_width.py --model qwen14b" "e693_qwen14b|scripts/e693_weak_ladder.py --model qwen14b --lite" "e685_qwen14b_b24|scripts/e685_readers_ledger.py --model qwen14b --block 24 --lite"; do
  tag=${j%%|*}; cmd=${j#*|}; echo "$(date +%H:%M:%S) launching $tag"; /venv/main/bin/python $cmd > logs/$tag.log 2>&1; echo "$(date +%H:%M:%S) $tag done, exit $?"
done
echo "$(date +%H:%M:%S) all 14B jobs done"
