#!/bin/bash
# session 157 chain: the small-job scheduler, its retries, then the two large ledger jobs alone (each retried once if its result is missing)
cd /workspace/wdd; export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1
until grep -q "queue done" logs/gpuq_s157.log 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) scheduler done; retries"; bash scripts/retry.sh scripts/jobs_s157.txt
echo "$(date +%H:%M:%S) large jobs"; while IFS='|' read -r mem tag cmd; do echo "$(date +%H:%M:%S) launching $tag"; bash -c "$cmd" > "logs/$tag.log" 2>&1; echo "$(date +%H:%M:%S) $tag exit $?"; done < scripts/jobs_s157_14b.txt
bash scripts/retry.sh scripts/jobs_s157_14b.txt; echo "$(date +%H:%M:%S) all session 157 jobs done"
