#!/bin/bash
# usage: wave2.sh <parallel> <jobfile>  (resume-safe: uses runjob.sh)
P=$1; J=$2; B=$(basename $J)
nohup sh -c "cat $J | xargs -P $P -I{} bash /workspace/wdd/scripts/runjob.sh '{}'" > /workspace/wdd/logs/$B.runner.log 2>&1 &
echo "wave $B launched P=$P pid $!"
