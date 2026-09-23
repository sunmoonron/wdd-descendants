#!/bin/bash
# usage: waitrun.sh "<glob of result json>" <count> <HHMM deadline> <command...>; waits until the glob matches count files or the deadline passes, then runs the command
g=$1; n=$2; dl=$3; shift 3
while [ "$(ls $g 2>/dev/null | wc -l)" -lt "$n" ] && [ "$(date +%H%M)" -lt "$dl" ]; do sleep 30; done
echo "$(date +%H:%M:%S) waited: $(ls $g 2>/dev/null | wc -l) of $n files"; exec "$@"
