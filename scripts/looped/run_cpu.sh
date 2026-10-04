#!/usr/bin/env bash
# CPU-side steps that run alongside the GPU pipeline (16 threads); logs/<name>.log, exit codes in logs/status_cpu.txt
cd /data/loopwdd
step() { name=$1; shift; echo "$(date +%T) start $name" >> logs/status_cpu.txt
  ./run "LW_DEVM=cpu LW_DEVD=cpu LW_THREADS=16 python code/$*" > logs/$name.log 2>&1; echo "$(date +%T) end $name exit $?" >> logs/status_cpu.txt; }
step e8_ouro e8_boundary.py
until grep -q "end e2_ouro" logs/status.txt; do sleep 30; done
step e6_ouro e6_analysis.py
step e9_ouro e9_preds.py
echo "$(date +%T) CPU DONE" >> logs/status_cpu.txt
