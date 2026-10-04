#!/usr/bin/env bash
# the full pipeline, in order of value; each step logs to logs/<name>.log and its exit code to logs/status.txt
cd /data/loopwdd
step() { name=$1; shift; echo "$(date +%T) start $name" >> logs/status.txt
  ./run "python code/$*" > logs/$name.log 2>&1; echo "$(date +%T) end $name exit $?" >> logs/status.txt; }
step e2_ouro e2_codes.py ouro
step e3_ouro_single e3_splice.py ouro single
step e2_smol e2_codes.py smol
step e3_smol_single e3_splice.py smol single
step e4_ouro e4_perturb.py ouro
step e4_smol e4_perturb.py smol
step e5_ouro e5_ledger.py
step e3_ouro_compound e3_splice.py ouro compound
step e3_smol_compound e3_splice.py smol compound
echo "$(date +%T) ALL DONE" >> logs/status.txt
