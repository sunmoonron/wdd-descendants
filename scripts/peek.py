"""peek: print the summary line of every finished result JSON (box-side), newest last; optional prefix filter"""
import json, glob, os, sys
pre = sys.argv[1] if len(sys.argv) > 1 else "e6"
for f in sorted(glob.glob(f"/workspace/wdd/results/{pre}*.json"), key=os.path.getmtime):
    if "prereg" in f: continue
    try: d = json.load(open(f)); print(f"\n## {os.path.basename(f)} ({d.get('seconds', 0):.0f}s)\n{d.get('summary', '')[:3000]}")
    except Exception as e: print(f"\n## {os.path.basename(f)}: {type(e).__name__} {e}")
