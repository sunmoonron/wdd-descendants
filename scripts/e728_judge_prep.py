"""Build the blind judge's packet from results/e728_qwen35.json: shuffled, unlabeled sample sets.
Sets: stage 3 wdd/random rows at nat10 and zoo0.8 (40), the eight zoo specs at zoo0.8 (positive controls for the judge),
the eight random controls at zoo0.8, and the base answers (negative control). The key file maps set ids back."""
import json, random, sys, os
src = sys.argv[1]; out_dir = sys.argv[2]; NW = 45; NT = 6
d = json.load(open(src)); sets = []
for grp in ("wdd", "random"):
    for i, e in enumerate(d["stage3"][grp]):
        for cond in ("nat1", "nat3", "zoo0.8"):
            sets.append({"label": f"stage3 {grp} #{i} L{e['row']['layer']} N{e['row']['neuron']} {cond}", "texts": e[cond]["texts"]})
for grp in ("wdd", "random"):
    for i, e in enumerate(d["stage3b"][grp]):
        sets.append({"label": f"stage3b {grp} #{i} L{e['row']['layer']} N{e['row']['neuron']} zoo0.8", "texts": e["zoo0.8"]["texts"]})
for k, v in d["stage2"].items():
    dose = "zoo0.8" if "zoo0.8" in v["doses"] else sorted(v["doses"])[0]
    sets.append({"label": f"stage2 {k} {dose}", "texts": v["doses"][dose]["texts"]})
sets.append({"label": "base answers", "texts": [b["text"] for b in d["base_answers"]]})
rng = random.Random(7281); rng.shuffle(sets)
key = {}; packet = []
for i, s in enumerate(sets):
    sid = f"S{i + 1:02d}"; key[sid] = s["label"]
    packet.append({"id": sid, "samples": [" ".join(t.split()[:NW]) for t in s["texts"][:NT]]})
prompts = [b["prompt"] for b in d["base_answers"]][:NT]
os.makedirs(out_dir, exist_ok=True)
json.dump({"prompts": prompts, "sets": packet}, open(os.path.join(out_dir, "judge_packet.json"), "w"), indent=1)
json.dump(key, open(os.path.join(out_dir, "judge_key.json"), "w"), indent=1)
print("sets", len(packet), "words", sum(len(t.split()) for s in packet for t in s["samples"]))
