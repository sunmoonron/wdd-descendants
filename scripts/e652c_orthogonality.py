"""Quick decisive check: are a seed's top co-firing block-4 rows near-orthogonal in their write columns (e87)? If so the
masking mechanism (co-firers shifting each other's consistency context) and the coherent-sum both fail by construction."""
import sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False); name = sys.argv[1]; dev = "cuda:0" if torch.cuda.is_available() else "cpu"
M = {"pythia160": "EleutherAI/pythia-160m", "pythia410": "EleutherAI/pythia-410m"}
tok = AutoTokenizer.from_pretrained(M[name]); model = AutoModelForCausalLM.from_pretrained(M[name], dtype=torch.float32).to(dev).eval()
L = model.gpt_neox.layers; b = 4; W = L[b].mlp.dense_4h_to_h.weight.detach().float(); DFF = W.shape[1]
import datasets
lines = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.split()) > 40][:24]
cap = {}; h = L[b].mlp.dense_4h_to_h.register_forward_pre_hook(lambda m, a: cap.setdefault("a", []).append(a[0][:, 1:].detach().float().reshape(-1, DFF)))
for t in lines: model(tok(t, return_tensors="pt").input_ids[:, :192].to(dev))
h.remove(); A = torch.cat(cap["a"]); An = (A - A.mean(0)) / (A.std(0) + 1e-6); act = A.abs().mean(0)
Wn = W / W.norm(dim=0, keepdim=True).clamp_min(1e-12)
res = []
for seed in act.topk(20).indices.tolist():
    corr = (An[:, seed:seed+1] * An).mean(0); corr[seed] = -9; co = corr.topk(15).indices.tolist()
    cos = (Wn[:, seed] @ Wn[:, co]).abs(); res.append(float(cos.mean()))
print(f"[{name}] block {b}: mean |write cosine| between a high-use seed and its 15 most co-firing rows, over 20 seeds = {np.mean(res):.3f} (median {np.median(res):.3f}); share of seeds with mean |cos| < 0.1 = {np.mean([r<0.1 for r in res]):.2f}")
