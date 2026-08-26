"""Greedy generation through our runtime — a collapse detector, not a quality measure.

    .venv/bin/python scripts/smoke_generate.py <MODEL_DIR> [max_tokens]
"""
import sys, time
import mlx.core as mx
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from qwen38_flash_next_mlx.load import load
from mlx_lm import generate

try:
    mx.set_wired_limit(int(440e9))
except Exception as e:
    print("[warn]", e)
path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
t0 = time.time(); model, tok = load(path, lazy=True); print(f"[smoke] loaded in {time.time()-t0:.0f}s", flush=True)
PROMPTS = [
    "The capital of France is",
    "Write a Python function that merges overlapping intervals.",
    "Explain in two sentences why the sky appears blue.",
]
for p in PROMPTS:
    msgs = [{"role": "user", "content": p}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    t0 = time.time()
    out = generate(model, tok, prompt=prompt, max_tokens=n, verbose=False)
    print(f"\n=== {p}\n{out}\n[{time.time()-t0:.1f}s, peak {mx.get_peak_memory()/1e9:.0f} GB]", flush=True)
