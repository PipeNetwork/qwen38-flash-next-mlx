"""Load a published build the way the card says to — stock mlx-lm + model_file — and generate."""
import sys, time
import mlx.core as mx
from mlx_lm import load, generate
path = sys.argv[1]
t0 = time.time(); model, tok = load(path, trust_remote_code=True, lazy=True); print(f"[stock] loaded via mlx_lm.load in {time.time()-t0:.0f}s: {type(model).__module__}")
prompt = tok.apply_chat_template([{"role": "user", "content": "In one sentence, what is a mixture-of-experts model?"}], add_generation_prompt=True, tokenize=False)
print(generate(model, tok, prompt=prompt, max_tokens=120, verbose=False)); print(f"[stock] peak {mx.get_peak_memory()/1e9:.0f} GB")
