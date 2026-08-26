"""Teacher-forced, module-by-module comparison: where does the MLX port first diverge?"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch, mlx.core as mx
from test_parity import TINY, build_hf, build_mlx

torch.manual_seed(1)
B, T = 2, 12
ids = torch.randint(2, TINY["vocab_size"], (B, T)); ids[0, 5] = 1; ids[1, 0] = 1
hf = build_hf()
rec = {}
def hook(name):
    def f(mod, inp, out):
        rec[name] = (tuple(i.detach().float().numpy() for i in inp if torch.is_tensor(i)),
                     tuple(o.detach().float().numpy() for o in (out if isinstance(out, tuple) else (out,)) if torch.is_tensor(o)))
    return f
for name, mod in hf.named_modules():
    if name: mod.register_forward_hook(hook(name))
with torch.no_grad():
    ref_logits = hf(input_ids=ids).logits.float().numpy()

model = build_mlx(hf)
def d(a, b, label):
    a = np.array(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape: print(f"  {label:48s} SHAPE {a.shape} vs {b.shape}"); return
    print(f"  {label:48s} max|d| {np.abs(a-b).max():.3e}   scale {np.abs(b).max():.3e}")

m = model.model
x = mx.array(ids.numpy())
h = m.embed_tokens(x); d(h, rec["model.embed_tokens"][1][0], "embed")
h = mx.tile(h, (1, 1, m.hc))
full_idx = [i for i, l in enumerate(m.layers) if l.layer_type == "full_attention"]
from mlx_lm.models.base import create_attention_mask
mask = create_attention_mask(h, None)
eos = TINY["eos_token_id"]
prev_ctx = mx.full((B, 2), eos, x.dtype)
for i, layer in enumerate(m.layers):
    p = f"model.layers.{i}"
    hin = rec[p][0][0]
    d(h, hin, f"layer{i} input")
    hh = mx.array(hin)  # teacher-force the layer input
    if layer.ple is not None:
        pin = rec[p + ".ple"][0][0]
        emb_ref = rec[p + ".ple.ple_embedding"][1][0]
        emb = layer.ple.ple_embedding(x, prev_ctx)
        d(emb, emb_ref, f"layer{i} ple.ngram_embedding")
        ple_out = layer.ple(mx.array(pin), x, prev_ctx, None)
        d(ple_out, rec[p + ".ple"][1][0], f"layer{i} ple")
        hh = hh + mx.array(rec[p + ".ple"][1][0])
    mixed, hyper, inject = layer.attn_hyper_connection(hh)
    r = rec[p + ".attn_hyper_connection"][1]
    d(mixed, r[0], f"layer{i} attn_hc.mixed"); d(inject, r[2], f"layer{i} attn_hc.inject")
    xin = mx.array(r[0])
    if layer.layer_type == "linear_attention":
        a = layer.linear_attn(xin, None, None); d(a, rec[p + ".linear_attn"][1][0], f"layer{i} linear_attn")
        aref = rec[p + ".linear_attn"][1][0]
    else:
        a = layer.self_attn(xin, m.rope, mask, None, None); d(a, rec[p + ".self_attn"][1][0], f"layer{i} self_attn")
        aref = rec[p + ".self_attn"][1][0]
    a = mx.array(aref)
    h2 = mx.array(r[1]) + (a[..., None, :] * mx.array(r[2])[..., None]).reshape(*a.shape[:-1], -1)
    mixed2, hyper2, inject2 = layer.mlp_hyper_connection(h2)
    r2 = rec[p + ".mlp_hyper_connection"][1]
    d(mixed2, r2[0], f"layer{i} mlp_hc.mixed")
    mo = layer.mlp(mx.array(r2[0])); d(mo, rec[p + ".mlp"][1][0], f"layer{i} mlp")
    d(layer(hh, m.rope, mask, None, None, None, x, prev_ctx) if layer.ple is None else layer(mx.array(hin), m.rope, mask, None, None, None, x, prev_ctx), rec[p][1][0], f"layer{i} FULL (teacher-forced input)")
    h = mx.array(rec[p][1][0])
fin = m.hyper_connection_mixer(h); d(fin, rec["model.hyper_connection_mixer"][1][0], "final mixer")
d(model.lm_head(mx.array(rec["model.hyper_connection_mixer"][1][0])), ref_logits, "lm_head")
