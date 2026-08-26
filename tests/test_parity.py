"""Numerical parity of this package's `qwen4_exp` runtime against transformers, at tiny scale.

The 360 GB checkpoint loads here, but a random tiny model is what lets every fragile path be
exercised in seconds and be *broken on purpose*. The config is chosen so the details are live:
norm weights perturbed off their initialisation (a norm-variant substitution becomes visible), an
EOS mid-sequence (the n-gram segment reset fires), a sequence longer than the indexer budget (block
selection fires), more experts than top-k, and the full-attention layer last so a selection
tie-break cannot propagate into later layers.

    .venv/bin/python tests/test_parity.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mlx.core as mx
from mlx.utils import tree_flatten

from qwen38_flash_next_mlx import qwen4_exp as q

TINY = dict(
    vocab_size=128, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=32,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
    linear_value_head_dim=8, linear_conv_kernel_dim=4,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32, hc_count=4, hc_lowrank=8,
    ple_layer_ids=[2], ple_embed_dim=64, ple_conv_kernel_size=4, ngram_size=3,
    heads_per_ngram=8, ngram_vocab_size_base=1000, make_ngram_vocab_size_divisible_by=8,
    split_ngram_parts=4, seed=1234,
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=64,
    indexer_compress_ratio=2, output_gate_type="sigmoid", eos_token_id=1, bos_token_id=1,
    rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [2, 1, 1], "mrope_interleaved": True, "rope_type": "default"},
    rms_norm_eps=1e-6, tie_word_embeddings=False,
)
TINY_SPARSE = {**TINY, "indexer_budget": 8}   # 4 blocks of 2 tokens; kv_len 12 > budget

B, T = 2, 12


def inputs():
    torch.manual_seed(1)
    ids = torch.randint(2, TINY["vocab_size"], (B, T))
    ids[0, 5] = TINY["eos_token_id"]  # n-gram context must reset at this boundary
    ids[1, 0] = TINY["eos_token_id"]
    return ids


def build_hf(cfg, seed=0):
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
    torch.manual_seed(seed)
    model = Qwen4ExpForCausalLM(Qwen4ExpTextConfig(**cfg)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.add_(0.3 * torch.randn_like(p))  # off-init: x/rms*w, *(1+w) and x/rms now differ
            elif "block_inject_weight" in name or "input_mix_weight" in name:
                p.mul_(3.0)  # make the residual gates visibly data-dependent
    return model


def hf_release_layout(model, n_shards):
    """The tiny model's tensors, named and laid out exactly as the HF release ships them:
    `model.language_model.` prefix, sharded n-gram table, fused experts, torch conv layout —
    so the runtime's own sanitize is what gets tested."""
    weights = {}
    for k, v in model.state_dict().items():
        v = v.detach().numpy() if not v.is_floating_point() else v.detach().float().numpy()
        if k.startswith("model."):
            k = "model.language_model." + k[len("model."):]
        if k.endswith("ple_embedding.ngram_embedding.weight"):
            rows = v.shape[0] // n_shards
            for i in range(n_shards):
                weights[k.replace("ngram_embedding.weight", f"ngram_embedding.shard_{i}.weight")] = \
                    mx.array(v[i * rows:(i + 1) * rows])
        else:
            weights[k] = mx.array(v)
    return weights


def build_mlx(hf_model, cfg, weights=None):
    args = q.ModelArgs.from_dict({"text_config": dict(cfg), "vision_config": {}})
    model = q.Model(args)
    weights = weights if weights is not None else model.sanitize(hf_release_layout(hf_model, cfg["split_ngram_parts"]))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def logits_hf(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits.float().numpy()


def logits_mlx(model, ids):
    return np.array(model(mx.array(ids.numpy())).astype(mx.float32))


def report(label, out, ref, rows=None):
    if rows is not None:
        out, ref = out[rows], ref[rows]
    scale = float(np.abs(ref).max())
    delta = float(np.abs(out - ref).max())
    ok = delta < 1e-4 * max(scale, 1.0)
    print(f"  {label:58s} max|delta| {delta:.3e}  (scale {scale:.3e})  {'OK' if ok else 'FAIL'}")
    return ok


def sparse_masks(hf, model, ids):
    """(reference mask, our mask, per-query score vectors) for the full-attention layer."""
    rec = {}
    layer = hf.model.layers[3]
    layer.self_attn.indexer.register_forward_hook(lambda m, i, o: rec.update(idx=(i, o)))
    layer.self_attn.register_forward_hook(lambda m, i, o: rec.update(x=i[0]))
    logits_hf(hf, ids)
    (hidden, (cos, sin), amask), ref_mask = rec["idx"][0][:3], rec["idx"][1]
    ref = ref_mask.numpy() & np.tril(np.ones((T, T), bool))[None, None]
    attn = model.model.layers[3].self_attn
    ours = np.array(attn.indexer(mx.array(hidden.float().numpy()), model.model.rope, None, 0))
    # scores, recomputed the reference way, to classify disagreements as ties or not
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb
    ix = layer.self_attn.indexer
    r = ix.compress_ratio
    scores = {}
    with torch.no_grad():
        qk = ix.index_qk_proj(hidden)
        qq, tk = torch.split(qk, [ix.index_n_heads * ix.index_head_dim, ix.index_head_dim], -1)
        qq = apply_rotary_pos_emb(ix.q_layernorm(qq.reshape(B, T, -1, ix.index_head_dim)), cos=cos, sin=sin, unsqueeze_dim=2)
        raw = tk.reshape(B, T, ix.index_head_dim)
        for b in range(B):
            for qi in range(T):
                nb = (qi + 1) // r
                if nb == 0:
                    continue
                blocks = torch.arange(nb * r).view(nb, r)
                kg = ix.k_layernorm(raw[b][blocks.flatten()].view(nb, r, -1).float().mean(1))
                bk = apply_rotary_pos_emb(kg.unsqueeze(1), cos=cos[b][blocks[:, 0]], sin=sin[b][blocks[:, 0]]).squeeze(1)
                sc = torch.relu(torch.matmul(qq[b, qi].float(), bk.float().T).T).sum(-1) / math.sqrt(ix.index_head_dim)
                scores[(b, qi)] = sc.numpy()
    return ref, ours, scores


def main():
    ids = inputs()
    all_ok = True

    print("[1] dense path (kv_len <= indexer budget): exact parity")
    hf = build_hf(TINY)
    model = build_mlx(hf, TINY)
    ref = logits_hf(hf, ids)
    all_ok &= report("logits", logits_mlx(model, ids), ref)

    print("[2] sanitize is idempotent and load-bearing")
    raw = hf_release_layout(hf, TINY["split_ngram_parts"])
    once = model.sanitize(raw)
    twice = model.sanitize(dict(once))
    same = all(k in twice and once[k].shape == twice[k].shape and bool(mx.array_equal(once[k], twice[k])) for k in once) and len(once) == len(twice)
    print(f"  {'second sanitize pass changes nothing':58s} {'OK' if same else 'FAIL'}")
    all_ok &= same
    unshifted = {k: (v - 1.0 if k.endswith(model.CENTERED_NORMS) else v) for k, v in once.items()}
    ctrl = build_mlx(hf, TINY, weights=unshifted)
    d = float(np.abs(logits_mlx(ctrl, ids) - ref).max())
    print(f"  {'control: skip the (1+w) shift -> logits move by':58s} {d:.3e}  {'OK' if d > 1e-2 else 'FAIL'}")
    all_ok &= d > 1e-2

    print("[3] n-gram hash is load-bearing (checkpoint multipliers vs seed-0 recomputation)")
    ple = model.model.layers[1].ple.ple_embedding
    keep = ple.layer_multipliers
    seed0 = q.NGramEmbedding(q.ModelArgs.from_dict({"text_config": {**TINY, "seed": 0}, "vision_config": {}}).text, TINY["ple_embed_dim"], 0)
    ple.layer_multipliers = seed0.layer_multipliers
    d = float(np.abs(logits_mlx(model, ids) - ref).max())
    ple.layer_multipliers = keep
    print(f"  {'control: seed-0 multipliers -> logits move by':58s} {d:.3e}  {'OK' if d > 1e-3 else 'FAIL'}")
    all_ok &= d > 1e-3

    print("[4] sparse path (kv_len 12 > budget 8): selection parity up to zero-score ties")
    hf_s = build_hf(TINY_SPARSE)
    model_s = build_mlx(hf_s, TINY_SPARSE)
    ref_mask, our_mask, scores = sparse_masks(hf_s, model_s, ids)
    tie_rows, bad = [], []
    for b in range(B):
        for qi in range(T):
            if (ref_mask[b, 0, qi] != our_mask[b, 0, qi]).any():
                sc = scores.get((b, qi))
                blocks = np.nonzero(ref_mask[b, 0, qi] != our_mask[b, 0, qi])[0] // TINY_SPARSE["indexer_compress_ratio"]
                if sc is not None and all(sc[blk] == 0.0 for blk in set(blocks.tolist()) if blk < len(sc)):
                    tie_rows.append((b, qi))
                else:
                    bad.append((b, qi))
    print(f"  {'queries differing only by zero-score tie-break':58s} {len(tie_rows)}")
    print(f"  {'queries differing otherwise':58s} {len(bad)}  {'OK' if not bad else 'FAIL'}")
    all_ok &= not bad
    ref_s = logits_hf(hf_s, ids)
    keep_rows = np.ones((B, T), bool)
    for b, qi in tie_rows:
        keep_rows[b, qi] = False
    all_ok &= report("logits (tie-affected positions excluded)", logits_mlx(model_s, ids), ref_s, rows=keep_rows)
    dense = build_mlx(hf_s, {**TINY_SPARSE, "indexer_budget": 64})
    d = float(np.abs(logits_mlx(dense, ids) - ref_s).max())
    print(f"  {'control: dense attention instead of sparse -> logits move by':58s} {d:.3e}  {'OK' if d > 1e-3 else 'FAIL'}")
    all_ok &= d > 1e-3

    print("[5] token-by-token decode == single forward (sparse config, cached indexer/PLE/SSM)")
    cache = model_s.make_cache()
    steps = []
    for t in range(T):
        steps.append(np.array(model_s(mx.array(ids[:, t:t + 1].numpy()), cache=cache).astype(mx.float32)))
    inc = np.concatenate(steps, axis=1)
    single = logits_mlx(model_s, ids)
    all_ok &= report("incremental vs single-shot (tie rows excluded)", inc, single, rows=keep_rows)
    cache = model_s.make_cache()
    a = np.array(model_s(mx.array(ids[:, :5].numpy()), cache=cache).astype(mx.float32))
    b_ = np.array(model_s(mx.array(ids[:, 5:].numpy()), cache=cache).astype(mx.float32))
    all_ok &= report("chunked prefill 5+7 vs single-shot (tie rows excluded)", np.concatenate([a, b_], 1), single, rows=keep_rows)

    print("\nALL OK" if all_ok else "\nSOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
