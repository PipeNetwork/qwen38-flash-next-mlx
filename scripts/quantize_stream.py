"""Quantize Qwen3.8-Flash-Next shard by shard, never holding more than one tensor.

The bf16 release is 360 GB against 512 GB of memory, and the usual `load -> nn.quantize -> save`
would hold both the source and the result. This does the same arithmetic per tensor, streaming.

Which tensors get quantized is *derived* from the runtime: a full-size `Model` is built lazily
(MLX allocates nothing until evaluation) and every leaf that defines `to_quantized` — the exact
question `nn.quantize` asks — is recorded, then filtered by the recipe below. Output keys follow
the mlx-vlm layout (`language_model.model.*`, `language_model.lm_head`, `vision_tower.*`) that the
community builds use; our runtime's sanitize accepts it, and the vision tower is carried in bf16
for a future multimodal runtime even though this one is text-only.

Recipe:
  * routed experts (`switch_mlp`, 96.6% of the 125B)      --bits / --expert-bits, group 64
  * n-gram tables (128 shards x [2,500,012 x 160], 51 B)   --ngram-bits, group 32 (160 % 64 != 0)
  * everything else quantizable                             --bits / --other-bits, group 64
  * routing / gating / selection weights stay bf16: the MoE router, `shared_expert_gate`,
    `block_inject_weight` (residual write gates), `in_proj_a`/`in_proj_b` (DeltaNet decay and
    beta), `indexer.index_qk_proj` (block selection) — ~36 M parameters in total.

    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits 4
    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits 4 --other-bits 8   # mixed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen38_flash_next_mlx.qwen4_exp import Model, ModelArgs

AUX_FILES = (
    "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "chat_template.jinja", "preprocessor_config.json",
    "video_preprocessor_config.json", "LICENSE",
)
GATING_BF16 = ("shared_expert_gate", "block_inject_weight", "in_proj_a", "in_proj_b", "indexer.index_qk_proj")


def recipe(path: str, args) -> dict | None:
    """Quantization parameters for a module path, or None to leave it in bf16."""
    if path.endswith("mlp.gate") or path.endswith(GATING_BF16):
        return None
    if ".ngram_embedding.shard_" in path:
        return {"group_size": args.ngram_group_size, "bits": args.ngram_bits}
    if ".switch_mlp." in path:
        return {"group_size": args.group_size, "bits": args.expert_bits}
    return {"group_size": args.group_size, "bits": args.other_bits}


def quantizable_paths(margs: ModelArgs, args) -> dict[str, dict]:
    model = Model(margs)  # lazy: nothing is allocated
    out = {}
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not hasattr(module, "to_quantized"):
            continue
        params = recipe(path, args)
        if params is None:
            continue
        if module.weight.shape[-1] % params["group_size"]:
            raise SystemExit(f"{path}: in-dim {module.weight.shape[-1]} not divisible by group {params['group_size']}")
        out[path] = params
    return out


def materialise(x):
    with mx.stream(mx.cpu):  # memory-mapped read: keep it off the Metal command buffer
        mx.eval(x)
    return x


def quantize(w, group_size, bits):
    try:
        out = mx.quantize(materialise(w), group_size=group_size, bits=bits); mx.eval(out); return out
    except RuntimeError as err:
        if "Timeout" not in str(err):
            raise
        with mx.stream(mx.cpu):
            out = mx.quantize(w, group_size=group_size, bits=bits); mx.eval(out); return out


def out_key(internal: str) -> str:
    """Runtime-internal path -> published (mlx-vlm style) key."""
    if internal.startswith("model."):
        return "language_model." + internal
    if internal.startswith("lm_head"):
        return "language_model." + internal
    return internal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--expert-bits", type=int)
    ap.add_argument("--other-bits", type=int)
    ap.add_argument("--ngram-bits", type=int)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--ngram-group-size", type=int, default=32)
    ap.add_argument("--shard-gb", type=float, default=10.0)
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()
    args.expert_bits = args.expert_bits or args.bits
    args.other_bits = args.other_bits or args.bits
    args.ngram_bits = args.ngram_bits or args.bits

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    raw_cfg = json.load(open(src / "config.json"))
    margs = ModelArgs.from_dict(raw_cfg)
    model = Model(margs)
    qpaths = quantizable_paths(margs, args)
    print(f"quantizable modules: {len(qpaths)}  (experts {args.expert_bits}b, other {args.other_bits}b, "
          f"n-gram {args.ngram_bits}b/g{args.ngram_group_size})", flush=True)

    index = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(index.values()))
    if args.limit_shards:
        shards = shards[: args.limit_shards]

    target = args.shard_gb * 1e9
    out_index, pending, pending_bytes = {}, {}, 0
    out_n = total_out = 0
    counts = {"quantized": 0, "bf16": 0, "vision": 0, "dropped": 0}
    started = time.time()

    def flush():
        nonlocal pending, pending_bytes, out_n, total_out
        if not pending:
            return
        out_n += 1
        name = f"model-{out_n:05d}.safetensors"
        mx.save_safetensors(str(dst / name), pending, metadata={"format": "mlx"})
        for key in pending:
            out_index[key] = name
        size = (dst / name).stat().st_size
        total_out += size
        print(f"  -> {name}  {len(pending)} tensors  {size/1e9:.2f} GB  (total {total_out/1e9:.1f} GB, {time.time()-started:.0f}s)", flush=True)
        pending, pending_bytes = {}, 0

    for i, shard in enumerate(shards, 1):
        loaded = mx.load(str(src / shard))
        vision = {"vision_tower." + k[len("model.visual."):]: v for k, v in loaded.items() if k.startswith("model.visual.")}
        counts["vision"] += len(vision)
        counts["dropped"] += sum(k.startswith("mtp.") for k in loaded)
        text = model.sanitize({k: v for k, v in loaded.items() if not k.startswith("model.visual.")})
        print(f"[{i}/{len(shards)}] {shard}: {len(text)} text + {len(vision)} vision tensors", flush=True)
        emit_all = dict(vision)
        for key, value in text.items():
            module = key.rsplit(".", 1)[0]
            if key.endswith(".weight") and module in qpaths:
                p = qpaths[module]
                w, scales, biases = quantize(value, p["group_size"], p["bits"])
                ok = out_key(module)
                emit_all[ok + ".weight"], emit_all[ok + ".scales"], emit_all[ok + ".biases"] = w, scales, biases
                counts["quantized"] += 1
            else:
                emit_all[out_key(key)] = materialise(value)  # source dtype kept (bf16 / int64)
                counts["bf16"] += 1
        for k, v in emit_all.items():
            pending[k] = v
            pending_bytes += v.nbytes
            if pending_bytes >= target:
                flush()
        del loaded, text, emit_all
        mx.clear_cache()
    flush()

    quant = {"group_size": args.group_size, "bits": args.bits}
    # Keyed by the runtime's *internal* module paths (`model.layers...`): that is what mlx-lm's
    # loader (and ours) looks up when replaying per-module overrides, regardless of how the
    # tensors themselves are named on disk.
    for path, p in qpaths.items():
        if p != {"group_size": args.group_size, "bits": args.bits}:
            quant[path] = p
    cfg = dict(raw_cfg)
    cfg["quantization"] = quant
    cfg["quantization_config"] = quant
    cfg["model_file"] = "qwen4_exp.py"
    json.dump(cfg, open(dst / "config.json", "w"), indent=2)
    json.dump({"metadata": {"total_size": total_out}, "weight_map": out_index}, open(dst / "model.safetensors.index.json", "w"), indent=2)
    for name in AUX_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    shutil.copy2(Path(__file__).resolve().parents[1] / "qwen38_flash_next_mlx" / "qwen4_exp.py", dst / "qwen4_exp.py")
    print(f"\n{out_n} shards, {total_out/1e9:.1f} GB, {(time.time()-started)/60:.1f} min; {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
