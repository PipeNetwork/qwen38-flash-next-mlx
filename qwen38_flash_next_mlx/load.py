"""Load Qwen3.8-Flash-Next through this package's runtime, from a raw HF or a converted checkpoint.

`mlx_lm.load` resolves the model class from the installed registry, so it cannot pick up the fixed
`qwen4_exp.py` here; this does the same job with our class. Quantized checkpoints carry a
`quantization` map in config.json (uniform or per-module), which is replayed with `nn.quantize`
exactly as mlx-lm does, so mixed-precision builds reload as built.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .qwen4_exp import Model, ModelArgs


def load_model(path, lazy: bool = False, strict: bool = True):
    path = Path(path)
    with open(path / "config.json") as fh:
        config = json.load(fh)
    model = Model(ModelArgs.from_dict(config))

    weights = {}
    for wf in sorted(glob.glob(str(path / "*.safetensors"))):
        weights.update(mx.load(wf))
    weights = model.sanitize(weights)

    if (quantization := config.get("quantization")) is not None:
        def class_predicate(p, m):
            for key in (p, "language_model." + p):  # internal path, or the published name
                if key in quantization:
                    return quantization[key]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights  # what the checkpoint actually holds
        nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"],
                    class_predicate=class_predicate)

    model.load_weights(list(weights.items()), strict=strict)
    if not lazy:
        mx.eval(model.parameters())
    model.eval()
    return model, config


def load(path, lazy: bool = False):
    from mlx_lm.utils import load_tokenizer
    model, config = load_model(path, lazy=lazy)
    tokenizer = load_tokenizer(Path(path), eos_token_ids=config.get("text_config", config).get("eos_token_id"))
    return model, tokenizer
