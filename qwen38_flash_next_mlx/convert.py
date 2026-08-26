"""HF checkpoint -> MLX (mlx-lm `qwen4_exp`) tensor naming.

The upstream `qwen4_exp.py` sanitize only understands checkpoints that were *already* converted
(it strips a `language_model.` prefix and transposes convolutions, nothing else). The HF release
differs in three ways it does not handle:

* every text tensor sits under `model.language_model.`;
* routed experts are one fused `experts.gate_up_proj` [E, 2I, H] (gate rows first — see
  `Qwen4ExpTextExperts.forward`: `.chunk(2, dim=-1)`) plus `experts.down_proj` [E, H, I], where
  mlx-lm's `SwitchGLU` wants `switch_mlp.{gate,up,down}_proj.weight`;
* the 4B-parameter MTP head (`mtp.*`) is present and must be dropped, since strict loading
  rejects unknown tensors.

The vision tower (`model.visual.*`, 0.4B) is kept under `vision_tower.` so a future mlx-vlm
runtime finds it; the text runtime ignores that prefix.
"""

from __future__ import annotations

import mlx.core as mx

TEXT_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."


def sanitize_hf(weights: dict) -> dict:
    out = {}
    for k, v in weights.items():
        if k.startswith("mtp."):
            continue
        if k.startswith(VISION_PREFIX):
            out["vision_tower." + k[len(VISION_PREFIX):]] = v
            continue
        if k.startswith(TEXT_PREFIX):
            k = "model." + k[len(TEXT_PREFIX):]
        if k.endswith(".mlp.experts.gate_up_proj"):
            base = k[: -len("experts.gate_up_proj")]
            inter = v.shape[1] // 2
            out[base + "switch_mlp.gate_proj.weight"] = v[:, :inter, :]
            out[base + "switch_mlp.up_proj.weight"] = v[:, inter:, :]
            continue
        if k.endswith(".mlp.experts.down_proj"):
            out[k[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight"] = v
            continue
        if k.endswith("conv1d.weight") and v.ndim == 3 and v.shape[1] == 1:
            v = v.transpose(0, 2, 1)  # torch (C, 1, K) -> mlx (C, K, 1)
        out[k] = v
    return out
