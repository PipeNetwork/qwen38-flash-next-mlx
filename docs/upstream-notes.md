# Notes for ml-explore/mlx-lm#1788 (qwen4_exp)

Found by tiny-config parity against `transformers` 5.16 (`tests/test_parity.py`), not by reading.

1. **RMSNorm variant.** `Qwen4ExpTextRMSNorm` is `x/rms * (1 + w)` with zero-init `w`; the PR's
   `RMSNorm` multiplies by `w`. Affects `hc_norm` (all hyper-connections + final mixer),
   `q_norm`/`k_norm`, `indexer.{q,k}_layernorm`, `ple.norm_{key,query,conv}`. `linear_attn.norm`
   (gated, ones-init) is correct as is. Real weights: `layers.1.attn_hyper_connection.hc_norm`
   mean −0.089, 55% negative. Fix used here: fold `+1` at sanitize when converting from the raw HF
   layout (mlx-lm's qwen3_next/qwen3_5 convention), gated on raw-HF markers so it is idempotent.
2. **`seed` default.** `TextArgs.seed = 0`; transformers default is 1234 and config.json has no
   `seed`. The checkpoint's `layer_multipliers` = `[23703573157769, 20109073645365, 8052911324071]`
   = seed 1234. Fix: default 1234 and hash with the loaded int64 buffers.
3. **Sparse prefill mask.** Reference forms blocks per query over its visible prefix; the trailing
   partial block (incl. the query itself) is always visible; result is ANDed with causal. The PR
   uses global blocks, marks the global tail visible for every query (future leak), never adds the
   per-query partial block, and replaces a `"causal"` string mask instead of combining. Decode
   (S=1) was already exact. Fix in `QSAIndexer.__call__` + `Attention.__call__`.
4. **DeltaNet q/k normalisation.** `rms_norm(x, None, 1e-6)` is `l2norm` with `eps*d`; the
   reference is `x * rsqrt(sum(x^2) + 1e-6)`. Same as mlx-lm #1624 for KDA.
5. **Loading the HF release.** `sanitize` must map `model.language_model.` → `model.`, split
   `experts.gate_up_proj` [E, 2I, H] into `switch_mlp.gate_proj` / `up_proj` (gate rows first),
   rename `experts.down_proj`, drop `mtp.*`. As submitted it only loads pre-converted checkpoints.
6. **`quant_predicate` arity.** mlx-lm calls `predicate(path, module)`; the PR defines three args.
7. **`model_file` loader vs `from __future__ import annotations`** (mlx-lm `utils.load_model`): the
   custom module is executed via `spec.loader.exec_module` without being registered in
   `sys.modules`, so any `@dataclass` in a file that uses postponed annotations fails in
   `dataclasses._is_type` (`sys.modules.get(cls.__module__)` is `None`). The PR file has that
   import; a checkpoint bundling it as `model_file` cannot load until the import is removed (or
   the loader does `sys.modules[spec.name] = module` before `exec_module`).
