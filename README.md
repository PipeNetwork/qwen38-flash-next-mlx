# qwen38-flash-next-mlx

MLX (Apple Silicon) runtime and quantization tooling for
[**Qwen/Qwen3.8-Flash-Next**](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — 125B-A6B
hybrid Gated-DeltaNet / Qwen-Sparse-Attention MoE with gated residual streams and a 51B-parameter
hashed n-gram embedding (`model_type: qwen4_exp`).

Published builds: **[pipenetwork/Qwen3.8-Flash-Next MLX](https://huggingface.co/collections/pipenetwork/qwen38-flash-next-mlx-6a8f26fa7137017a3b323c5b)**
(8-bit, 6-bit, mixed 4/8-bit, 4-bit; see [Measurements](#measurements)).

## Why this exists

`qwen4_exp` is carried by no released mlx-lm or mlx-vlm. An mlx-lm pull request
([ml-explore/mlx-lm#1788](https://github.com/ml-explore/mlx-lm/pull/1788)) adds it; validating that
PR against `transformers` 5.16 at tiny scale — rather than judging it by whether it generates
fluent text — found three bugs that keep the text fluent while making it wrong:

| | reference (`transformers`) | PR #1788 | consequence |
|---|---|---|---|
| RMSNorm | `x/rms · (1 + w)`, `w` zero-initialised (`hc_norm`, `q_norm`, `k_norm`, indexer and PLE norms) | `x/rms · w` | with the real weights (mean ≈ −0.09, half negative) half the channels of every residual-stream read are sign-flipped |
| n-gram hash seed | `1234` — the transformers default; `config.json` carries no `seed` | `0` | the three hash multipliers differ, so every bigram/trigram lookup hits an unrelated row of the 51B table. The checkpoint's own `layer_multipliers` buffer `[23703573157769, 20109073645365, 8052911324071]` is exactly seed 1234 |
| sparse-attention prefill | per query: blocks over its own visible prefix, its partial trailing block always visible, ANDed with causal | global blocks; the global tail is visible to *all* queries (future leak), the query's own partial block is dropped, and the causal mask is replaced rather than combined | wrong hidden states for every position past 2048 during prefill; decode was exact |

Plus two things that make the PR unable to load the release at all: its `sanitize` expects an
already-converted layout (it never splits `experts.gate_up_proj`, never drops `mtp.*`, never sees
the `model.language_model.` prefix), and its `quant_predicate` has the wrong arity for mlx-lm.

`qwen38_flash_next_mlx/qwen4_exp.py` is the PR with those fixed, plus an exact `l2norm` for the
DeltaNet q/k (the PR's `rms_norm(·, eps)` is `l2norm` with `eps·d`, visible when activations are
small — the same issue mlx-lm fixed for KDA in #1624), and a sanitize that reads the HF release
directly. It is the file bundled in every published checkpoint (`model_file` in `config.json`).

## Validation

```bash
./scripts/run_tests.sh
```

A random tiny model with every fragile path live — norms perturbed off their initialisation, an
EOS mid-sequence so the n-gram segment reset fires, a sequence longer than the indexer budget so
block selection fires, more experts than top-k, the full-attention layer last — checked against
`transformers`, with each fix asserted to be load-bearing by breaking it:

```
[1] dense path (kv_len <= indexer budget): exact parity
  logits                                        max|delta| 1.043e-07  (scale 2.868e-01)  OK
[2] sanitize is idempotent and load-bearing
  control: skip the (1+w) shift -> logits move by            2.879e-01  OK
[3] n-gram hash: seed-0 multipliers -> logits move by          3.787e-02  OK
[4] sparse path (kv_len 12 > budget 8): selection parity up to zero-score ties
  queries differing only by zero-score tie-break             1
  queries differing otherwise                                0  OK
  logits (tie-affected positions excluded)      max|delta| 8.941e-08  OK
  control: dense attention instead of sparse -> logits move by 2.988e-02  OK
[5] token-by-token decode == single forward
  incremental vs single-shot                    max|delta| 4.470e-08  OK
  chunked prefill 5+7 vs single-shot            max|delta| 0.000e+00  OK
```

The one selection disagreement is a tie: the indexer scores are `relu`-summed, three candidate
blocks score exactly 0.0, and `torch.topk` and `mx.argpartition` break the tie differently. The
reference is not device-deterministic there either, so the test excludes those rows rather than
pretend they are decidable.

On the real checkpoint the bf16 model loads through this runtime with zero missing and zero
unexpected tensors, and generates coherently (*The capital of France is* → `Paris`; merge-intervals
code; Rayleigh scattering in two sentences) at ~24 tok/s, 354 GB resident.

## Building the quants

```bash
python scripts/quantize_stream.py --src Qwen3.8-Flash-Next-src --dst out/Qwen3.8-Flash-Next-MLX-4bit --bits 4
python scripts/quantize_stream.py --src ... --dst out/...-mixed-4_8bit --bits 4 --other-bits 8
```

Shard by shard, never holding more than one tensor: the 360 GB bf16 release would not fit next to
its own quantized copy. Which tensors are quantized is *derived* from the runtime (every leaf with
`to_quantized`, then a recipe), not guessed from names. The recipe:

* routed experts (120.8B, 96.6% of the model): `--bits`, group 64;
* the 128 n-gram shards (`[2,500,012 × 160]` each, 51.2B): `--ngram-bits`, **group 32** —
  160 is not a multiple of 64, and left unquantized the tables alone are 102 GB;
* everything else quantizable (~4.2B): `--other-bits`, group 64;
* kept in bfloat16 (~36M parameters): the MoE router, `shared_expert_gate`, the residual write
  gates, DeltaNet `in_proj_a`/`in_proj_b`, the indexer projection — every weight that decides
  *what* is computed rather than computing it;
* dropped: the 4B MTP head. Carried in bfloat16: the 0.4B vision tower (this runtime is text-only).

Output keys follow the mlx-vlm layout the community builds use (`language_model.model.*`,
`vision_tower.*`); the runtime's sanitize accepts both that and the raw HF layout, and the `+1`
norm fold is gated on raw-HF markers so it cannot be applied twice.

## Measurements

`scripts/ppl_corpus.py` tokenizes wikitext-2 (test) once; `scripts/ppl_large.py` teacher-forces
every build on the same 145 windows of 2048 tokens; `scripts/ppl_compare.py` does the paired
bootstrap. Results are in the model cards and below.

<!-- measurements -->
| build | size | perplexity | ΔNLL/token vs bf16 [95% CI] | windows worse |
|---|---:|---:|---|---:|
| bfloat16 (upstream) | 360.0 GB | 4.4708 | — | — |
| [8bit](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-8bit) | 192.2 GB | 4.4749 | +0.0009 [−0.0003, +0.0021] | 73/145 |
| [6bit](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-6bit) | 148.0 GB | 4.4767 | +0.0013 [−0.0003, +0.0029] | 81/145 |
| [mixed-4_8bit](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit) | 106.2 GB | 4.5286 | +0.0128 [+0.0109, +0.0148] | 128/145 |
| [4bit](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-4bit) | 103.8 GB | 5.3914 | +0.1872 [+0.1778, +0.1968] | 145/145 |
<!-- /measurements -->

**6-bit and 8-bit are statistically indistinguishable from bfloat16; uniform 4-bit costs +20.6% while the mixed 4/8-bit build costs +1.3% — the recommended build below 150 GB.**

### Where the 4-bit damage comes from

One group at a time moved back to 8-bit from the uniform 4-bit build, same windows, same runtime (the ablation builds are not published):

| everything 4-bit except… | perplexity | vs bfloat16 |
|---|---:|---:|
| — (uniform 4-bit) | 5.3914 | +20.6% |
| hyper-connection read gates (`input_mix_weight_down/up`, 0.6B) at 8-bit | 4.9744 | +11.3% |
| attention, DeltaNet, shared experts, PLE projections (~2.3B) at 8-bit | 4.8969 | +9.5% |
| `embed_tokens` and `lm_head` (1.3B) at 8-bit | 5.2843 | +18.2% |
| all three at 8-bit (= mixed-4_8bit) | 4.5286 | +1.3% |

No single group is responsible: the hyper-connection gates and the attention/DeltaNet projections each carry about half of the loss and the effects are roughly additive, so every non-expert weight is worth its 8 bits. Per parameter, these ~4B weights are roughly 20x more quantization-sensitive than the 121B of routed experts.

## Layout

| path | what |
|---|---|
| `qwen38_flash_next_mlx/qwen4_exp.py` | the runtime (PR #1788 + fixes), also bundled in each checkpoint |
| `qwen38_flash_next_mlx/load.py` | loader that replays per-module quantization from `config.json` |
| `scripts/quantize_stream.py` | streaming quantizer |
| `scripts/ppl_corpus.py`, `ppl_large.py`, `ppl_compare.py`, `ppl_table.py` | evaluation |
| `scripts/smoke_generate.py` | greedy generation, collapse detector |
| `scripts/upload.py`, `make_collection.py` | publishing, card rendered from the measurements |
| `tests/test_parity.py` | the validation above; `tests/debug_modules.py` locates a divergence module by module |
| `docs/upstream-notes.md` | findings written up for the PR |
