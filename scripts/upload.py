"""Publish a built Qwen3.8-Flash-Next MLX quant to the Hub, with a card rendered from the measurements.

    .venv/bin/python scripts/upload.py --dir <build dir> --repo pipenetwork/<name> [--yes]

Nothing uploads without --yes. The card's quality table is generated from ppl_results.json by the
paired bootstrap in ppl_table.py, so the numbers on the Hub are the numbers that were measured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppl_table import markdown, rows

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "Qwen/Qwen3.8-Flash-Next"
CODE_REPO = "https://github.com/PipeNetwork/qwen38-flash-next-mlx"
PR = "https://github.com/ml-explore/mlx-lm/pull/1788"
ANCHOR = "Qwen3.8-Flash-Next-src"
ORDER = ["Qwen3.8-Flash-Next-src", "Qwen3.8-Flash-Next-MLX-8bit", "Qwen3.8-Flash-Next-MLX-6bit",
         "Qwen3.8-Flash-Next-MLX-mixed-4_8bit", "Qwen3.8-Flash-Next-MLX-4bit"]
LABELS = {"Qwen3.8-Flash-Next-src": "bfloat16 (upstream)"}
for n in ORDER[1:]:
    LABELS[n] = f"[{n.split('MLX-')[1]}](https://huggingface.co/pipenetwork/{n})"

CARD = """---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: {upstream}
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- qwen4_exp
- mixture-of-experts
- {bits_tag}
pipeline_tag: text-generation
library_name: mlx
---

# {repo_name}

MLX (Apple Silicon) build of [**Qwen3.8-Flash-Next**](https://huggingface.co/{upstream}) —
125B-A6B hybrid Gated-DeltaNet / sparse-attention MoE with a 51B-parameter hashed n-gram
embedding — quantized to **{recipe}**.

**These files are modified**: the weights are converted to MLX and quantized; the architecture is
unchanged. The 4B multi-token-prediction head is not included. The vision tower is carried in
bfloat16 but the runtime below is text-only.

## Runtime

`qwen4_exp` is in **no released mlx-lm** ({pr} is open and unmerged), so this repository ships its
own `qwen4_exp.py` and declares it via `model_file`:

```bash
pip install -U mlx-lm
mlx_lm.generate --model pipenetwork/{repo_name} --trust-remote-code \\
  --prompt "Write a Python function that merges overlapping intervals." --max-tokens 300
```
```python
from mlx_lm import load, generate
model, tokenizer = load("pipenetwork/{repo_name}", trust_remote_code=True)
```

The bundled runtime is the open PR **with three numerical fixes** found while validating it against
`transformers` 5.16 (details and tests in [{code_repo}]({code_repo})):

| what | reference | the PR as submitted | effect |
|---|---|---|---|
| RMSNorm variant | `x/rms · (1 + w)`, zero-initialised | `x/rms · w` | half the channels of every residual read sign-flipped |
| n-gram hash seed | `1234` (transformers default; not in config.json) | `0` | every bigram/trigram looks up an unrelated row |
| sparse-attention prefill | per-query blocks, own partial block visible, causal | global blocks; leaks future tokens, drops the query's own | wrong hidden states for prompts > 2048 tokens |

This checkpoint follows the mlx-lm convention for the norm fix: the `+1` is folded into the stored
norm weights at conversion, and the runtime multiplies by `w`. Tiny-config parity against
`transformers` is **1e-7** on the dense path, exact on cached decode and chunked prefill.

## Size and what is quantized

**{gb:.1f} GB** on disk (bfloat16 upstream: 360.0 GB).

| group | share of parameters | this build |
|---|---:|---|
| routed experts (`switch_mlp`) | 120.8B (96.6% of the 125B) | {expert_bits}-bit, group 64 |
| n-gram embedding tables (128 shards × [2,500,012 × 160]) | 51.2B (separate) | {ngram_bits}-bit, group 32 |
| attention, DeltaNet, hyper-connections, shared experts, embeddings, `lm_head` | ~4.2B | {other_bits}-bit, group 64 |
| MoE router, `shared_expert_gate`, residual write gates, DeltaNet `in_proj_a/b`, indexer projection | 36M | bfloat16 |
| vision tower | 0.4B | bfloat16 (unused by this runtime) |

The n-gram tables need group size 32 because their row width (160) is not a multiple of 64; left
in bfloat16 they alone would be 102 GB.

## Quality

Perplexity on wikitext-2 (test), {tokens:,} tokens in {windows} windows of {seq}, every build scored
on **identical** windows through this runtime. Perplexity varies far more between windows than
between quants, so the comparison that means anything is paired: per-window NLL differences
against bfloat16, bootstrapped over one shared index set (20,000 resamples).

{table}

Read the interval, not the point estimate: an interval that straddles zero is a build that is
statistically indistinguishable from bfloat16 on this corpus; "windows worse" counts how many of
the {windows} windows the build lost outright.

{recommendation}

{ablation}

Greedy generation (a collapse detector, not a ranking) is coherent on every published build.

## License

[Qwen Community License 1.0](LICENSE), as the upstream model. Port code: [{code_repo}]({code_repo}).
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--results", default=str(ROOT / "ppl_results.json"))
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--card-only", action="store_true", help="push only README.md (re-rendered)")
    args = parser.parse_args()

    d = Path(args.dir)
    cfg = json.load(open(d / "config.json"))
    q = cfg["quantization"]
    overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
    expert_bits = next((v["bits"] for k, v in overrides.items() if ".switch_mlp." in k), q["bits"])
    ngram_bits = next((v["bits"] for k, v in overrides.items() if "ngram_embedding" in k), q["bits"])
    other_bits = next((v["bits"] for k, v in overrides.items() if ".switch_mlp." not in k and "ngram_embedding" not in k), q["bits"])
    recipe = f"{expert_bits}-bit" if expert_bits == other_bits else f"{expert_bits}-bit experts / {other_bits}-bit everything else"
    gb = sum(p.stat().st_size for p in d.iterdir() if p.is_file()) / 1e9

    sizes = {}
    for n in ORDER:
        p = Path("/Users/david/llm/qwen38-flash-next-out") / n
        if p.exists():
            sizes[n] = sum(f.stat().st_size for f in p.iterdir() if f.is_file()) / 1e9
    sizes[ANCHOR] = 360.0
    res = json.load(open(args.results))
    a = res[ANCHOR]
    table = markdown(rows(args.results, ANCHOR, ORDER, sizes, LABELS), "bf16")
    ppl = {n: res[n]["perplexity"] for n in ORDER if n in res}
    pct = lambda n: 100 * (ppl[n] / ppl[ANCHOR] - 1)
    recommendation = ""
    if all(n in ppl for n in ORDER):
        m, u = "Qwen3.8-Flash-Next-MLX-mixed-4_8bit", "Qwen3.8-Flash-Next-MLX-4bit"
        recommendation = (
            f"**The mixed 4/8-bit build is the one to use at this size.** Routed experts are 96.6% of the "
            f"parameters, yet quantizing everything to 4-bit costs {pct(u):+.1f}% while keeping only the ~3% of "
            f"non-expert weights (attention, DeltaNet, hyper-connections, shared experts, embeddings) at 8-bit "
            f"brings that to {pct(m):+.1f}% for {sizes[m]-sizes[u]:.1f} GB more — the uniform 4-bit build is "
            f"dominated and is published for completeness. 6-bit and 8-bit are statistically indistinguishable "
            f"from bfloat16 on this corpus.")

    ablation = ""
    ABL = [("abl-4bit-hc8", "hyper-connection read gates (`input_mix_weight_down/up`, 0.6B)"),
           ("abl-4bit-attn8", "attention, DeltaNet, shared experts, PLE projections (~2.3B)"),
           ("abl-4bit-embed8", "`embed_tokens` and `lm_head` (1.3B)")]
    if all(n in res for n, _ in ABL) and all(n in ppl for n in ORDER):
        from ppl_table import paired
        u = "Qwen3.8-Flash-Next-MLX-4bit"
        lines = ["| everything 4-bit except… | perplexity | vs bfloat16 |", "|---|---:|---:|",
                 f"| — (uniform 4-bit) | {ppl[u]:.4f} | {pct(u):+.1f}% |"]
        for n, what in ABL:
            lines.append(f"| {what} at 8-bit | {res[n]['perplexity']:.4f} | {100*(res[n]['perplexity']/ppl[ANCHOR]-1):+.1f}% |")
        m = "Qwen3.8-Flash-Next-MLX-mixed-4_8bit"
        lines.append(f"| all three at 8-bit (= mixed-4_8bit) | {ppl[m]:.4f} | {pct(m):+.1f}% |")
        ablation = ("### Where the 4-bit damage comes from\n\nOne group at a time moved back to 8-bit from the uniform "
                    "4-bit build, same windows, same runtime (the ablation builds are not published):\n\n" + "\n".join(lines) +
                    "\n\nNo single group is responsible: the hyper-connection gates and the attention/DeltaNet "
                    "projections each carry about half of the loss and the effects are roughly additive, so every "
                    "non-expert weight is worth its 8 bits. Per parameter, these ~4B weights are roughly 20x more "
                    "quantization-sensitive than the 121B of routed experts.")
    card = CARD.format(upstream=UPSTREAM, code_repo=CODE_REPO, pr=PR, repo_name=args.repo.split("/")[-1],
                       bits_tag=f"{expert_bits}-bit", recipe=recipe, gb=gb, expert_bits=expert_bits,
                       ngram_bits=ngram_bits, other_bits=other_bits, tokens=a["tokens"], windows=a["windows"],
                       seq=a["seq_len"], table=table, recommendation=recommendation, ablation=ablation)
    (d / "README.md").write_text(card)
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    print(f"repo   {args.repo}\ndir    {d}\nfiles  {len(files)}, {gb:.1f} GB\n")
    print(table)
    if not args.yes:
        print("\ndry run — pass --yes to upload")
        return 0
    from huggingface_hub import HfApi
    api = HfApi()
    if args.card_only:
        api.upload_file(path_or_fileobj=str(d / "README.md"), path_in_repo="README.md", repo_id=args.repo, repo_type="model")
        print(f"\ncard refreshed https://huggingface.co/{args.repo}")
        return 0
    api.create_repo(args.repo, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(d), repo_id=args.repo, repo_type="model")
    print(f"\nuploaded https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
