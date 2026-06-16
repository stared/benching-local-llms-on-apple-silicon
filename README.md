# llaming — local coding LLMs on Apple Silicon

Benchmarks and notes for running local coding models on an **Apple M5 Max / 128 GB**:
which model, which engine (**llama.cpp vs MLX**), and how fast — measured with an
apples-to-apples throughput harness.

> This README is the summary. The full working log — setup, methodology, every caveat —
> is in **[NOTES.md](NOTES.md)**.

## Takeaways

- **Daily driver: Qwen3.6-35B-A3B (Q8) on llama.cpp with MTP — ~100 tok/s.** Fast, and
  barely slows down at long context.
- **MTP (speculative decoding) helps unevenly:** +75% on the **dense 27B** (17→32 tok/s),
  only +12% on the **MoE 35B-A3B**. Lossless either way.
- **llama.cpp beats MLX** for these models on the M5 Max (+10–24%) — the opposite of the
  usual "MLX is faster on Mac" claim. Measured, not assumed.
- **Capability:** these local models sit around the **cloud frontier of mid-2025 to
  early-2026** — but still a generation behind today's best on hard benchmarks.

## Results

8-bit · M5 Max 128 GB · decode = total tok/s (reasoning included) · median of 3 runs ·
2026-06-14. Full numbers in [`bench_results.md`](bench_results.md).

| Model | Engine | MTP | Decode @128 | Decode @8k | Memory |
|---|---|:-:|--:|--:|--:|
| **Qwen3.6-35B-A3B** (MoE) | llama.cpp | ✓ | **105** | **97** | 45 GB |
| | llama.cpp | | 93 | 87 | 44 GB |
| | MLX | | 85 | 79 | 37 GB |
| **Qwen3.6-27B** (dense) | llama.cpp | ✓ | 32 | 30 | 42 GB |
| | llama.cpp | | 18 | 17 | 41 GB |
| | MLX | | 17 | 17 | 28 GB |
| **DeepSeek-V4-Flash** | ds4 (q2–q4, 91 GB) | | 33 | 28 | 103 GB |

## How good is "local" really?

One table per model: vendor **self-claim** vs **independent** eval (where one exists),
against the **nearest dated frontier model**. Scores link to the page they're read from.
**Comparisons are scaffold/tool-dependent — read as ±several points, not exact.**

#### Qwen3.6-35B-A3B (MoE) — SoTA level of early-mid 2025

| Benchmark | Self-claim | Indep. | Nearest frontier | Score | Released |
|---|--:|--:|---|--:|---|
| SWE-bench Verified | [73.4][qwen] | — | Claude 4 Sonnet | [72.7][swev] | May '25 |
| SWE-bench Pro\* | [49.5][qwen] | — | Claude Sonnet 4 (norm.) | [42.7][swepro] | May '25 |
| GPQA Diamond | [86.0][qwen] | — | Grok 4 | [87][gpqa] | Jul '25 |
| HLE (no-tools) | [21.4][qwen] | — | o3 (high) | [20.6][hletext] | Apr '25 |

#### Qwen3.6-27B (dense) — SoTA level of mid 2025

| Benchmark | Self-claim | Indep. | Nearest frontier | Score | Released |
|---|--:|--:|---|--:|---|
| SWE-bench Verified | [77.2][qwen] | — | Claude 4 Sonnet → Opus 4.5 | [72.7][swev]–[80.9][swev] | May–Nov '25 |
| SWE-bench Pro\* | [53.5][qwen] | — | Claude Sonnet 4.5 (norm.) | [43.6][swepro] | Sep '25 |
| GPQA Diamond | [87.8][qwen] | — | Grok 4 | [87][gpqa] | Jul '25 |
| HLE (no-tools) | [24.0][qwen] | — | o3 → GPT-5 | [20.6][hletext]–[26.3][hletext] | Apr–Aug '25 |

#### DeepSeek-V4-Flash — SoTA level of late 2025 to early 2026

| Benchmark | Self-claim | Indep. | Nearest frontier | Score | Released |
|---|--:|--:|---|--:|---|
| SWE-bench Verified | [78.6][dsv4] | — | Claude 4 Sonnet → Opus 4.5 | [72.7][swev]–[80.9][swev] | May–Nov '25 |
| SWE-bench Pro‡ | [52.3][dsv4] | — | Claude Opus 4.6 | [51.9][swepro] | Feb '26 |
| GPQA Diamond | [87.4][dsv4] | [86.7][aa] | Grok 4 | [87][gpqa] | Jul '25 |
| HLE (no-tools) | [29.4][dsv4] | [27.8][aa] | GPT-5.2 | [28.5][hletext] | Dec '25 |
| LiveCodeBench v6 | [88.4][dsv4] | — | GPT-5.2 Codex | [88][lcb] | Jan '26 |

\* SWE-bench Pro: Qwen reports on its *refined* set (~+11 vs Scale's public set); "nearest"
uses the normalized value. All Qwen figures are vendor self-claim — OpenRouter/AA pages are
JS-rendered, so I couldn't pull independent Qwen numbers. DS4-Flash is full-precision "High"
(its "Max" mode adds a few points); your 91 GB 2–4-bit quant scores lower.
‡ DS4-Flash's SWE-Pro is on DeepSeek's own scaffold, so the Opus 4.6 match is cross-scaffold.

**Reality check:** the live frontier still leads the unsaturated tests by a wide margin —
HLE [53.3][hle] (Fable 5, Jun '26) and SWE-bench Pro [59.1][swepro] (GPT-5.4) sit well above
anything here. Strong for 27–35B on a laptop, not the frontier.

[qwen]: https://huggingface.co/Qwen/Qwen3.6-27B
[aa]: https://openrouter.ai/deepseek/deepseek-v4-flash#benchmarks
[swev]: https://llm-stats.com/benchmarks/swe-bench-verified
[gpqa]: https://epoch.ai/benchmarks/gpqa-diamond
[hle]: https://artificialanalysis.ai/evaluations/humanitys-last-exam
[swepro]: https://labs.scale.com/leaderboard/swe_bench_pro_public
[dsv4]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
[lcb]: https://www.vals.ai/benchmarks/lcb
[hletext]: https://labs.scale.com/leaderboard/humanitys_last_exam_text_only

## Quick start

**1. Download the model** — get the **MTP build**. It's a superset: it runs as a normal
model *and* unlocks speculative decoding, so it's the only file you need.

```bash
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-Q8_0.gguf --local-dir ~/models
```

**2. Serve it** (exposes an OpenAI-compatible API on `:8080`):

```bash
llama-server -m ~/models/Qwen3.6-35B-A3B-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \   # MTP on; drop these for baseline
  -ngl 999 -fa on -c 65536 --parallel 1 --jinja \
  --host 127.0.0.1 --port 8080
```

**3. Point your coding agent** (opencode, Pi, Cline, aider, …) at
`http://127.0.0.1:8080/v1` — API key: any string, model: the file name.

**4. Benchmark it yourself:**

```bash
./bench_prep.sh        # quiet the machine first
python3 bench.py       # full sweep, ~20 min
```

## Models

**Download the MTP GGUF builds — nothing else needed.** Each is a superset: run it
plainly for the baseline, or add `--spec-type draft-mtp` for speculative decoding. No
separate non-MTP download.

- [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) — fast MoE, the daily driver
- [`unsloth/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) — dense, stronger coder, slower

(MLX 8-bit — [`mlx-community/Qwen3.6-35B-A3B-8bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit),
[`27B`](https://huggingface.co/mlx-community/Qwen3.6-27B-8bit) — is the alternative, but
benchmarks slower here.)

## Files

- [`NOTES.md`](NOTES.md) — full working log: setup, methodology, caveats, benchmark tables
- [`bench.py`](bench.py) — throughput harness; drives each engine's OpenAI `/v1` server
- [`bench_prep.sh`](bench_prep.sh) — quiets the machine before benchmarking
- [`bench_native.sh`](bench_native.sh) — engine-only cross-check (`llama-bench`, `mlx_lm.generate`)
- [`bench_results.md`](bench_results.md) — latest measured run

## Sources

- [How to set up a local coding agent on macOS](https://ikyle.me/blog/2026/how-to-setup-a-local-coding-agent-on-macos) ([HN discussion](https://news.ycombinator.com/item?id=48507020))
- Benchmark figures: Qwen3.6 model card, Artificial Analysis, Scale SWE-bench Pro — full citations in [NOTES.md](NOTES.md)
</content>
