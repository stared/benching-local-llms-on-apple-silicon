# llaming — local coding LLMs on Apple Silicon

Benchmarks and field notes for running **Qwen3.6** (and friends) as a local coding-agent
backend on an **Apple M5 Max / 128 GB**, comparing **llama.cpp vs MLX**, with an
apples-to-apples throughput harness.

> For the detailed working log (setup commands, methodology, every caveat), see
> **[NOTES.md](NOTES.md)**. This README is the summary.

## TL;DR

- **Fastest good setup: `llama.cpp` + Qwen3.6-35B-A3B Q8 + MTP ≈ 97–105 tok/s.** Fast
  prefill, barely sags to 8k context. This is the daily driver.
- **MTP (speculative decoding) is architecture-dependent:** huge on the **dense 27B
  (+75%, 17→31 t/s)**, modest on the **MoE 35B-A3B (+12%)**. Lossless either way.
- **llama.cpp beats MLX** for these Qwen models on M5 Max (+10–24% decode) — counter to
  the usual "MLX is faster on Mac" assumption. Measured, not assumed.
- **Capability:** these local models land around the **cloud frontier of mid-2025 →
  early-2026** depending on model/benchmark — but stay **~a generation behind the live
  2026 frontier** on hard, unsaturated tests (HLE, Terminal-Bench, SWE-Pro).

## Latest throughput results

M5 Max 128 GB · 8-bit · decode = total tok/s (reasoning included) · median of 3 runs
(2026-06-14):

| setup | decode @128 | decode @8k | RSS |
|---|--:|--:|--:|
| **llama.cpp 35B-A3B Q8 +MTP** | **104.6** | **96.8** | 45 GB |
| llama.cpp 35B-A3B Q8 (baseline) | 92.9 | 86.9 | 44 GB |
| MLX 35B-A3B 8bit | 84.5 | 79.4 | 37 GB |
| llama.cpp 27B Q8 +MTP | 31.7 | 29.9 | 42 GB |
| MLX 27B 8bit | 17.3 | 17.1 | 28 GB |
| ds4 DeepSeek-V4-Flash q2–q4 (91 GB) | 33.0 | 27.8 | 103 GB |

## How good is "local" really?

Mapped against *dated* frontier SOTA across 7 benchmarks (see [NOTES.md](NOTES.md) for
the full table + caveats):

- **Qwen3.6-35B-A3B (fast MoE)** ≈ early-2025 frontier (o3 / Gemini 2.5 era)
- **Qwen3.6-27B (quality)** ≈ mid-2025 frontier (Grok 4 / Claude Sonnet 4.5)
- **DS4-Flash (full precision)** ≈ mid-2025 → early-2026 (the quantized 91 GB local build is lower)

Reality check on the unsaturated benchmarks: HLE today ~53 vs our best ~28 (**<½**);
Terminal-Bench 2.0 ~83 vs our 59. Remarkable for 27–35B on a laptop — but not the live frontier.

## Quick start

**Serve a model** (downloads the GGUF on first run, exposes an OpenAI `/v1` API):

```bash
# fast MoE, with MTP speculative decoding (needs the -MTP-GGUF build)
llama-server -m ~/models/Qwen3.6-35B-A3B-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 999 -fa on -c 65536 --parallel 1 --jinja \
  --host 127.0.0.1 --port 8080
```

**Point your coding agent** (opencode, Pi, Cline, aider, …) at:
`http://127.0.0.1:8080/v1` · API key: any string · model: the `-a` alias.

**Run the benchmark** (starts each server itself, one model at a time):

```bash
./bench_prep.sh        # quiet the machine (close apps, kill stray servers)
python3 bench.py       # full sweep across all configs, ~20 min
python3 bench.py --only +MTP --mtp-nmax 3   # filter + tune draft depth
```

## Files

| file | purpose |
|---|---|
| `README.md` | this summary |
| `NOTES.md` | detailed ongoing working log (setup, methodology, caveats, full benchmark tables) |
| `bench.py` | apples-to-apples throughput harness via each engine's OpenAI `/v1` server |
| `bench_prep.sh` | quiets the machine before benchmarking |
| `bench_native.sh` | engine-only cross-check (`llama-bench` + `mlx_lm.generate`) |

## Models

- Qwen3.6 GGUF: [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) · [`unsloth/Qwen3.6-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)
- MTP builds (for speculative decoding): [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) · [`unsloth/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)
- MLX: [`mlx-community/Qwen3.6-35B-A3B-8bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit) · [`mlx-community/Qwen3.6-27B-8bit`](https://huggingface.co/mlx-community/Qwen3.6-27B-8bit)

## Sources

- [How to set up a local coding agent on macOS](https://ikyle.me/blog/2026/how-to-setup-a-local-coding-agent-on-macos) ([HN discussion](https://news.ycombinator.com/item?id=48507020))
- Qwen3.6 model card · Artificial Analysis · Scale SWE-bench Pro (see [NOTES.md](NOTES.md) for full citations)
</content>
