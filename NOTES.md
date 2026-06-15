# Working notes: local coding models with llama.cpp (vs MLX)

Ongoing working log — setup commands, methodology, caveats, and full benchmark tables,
appended over time. For the reader-facing summary see **[README.md](README.md)**.

## Hardware

- Apple **M5 Max**, **128 GB** unified memory → effectively no memory pressure for
  these models, even at 8-bit with long context.
- `llama.cpp` already installed via Homebrew (`llama-server`, `llama-cli`),
  version 9430 (built with Metal).

## What we run today (MLX)

Served with `mlx_lm.server`, ~100 t/s each:

| Model                          | Type         | Quant | Notes        |
|--------------------------------|--------------|-------|--------------|
| `mlx-community/Qwen3.6-27B-8bit`     | dense        | 8-bit | ~100 t/s |
| `mlx-community/Qwen3.6-35B-A3B-8bit` | MoE (A3B act)| 8-bit | ~100 t/s |

## llama.cpp equivalents (verified repos, 8-bit = apples-to-apples with MLX)

llama.cpp uses **GGUF** instead of MLX. `llama-server` downloads straight from
Hugging Face with `-hf REPO:QUANT`. Verified Q8_0 (8-bit) files, both single-file:

| MLX model (current)            | GGUF repo (Q8_0)                       | File                              | Size    |
|--------------------------------|----------------------------------------|-----------------------------------|---------|
| Qwen3.6-35B-A3B-8bit (MoE)     | `unsloth/Qwen3.6-35B-A3B-GGUF`         | `Qwen3.6-35B-A3B-Q8_0.gguf`       | 36.9 GB |
| Qwen3.6-27B-8bit (dense)       | `unsloth/Qwen3.6-27B-GGUF`             | `Qwen3.6-27B-Q8_0.gguf`           | 28.6 GB |

Alt publisher: `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`, `bartowski/Qwen_Qwen3.6-27B-GGUF`
(also carry Q8_0). MTP draft variants exist (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) but
see note below — not worth it for the MoE.

### Quant choice (from the HN thread)

- **MoE** (35B-A3B): low active params → quantize *less* aggressively. Use **Q5_K_M**
  or higher. To match our current 8-bit quality use **Q8_0**; with 128 GB it fits easily.
- **Dense** (27B): **Q4_K_M** is "adequate", **Q8_0** to match MLX 8-bit.
- Speculative/MTP draft decoding helps **dense** models, but "doesn't add much" for
  low-active-param **MoE** — skip the draft model for 35B-A3B.

### Serve the MoE 35B-A3B (Q8_0)

```bash
llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0 \    # 36.9 GB, single file
  -a qwen3.6-35b-a3b \
  -ngl 999 -fa on \
  -c 65536 --parallel 1 \
  --jinja \                                  # enable chat template + tool calling
  --host 127.0.0.1 --port 8080
```

### Serve the dense 27B (Q8_0)

```bash
llama-server \
  -hf unsloth/Qwen3.6-27B-GGUF:Q8_0 \        # 28.6 GB, single file
  -a qwen3.6-27b \
  -ngl 999 -fa on \
  -c 65536 --parallel 1 \
  --jinja \
  --host 127.0.0.1 --port 8081
```

Key flags:
- `-ngl 999` → offload all layers to Metal GPU.
- `-fa on` → flash attention (faster, less memory).
- `-c 65536` → 64k context (raise it; 128 GB allows much more).
- `--jinja` → use the model's chat template; **required** for proper tool/function
  calling from coding agents.
- `-a NAME` → model alias reported on the API.
- `--parallel 1` → single sequence (max speed for one user).

## Pointing a coding agent at it

`llama-server` exposes an **OpenAI-compatible** API at `http://127.0.0.1:8080/v1`.
Point any agent (opencode, Pi, Cline, aider, etc.) at:

- Base URL: `http://127.0.0.1:8080/v1`
- API key: any non-empty string (e.g. `local`)
- Model: the alias from `-a` (e.g. `qwen3.6-35b-a3b`)

Tip from the article: Qwen/Gemma "do not need" the huge default agent system prompt —
trim it to improve speed and reliability.

## Expected performance

Per the blog/HN on M-series: llama.cpp ~70 t/s, MLX ~80 t/s on Qwen3-Coder class
models. MLX tends to edge out llama.cpp for pure decode on Apple Silicon — so our
current ~100 t/s on MLX is expected to be **as good or better** than llama.cpp here.
Reasons to still use llama.cpp: GGUF ecosystem, draft/MTP speculative decoding for
dense models, multimodal `mmproj`, and broader agent/tooling compatibility.

## Benchmark to compare apples-to-apples

```bash
llama-bench -m <path-to.gguf> -ngl 999 -fa 1
```

## Benchmarking (llama.cpp vs MLX, 8-bit)

Three files:
- `bench_prep.sh`  — quiets the machine (quits heavy apps, kills stray servers, prints state).
- `bench.py`       — apples-to-apples via each engine's OpenAI `/v1` server (stdlib only).
- `bench_native.sh`— engine-only cross-check (`llama-bench` + `mlx_lm.generate --verbose`).

```bash
./bench_prep.sh            # close apps, then optionally pause Spotlight (printed)
python3 bench.py           # all 4 models, context sweep 128/1k/4k/8k -> ~18-25 min
./bench_native.sh          # sanity cross-check without server overhead
```

Useful flags: `--only 27B` (filter), `--prompt-tokens 128,2048,8192` (context sizes),
`--gen-tokens 800`, `--runs 5`. Default sweeps **128 / 1k / 4k / 8k** prompt tokens.

A fifth config, **ds4 DeepSeek-V4-Flash** (`~/not-my-repos/ds4`, 91 GB), is included
but opt-in — run `python3 bench.py --only ds4`. It's measured with the *same*
trial()/sweep, so numbers line up, but note two things: (1) it's a different, much
larger model — a whole-setup comparison, not engine-isolated; (2) ds4's real edge is
warm suffix-only re-prefill (disk KV cache), which this cold-prefill bench does not
exercise, so its prefill number here is pessimistic vs a real agent session.

`bench.py` launches one server at a time (never two models resident, llama.cpp at
`-c 16384` so 8k prompt + gen fits), warms it up, and reports medians: **decode t/s**
(headline), **prefill t/s**, **TTFT**, process **RSS peak**, **system memory delta**
(catches GPU buffers), and **cold load time**. Saved to `bench_results.{json,md}`.

### Fair-benchmark checklist
- Only ONE model loaded at a time — both engines share the 40-core GPU + unified RAM.
- AC power, lid open, Low Power Mode off (verified off).
- Quit Beeper/Slack/Signal/Dropbox/Cursor/Claude-app/browsers; pause Spotlight.
- Optional: reboot first (10-day uptime had 41k pageouts) for cleanest peak-memory.
- Expect MLX to edge llama.cpp on decode; llama.cpp competitive on prefill.

## Results (2026-06-14, M5 Max 128 GB, full sweep)

8-bit Q8 throughput, decode = total tok/s (reasoning incl.), median of 3 runs.
Numbers below are the interactive (128-tok) and long-context (8k) decode rates.

| setup | decode @128 | decode @8k | prefill @8k | RSS |
|---|--:|--:|--:|--:|
| **llama.cpp 35B-A3B Q8 +MTP** | **104.6** | **96.8** | 1990 | 45 GB |
| llama.cpp 35B-A3B Q8 (baseline) | 92.9 | 86.9 | 2462 | 44 GB |
| MLX 35B-A3B 8bit | 84.5 | 79.4 | 3093 | 37 GB |
| llama.cpp 27B Q8 +MTP | 31.7 | 29.9 | 521 | 42 GB |
| llama.cpp 27B Q8 (baseline) | 18.4 | 17.0 | 567 | 41 GB |
| MLX 27B 8bit | 17.3 | 17.1 | 637 | 28 GB |
| ds4 DeepSeek-V4-Flash q2-q4 (91 GB) | 33.0 | 27.8 | 366 | 103 GB* |

\* ds4 RSS reads 0.3 GB (mmap); 103 GB is the system-memory footprint.

### Findings
- **MTP speedup is architecture-dependent:** dense 27B **+75% (≈1.75×, 17→31 t/s)**;
  MoE 35B-A3B **+12% (87→97 t/s)**. Big forward pass (dense) = more to gain from
  speculation; cheap 3B-active MoE = less. MTP is lossless (big model verifies).
- **llama.cpp beats MLX here** for these Qwen models on M5 Max: MoE +10% baseline,
  +24% with MTP; dense roughly tied until MTP makes llama ≈1.8× MLX. (Opposite of
  the common "MLX is faster on Mac" assumption — verified, not assumed.)
- **Daily driver: llama.cpp 35B-A3B Q8 + MTP** — ~100 t/s, fast prefill, barely sags
  to 8k. Dense 27B is only interactive *with* MTP, and the MoE is still ~3× faster.
- **ds4** is sane after the total-tok/s fix (~30 t/s, matches its own note) —
  impressive for a 91 GB frontier-class model, but costs 103 GB resident.
- **Memory: trust RSS, not the bench's sys-Δ column** (sys-Δ drifts as prior runs
  free memory). llama trades ~8 GB more RAM than MLX for its speed.
- MTP slightly lowers prefill and adds ~1 s load — negligible vs the decode win.

## How our local models compare to dated frontier SOTA (2026-06-15)

Question: a local Mac model today ≈ the cloud frontier of *when*? Answer depends
heavily on the benchmark — SWE-bench Verified is saturated/contaminated and
flatters; harder unsaturated benchmarks (HLE, SWE-Pro, Terminal-Bench) place our
models earlier and reveal a real gap to the live 2026 frontier.

Local model scores are vendor self-reported (Qwen card) unless marked **(AA)** =
Artificial Analysis independent. Dated match = which frontier model first hit ~that
score, and when.

| benchmark (↓ less saturated) | 35B-A3B | 27B | DS4-Flash | best local ≈ era |
|---|--|--|--|--|
| SWE-bench Verified | 73.4 | 77.2 | ~79–81 | 27B = Claude Sonnet 4.5 (Sep'25); DS4F ≈ Opus 4.5 (Dec'25) |
| GPQA Diamond | 86.0 | 87.8 | 86.7 **(AA)** | all ≈ Grok 4 / Gemini 2.5 — mid'25 (DeepSeek self-claim 91.3 was inflated) |
| HLE (hardest) | 21.4 | 24.0 | 27.8 **(AA)** | 35B≈o3 early'25; 27B≈Grok4 Jul'25; DS4F≈GPT-5.2 Jan'26 |
| SWE-bench Pro | 49.5 | 53.5 | n/a | **on Qwen's refined set, +11 hot vs Scale**; normalized 27B≈42≈Sonnet 4/4.5 mid'25 |
| Terminal-Bench 2.0 | 51.5 | 59.3 | n/a | 27B = Claude Opus 4.5 (59.3 exact, Dec'25); frontier today 82.7 (GPT-5.5) |
| LiveCodeBench v6 | 80.4 | 83.9 | 87.5–91.6 | near-frontier, window-dependent |
| AIME 2026 | 92.7 | 94.1 | n/a | ≈ frontier (math saturating) |

### Dated-frontier placement (synthesis)
- **Qwen3.6-35B-A3B (fast MoE)** ≈ **early-2025** frontier (o3 / Gemini 2.5 era).
- **Qwen3.6-27B (quality)** ≈ **mid-2025** frontier (Grok 4 / Sonnet 4.5, ~Jul–Sep'25).
- **DS4-Flash (full precision)** spans **mid-2025 (GPQA 86.7) → early-2026 (HLE 27.8)**;
  edges the 27B on paper but not uniformly.

### Load-bearing caveats
- **SWE-Verified is contaminated/saturated** (OpenAI audit: gold patches in training
  sets). Its "Sep-2025" read is the rosiest; trust SWE-Pro / HLE more.
- **SWE-Pro set mismatch:** Qwen reports on their *refined* set, which runs ~+11 pts
  hot vs Scale's public set (anchored on Opus 4.5: 57.1 vs 45.9). Raw 53.5 does NOT
  beat Opus 4.5 — normalized it's ~42 ≈ Sonnet 4/4.5.
- **Vendor inflation, caught:** DeepSeek self-claimed GPQA 91.3; independent AA = 86.7.
- **DS4-Flash quant trap:** all DS4F numbers are full-precision API. We run the ds4
  repo's **91 GB 2–4-bit imatrix build** (~30 t/s, 103 GB RAM) — real local quality
  sits *below* every published number, by an unmeasured amount.
- **Terminal-Bench is scaffold-sensitive** (Opus 4.6 reads 59→80 by harness).

### Bottom line
Local-on-a-Mac today ≈ **mid-2025-to-early-2026 frontier** depending on model and axis
— remarkable for 27–35B on a laptop. But on the unsaturated hard tests the live 2026
frontier is a clear generation ahead: **HLE ~53 today vs our best ~28 (<½); Terminal-
Bench 2.0 ~83 vs our 59**. To remove all scaffold/set/quant caveats at once, run the
benchmarks locally against our actual builds (`bench.py` is throughput-only so far —
quality evals would be a separate harness).

## References

- https://ikyle.me/blog/2026/how-to-setup-a-local-coding-agent-on-macos
- https://news.ycombinator.com/item?id=48507020
</content>
</invoke>
