# Running local coding models with llama.cpp (vs MLX)

Notes for replacing/complementing the MLX setup with llama.cpp on this Mac.

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

## References

- https://ikyle.me/blog/2026/how-to-setup-a-local-coding-agent-on-macos
- https://news.ycombinator.com/item?id=48507020
</content>
</invoke>
