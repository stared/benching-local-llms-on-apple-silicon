#!/usr/bin/env bash
# Native engine-only cross-check (no HTTP server overhead), as a sanity check
# against bench.py's server-path numbers. llama.cpp has llama-bench built in;
# MLX uses mlx_lm.generate --verbose which prints prompt/gen tok/s + peak memory.
set -u
MODELS=~/models

echo "############ llama.cpp (llama-bench) ############"
# pp512 = prompt-processing (prefill) speed, tg128 = token-generation (decode) speed
for f in Qwen3.6-35B-A3B-Q8_0.gguf Qwen3.6-27B-Q8_0.gguf; do
  echo "---- $f ----"
  llama-bench -m "$MODELS/$f" -ngl 999 -fa 1 -p 512 -n 128 -r 3
done

echo
echo "############ MLX (mlx_lm.generate --verbose) ############"
# --verbose prints: Prompt tok/s, Generation tok/s, Peak memory
for m in mlx-community/Qwen3.6-35B-A3B-8bit mlx-community/Qwen3.6-27B-8bit; do
  echo "---- $m ----"
  mlx_lm.generate --model "$m" \
    --prompt "Write a Python function that merges two sorted lists. Explain briefly." \
    --max-tokens 256 --temp 0 --verbose True
done
