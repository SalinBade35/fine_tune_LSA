#!/usr/bin/env bash
set -e
mlx_lm.lora \
  --model mlx-community/Qwen3-8B-4bit \
  --train \
  --data ./data \
  --adapter-path ./adapters \
  --batch-size 1 \
  --num-layers 16 \
  --iters 1500 \
  --learning-rate 1e-5 \
  --max-seq-length 1024 \
  --steps-per-eval 150 \
  --save-every 300 \
  --grad-checkpoint