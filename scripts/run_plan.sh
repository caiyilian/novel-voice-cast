#!/bin/bash
# Phase 1 — 插图规划：分析全卷小说，输出插图计划
# 用法: bash scripts/run_plan.sh
# 断点续跑: bash scripts/run_plan.sh --resume

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR" && source .venv/bin/activate && exec python scripts/run_illustration_plan.py \
    --novel novels/novel.txt \
    --labels novels/labels.txt \
    --output output/illustration_plan.json \
    "$@"