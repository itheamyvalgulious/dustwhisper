#!/bin/bash
source .venv/bin/activate
python scripts/benchmark_engine.py --scenario random_materials --width 1920 --height 1080 --warmup 3 --frames 16 --json "$@"
