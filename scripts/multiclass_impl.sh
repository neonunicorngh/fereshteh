#!/bin/bash
set -e

cd "$(dirname "$0")/.."

KEY="hw02"

for i in 1 2 3 4 5
do
  echo "Run $i/5 ..."
  uv run python scripts/multiclass_impl.py --keyword "$KEY" --epochs 30 --eta 0.001 --batch-size 2048
done

uv run python scripts/multiclass_eval.py --keyword "$KEY"
