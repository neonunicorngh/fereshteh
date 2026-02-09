#!/bin/bash

# ===============================
# HW02 Question 7 background run
# ===============================

echo "Starting HW02Q7 animation run..."
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

# Activate uv virtual environment
source .venv/bin/activate

# Run the animation script in background
nohup python scripts/binaryclassification_animate_impl.py \
    > hw02q7_run.log 2>&1 &

echo "Process started in background."
echo "Logs are being written to hw02q7_run.log"
echo "End command time: $(date)"
