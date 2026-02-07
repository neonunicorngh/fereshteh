#!/bin/bash

SCRIPT_NAME="binaryclassification_impl.py"

echo "Starting the binary classification training task..."

if [ -f "$SCRIPT_NAME" ]; then
    python "$SCRIPT_NAME"
else
    echo "Error: $SCRIPT_NAME not found!"
    exit 1
fi

echo "Process finished at $(date)"
