#!/bin/bash
set -e

if [ "$1" = "train" ]; then
    shift
    python sagemaker/code/train.py "$@"
elif [ "$1" = "serve" ]; then
    shift
    python sagemaker/code/serve.py "$@"
else
    exec "$@"
fi
