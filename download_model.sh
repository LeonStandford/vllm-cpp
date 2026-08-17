#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"

if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

hf download meta-llama/Llama-3.2-1B-Instruct \
    model.safetensors tokenizer.json tokenizer_config.json \
    --local-dir .cache/huggingface/download
