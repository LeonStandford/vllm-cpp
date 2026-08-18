#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# the engine reads .cache/huggingface/download/* relative to the repo root
./build/tiny-vllm "$@"
