#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/build.sh
./build/tiny-vllm "What is the capital of France?"
