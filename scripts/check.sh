#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

compute-sanitizer ./build/tiny-vllm "$@"
