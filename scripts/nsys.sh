#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf report1.nsys-rep report1.sqlite

./scripts/test.sh
# --output has to come before the app path, otherwise nsys hands it to tiny-vllm,
# which would take it for a prompt
nsys profile --output=report1.nsys-rep ./build/tiny-vllm
nsys stats report1.nsys-rep
