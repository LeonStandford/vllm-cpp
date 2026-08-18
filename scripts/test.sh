#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/build.sh
./scripts/run.sh "$@"
