#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

NCU=${NCU:-$(command -v ncu || true)}
if [ -z "$NCU" ]; then
    for candidate in /usr/local/cuda/bin/ncu /opt/cuda/bin/ncu; do
        if [ -x "$candidate" ]; then
            NCU=$candidate
            break
        fi
    done
fi
if [ -z "$NCU" ]; then
    echo "ncu not found - install Nsight Compute or set NCU=/path/to/ncu" >&2
    exit 1
fi

sudo "$NCU" --target-processes all ./build/tiny-vllm "$@"
