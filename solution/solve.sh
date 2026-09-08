#!/bin/bash
# Oracle solution. Produces every artifact instruction.md asks for and is graded by
# the same verifier as any agent's output.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Deterministic cuBLAS reductions; every seed is fixed inside the Python.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

python "${SCRIPT_DIR}/run_all.py"
