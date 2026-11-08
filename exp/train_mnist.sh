#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${PROJECT_ROOT}/res/models/mnist/posterior"
SNAPSHOT_DIR="${RUN_DIR}/snapshots"
export PYTHONPATH="${PROJECT_ROOT}/src/engression:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

args=(
    --config "${PROJECT_ROOT}/exp/configs/train_mnist.yaml"
    --result-dir "${RUN_DIR}"
    --snapshot-dir "${SNAPSHOT_DIR}"
)
if [[ -f "${RUN_DIR}/latest.pt" ]]; then
    args+=(--resume "${RUN_DIR}/latest.pt")
fi
"${PYTHON_BIN}" -u -m exp.train_mnist "${args[@]}" data.download=true "$@"
