#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src/engression:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN:-python}" -m exp.sample_diffusion_8gauss_controls \
    --config "exp/configs/diffusion-8gauss-controls.yaml" \
    --device "${DEVICE:-cuda}" --all "$@"
