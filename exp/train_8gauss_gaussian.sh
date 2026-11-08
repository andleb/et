#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RESULT_DIR="${PROJECT_ROOT}/res/models/score_and_diffusion/gaussian_isotropic"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN:-python}" -u exp/train_8gauss.py \
  --config "${PROJECT_ROOT}/exp/configs/train_8gauss_gaussian_isotropic.yaml" \
  --mode gaussian \
  --result_dir "${RESULT_DIR}"
