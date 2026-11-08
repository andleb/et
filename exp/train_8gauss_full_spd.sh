#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src/engression:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for seed in 42 43; do
    result_dir="${PROJECT_ROOT}/res/models/score_and_diffusion/full_spd_seed${seed}"
    "${PYTHON_BIN:-python}" -u exp/train_8gauss.py \
        --config "${PROJECT_ROOT}/exp/configs/train_8gauss_full_spd.yaml" \
        --mode generalized \
        --seed "${seed}" \
        --result_dir "${result_dir}" \
        "$@"
done
