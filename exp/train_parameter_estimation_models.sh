#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src/engression:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

case "${1:---all}" in
    --all) TRAIN_POSTERIOR=1; TRAIN_SCORE=1 ;;
    --posterior) TRAIN_POSTERIOR=1; TRAIN_SCORE=0 ;;
    --score) TRAIN_POSTERIOR=0; TRAIN_SCORE=1 ;;
    *) echo "Usage: $0 [--all|--posterior|--score]" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"

if (( TRAIN_POSTERIOR )); then
    for SEED in 42 43 46; do
        RESULT_DIR="${PROJECT_ROOT}/res/models/parameter_estimation/posterior_seed${SEED}"
        mkdir -p "${RESULT_DIR}"
        "${PYTHON_BIN}" -u exp/train_8gauss.py \
            --config "${PROJECT_ROOT}/exp/configs/train_8gauss_parameter_estimation_posterior.yaml" \
            --mode generalized \
            --seed "${SEED}" \
            --result_dir "${RESULT_DIR}" \
            2>&1 | tee "${RESULT_DIR}/train.out"
    done
fi

if (( TRAIN_SCORE )); then
    for SEED in 10042 11042 12042 13042 14042; do
        RESULT_DIR="${PROJECT_ROOT}/res/models/parameter_estimation/score_seed${SEED}"
        mkdir -p "${RESULT_DIR}"
        "${PYTHON_BIN}" -u exp/train_parameter_estimation_score.py \
            --config "${PROJECT_ROOT}/exp/configs/train_8gauss_parameter_estimation_score.yaml" \
            --result_dir "${RESULT_DIR}" \
            "training.seed=${SEED}" \
            2>&1 | tee "${RESULT_DIR}/train.out"
    done
fi
