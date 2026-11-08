#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MNIST_DEVICE="${MNIST_DEVICE:-cuda}"
export PYTHONPATH="${PROJECT_ROOT}/src/engression:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

paths=(
    gaussian-isotropic
    gaussian-non-diagonal
    generalized-isotropic
    generalized-non-diagonal
    joint
)
if [[ $# -gt 0 ]]; then
    paths=("$@")
fi
for path in "${paths[@]}"; do
    config="exp/configs/diffusion-mnist-${path}.yaml"
    if [[ ! -f "${config}" ]]; then
        echo "Unknown MNIST production path: ${path}" >&2
        exit 2
    fi
    "${PYTHON_BIN}" -u -m exp.sample_diffusion_mnist \
        --config "${config}" --device "${MNIST_DEVICE}"
done
