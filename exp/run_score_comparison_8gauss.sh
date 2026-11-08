#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p exp/logs
LOG_PATH="exp/logs/score_comparison_8gauss_80k.log"
nohup python -u exp/score_comparison_8gauss.py "$@" >> "$LOG_PATH" 2>&1 < /dev/null &
printf 'Started score comparison (PID %s). Log: %s\n' "$!" "$LOG_PATH"
