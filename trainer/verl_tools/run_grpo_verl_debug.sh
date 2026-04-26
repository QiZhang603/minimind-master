#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/out/verl_grpo_debug_ckpt}"
TIMELINE_FILE="${TIMELINE_FILE:-${CKPT_DIR}/ray_timeline.json}"

mkdir -p "${CKPT_DIR}"

# This debug preset keeps run time short and emits timeline for Perfetto/Chrome trace.
CKPT_DIR="${CKPT_DIR}" \
VERL_EXPERIMENT="${VERL_EXPERIMENT:-minimind-grpo-debug}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}" \
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-2}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}" \
bash "${SCRIPT_DIR}/run_grpo_verl_2x4090.sh" \
  +ray_kwargs.timeline_json_file="${TIMELINE_FILE}" \
  +ray_kwargs.ray_init.include_dashboard=True \
  +ray_kwargs.ray_init.dashboard_port=8265 \
  "$@"

cat <<EOF
Ray debug quick tips:
1) Open dashboard: http://<your_server_ip>:8265
2) Timeline file: ${TIMELINE_FILE}
3) Open timeline with https://ui.perfetto.dev
EOF
