#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/dataset/verl/rlaif_train.parquet}"
VAL_FILE="${VAL_FILE:-${REPO_ROOT}/dataset/verl/rlaif_val.parquet}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/minimind-3}"
REWARD_FN_PATH="${REWARD_FN_PATH:-${SCRIPT_DIR}/reward_fn_minimind.py}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/out/verl_grpo_ckpt}"

WANDB_PROJECT="${WANDB_PROJECT:-MiniMind-GRPO-verl}"
WANDB_EXPERIMENT="${WANDB_EXPERIMENT:-minimind-grpo-2x4090}"

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
LR="${LR:-1e-6}"
ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-vllm}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.55}"

export MINIMIND_REWARD_MODEL_PATH="${MINIMIND_REWARD_MODEL_PATH:-${REPO_ROOT}/internlm2-1_8b-reward}"
export TOKENIZERS_PARALLELISM="true"
export HYDRA_FULL_ERROR=1

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Train parquet not found: ${TRAIN_FILE}"
  echo "Please run trainer/verl_tools/convert_minimind_rlaif_to_verl.py first."
  exit 1
fi

if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Val parquet not found, fallback to train parquet: ${TRAIN_FILE}"
  VAL_FILE="${TRAIN_FILE}"
fi

mkdir -p "${CKPT_DIR}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name="${ROLLOUT_ENGINE}" \
  actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.use_kl_in_reward=False \
  reward.custom_reward_function.path="${REWARD_FN_PATH}" \
  reward.custom_reward_function.name=compute_score_minimind \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_EXPERIMENT}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  +ray_kwargs.ray_init.num_gpus=2 \
  +ray_kwargs.timeline_json_file="${CKPT_DIR}/ray_timeline.json" \
  "$@"
