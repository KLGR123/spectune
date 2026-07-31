#!/usr/bin/env bash
# Example GRPO launch for Spectune tool-agent RL via verl.
# Requires: spectune installed editable, verl checkout, GPU rollout backend.
set -xeuo pipefail

SPECTUNE_ROOT=${SPECTUNE_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
VERL_ROOT=${VERL_ROOT:-"$(cd "${SPECTUNE_ROOT}/../verl" && pwd)"}
TRAIN_FILE=${TRAIN_FILE:-"${SPECTUNE_ROOT}/outputs/verl/train.parquet"}
TEST_FILE=${TEST_FILE:-"${SPECTUNE_ROOT}/outputs/verl/test.parquet"}
TOOL_CONFIG=${TOOL_CONFIG:-"${SPECTUNE_ROOT}/examples/verl/tools_config.yaml"}
MODEL_PATH="models--Qwen--Qwen3-30B-A3B-Instruct-2507"
CKPTS_DIR="/fs_mol/liujiarun/checkpoints"
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing ${TRAIN_FILE}. Compile artifacts first (see examples/verl/README.md)." >&2
  exit 1
fi
if [[ ! -f "${TOOL_CONFIG}" ]]; then
  python - <<PY
from spectune.tools.catalog import DEFAULT_RL_TOOL_NAMES
from spectune.tools.verl import write_tools_config
write_tools_config("${TOOL_CONFIG}", tool_names=DEFAULT_RL_TOOL_NAMES)
print("wrote ${TOOL_CONFIG}")
PY
fi

cd "${VERL_ROOT}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.train_batch_size=16 \
  data.max_prompt_length=4096 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=16 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  reward.custom_reward_function.path=pkg://spectune.reward.verl \
  reward.custom_reward_function.name=compute_score \
  trainer.project_name=spectune-grpo \
  trainer.experiment_name=nmrexp-tool-agent \
  trainer.n_gpus_per_node="${NDEVICES_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="${CKPTS_DIR}"
