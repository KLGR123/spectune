#!/usr/bin/env bash

set -xeuo pipefail

NNODES=1
NDEVICES_PER_NODE=4
PROJECT_NAME=spectune-debug
EXPERIMENT_NAME=grpo-nmrexp-20k-qwen3-4b-base

# fill in the following paths
export SPECTUNE_ROOT=/fs_mol/liujiarun/spectune
export VERL_ROOT=/fs_mol/liujiarun/verl
export TRAIN_FILE=$SPECTUNE_ROOT/outputs/datasets/verl/train.parquet
export TEST_FILE=$SPECTUNE_ROOT/outputs/datasets/verl/test.parquet
export TOOL_CONFIG=$SPECTUNE_ROOT/examples/grpo/tools_config.yaml
export MODEL_PATH=/fs_mol/guojianz/projects/FunMG/LLM/model_weight/qwen/models--Qwen--Qwen3-4B-Instruct-2507
export TENSORBOARD_DIR=$SPECTUNE_ROOT/outputs/tensorboard/$PROJECT_NAME/$EXPERIMENT_NAME

export VERL_DEBUG=1


cd "${VERL_ROOT}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  +algorithm.filter_groups.enable=True \
  +algorithm.filter_groups.metric=acc \
  +algorithm.filter_groups.max_num_gen_batches=4 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.train_batch_size=2 \
  data.max_prompt_length=4096 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  reward_model.reward_manager=naive \
  custom_reward_function.path="${SPECTUNE_ROOT}/spectune/reward/verl.py" \
  custom_reward_function.name=compute_score \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NDEVICES_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq=5 \
  trainer.test_freq=2 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.resume_mode=auto \
  trainer.default_local_dir="${SPECTUNE_ROOT}/outputs/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
  trainer.logger='["console","tensorboard"]'