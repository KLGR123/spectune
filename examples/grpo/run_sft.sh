#!/usr/bin/env bash
# SFT training on rollout-filtered NMRexp data using verl fsdp_sft_trainer.

set -xeuo pipefail

NDEVICES_PER_NODE=8
PROJECT_NAME=spectune
EXPERIMENT_NAME=sft-nmrexp-rollout-qwen3-8b

# fill in the following paths
export SPECTUNE_ROOT=/fs_mol/liujiarun/spectune
export VERL_ROOT=/fs_mol/liujiarun/verl
export TRAIN_FILE=$SPECTUNE_ROOT/outputs/datasets/verl/nmrexp_sft_rollout.parquet
export VAL_FILE=$SPECTUNE_ROOT/outputs/datasets/verl/test.parquet
export MODEL_PATH=/fs_mol/liujiarun/models/qwen3-8b
export SAVE_PATH=$SPECTUNE_ROOT/outputs/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME
export TENSORBOARD_DIR=$SPECTUNE_ROOT/outputs/tensorboard/$PROJECT_NAME/$EXPERIMENT_NAME

cd "${VERL_ROOT}"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node="${NDEVICES_PER_NODE}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TRAIN_FILE}" \
    data.val_max_samples=500 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=16384 \
    model.partial_pretrain="${MODEL_PATH}" \
    use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    optim.lr=2e-5 \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=2 \
    trainer.save_freq=200 \
    trainer.logger='["console","tensorboard"]'\
    model.strategy=fsdp \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.offload_params=True \
