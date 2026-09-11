#!/usr/bin/env bash
# SFT training on rollout-filtered NMRexp data using verl fsdp_sft_trainer.


set -xeuo pipefail

NDEVICES_PER_NODE=8
PROJECT_NAME=sft
EXPERIMENT_NAME=sft-all-rollout-qwen3-32b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPECTUNE_ROOT="${SPECTUNE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export VERL_ROOT="${VERL_ROOT:-$(cd "$SPECTUNE_ROOT/verl" && pwd)}"
export MODEL_PATH=/fs_mol/liujiarun/models/qwen3-32b
export TRAIN_FILE=$SPECTUNE_ROOT/outputs/datasets/verl/sft.parquet

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SAVE_PATH=$SPECTUNE_ROOT/outputs/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME
export TENSORBOARD_DIR=$SPECTUNE_ROOT/outputs/tensorboard/$PROJECT_NAME/$EXPERIMENT_NAME

mkdir -p "${SAVE_PATH}" "${TENSORBOARD_DIR}"

cd "${VERL_ROOT}"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node="${NDEVICES_PER_NODE}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=[${TRAIN_FILE}] \
    data.val_files=[${TRAIN_FILE}] \
    data.val_max_samples=200 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=32678 \
    data.truncation=right \
    model.partial_pretrain="${MODEL_PATH}" \
    use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    optim.lr=5e-6 \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=3 \
    trainer.save_freq=50 \
    trainer.logger='["console","tensorboard"]'\
    model.fsdp_config.model_dtype=bf16 \
    model.strategy=fsdp \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.offload_params=True \
    model.use_liger=True \
    +model.use_fused_kernels=True \