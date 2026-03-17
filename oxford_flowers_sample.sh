#!/bin/bash
# Multi-GPU sampling script for improved-diffusion models on Oxford Flowers
# Adapted to use MODEL_FLAGS and DIFFUSION_FLAGS like the example

set -euo pipefail

# Specify which GPUs to use (comma-separated). nproc_per_node is inferred from this.
export CUDA_VISIBLE_DEVICES=2
NPROC=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
MASTER_PORT=29511

# Directory where training logs/checkpoints were saved
LOG_ROOT="/data/diffusion/repos/improved-diffusion/logs_improved_diffusion_flowers"

# Sampling parameters
NUM_SAMPLES=8200   # total samples to generate
BATCH_SIZE=512       # samples per GPU per iteration

# Define model and diffusion flags matching your training config
MODEL_FLAGS="--image_size 64 --num_channels 64 --num_res_blocks 2 --dropout 0.1 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

for REG in 0.0 0.3; do
  EXP_DIR="${LOG_ROOT}/oxford_flowers_reg${REG}"
  # Find the most recent model checkpoint for this reg value
  CKPT=$(ls "${EXP_DIR}" | grep '^ema_' | sort | tail -n1)
  CKPT_PATH="${EXP_DIR}/${CKPT}"

  # Directory to save generated samples
  SAMPLE_DIR="${EXP_DIR}/samples"
  mkdir -p "${SAMPLE_DIR}"
  export OPENAI_LOGDIR="${SAMPLE_DIR}"

  echo "Sampling from ${CKPT_PATH} with reg_val=${REG} on GPUs ${CUDA_VISIBLE_DEVICES}"
  torchrun --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    scripts/image_sample.py \
      --model_path "${CKPT_PATH}" \
      ${MODEL_FLAGS} \
      ${DIFFUSION_FLAGS} \
      --num_samples "${NUM_SAMPLES}" \
      --batch_size "${BATCH_SIZE}"
done
