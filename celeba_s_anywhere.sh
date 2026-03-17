#!/bin/bash
# Sample CelebA 64×64 models for explicit checkpoint paths
# (for reg_val = 0.0 and 0.3)

set -euo pipefail

########################################
# GPU / PROCESS SETTINGS
########################################

# GPUs used for sampling
SAMPLE_GPUS="0,1,2,3,4,5,6,7"        # edit to what you want, e.g. "0,1"
SAMPLE_NPROC=8         # torchrun processes (set = number of GPUs if multi-GPU)
MASTER_PORT=29811      # change if conflicting

export CUDA_VISIBLE_DEVICES=${SAMPLE_GPUS}

########################################
# PATHS / SETTINGS (EDIT HERE)
########################################

# CelebA 64×64 data + logs
DATA_DIR="/data/diffusion/repos/ddim/ddim_celeba/datasets/celeba/celeba/prepocessed_imgs"
LOG_ROOT="/data/diffusion/repos/improved-diffusion/logs_improved_diffusion_celeba64"

REAL_DIR="${DATA_DIR}"
IMAGE_SIZE=256

mkdir -p "${LOG_ROOT}"

########################################
# CHECKPOINT PATHS (EDIT HERE)
########################################
# Set the exact checkpoint paths you want to sample.
# One path per reg value. If you don't want to sample a reg, leave its value empty.

declare -A CKPTS

# Example paths – update to your actual ema checkpoints
CKPTS["0.0"]="/data/diffusion/repos/improved-diffusion/logs_improved_diffusion_celeba64/celeba_reg0.0/ema_0.9999_300000.pt"
CKPTS["0.3"]="/data/diffusion/repos/improved-diffusion/logs_improved_diffusion_celeba64/celeba_reg0.3_cosine/ema_0.9999_083335.pt"

########################################
# SAMPLING CONFIG
########################################

NUM_SAMPLES=50000
BATCH_SIZE_SAMPLE=128   # reduce if OOM

# Match CelebA training config
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 2 --attention_resolutions 16 --dropout 0.3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

########################################
# LOOP OVER REG VALUES
########################################

for REG in 0.3; do
    CKPT_PATH="${CKPTS[$REG]:-}"

    if [ -z "${CKPT_PATH}" ]; then
        echo "[Info] Skipping reg=${REG}: no checkpoint path specified."
        continue
    fi

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "[Warn] Skipping reg=${REG}: checkpoint not found at ${CKPT_PATH}" >&2
        continue
    fi

    EXP_DIR="$(dirname "${CKPT_PATH}")"
    CKPT_BASE="$(basename "${CKPT_PATH}")"
    CKPT_TAG="${CKPT_BASE%.*}"   # strip .pt
    SAMPLES_DIR="${EXP_DIR}/samples_${CKPT_TAG}_2"

    mkdir -p "${SAMPLES_DIR}"
    export OPENAI_LOGDIR="${SAMPLES_DIR}"

    echo "==== [Sampling][CelebA64] reg_val=${REG} ===="
    echo "Checkpoint : ${CKPT_PATH}"
    echo "Samples dir: ${SAMPLES_DIR}"

    torchrun --standalone --nproc_per_node=${SAMPLE_NPROC} --master_port=${MASTER_PORT} \
        scripts/image_sample.py \
            --model_path "${CKPT_PATH}" \
            ${MODEL_FLAGS} \
            ${DIFFUSION_FLAGS} \
            --num_samples "${NUM_SAMPLES}" \
            --batch_size "${BATCH_SIZE_SAMPLE}"

done

echo "==== Done (sampling for explicit CelebA64 checkpoints) ===="