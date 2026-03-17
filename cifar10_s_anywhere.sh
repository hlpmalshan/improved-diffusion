#!/bin/bash
# Sample + evaluate CIFAR-10 models for explicit checkpoint paths
# (for reg_val = 0.0 and 0.3)

set -euo pipefail

########################################
# GPU / PROCESS SETTINGS
########################################

SAMPLE_GPUS="0, 1, 2, 3, 4, 5, 6, 7"       # GPUs used for sampling
SAMPLE_NPROC=8        # torchrun processes
MASTER_PORT=29611     # change if conflicting

export CUDA_VISIBLE_DEVICES=${SAMPLE_GPUS}

########################################
# PATHS / EVAL SETTINGS
########################################

DATA_DIR="/data/diffusion/repos/ddim/ddim_cifar10/datasets/cifar10/train"
LOG_ROOT="/data/diffusion/repos/improved-diffusion/logs_improved_diffusion_cifar10"
EVAL_SCRIPT="evaluation.py"        # metrics script
REAL_DIR="${DATA_DIR}"
IMAGE_SIZE=32

mkdir -p "${LOG_ROOT}"

########################################
# CHECKPOINT PATHS (EDIT HERE)
########################################
# Set the exact checkpoint paths you want to evaluate.
# One path per reg value. If you don't want to evaluate a reg, leave its value empty.

declare -A CKPTS

CKPTS["0.0"]="/data/diffusion/repos/improved-diffusion/logs_improved_diffusion_cifar10/cifar10_reg0.0/ema_0.9999_101600.pt"
CKPTS["0.3"]="/data/diffusion/repos/improved-diffusion/logs_improved_diffusion_cifar10/cifar10_reg0.3/ema_0.9999_101600.pt"

########################################
# SAMPLING CONFIG
########################################

NUM_SAMPLES=50000
BATCH_SIZE_SAMPLE=256

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

########################################
# CSV OUTPUT
########################################

OUT_CSV="${LOG_ROOT}/metrics_cifar10_custom_ckpts.csv"

if [ ! -f "${OUT_CSV}" ]; then
    echo "dataset,reg_val,exp_dir,ckpt,samples_dir,fid,is_mean,is_std,precision,recall,density,coverage" > "${OUT_CSV}"
fi

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
    CKPT_TAG="${CKPT_BASE%.*}"   # strip extension
    SAMPLES_DIR="${EXP_DIR}/samples_${CKPT_TAG}"

    mkdir -p "${SAMPLES_DIR}"
    export OPENAI_LOGDIR="${SAMPLES_DIR}"

    echo "==== [Sampling][CIFAR10] reg_val=${REG} ===="
    echo "Checkpoint : ${CKPT_PATH}"
    echo "Samples dir: ${SAMPLES_DIR}"

    torchrun --standalone --nproc_per_node=${SAMPLE_NPROC} --master_port=${MASTER_PORT} \
        scripts/image_sample.py \
            --model_path "${CKPT_PATH}" \
            ${MODEL_FLAGS} \
            ${DIFFUSION_FLAGS} \
            --num_samples "${NUM_SAMPLES}" \
            --batch_size "${BATCH_SIZE_SAMPLE}"

    # echo "==== [Evaluating][CIFAR10] reg_val=${REG} ===="

    # OUTPUT=$(python "${EVAL_SCRIPT}" \
    #     --real_dir "${REAL_DIR}" \
    #     --gen_dir "${SAMPLES_DIR}" \
    #     --image_size "${IMAGE_SIZE}")

    # FID=$(echo "${OUTPUT}"         | awk -F': ' '/^FID:/{print $2}')
    # IS_LINE=$(echo "${OUTPUT}"     | awk -F': ' '/^Inception Score:/{print $2}')
    # IS_MEAN=$(echo "${IS_LINE}"    | awk '{print $1}')
    # IS_STD=$(echo "${IS_LINE}"     | awk '{print $3}')
    # PREC=$(echo "${OUTPUT}"        | awk -F': ' '/^Precision:/{print $2}')
    # REC=$(echo "${OUTPUT}"         | awk -F': ' '/^Recall:/{print $2}')
    # DENS=$(echo "${OUTPUT}"        | awk -F': ' '/^Density:/{print $2}')
    # COV=$(echo "${OUTPUT}"         | awk -F': ' '/^Coverage:/{print $2}')

    # echo "cifar10,${REG},${EXP_DIR},${CKPT_BASE},${SAMPLES_DIR},${FID},${IS_MEAN},${IS_STD},${PREC},${REC},${DENS},${COV}" >> "${OUT_CSV}"

done

echo "==== Done (sample + eval for explicit CIFAR-10 checkpoints) ===="
