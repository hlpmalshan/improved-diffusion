#!/bin/bash
# End-to-end pipeline for CIFAR-10 on improved-diffusion:
# 1) Train models for reg_val in {0.3, 0.0}
# 2) Sample images from trained EMA checkpoints
# 3) Evaluate FID / IS / PRDC and append to CSV

set -euo pipefail

########################################
# GPU SETTINGS (EDIT HERE)
########################################

# GPUs used for training (comma-separated)
TRAIN_GPUS="0,1,2"
TRAIN_NPROC=3

# GPUs used for sampling (comma-separated)
SAMPLE_GPUS="5"
SAMPLE_NPROC=1

########################################
# COMMON PATHS / SETTINGS
########################################

DATA_DIR="/data/diffusion/repos/ddim/ddim_cifar10/datasets/cifar10/train"
LOG_ROOT="/data/diffusion/repos/improved-diffusion/logs_improved_diffusion_cifar10"
EVAL_SCRIPT="evaluation.py" # your metrics script
REAL_DIR="${DATA_DIR}"
IMAGE_SIZE=32

mkdir -p "${LOG_ROOT}"

########################################
# 1) TRAINING
########################################

# CIFAR-10 style hyperparameters (from improved-diffusion example)
BATCH_SIZE=128
IMG_SIZE=32
NUM_CH=128
RES_BLOCKS=3
DROPOUT=0.3
DIFF_STEPS=1000
NOISE_SCHED="cosine"
LR=1e-4
EMA_RATE=0.9999
WEIGHT_DECAY=0.0
TOTAL_ITERS=508000         # adjust as needed
SAVE_FREQ=101600            # checkpoint/save interval

echo "==== [Stage 1] Training CIFAR-10 models (reg=0.3, 0.0) on GPUs ${TRAIN_GPUS} ===="

# for REG in 0.3 0.0; do
#     EXP_DIR="${LOG_ROOT}/cifar10_reg${REG}"
#     mkdir -p "${EXP_DIR}"
#     export OPENAI_LOGDIR="${EXP_DIR}"

#     echo "[Train][CIFAR10] reg_val=${REG}"
#     CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} \
#     torchrun --standalone --nproc_per_node=${TRAIN_NPROC} --master_port=29601 \
#       scripts/image_train.py \
#         --data_dir "${DATA_DIR}" \
#         --batch_size "${BATCH_SIZE}" \
#         --image_size "${IMG_SIZE}" \
#         --num_channels "${NUM_CH}" \
#         --num_res_blocks "${RES_BLOCKS}" \
#         --dropout "${DROPOUT}" \
#         --learn_sigma True \
#         --diffusion_steps "${DIFF_STEPS}" \
#         --noise_schedule "${NOISE_SCHED}" \
#         --lr "${LR}" \
#         --ema_rate "${EMA_RATE}" \
#         --weight_decay "${WEIGHT_DECAY}" \
#         --lr_anneal_steps "${TOTAL_ITERS}" \
#         --save_interval "${SAVE_FREQ}" \
#         --log_interval 10 \
#         --reg_val "${REG}"
# done

########################################
# 2) SAMPLING
########################################

echo "==== [Stage 2] Sampling from CIFAR-10 models on GPUs ${SAMPLE_GPUS} ===="

# Sampling configuration
NUM_SAMPLES=50000              # typical CIFAR-10 eval
BATCH_SIZE_SAMPLE=512

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

export CUDA_VISIBLE_DEVICES=${SAMPLE_GPUS}
MASTER_PORT=29611

for REG in 0.0; do
  EXP_DIR="${LOG_ROOT}/cifar10_reg${REG}"

  if [ ! -d "${EXP_DIR}" ]; then
      echo "[Sample][CIFAR10] Skip reg=${REG}: missing ${EXP_DIR}" >&2
      continue
  fi

  CKPT=$(ls "${EXP_DIR}" | grep '^ema_' | sort | tail -n1 || true)
  if [ -z "${CKPT}" ]; then
      echo "[Sample][CIFAR10] Skip reg=${REG}: no ema_ checkpoint in ${EXP_DIR}" >&2
      continue
  fi

  CKPT_PATH="${EXP_DIR}/${CKPT}"
  SAMPLE_DIR="${EXP_DIR}/samples"
  mkdir -p "${SAMPLE_DIR}"
  export OPENAI_LOGDIR="${SAMPLE_DIR}"

  echo "[Sample][CIFAR10] reg_val=${REG} from ${CKPT_PATH} -> ${SAMPLE_DIR}"
  torchrun --standalone --nproc_per_node=${SAMPLE_NPROC} --master_port=${MASTER_PORT} \
    scripts/image_sample.py \
      --model_path "${CKPT_PATH}" \
      ${MODEL_FLAGS} \
      ${DIFFUSION_FLAGS} \
      --num_samples "${NUM_SAMPLES}" \
      --batch_size "${BATCH_SIZE_SAMPLE}"
done

########################################
# 3) EVALUATION
########################################

echo "==== [Stage 3] Evaluating CIFAR-10 FID / IS / PRDC ===="

OUT_CSV="${LOG_ROOT}/metrics_cifar10.csv"

if [ ! -f "${OUT_CSV}" ]; then
    echo "dataset,reg_val,exp_dir,ckpt,samples_dir,fid,is_mean,is_std,precision,recall,density,coverage" > "${OUT_CSV}"
fi

for REG in 0.3 0.0; do
    EXP_DIR="${LOG_ROOT}/cifar10_reg${REG}"
    SAMPLES_DIR="${EXP_DIR}/samples"

    if [ ! -d "${SAMPLES_DIR}" ]; then
        echo "[Eval][CIFAR10] Skip reg=${REG}: missing ${SAMPLES_DIR}" >&2
        continue
    fi

    CKPT=$(ls "${EXP_DIR}" | grep '^model' | sort | tail -n1 || true)

    echo "[Eval][CIFAR10] reg_val=${REG}"

    OUTPUT=$(python "${EVAL_SCRIPT}" \
        --real_dir "${REAL_DIR}" \
        --gen_dir "${SAMPLES_DIR}" \
        --image_size "${IMAGE_SIZE}")

    FID=$(echo "${OUTPUT}"         | awk -F': ' '/^FID:/{print $2}')
    IS_LINE=$(echo "${OUTPUT}"     | awk -F': ' '/^Inception Score:/{print $2}')
    IS_MEAN=$(echo "${IS_LINE}"    | awk '{print $1}')
    IS_STD=$(echo "${IS_LINE}"     | awk '{print $3}')
    PREC=$(echo "${OUTPUT}"        | awk -F': ' '/^Precision:/{print $2}')
    REC=$(echo "${OUTPUT}"         | awk -F': ' '/^Recall:/{print $2}')
    DENS=$(echo "${OUTPUT}"        | awk -F': ' '/^Density:/{print $2}')
    COV=$(echo "${OUTPUT}"         | awk -F': ' '/^Coverage:/{print $2}')

    echo "cifar10,${REG},${EXP_DIR},${CKPT},${SAMPLES_DIR},${FID},${IS_MEAN},${IS_STD},${PREC},${REC},${DENS},${COV}" >> "${OUT_CSV}"
done

echo "==== Done (train → sample → eval for CIFAR-10) ===="
