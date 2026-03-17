#!/bin/bash
# End-to-end pipeline for Oxford Flowers on improved-diffusion:
# 1) Train models for reg_val in {0.3, 0.0}
# 2) Sample images from trained EMA checkpoints
# 3) Evaluate FID / IS / PRDC and append to CSV

set -euo pipefail

########################################
# GPU SETTINGS (EDIT HERE)
########################################

# GPUs used for training (comma-separated)
TRAIN_GPUS="2,5"
TRAIN_NPROC=2              # number of processes to match number of GPUs

# GPUs used for sampling (comma-separated)
SAMPLE_GPUS="2"
SAMPLE_NPROC=1             # usually 1 for single-GPU sampling

########################################
# COMMON PATHS / SETTINGS
########################################

DATA_DIR="/data/diffusion/repos/ddim/ddim_ox_flow/datasets/oxford_flowers/flowers-102/jpg"
LOG_ROOT="/data/diffusion/repos/improved-diffusion/logs_improved_diffusion_flowers"
EVAL_SCRIPT="evaluation.py"
REAL_DIR="${DATA_DIR}"
IMAGE_SIZE=64

mkdir -p "${LOG_ROOT}"

########################################
# 1) TRAINING
########################################

BATCH_SIZE=64
IMG_SIZE=64
NUM_CH=64
RES_BLOCKS=2
DROPOUT=0.1
DIFF_STEPS=1000
NOISE_SCHED="cosine"
LR=5e-5
EMA_RATE=0.999
WEIGHT_DECAY=0.0
TOTAL_ITERS=51000
SAVE_FREQ=20400

echo "==== [Stage 1] Training models (reg=0.3, 0.0) on GPUs ${TRAIN_GPUS} ===="

for REG in 0.3 0.0; do
    EXP_DIR="${LOG_ROOT}/oxford_flowers_reg${REG}"
    mkdir -p "${EXP_DIR}"
    export OPENAI_LOGDIR="${EXP_DIR}"

    echo "[Train] reg_val=${REG}"
    CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} \
    torchrun --standalone --nproc_per_node=${TRAIN_NPROC} --master_port=29501 \
      scripts/image_train.py \
        --data_dir "${DATA_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --image_size "${IMG_SIZE}" \
        --num_channels "${NUM_CH}" \
        --num_res_blocks "${RES_BLOCKS}" \
        --dropout "${DROPOUT}" \
        --learn_sigma True \
        --diffusion_steps "${DIFF_STEPS}" \
        --noise_schedule "${NOISE_SCHED}" \
        --lr "${LR}" \
        --ema_rate "${EMA_RATE}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --lr_anneal_steps "${TOTAL_ITERS}" \
        --save_interval "${SAVE_FREQ}" \
        --log_interval 10 \
        --reg_val "${REG}"
done

########################################
# 2) SAMPLING
########################################

echo "==== [Stage 2] Sampling from trained models on GPUs ${SAMPLE_GPUS} ===="

NUM_SAMPLES=8200
BATCH_SIZE_SAMPLE=512
MODEL_FLAGS="--image_size 64 --num_channels 64 --num_res_blocks 2 --dropout 0.1 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

export CUDA_VISIBLE_DEVICES=${SAMPLE_GPUS}
MASTER_PORT=29511

for REG in 0.0 0.3; do
  EXP_DIR="${LOG_ROOT}/oxford_flowers_reg${REG}"
  CKPT=$(ls "${EXP_DIR}" | grep '^ema_' | sort | tail -n1 || true)
  if [ -z "${CKPT}" ]; then
      echo "[Sample] Skip reg=${REG}: no ema_ checkpoint" >&2
      continue
  fi
  CKPT_PATH="${EXP_DIR}/${CKPT}"
  SAMPLE_DIR="${EXP_DIR}/samples"
  mkdir -p "${SAMPLE_DIR}"
  export OPENAI_LOGDIR="${SAMPLE_DIR}"

  echo "[Sample] reg_val=${REG} from ${CKPT_PATH} -> ${SAMPLE_DIR}"
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

echo "==== [Stage 3] Evaluating FID / IS / PRDC ===="

OUT_CSV="${LOG_ROOT}/metrics_oxford_flowers.csv"

if [ ! -f "${OUT_CSV}" ]; then
    echo "dataset,reg_val,exp_dir,ckpt,samples_dir,fid,is_mean,is_std,precision,recall,density,coverage" > "${OUT_CSV}"
fi

for REG in 0.0 0.3; do
    EXP_DIR="${LOG_ROOT}/oxford_flowers_reg${REG}"
    SAMPLES_DIR="${EXP_DIR}/samples"
    CKPT=$(ls "${EXP_DIR}" | grep '^model' | sort | tail -n1 || true)

    if [ ! -d "${SAMPLES_DIR}" ]; then
        echo "[Eval] Skip reg=${REG}: missing ${SAMPLES_DIR}" >&2
        continue
    fi

    echo "[Eval] Oxford Flowers reg_val=${REG}"

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

    echo "oxford_flowers,${REG},${EXP_DIR},${CKPT},${SAMPLES_DIR},${FID},${IS_MEAN},${IS_STD},${PREC},${REC},${DENS},${COV}" >> "${OUT_CSV}"
done

echo "==== Done (train → sample → eval for Oxford Flowers) ===="
