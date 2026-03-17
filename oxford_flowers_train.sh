#!/bin/bash
# Distributed training for improved diffusion on Oxford Flowers
# Assumes `image_train.py` accepts a `--reg_val` argument

set -euo pipefail

DATA_DIR="/data/diffusion/repos/ddim/ddim_ox_flow/datasets/oxford_flowers/flowers-102/jpg"
LOG_ROOT="/data/diffusion/repos/improved-diffusion/logs_improved_diffusion_flowers"
mkdir -p "${LOG_ROOT}"

# Hyperparameters
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
TOTAL_ITERS=51000          # lr_anneal_steps
SAVE_FREQ=20400            # checkpoint/save interval

for REG in 0.3 0.0; do
    EXP_DIR="${LOG_ROOT}/oxford_flowers_reg${REG}"
    mkdir -p "${EXP_DIR}"
    export OPENAI_LOGDIR="${EXP_DIR}"

    echo "Training improved diffusion with reg_val=${REG}"
    # distributed run on GPUs 0,1,2
    CUDA_VISIBLE_DEVICES=2,5 \
    torchrun --standalone --nproc_per_node=2 --master_port=29501 \
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
