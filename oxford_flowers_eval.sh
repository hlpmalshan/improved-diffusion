#!/bin/bash
# Evaluate Oxford Flowers runs and append FID / IS / PRDC to a CSV
# Assumes:
#  - evaluation script is saved as eval_metrics.py (the code you pasted)
#  - samples are in: ./logs_improved_diffusion_flowers/oxford_flowers_reg*/samples/*.png
#  - real images dir as below
#  - reg_val in {0.0, 0.3}

set -euo pipefail

EVAL_SCRIPT="evaluation.py"
REAL_DIR="/data/diffusion/repos/ddim/ddim_ox_flow/datasets/oxford_flowers/flowers-102/jpg"
LOG_ROOT="/data/diffusion/repos/improved-diffusion/logs_improved_diffusion_flowers"
OUT_CSV="${LOG_ROOT}/metrics_oxford_flowers.csv"

IMAGE_SIZE=64

# Init CSV with header once
if [ ! -f "${OUT_CSV}" ]; then
    echo "dataset,reg_val,exp_dir,ckpt,samples_dir,fid,is_mean,is_std,precision,recall,density,coverage" > "${OUT_CSV}"
fi

for REG in 0.0 0.3; do
    EXP_DIR="${LOG_ROOT}/oxford_flowers_reg${REG}"
    SAMPLES_DIR="${EXP_DIR}/samples"

    if [ ! -d "${EXP_DIR}" ]; then
        echo "Skip reg=${REG}: missing ${EXP_DIR}" >&2
        continue
    fi
    if [ ! -d "${SAMPLES_DIR}" ]; then
        echo "Skip reg=${REG}: missing samples dir ${SAMPLES_DIR}" >&2
        continue
    fi

    # Latest checkpoint (optional, for logging/provenance)
    CKPT=$(ls "${EXP_DIR}" | grep '^model' | sort | tail -n1 || true)

    echo "[Eval] Oxford Flowers reg_val=${REG}"

    OUTPUT=$(python "${EVAL_SCRIPT}" \
        --real_dir "${REAL_DIR}" \
        --gen_dir "${SAMPLES_DIR}" \
        --image_size "${IMAGE_SIZE}")

    # Parse metrics from eval script stdout
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
