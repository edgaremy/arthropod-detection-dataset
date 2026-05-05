#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

read -r -a MODES_ARR <<< "${MODES:-transfer scratch}"
read -r -a SIZES_ARR <<< "${SIZES:-100 500 1000 2000}"
read -r -a FOLDS_ARR <<< "${FOLDS:-0 1 2 3 4}"
read -r -a DEVICE_ARR <<< "${DEVICES:-0 1}"

EPOCHS="${EPOCHS:-20}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
LR0="${LR0:-0.001}"
LRF="${LRF:-0.01}"
FREEZE="${FREEZE:-10}"
BASE_MODEL="${BASE_MODEL:-yolo11l.pt}"
TRANSFER_WEIGHTS="${TRANSFER_WEIGHTS:-runs/arthro_and_flatbug/train/weights/best.pt}"
VAL_COMMON_TEST="${VAL_COMMON_TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

for mode in "${MODES_ARR[@]}"; do
    if [[ "$mode" != "transfer" && "$mode" != "scratch" ]]; then
        echo "Invalid mode: $mode (expected transfer or scratch)" >&2
        exit 1
    fi

    for size in "${SIZES_ARR[@]}"; do
        for fold in "${FOLDS_ARR[@]}"; do
            run_name="${mode}_11l_OOD-split${size}-fold${fold}"
            train_dir="runs/fine_tuning/${run_name}/train"
            best_weights="${train_dir}/weights/best.pt"
            last_weights="${train_dir}/weights/last.pt"

            if [[ "$SKIP_EXISTING" == "1" ]] && [[ -f "$best_weights" || -f "$last_weights" ]]; then
                echo "Skipping mode=$mode size=$size fold=$fold (existing run artifacts found in ${train_dir})"
                continue
            fi

            cmd=(
                python training/fine-tuning/finetune_on_OOD_subset_case.py
                --mode "$mode"
                --size "$size"
                --fold "$fold"
                --epochs "$EPOCHS"
                --warmup-epochs "$WARMUP_EPOCHS"
                --lr0 "$LR0"
                --lrf "$LRF"
                --freeze "$FREEZE"
                --base-model "$BASE_MODEL"
                --transfer-weights "$TRANSFER_WEIGHTS"
                --device "${DEVICE_ARR[@]}"
            )

            if [[ "$VAL_COMMON_TEST" == "1" ]]; then
                cmd+=(--val-common-test)
            fi

            echo "Running mode=$mode size=$size fold=$fold"
            if [[ "$DRY_RUN" == "1" ]]; then
                printf 'DRY_RUN: %q ' "${cmd[@]}"
                echo
            else
                "${cmd[@]}"
            fi
        done
    done
done
