#!/bin/bash
set -euo pipefail

MODEL_OVERRIDE=0
MODEL="google/gemma-2-9b"
CONFIG_OVERRIDE=""
OUTPUT_ROOT=""
WIDTH="131k"

declare -a EXTRA_DATASETS=()

SVA_DATASETS=("nounpp" "rc" "simple" "within_rc")

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [DATASET ...]

Options:
  --model MODEL           Override the default model (default: google/gemma-2-9b)
  --config CONFIG         Override the sweep config yaml
  --output-root PATH      Override the base output directory
  --width WIDTH           SAE width: 16k | 65k | 131k (default: 16k)
  -h, --help              Show this message

Datasets: nounpp rc simple within_rc
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --model"; exit 1; }
            MODEL="$1"
            MODEL_OVERRIDE=1
            ;;
        --config)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --config"; exit 1; }
            CONFIG_OVERRIDE="$1"
            ;;
        --output-root)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --output-root"; exit 1; }
            OUTPUT_ROOT="$1"
            ;;
        --width)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --width"; exit 1; }
            WIDTH="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            EXTRA_DATASETS+=("$1")
            ;;
    esac
    shift
done

if [[ "$WIDTH" != "16k" && "$WIDTH" != "65k" && "$WIDTH" != "131k" ]]; then
    echo "Invalid width: $WIDTH (must be 16k, 65k, or 131k)" >&2
    exit 1
fi

declare -a SELECTED_DATASETS=()
if [[ ${#EXTRA_DATASETS[@]} -gt 0 ]]; then
    for ds in "${EXTRA_DATASETS[@]}"; do
        valid=0
        for candidate in "${SVA_DATASETS[@]}"; do
            if [[ "$candidate" == "$ds" ]]; then
                valid=1
                break
            fi
        done
        if [[ $valid -eq 0 ]]; then
            echo "Invalid dataset '$ds'" >&2
            exit 1
        fi
        SELECTED_DATASETS+=("$ds")
    done
else
    SELECTED_DATASETS=("${SVA_DATASETS[@]}")
fi

CONFIG_PATH="sweep/fc_evaluation/sva_pair.yaml"

if [[ -n "$CONFIG_OVERRIDE" ]]; then
    CONFIG_PATH="$CONFIG_OVERRIDE"
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="${ARTIFACTS_DIR}/nap_results_${WIDTH}/"
fi

RESULTS_ROOT="${OUTPUT_ROOT}/sva/pair"

COMMON_BASE_ARGS="--config ${CONFIG_PATH} \
    --method nap \
    --num_train_examples 300 \
    --model ${MODEL} \
    --width ${WIDTH}"

run_experiment() {
    local dataset="$1"
    local description="$2"
    local extra_args="$3"
    local base_args="$4"

    echo "# [$dataset] $description"
    echo "python scripts/train.py $base_args $extra_args"
    python scripts/train.py $base_args $extra_args

    echo "python scripts/eval.py $base_args $extra_args"
    python scripts/eval.py $base_args $extra_args
    echo
}


run_dataset() {
    local dataset="$1"
    local save_dir="${RESULTS_ROOT}/${dataset}"
    local base_args="$COMMON_BASE_ARGS --dataset $dataset --save_path ${save_dir}"

    run_experiment "$dataset" "IG baseline (MLP acts)" "--steps 10 --use_neurons --use_mlp_acts --disable_stop_grad" "$base_args"
    run_experiment "$dataset" "IG baseline (resid.)" "--steps 10 --use_neurons --effect_method ig --submodule_types resid --disable_stop_grad" "$base_args"
    run_experiment "$dataset" "IG baseline (SAE + resid.)" "--steps 10 --effect_method ig --submodule_types resid --auc_test_handle_errors default --disable_stop_grad" "$base_args"
    run_experiment "$dataset" "IG baseline (SAE output, default errors)" "--steps 10 --auc_test_handle_errors default --disable_stop_grad" "$base_args"
    run_experiment "$dataset" "IG baseline (MLP outputs)" "--steps 10 --use_neurons --disable_stop_grad" "$base_args"
}

mkdir -p "$RESULTS_ROOT"
for dataset in "${SELECTED_DATASETS[@]}"; do
    run_dataset "$dataset"
done
