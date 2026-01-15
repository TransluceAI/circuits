#!/bin/bash
set -euo pipefail

# Script to run FC experiments for different dataset families.
# Pairing modes: pair | nopair
# Dataset groups: sva | user_modeling
#
# Examples:
#   ./pair_combined.sh                       # default: pair + sva
#   ./pair_combined.sh --pair-mode nopair --dataset-group user_modeling
#   ./pair_combined.sh --datasets nounpp rc  # subset within current group

PAIR_MODE="pair"
DATASET_GROUP="sva"
RUN_FULL=0
MODEL_OVERRIDE=0
MODEL=""
CONFIG_OVERRIDE=""
OUTPUT_ROOT=""

declare -a EXTRA_DATASETS=()

SVA_DATASETS=("nounpp" "rc" "simple" "within_rc")
USER_DATASETS_BASE=("gender" "country" "occupation" "religion" "obituary")

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [DATASET ...]

Options:
  --pair-mode MODE        pair | nopair (default: pair)
  --dataset-group GROUP   sva | user_modeling (default: sva)
  --model MODEL           Override the default model for the selected group
  --config CONFIG         Override the sweep config yaml
  --output-root PATH      Override the base output directory
  --full                  Run the extended experiment suite
  -h, --help              Show this message

Datasets:
  sva: nounpp rc simple within_rc
  user_modeling: gender country occupation religion obituary
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pair-mode)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --pair-mode"; exit 1; }
            PAIR_MODE="$1"
            ;;
        --dataset-group)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --dataset-group"; exit 1; }
            DATASET_GROUP="$1"
            ;;
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
        --full)
            RUN_FULL=1
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

if [[ "$PAIR_MODE" != "pair" && "$PAIR_MODE" != "nopair" ]]; then
    echo "Invalid pair mode: $PAIR_MODE" >&2
    exit 1
fi

if [[ "$DATASET_GROUP" != "sva" && "$DATASET_GROUP" != "user_modeling" ]]; then
    echo "Invalid dataset group: $DATASET_GROUP" >&2
    exit 1
fi

# Determine datasets for this run
declare -a AVAILABLE_DATASETS=()
declare -a DEFAULT_DATASETS=()
if [[ "$DATASET_GROUP" == "sva" ]]; then
    AVAILABLE_DATASETS=("${SVA_DATASETS[@]}")
    DEFAULT_DATASETS=("${SVA_DATASETS[@]}")
else
    if [[ "$PAIR_MODE" == "pair" ]]; then
        DEFAULT_DATASETS=("${USER_DATASETS_BASE[@]}")
    else
        DEFAULT_DATASETS=("${USER_DATASETS_BASE[@]}")
    fi
    AVAILABLE_DATASETS=("${DEFAULT_DATASETS[@]}" "${USER_DATASETS_BASE[@]/%/_nopair}")
fi

map_dataset_name() {
    local name="$1"
    local valid=1
    local -a allowed
    if [[ "$DATASET_GROUP" == "user_modeling" ]]; then
        allowed=("${USER_DATASETS_BASE[@]}" "${USER_DATASETS_BASE[@]/%/_nopair}")
    else
        allowed=("${SVA_DATASETS[@]}")
    fi
    for candidate in "${allowed[@]}"; do
        if [[ "$candidate" == "$name" ]]; then
            valid=0
            break
        fi
    done
    if [[ $valid -ne 0 ]]; then
        echo "Invalid dataset '$name' for group '$DATASET_GROUP'" >&2
        exit 1
    fi
    if [[ "$DATASET_GROUP" == "user_modeling" ]]; then
        local base="${name%_nopair}"
        if [[ "$PAIR_MODE" == "pair" ]]; then
            echo "$base"
        else
            echo "${base}_nopair"
        fi
    else
        echo "$name"
    fi
}

declare -a SELECTED_DATASETS=()
if [[ ${#EXTRA_DATASETS[@]} -gt 0 ]]; then
    for ds in "${EXTRA_DATASETS[@]}"; do
        SELECTED_DATASETS+=("$(map_dataset_name "$ds")")
    done
else
    SELECTED_DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# Determine config path and default model
case "${DATASET_GROUP}_${PAIR_MODE}" in
    sva_pair)
        CONFIG_PATH="sweep/fc_evaluation/sva_pair.yaml"
        DEFAULT_MODEL="meta-llama/Llama-3.1-8B"
        ;;
    sva_nopair)
        CONFIG_PATH="sweep/fc_evaluation/sva_nopair.yaml"
        DEFAULT_MODEL="meta-llama/Llama-3.1-8B"
        ;;
    user_modeling_pair)
        CONFIG_PATH="sweep/fc_evaluation/user_modeling_pair.yaml"
        DEFAULT_MODEL="meta-llama/Llama-3.1-8B-Instruct"
        ;;
    user_modeling_nopair)
        CONFIG_PATH="sweep/fc_evaluation/user_modeling_nopair.yaml"
        DEFAULT_MODEL="meta-llama/Llama-3.1-8B-Instruct"
        ;;
    *)
        echo "Unsupported configuration combination." >&2
        exit 1
        ;;
esac

if [[ -n "$CONFIG_OVERRIDE" ]]; then
    CONFIG_PATH="$CONFIG_OVERRIDE"
fi

if [[ $MODEL_OVERRIDE -eq 0 ]]; then
    MODEL="$DEFAULT_MODEL"
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="${ARTIFACTS_DIR}/results"
fi

RESULTS_ROOT="${OUTPUT_ROOT}/${DATASET_GROUP}/${PAIR_MODE}"

COMMON_BASE_ARGS="--config ${CONFIG_PATH} \
    --method nap \
    --num_train_examples 300 \
    --model ${MODEL} \
    --force_eval"

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

run_extended_experiments() {
    local dataset="$1"
    local base_args="$2"
    run_experiment "$dataset" "Stop grad + RelP grad (MLP acts)" "--steps 1 --use_stop_grad --use_relp_grad --use_neurons --use_mlp_acts" "$base_args"
    run_experiment "$dataset" "IG baseline (MLP acts)" "--steps 10 --use_neurons --use_mlp_acts --disable_stop_grad" "$base_args"
    run_experiment "$dataset" "Stop grad + RelP grad (resid.)" "--steps 1 --use_stop_grad --use_relp_grad --use_neurons --submodule_types resid" "$base_args"
    run_experiment "$dataset" "Stop grad + RelP grad + SAE (resid.)" "--steps 1 --use_stop_grad --use_relp_grad --auc_test_handle_errors default --submodule_types resid" "$base_args"
    run_experiment "$dataset" "IG baseline (attn.)" "--steps 10 --use_neurons --effect_method ig --submodule_types attn --disable_stop_grad" "$base_args"
    run_experiment "$dataset" "IG baseline (SAE + attn.)" "--steps 10 --effect_method ig --submodule_types attn --auc_test_handle_errors default --disable_stop_grad" "$base_args"
    run_experiment "$dataset" "IG baseline (resid.)" "--steps 10 --use_neurons --effect_method ig --submodule_types resid --disable_stop_grad" "$base_args"
    run_experiment "$dataset" "IG baseline (SAE + resid.)" "--steps 10 --effect_method ig --submodule_types resid --auc_test_handle_errors default --disable_stop_grad" "$base_args"
    run_experiment "$dataset" "IG baseline (transcoder)" "--steps 10 --effect_method ig --use_transcoder" "$base_args"
    # run_experiment "$dataset" "IG-inputs baseline" "--steps 10 --use_neurons --use_mlp_acts --effect_method ig-inputs --disable_stop_grad" "$base_args"
    run_experiment "$dataset" "Stop grad + RelP grad + IG" "--steps 10 --use_stop_grad --use_relp_grad --use_neurons --use_mlp_acts" "$base_args"
    run_experiment "$dataset" "Stop grad + RelP grad + SAE" "--steps 1 --use_stop_grad --use_relp_grad --auc_test_handle_errors default" "$base_args"
    run_experiment "$dataset" "Stop grad + RelP grad (no half rule)" "--steps 1 --use_stop_grad --use_relp_grad --use_neurons --use_mlp_acts --disable_half_rule" "$base_args"
    run_experiment "$dataset" "Delta" "--steps 1 --effect_method delta --use_neurons --use_mlp_acts" "$base_args"
    run_experiment "$dataset" "Stop grad (MLP acts)" "--steps 1 --use_stop_grad --use_neurons --use_mlp_acts" "$base_args"
    run_experiment "$dataset" "Stop grad + Shapley grad (MLP acts)" "--steps 1 --use_stop_grad --use_shapley_grad --use_neurons --use_mlp_acts" "$base_args"
    run_experiment "$dataset" "IG baseline (MLP outputs)" "--steps 10 --use_neurons --disable_stop_grad" "$base_args"
    run_experiment "$dataset" "Stop grad (MLP outputs)" "--steps 1 --use_stop_grad --use_neurons" "$base_args"
    run_experiment "$dataset" "Stop grad + Shapley grad (MLP outputs)" "--steps 1 --use_stop_grad --use_shapley_grad --use_neurons" "$base_args"
    run_experiment "$dataset" "Stop grad + RelP grad (MLP outputs)" "--steps 1 --use_stop_grad --use_relp_grad --use_neurons" "$base_args"
}

run_dataset() {
    local dataset="$1"
    local save_dir="${RESULTS_ROOT}/${dataset}"
    local base_args="$COMMON_BASE_ARGS --dataset $dataset --save_path ${save_dir}"

    run_experiment "$dataset" "IG baseline (SAE output, default errors)" "--steps 10 --auc_test_handle_errors default --disable_stop_grad" "$base_args"

    if [[ $RUN_FULL -eq 1 ]]; then
        run_extended_experiments "$dataset" "$base_args"
    fi
}

mkdir -p "$RESULTS_ROOT"
for dataset in "${SELECTED_DATASETS[@]}"; do
    run_dataset "$dataset"
done
