#!/bin/bash

# Check if required arguments are provided
if [ -z "$1" ]; then
    echo "Error: Dataset argument required"
    echo "Usage: [CUDA_VISIBLE_DEVICES=0,1,2,3] $0 <dataset> [effect_method] [--use_stop_grad_on_mlps]"
    echo "Example: $0 within_rc jvp"
    echo "Example: $0 within_rc jvp-ig-inputs"
    echo "Example: CUDA_VISIBLE_DEVICES=0,1,2,3 $0 within_rc jvp-ig-inputs"
    exit 1
fi

DATASET="$1"
EFFECT_METHOD="${2:-jvp}"  # Default to "jvp" if not provided

# Determine IG steps based on effect method
IG_ARGS=""
if [[ "$EFFECT_METHOD" == *"ig-inputs"* ]] || [[ "$EFFECT_METHOD" == *"conductance"* ]]; then
    IG_ARGS="--ig_steps 5"
fi

# Check for --use_stop_grad_on_mlps flag
STOP_GRAD_MLPS_ARG=""
if [[ "$3" == "--use_stop_grad_on_mlps" ]]; then
    STOP_GRAD_MLPS_ARG="--use_stop_grad_on_mlps"
fi

# Determine number of GPUs from CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=1
else
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPU_ARRAY[@]}
fi

# Determine training command based on number of GPUs
if [ "$NUM_GPUS" -gt 1 ]; then
    TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS"
    echo "Using $NUM_GPUS GPUs for training"
else
    TRAIN_CMD="python"
fi

# For eval, always use first GPU only
EVAL_GPU="${GPU_ARRAY[0]:-0}"

# Shared base arguments (without save_path)
BASE_ARGS="--config sweep/fc_evaluation/sva.yaml \
    --method enap \
    --num_train_examples 300"

# Helper function to run train and eval with given arguments
run_experiment() {
    local description="$1"
    local dataset="$2"
    local extra_args="$3"

    # Construct dataset-specific save path
    local save_path="${ARTIFACTS_DIR}/enap_results/${dataset}_300"

    echo "# $description"
    echo "$TRAIN_CMD scripts/train.py $BASE_ARGS --save_path $save_path --dataset $dataset $extra_args"
    $TRAIN_CMD scripts/train.py $BASE_ARGS --save_path $save_path --dataset $dataset $extra_args

    echo "CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval.py $BASE_ARGS --save_path $save_path --dataset $dataset $extra_args"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval.py $BASE_ARGS --save_path $save_path --dataset $dataset $extra_args --component edges
    echo
}

run_experiment "$EFFECT_METHOD relp" "$DATASET" "--effect_method $EFFECT_METHOD $IG_ARGS $STOP_GRAD_MLPS_ARG --use_mlp_acts --use_stop_grad --use_relp_grad --edge_threshold 0.00 --topk_neurons 1000 --topk_edges 500000 --batch_size 1"
