"""
Train to get circuits for a given model and dataset.

Usage:
    python evaluate.py --config sweep/simple.yaml
"""

import hashlib
import json
import os

import torch
import torch as t
import yaml
from args.args import get_args, make_save_path, print_args
from circuits.utils.constants import (
    DATASET_TASK_MAPPING,
    DTYPE_MAPPING,
    INCLUDE_EMBED_MAPPING,
    METHOD_MAPPING,
    N_LAYERS_MAPPING,
    PARALLEL_ATTN_MAPPING,
    START_LAYER_MAPPING,
    THRESHOLD_RANGE_MAPPING,
)
from circuits.utils.data_loading_utils import load_examples_helper
from nnsight import LanguageModel


def main():
    # Parse arguments
    args = get_args()

    # Print parsed arguments
    print_args(args)

    # Induce circuit save path based on the config.
    save_path = make_save_path(args)
    if not os.path.exists(save_path):
        raise ValueError(f"Save path {save_path} does not exist.")

    # check if all metrics to report have jsons
    all_exists = True
    for metric in args.metrics_to_report:
        if not os.path.exists(os.path.join(save_path, f"{metric}.json")):
            all_exists = False
            break
    if args.force_eval:
        all_exists = False
    if all_exists:
        print("All json files found for metrics to report, skipping evaluation...")
        return

    start_layer = START_LAYER_MAPPING[args.model]
    n_layers = N_LAYERS_MAPPING[args.model]
    parallel_attn = PARALLEL_ATTN_MAPPING[args.model]
    include_embed = INCLUDE_EMBED_MAPPING[args.model]
    dtype = DTYPE_MAPPING[args.model]

    # Load model.
    model = LanguageModel(
        args.model,
        device_map=args.device,
        dispatch=True,
        attn_implementation="eager",
        torch_dtype=dtype,
    )

    # Init method, and load circuit.
    method = METHOD_MAPPING[args.method](
        model,
        args,
        mode="eval",
        n_layers=n_layers,
        dtype=dtype,
        include_embed=include_embed,
        parallel_attn=parallel_attn,
    )
    method.load(
        os.path.join(save_path, "train.pt"),
        start_layer=start_layer,
        use_weight_based_nodes=args.use_weight_based_nodes,
    )

    # Load examples.
    dataset_path = os.path.join(args.data_path, f"{args.dataset}_test.json")
    examples = load_examples_helper(
        args.dataset,
        dataset_path,
        args.num_auc_test_examples,
        model,
        use_min_length_only=True if DATASET_TASK_MAPPING[args.dataset] == "sva" else False,
        allow_length_mismatch=False if DATASET_TASK_MAPPING[args.dataset] == "sva" else True,
    )

    if examples is None:
        raise ValueError(f"Failed to load examples from {dataset_path}")

    # Print out randomly sampled 3 examples.
    print("Randomly sampled example:")
    print(examples[0])
    print("=" * 50)

    hash_str = (
        args.dataset
        + args.model
        + str([s.name for s in method.submodules])
        + str(THRESHOLD_RANGE_MAPPING[args.model])
        + str(args.auc_test_handle_errors)
        + str(args.use_neurons)
        + str(args.auc_test_random)
        + str(args.num_auc_test_examples)
        + str(args.suffix_length)
    )
    hash = hashlib.md5(hash_str.encode()).hexdigest()

    # inference faithfulness and completeness, and evaluate the results.
    if "auc" in args.metrics_to_report:
        print("Inferring faithfulness and completeness...")

        fc_sub_path = os.path.join(save_path, "faithfulness_and_completeness")
        if not os.path.exists(fc_sub_path):
            os.makedirs(fc_sub_path)

        # save eval config
        with open(os.path.join(fc_sub_path, f"{hash}.yaml"), "w") as f:
            yaml.dump(args.__dict__, f)

        # faithfulness and completeness
        if os.path.exists(os.path.join(fc_sub_path, f"{hash}.pt")) and not args.force_eval:
            print("Loading faithfulness and completeness results from disk...")
            faithfulness_and_completeness_results = t.load(os.path.join(fc_sub_path, f"{hash}.pt"))
        else:
            print("Inferring faithfulness and completeness...")
            faithfulness_and_completeness_results = method.inference_faithfulness_and_completeness(
                examples,
                thresholds=THRESHOLD_RANGE_MAPPING[args.model],
                component=args.component,
            )
            t.save(
                faithfulness_and_completeness_results,
                os.path.join(fc_sub_path, f"{hash}.pt"),
            )

        # write as json
        with open(os.path.join(fc_sub_path, f"{hash}.json"), "w") as f:
            json.dump(faithfulness_and_completeness_results, f)

    if "steering" in args.metrics_to_report:
        print("Inferring steering...")

        fc_sub_path = os.path.join(save_path, "steering")
        if not os.path.exists(fc_sub_path):
            os.makedirs(fc_sub_path)

        # save eval config
        with open(os.path.join(fc_sub_path, f"{hash}.yaml"), "w") as f:
            yaml.dump(args.__dict__, f)

        # steering at specific sites
        if os.path.exists(os.path.join(fc_sub_path, f"{hash}_steering.pt")):
            print("Loading steering results from disk...")
            steering_results = t.load(os.path.join(fc_sub_path, f"{hash}_steering.pt"))
        else:
            print("Inferring steering...")
            steering_results = method.inference_steering(
                examples, thresholds=THRESHOLD_RANGE_MAPPING[args.model]
            )
            t.save(
                steering_results,
                os.path.join(fc_sub_path, f"{hash}_steering.pt"),
            )

        # write as json
        with open(os.path.join(fc_sub_path, f"{hash}_steering.json"), "w") as f:
            json.dump(steering_results, f)


if __name__ == "__main__":
    main()
