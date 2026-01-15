import argparse
import os

import yaml


def print_args(args):
    print("Training Configuration:")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Effect method: {args.effect_method}")
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    print(f"Dictionary path: {args.dict_path}")
    print(f"Save path: {args.save_path}")
    print(f"Number of examples: {args.num_train_examples}")
    print(f"Steps: {args.steps}")
    print(f"Node threshold: {args.node_threshold}")
    print(f"Edge threshold: {args.edge_threshold}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Use neurons: {args.use_neurons}")
    print(f"No pair: {args.nopair}")
    print(f"Nodes only: {args.nodes_only}")
    print(f"Suffix length: {args.suffix_length}")
    print(f"Use Shapley gradient: {args.use_shapley_grad}")
    print(f"Use Shapley qk: {args.use_shapley_qk}")
    print(f"Use stop grad: {args.use_stop_grad}")
    print(f"Use relp grad: {args.use_relp_grad}")
    print(f"Disable half rule: {args.disable_half_rule}")
    print(f"Use stop grad on MLPs: {args.use_stop_grad_on_mlps}")
    print(f"Disable stop grad: {args.disable_stop_grad}")
    print(f"Top-k neurons: {args.topk_neurons}")
    print(f"Top-k edges: {args.topk_edges}")
    print(f"Edge weight type: {args.edge_weight_type}")
    print(f"IG steps: {args.ig_steps}")
    print(f"Force eval: {args.force_eval}")
    print("=" * 50)


def make_save_path(args):
    save_base = (
        f"{args.model.split('/')[-1]}_{args.dataset}_N{args.num_train_examples}_AGG{args.aggregation}_M{args.method}"
        + (f"_EFFECT{args.effect_method.upper()}" if args.effect_method != "ig" else "")
        + ("_USE_NEURONS" if args.use_neurons else "")
        + ("_USE_MLPACTS" if args.use_mlp_acts else "")
        + ("_USE_SHAPLEY_GRAD" if args.use_shapley_grad else "")
        + ("_USE_SHAPLEY_QK" if args.use_shapley_qk else "")
        + ("_USE_STOP_GRAD" if args.use_stop_grad else "")
        + ("_USE_RELP_GRAD" if args.use_relp_grad else "")
        + ("_USE_STOP_GRAD_ON_MLPS" if args.use_stop_grad_on_mlps else "")
        + ("_DISABLE_HALF_RULE" if args.disable_half_rule else "")
        + ("_DISABLE_STOP_GRAD" if args.disable_stop_grad else "")
        + ("_EDGE_THRESHOLD" + str(args.edge_threshold))
        + ("_TOPK_NEURONS" + str(args.topk_neurons))
        + ("_TOPK_EDGES" + str(args.topk_edges) if args.topk_edges is not None else "")
        + (
            f"_EDGE_WEIGHT_{args.edge_weight_type.upper()}"
            if args.edge_weight_type != "final_attr"
            else ""
        )
        + (f"_IG_STEPS{args.ig_steps}" if args.ig_steps is not None else "")
        + (
            "_HANDLE_" + args.auc_test_handle_errors.upper()
            if args.auc_test_handle_errors != "default"
            else ""
        )
        + (
            f"_STEPS{args.steps}"
            if (args.steps != 10 and not args.use_stop_grad)
            or (args.steps != 1 and args.use_stop_grad)
            else ""
        )
        + (
            f"_SUBMODULES_{','.join(args.submodule_types)}"
            if args.submodule_types != ["mlp"]
            else ""
        )
        + ("_USE_TRANSCODER" if args.use_transcoder else "")
        + (f"_WIDTH{args.width}" if args.width != "8x" else "")
    )
    return os.path.join(args.save_path, f"{save_base}")


def get_args() -> argparse.Namespace:
    """Parse command line arguments with YAML config support."""
    parser = argparse.ArgumentParser(description="Training script arguments")

    # Config file argument
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    #########################################################
    # Training arguments
    #########################################################

    # Model argument (string)
    parser.add_argument("--model", type=str, default="default_model", help="Model name or path")

    # Method argument (string)
    parser.add_argument("--method", type=str, default="clso", help="Method name")

    # Effect method argument (string) - for NAP method
    parser.add_argument(
        "--effect_method",
        type=str,
        default="ig",
        choices=[
            "ig",
            "random",
            "delta",
            "jvp",
            "clso",
            "jvp-ig-inputs",
            "jvp-conductance",
            "clso-ig-inputs",
            "clso-conductance",
        ],
        help="Effect computation method: 'ig', 'random', 'delta', 'jvp', 'clso', or with IG modes like 'jvp-ig-inputs', 'jvp-conductance'",
    )

    # Dataset argument (string)
    parser.add_argument(
        "--dataset", type=str, default="default_dataset", help="Dataset name or path"
    )

    # Data path (string)
    parser.add_argument("--data_path", type=str, default="data", help="Data path")

    # Dictionary argument (string)
    parser.add_argument(
        "--dict_path", type=str, default="default_dict_path", help="Path to dictionary"
    )

    # Save path (string)
    parser.add_argument("--save_path", type=str, default="circuits", help="Save path")

    # Optional method load path for eval
    parser.add_argument(
        "--method_load_path", type=str, default=None, help="Optional method load path for eval"
    )

    # Number of examples (integer)
    parser.add_argument(
        "--num_train_examples", type=int, default=100, help="Number of training examples"
    )

    # Steps (integer)
    parser.add_argument(
        "--steps", type=int, default=10, help="Steps for IG (should be 1 for linearised methods)"
    )

    # IG steps for integrated gradients (integer, optional)
    parser.add_argument(
        "--ig_steps",
        type=int,
        default=None,
        help="Number of integration steps for IG attribution (None means use regular gradients)",
    )

    # Node threshold (float)
    parser.add_argument("--node_threshold", type=float, default=0.2, help="Node threshold value")

    # Edge threshold (float)
    parser.add_argument("--edge_threshold", type=float, default=0.02, help="Edge threshold value")

    # Aggregation (string)
    parser.add_argument("--aggregation", type=str, default="sum", help="Aggregation method")

    # Use neurons (boolean)
    parser.add_argument("--use_neurons", action="store_true", help="Use neurons")

    # Batch size (integer)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    # Seed (integer)
    parser.add_argument("--seed", type=int, default=12, help="Seed")

    # Device (string)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")

    # Pair data or not (boolean)
    parser.add_argument("--nopair", action="store_true", help="Pair data or not")

    # Nodes only or not (boolean)
    parser.add_argument("--nodes_only", action="store_true", help="Nodes only or not")

    # Use subject model or not (boolean)
    parser.add_argument("--use_subject_model", action="store_true", help="Use subject model or not")

    # Suffix length (integer)
    parser.add_argument("--suffix_length", type=int, default=None, help="Suffix length")

    # Submodule types to include (list of strings)
    parser.add_argument(
        "--submodule_types",
        type=str,
        nargs="*",
        default=["embed", "attn", "mlp", "resid"],
        help="List of submodule types to include for attribution (space-separated: --submodule_types embed attn mlp resid)",
    )
    parser.add_argument(
        "--use_mlp_acts", action="store_true", help="Use MLP activations for attribution"
    )
    parser.add_argument("--use_transcoder", action="store_true", help="Use transcoder")
    parser.add_argument("--width", type=str, default="8x", help="Width")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument(
        "--force_eval",
        action="store_true",
        help="Force evaluation even if cached metrics exist",
    )

    # Our tracing methods specific arguments
    parser.add_argument(
        "--use_shapley_grad", action="store_true", help="Use Shapley gradient for MLP gate"
    )
    parser.add_argument(
        "--use_relp_grad", action="store_true", help="Use RelP gradient for MLP gate"
    )
    parser.add_argument("--use_shapley_qk", action="store_true", help="Use Shapley qk")
    parser.add_argument(
        "--use_stop_grad", action="store_true", help="Use stop gra on non-linear module"
    )
    parser.add_argument(
        "--use_stop_grad_on_mlps", action="store_true", help="Use stop grad on MLPs"
    )
    parser.add_argument("--disable_half_rule", action="store_true", help="Don't use half rule")
    parser.add_argument("--disable_stop_grad", action="store_true", help="Disable stop grad")
    parser.add_argument("--topk_neurons", type=int, default=100, help="Top-k neurons")
    parser.add_argument("--topk_edges", type=int, default=None, help="Top-k edges")
    parser.add_argument(
        "--edge_weight_type",
        type=str,
        default="final_attr",
        choices=["final_attr", "weight"],
        help="Edge weight type: 'final_attr' (final attribution) or 'weight' (raw weight)",
    )

    # Use edges or not (boolean)
    parser.add_argument("--component", type=str, default="nodes", help="Use edges or not")

    # Use weight based nodes or not (boolean)
    parser.add_argument(
        "--use_weight_based_nodes", action="store_true", help="Use weight based nodes or not"
    )

    #########################################################
    # List arguments
    #########################################################

    # Component types (list of strings) - space-separated: --components attn mlp embed
    parser.add_argument(
        "--metrics_to_report",
        type=str,
        nargs="*",
        default=[],
        help="List of metrics to report (space-separated: --metrics_to_report auc steering)",
    )

    #########################################################
    # AUC evaluation arguments
    #########################################################

    # Number of test examples (integer)
    parser.add_argument(
        "--num_auc_test_examples", type=int, default=40, help="Number of test examples"
    )

    # Test random or not (boolean)
    parser.add_argument("--auc_test_random", action="store_true", help="Test random or not")

    # Test handle errors or not (string)
    parser.add_argument(
        "--auc_test_handle_errors", type=str, default="default", help="Test handle errors or not"
    )

    # Test ablation type (string)
    parser.add_argument(
        "--auc_test_ablation_type", type=str, default="mean", help="Test ablation type"
    )

    #########################################################
    # Steering evaluation arguments
    #########################################################

    # Number of test examples (integer)

    # Steering intervention type (string)
    parser.add_argument(
        "--steering_intervention_type",
        type=str,
        default="addition",
        help="Steering intervention type",
    )

    # Steering batch size (integer)
    parser.add_argument("--steering_batch_size", type=int, default=10, help="Steering batch size")

    # Steering output length (integer)
    parser.add_argument(
        "--steering_output_length", type=int, default=128, help="Steering output length"
    )

    # Steering top-k nodes (integer)
    parser.add_argument("--steering_topk_nodes", type=int, default=10, help="Steering top-k nodes")

    # Steering top-k edges (integer)
    parser.add_argument("--steering_topk_edges", type=int, default=10, help="Steering top-k edges")

    # Steering factors (list of floats)
    parser.add_argument(
        "--steering_factors",
        type=float,
        nargs="*",
        default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0],
        help="List of steering factors (space-separated: --steering_factors 0.2 0.4 0.6)",
    )

    # Steering number of examples (integer)
    parser.add_argument(
        "--steering_num_of_examples", type=int, default=10, help="Steering number of examples"
    )

    # Steering split ratio (float)
    parser.add_argument(
        "--steering_split_ratio", type=float, default=0.5, help="Steering split ratio"
    )

    # LM judge model (string)
    parser.add_argument("--lm_judge_model", type=str, default="gpt-4o-mini", help="LM judge model")

    # LM judge temperature (float)
    parser.add_argument(
        "--lm_judge_temperature", type=float, default=1.0, help="LM judge temperature"
    )

    # First parse to get the config file path
    temp_args = parser.parse_args()

    # Load YAML config if provided and set as defaults
    if temp_args.config:
        with open(temp_args.config, "r") as f:
            config = yaml.safe_load(f)

        # Set YAML values as defaults in the parser
        for key, value in config.items():
            if hasattr(temp_args, key):
                # Find the action for this argument and set its default
                for action in parser._actions:
                    if action.dest == key:
                        action.default = value
                        break

    # Parse again with updated defaults - command line args will override YAML
    args = parser.parse_args()

    return args
