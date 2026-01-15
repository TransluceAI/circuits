"""
Train to get circuits for a given model and dataset.

Usage:
    python train.py --config sweep/simple.yaml
    torchrun --nproc_per_node=4 train.py --config sweep/simple.yaml  # Multi-GPU
"""

import os
import pickle
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from args.args import get_args, make_save_path, print_args
from circuits.core.grad import revert_stop_nonlinear_grad_for_llama, stop_nonlinear_grad_for_llama
from circuits.utils.constants import (
    DTYPE_MAPPING,
    INCLUDE_EMBED_MAPPING,
    METHOD_MAPPING,
    N_LAYERS_MAPPING,
    PARALLEL_ATTN_MAPPING,
)
from circuits.utils.data_loading_utils import load_examples_helper
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from util.subject import Subject, llama31_8B_instruct_config


def setup_distributed():
    """Initialize distributed training if multiple GPUs available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize process group with long timeout (4 hours)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=4),
        )

        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()

    # Parse arguments
    args = get_args()

    # Print parsed arguments (only on rank 0)
    if rank == 0:
        print_args(args)

    if is_distributed and rank == 0:
        print(f"Running distributed training on {world_size} GPUs")

    # Induce circuit save path based on the config.
    save_path = make_save_path(args)

    # if the directory does not exist, create it (only rank 0)
    if rank == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"Save path: {save_path}")

    # Synchronize to ensure directory exists
    if is_distributed:
        dist.barrier()

    # check if train.pt already exists (all ranks check)
    if os.path.exists(os.path.join(save_path, "train.pt")):
        if rank == 0:
            print("Train.pt already exists, skipping training...")
        cleanup_distributed()
        return

    n_layers = N_LAYERS_MAPPING[args.model]
    parallel_attn = PARALLEL_ATTN_MAPPING[args.model]
    include_embed = INCLUDE_EMBED_MAPPING[args.model]
    dtype = DTYPE_MAPPING[args.model]

    # Load model.
    if args.method in {"nap"}:
        if args.use_stop_grad:
            # if using stop grad, it means using our own implementation
            # load HF model, wrap it, and init as nnsight LanguageModel
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
            )

            # load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token
            # core HF model has stop gradient replacement model
            try:
                _ = revert_stop_nonlinear_grad_for_llama(model)
            except Exception:
                pass
            model = stop_nonlinear_grad_for_llama(
                model,
                args.use_shapley_grad,
                args.use_shapley_qk,
                args.use_relp_grad,
                not args.disable_half_rule,
            )

            # wrap it as nnsight LanguageModel
            model = LanguageModel(
                model,
                tokenizer=tokenizer,
                device_map="auto",
                dispatch=True,
                attn_implementation="eager",
                torch_dtype=dtype,
            )
        else:
            # by default, use nnsight backbone
            model = LanguageModel(
                args.model,
                device_map="auto",
                dispatch=True,
                attn_implementation="eager",
                torch_dtype=dtype,
            )
            assert isinstance(model, LanguageModel), "Model must be a LanguageModel"
    elif args.method in {"enap"}:
        # For distributed training, use specific device
        device_map = {"": local_rank} if is_distributed else "auto"
        model = Subject(
            llama31_8B_instruct_config,
            nnsight_lm_kwargs={
                "dispatch": True,
                "device_map": device_map,
                "attn_implementation": "eager",
            },
        )
    else:
        raise ValueError(f"Invalid method: {args.method}")

    # Load examples.
    dataset_path = os.path.join(args.data_path, f"{args.dataset}_train.json")
    examples = load_examples_helper(
        dataset=args.dataset,
        dataset_path=dataset_path,
        num_examples=args.num_train_examples,
        model=model,
        nopair=args.nopair,
    )
    num_examples = min([args.num_train_examples, len(examples)])
    if num_examples < args.num_train_examples:  # warn the user
        print(
            f"Total number of examples is less than {args.num_train_examples}. "
            f"Using {num_examples} examples instead."
        )

    # Print out randomly sampled example (only rank 0)
    if rank == 0:
        print("Randomly sampled example:")
        print(examples[0])
        print("=" * 50)

    # Split examples by rank for distributed training
    if is_distributed:
        # Ensure all ranks have the same total number of examples
        examples_per_rank = len(examples) // world_size
        start_idx = rank * examples_per_rank
        end_idx = start_idx + examples_per_rank if rank < world_size - 1 else len(examples)
        examples_for_rank = examples[start_idx:end_idx]
        print(
            f"Rank {rank}: Processing {len(examples_for_rank)} examples (indices {start_idx} to {end_idx})"
        )
        examples = examples_for_rank

    # Synchronize all processes before training
    if is_distributed:
        dist.barrier()

    # Initialize method.
    device = f"cuda:{local_rank}" if is_distributed else "cuda"
    method = METHOD_MAPPING[args.method](
        model,
        args,
        mode="train",
        n_layers=n_layers,
        dtype=dtype,
        include_embed=include_embed,
        parallel_attn=parallel_attn,
    )

    # Override device for distributed training
    if is_distributed:
        method.device = device

    # Train method.
    additional_args = {
        "batch_size": args.batch_size,
        "device": device,
        "seed": args.seed,
        "use_neurons": args.use_neurons,
        "nopair": args.nopair,
        "nodes_only": args.nodes_only,
        "n_layers": n_layers,
        "parallel_attn": parallel_attn,
        "include_embed": include_embed,
        "dtype": dtype,
        "suffix_length": args.suffix_length,
        "steps": args.steps,
        "use_transcoder": args.use_transcoder,
        "use_shapley_grad": args.use_shapley_grad,
        "use_shapley_qk": args.use_shapley_qk,
        "use_relp_grad": args.use_relp_grad,
        "use_stop_grad_on_mlps": args.use_stop_grad_on_mlps,
        "disable_stop_grad": args.disable_stop_grad,
        "edge_threshold": args.edge_threshold,
        "topk_neurons": args.topk_neurons,
        "is_distributed": is_distributed,
        "rank": rank,
        "world_size": world_size,
    }
    # Check if temp files already exist from previous run
    temp_dir = os.path.join(save_path, "temp_ddp")
    temp_file = os.path.join(temp_dir, f"rank_{rank}.pkl")

    if is_distributed and args.method == "enap" and os.path.exists(temp_file):
        print(f"Rank {rank}: Found existing results at {temp_file}, skipping training")
        result = None  # Will load from disk later
    else:
        # Run training
        result = method.train(examples, **additional_args)

        # Save results if distributed
        if is_distributed and args.method == "enap":
            # Clean up model from GPU before saving (no longer needed)
            del model
            if hasattr(method, "model"):
                del method.model
            torch.cuda.empty_cache()
            import gc

            gc.collect()

            # Save results to temporary files
            if rank == 0 and not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            dist.barrier()  # Wait for rank 0 to create directory

            # Each rank saves its results to disk using pickle
            with open(temp_file, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Rank {rank}: Saved results to {temp_file}")

    # Synchronize after all saves/checks
    if is_distributed and args.method == "enap":
        dist.barrier()

        if rank == 0:
            print("Loading and combining results from all ranks...")

            # Load results from all ranks
            running_nodes = []
            running_edges = []
            min_length = None

            for r in range(world_size):
                rank_file = os.path.join(temp_dir, f"rank_{r}.pkl")
                with open(rank_file, "rb") as f:
                    rank_result = pickle.load(f)
                running_nodes.extend(rank_result["running_nodes"])
                running_edges.extend(rank_result["running_edges"])
                if min_length is None:
                    min_length = rank_result["min_length"]

            print(f"Temp files kept in {temp_dir} for potential reuse")

            print(f"Total batches gathered: {len(running_nodes)}")

            # Reload model on rank 0 for processing
            model = Subject(
                llama31_8B_instruct_config,
                nnsight_lm_kwargs={
                    "dispatch": True,
                    "device_map": {"": 0},
                    "attn_implementation": "eager",
                },
            )
            method.model = model

            # Process all gathered edges
            from circuits.evals.enap import enap_mlp_nodes_to_sparse_act_nodes

            start_time = time.time()
            method.nodes_weight = enap_mlp_nodes_to_sparse_act_nodes(
                running_nodes,
                running_edges,
                method.model,
                batch_aggregation=method.aggregation,
                min_length=min_length,
                topk_edges=method.topk_edges,
                edge_weight_type=method.edge_weight_type,
            )
            elapsed_time = time.time() - start_time
            print(f"enap_mlp_nodes_to_sparse_act_nodes took {elapsed_time:.2f} seconds")

    # Synchronize before saving
    if is_distributed:
        dist.barrier()

    # Save method (only rank 0).
    if rank == 0:
        print(
            f"Saving {len(examples) * world_size if is_distributed else len(examples)} circuits to {save_path}/train.pt"
        )
        method.save(os.path.join(save_path, "train.pt"), examples)

    # Cleanup distributed
    cleanup_distributed()


if __name__ == "__main__":
    main()
