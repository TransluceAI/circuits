"""
Implementation of edge-based NAP.

We implement a simple trick to evaluate edges by turning edge eval to NAP eval. The trick is to
reassign node weights based on maximal weight of all incoming edges for the code.
"""

import time
from typing import Literal

import torch as t
from circuits.analysis.process_circuits import prepare_cis
from circuits.core.core import get_all_pairs_cl_so_effects_with_attributions
from circuits.core.jvp import ADAGConfig, get_all_pairs_cl_ja_effects_with_attributions
from circuits.utils.dictionary_loading_utils import load_saes_and_submodules
from circuits.utils.modeling_utils import SparseAct
from tqdm import tqdm
from util.subject import Subject

from .base import BaseMethod


def _aggregate_weights(weight_lists: list[t.Tensor], batch_aggregation: str) -> float:
    """Helper function to aggregate weight tensors across batches."""
    if batch_aggregation == "mean":
        return t.cat(weight_lists, dim=0).mean().item()
    elif batch_aggregation == "max":
        return t.cat(weight_lists, dim=0).max().item()
    elif batch_aggregation == "sum":
        return t.cat(weight_lists, dim=0).sum().item()
    else:
        raise ValueError(f"Invalid batch aggregation type: {batch_aggregation}")


def _process_weights_for_node(
    weight_dict: dict,
    batch_aggregation: str,
    layer_key: str,
    token_pos: int,
    neuron_idx: int,
    store: dict,
) -> None:
    """Helper function to process and store weights for a single node."""
    aggregated_weights = []
    for _, weight_list in weight_dict.items():
        aggregated_weight = _aggregate_weights(weight_list, batch_aggregation)
        aggregated_weights.append(aggregated_weight)

    store[layer_key].setdefault(token_pos, {})[neuron_idx] = t.tensor(aggregated_weights)


def enap_mlp_nodes_to_sparse_act_nodes(
    nodes,  # unused but kept for API compatibility
    edges,
    model,
    min_length=3,
    batch_aggregation: Literal["mean", "max", "sum"] = "mean",
    topk_edges: int | None = None,
    edge_weight_type: Literal["final_attr", "weight"] = "final_attr",
):
    """
    Convert MLP nodes and edges to SparseAct nodes with edge weight filtering.

    Args:
        nodes: Unused but kept for API compatibility
        edges: Edge data from circuit analysis
        model: The model being analyzed
        min_length: Minimum length for tensor initialization
        batch_aggregation: How to aggregate weights across batches ("mean", "max", "sum")
        topk_edges: If specified, only keep the top-k edges by weight magnitude
        edge_weight_type: Which edge weights to use ("final_attr" or "weight")

    Returns:
        Dictionary of SparseAct nodes with filtered edge weights stored in edge_weights attribute
    """
    # Suppress unused argument warnings
    _ = nodes
    # Assume model is a Subject; ignore layer -1 and final logit layer (layer == model.L)
    num_layers = model.L
    intermediate_size = model.I
    dtype = model.dtype
    device = model.model.device

    # Prepare per-layer SparseAct containers and edge weight storage
    sparse_act_nodes = {}
    incoming_edge_weights_store = {}
    outgoing_edge_weights_store = {}

    def ensure_layer(layer_idx: int):
        key = f"mlp_{layer_idx}"
        if key not in sparse_act_nodes:
            sparse_act_nodes[key] = SparseAct(
                act=t.zeros(min_length, intermediate_size, dtype=dtype, device=device),
                resc=t.zeros(min_length, 1, dtype=dtype, device=device),
            )
            incoming_edge_weights_store[key] = {}
            outgoing_edge_weights_store[key] = {}
        return key

    # Accumulate edge weights across all batches based on selected type
    # Global accumulators: tgt_key -> src_key -> [list of tensors from all batches]
    global_per_tgt_edge_raw = {}
    # Global accumulators: src_key -> tgt_key -> [list of tensors from all batches]
    global_per_src_edge_raw = {}

    total_examples = 0
    for batched_edges in edges:
        if len(batched_edges) == 0:
            continue
        # Determine batch size from any tensor field on first edge
        batch_size = int(batched_edges[0].final_attribution.shape[0])
        total_examples += batch_size

        for e in batched_edges:
            src = e.src  # NeuronIdx(layer, token, neuron)
            src_layer_idx = int(src.layer)
            src_token_pos = int(src.token)
            src_neuron_idx = int(src.neuron)

            tgt = e.tgt  # NeuronIdx(layer, token, neuron)
            tgt_layer_idx = int(tgt.layer)
            # skip layer -1 and the final logit layer num_layers
            if tgt_layer_idx < 0 or tgt_layer_idx >= num_layers:
                continue
            tgt_token_pos = int(tgt.token)
            tgt_neuron_idx = int(tgt.neuron)

            tgt_key = (tgt_layer_idx, tgt_token_pos, tgt_neuron_idx)
            src_key = (src_layer_idx, src_token_pos, src_neuron_idx)

            # Select the appropriate edge weight based on edge_weight_type
            if edge_weight_type == "final_attr":
                raw_weight = e.final_attribution.cpu().sum(dim=-1)  # [B, ] - keep on CPU
            else:  # "weight"
                raw_weight = e.weight.cpu()  # keep on CPU

            # Accumulate raw edge weights per source->target pair across all batches
            global_per_tgt_edge_raw.setdefault(tgt_key, {}).setdefault(src_key, []).append(
                raw_weight
            )
            global_per_src_edge_raw.setdefault(src_key, {}).setdefault(tgt_key, []).append(
                raw_weight
            )

    # Apply topk filtering if specified
    if topk_edges is not None:
        # Collect all edge weights to determine threshold
        all_edge_weights = []
        for tgt_key, src_dict in global_per_tgt_edge_raw.items():
            for src_key, weight_list in src_dict.items():
                aggregated_weight = _aggregate_weights(weight_list, batch_aggregation)
                all_edge_weights.append(abs(aggregated_weight))

        # Sort and determine threshold
        all_edge_weights_sorted = sorted(all_edge_weights, reverse=True)
        if len(all_edge_weights_sorted) > topk_edges:
            threshold = all_edge_weights_sorted[topk_edges - 1]
        else:
            threshold = 0.0  # Keep all edges if we have fewer than topk_edges

        # Filter edges based on threshold for both incoming and outgoing
        filtered_global_per_tgt_edge_raw = {}
        filtered_global_per_src_edge_raw = {}

        for tgt_key, src_dict in global_per_tgt_edge_raw.items():
            filtered_src_dict = {}
            for src_key, weight_list in src_dict.items():
                aggregated_weight = _aggregate_weights(weight_list, batch_aggregation)
                if abs(aggregated_weight) >= threshold:
                    filtered_src_dict[src_key] = weight_list
            if filtered_src_dict:
                filtered_global_per_tgt_edge_raw[tgt_key] = filtered_src_dict

        for src_key, tgt_dict in global_per_src_edge_raw.items():
            filtered_tgt_dict = {}
            for tgt_key, weight_list in tgt_dict.items():
                aggregated_weight = _aggregate_weights(weight_list, batch_aggregation)
                if abs(aggregated_weight) >= threshold:
                    filtered_tgt_dict[tgt_key] = weight_list
            if filtered_tgt_dict:
                filtered_global_per_src_edge_raw[src_key] = filtered_tgt_dict

        global_per_tgt_edge_raw = filtered_global_per_tgt_edge_raw
        global_per_src_edge_raw = filtered_global_per_src_edge_raw

    # Now apply aggregation after processing all batches
    # Process incoming edges
    for tgt_key in global_per_tgt_edge_raw.keys():
        tgt_layer_idx, tgt_token_pos, tgt_neuron_idx = tgt_key
        tgt_layer_key = ensure_layer(tgt_layer_idx)
        _process_weights_for_node(
            global_per_tgt_edge_raw[tgt_key],
            batch_aggregation,
            tgt_layer_key,
            tgt_token_pos,
            tgt_neuron_idx,
            incoming_edge_weights_store,
        )

    # Process outgoing edges
    for src_key in global_per_src_edge_raw.keys():
        src_layer_idx, src_token_pos, src_neuron_idx = src_key
        src_layer_key = ensure_layer(src_layer_idx)
        _process_weights_for_node(
            global_per_src_edge_raw[src_key],
            batch_aggregation,
            src_layer_key,
            src_token_pos,
            src_neuron_idx,
            outgoing_edge_weights_store,
        )

    # Ensure all layers exist (fill with zeros if absent)
    for i in range(num_layers):
        ensure_layer(i)

    # Set edge weights for all SparseAct nodes
    for key in sparse_act_nodes:
        sparse_act_nodes[key].incoming_edge_weights = incoming_edge_weights_store.get(key, {})
        sparse_act_nodes[key].outgoing_edge_weights = outgoing_edge_weights_store.get(key, {})
    return sparse_act_nodes


class ENAP(BaseMethod):
    """Implementation of edge-based NAP with MLP neurons."""

    def __str__(self):
        return f"ENAP-{self.effect_method}"

    def __init__(self, model, args, mode, **kwargs):
        super().__init__(model, args, mode, **kwargs)

        self.edge_threshold = getattr(args, "edge_threshold", 0.00)
        self.topk_neurons = getattr(args, "topk_neurons", 100)
        self.topk_edges = getattr(args, "topk_edges", None)
        self.edge_weight_type = getattr(args, "edge_weight_type", "final_attr")
        self.use_relp_grad = getattr(args, "use_relp_grad", False)
        self.use_shapley_grad = getattr(args, "use_shapley_grad", False)
        self.disable_stop_grad = getattr(args, "disable_stop_grad", False)
        self.use_stop_grad_on_mlps = getattr(args, "use_stop_grad_on_mlps", False)
        self.disable_half_rule = getattr(args, "disable_half_rule", False)

        # IG parameters
        self.ig_steps = getattr(args, "ig_steps", None)

        # valid the of the model is a Subject
        if mode == "train":
            assert isinstance(model, Subject), "mode train must be a Subject type model"

        # Effect computation method selection
        # Format: "jvp", "clso", "jvp-ig-inputs", "jvp-conductance", etc.
        self.effect_method = getattr(args, "effect_method", "jvp")

        # Extract IG mode from effect_method if it contains "-"
        if "-" in self.effect_method:
            self.ig_mode = "-".join(self.effect_method.split("-")[1:])
        else:
            self.ig_mode = "ig-inputs"  # default

        # Suffix length for token filtering
        self.suffix_length = getattr(args, "suffix_length", None)

        # main artifacts of the method
        self.nodes = None
        self.edges = None
        self.nodes_final_attribution = None
        self.nodes_weight = None

        # fake loading identity dictionary
        if mode == "eval":

            print("Loading identity dictionary...")
            self.submodules, self.dictionaries = load_saes_and_submodules(
                model,
                separate_by_type=True,
                include_embed=False,
                include_attn=False,
                include_mlp=True,  # TODO: currently only support mlp
                include_resid=False,
                neurons=True,
                device=self.device,
                dtype=self.dtype,
                module_dims=self.module_dims,
                use_mlp_acts=True,
                width=self.width,
            )

    def make_dataloader(self, examples, **kwargs):

        batches = []

        # process once for all
        clean_prefixes = [e["clean_prefix"] for e in examples]
        clean_answers = [[e["clean_answer"].strip()] for e in examples]

        cis, attention_masks, focus_tokens, keep_pos, starts = prepare_cis(
            self.model,
            self.tokenizer,
            clean_prefixes,
            [None] * len(clean_prefixes),
            k=1,
            system_prompt=None,
            true_answers=clean_answers,
            use_chat_format=False,
            verbose=False,
        )

        for i in tqdm(range(0, len(examples), self.batch_size), desc="Processing batches"):
            batches.append(
                (
                    cis[i : i + self.batch_size],
                    attention_masks[i : i + self.batch_size],
                    focus_tokens[i : i + self.batch_size],
                    keep_pos,
                    starts,
                )
            )
        return batches

    def train(self, examples, **kwargs):
        # Check if running in distributed mode
        is_distributed = kwargs.get("is_distributed", False)
        rank = kwargs.get("rank", 0)

        dataloader = self.make_dataloader(examples, **kwargs)

        running_nodes = []
        running_edges = []

        desc = f"Training (Rank {rank})" if is_distributed else "Training"
        for batch in tqdm(dataloader, desc=desc, disable=(is_distributed and rank != 0)):

            cis, attention_masks, focus_tokens, keep_pos, starts = batch

            if "jvp" in self.effect_method:
                # JVP method uses ADAGConfig
                config = ADAGConfig(
                    device=self.device,
                    verbose=self.verbose,
                    parent_threshold=None,
                    edge_threshold=self.edge_threshold,
                    node_attribution_threshold=None,
                    topk=None,
                    batch_aggregation="any",
                    topk_neurons=self.topk_neurons,
                    use_shapley_grad=self.use_shapley_grad,
                    use_relp_grad=self.use_relp_grad,
                    use_shapley_qk=False,
                    disable_stop_grad=self.disable_stop_grad,
                    use_stop_grad_on_mlps=self.use_stop_grad_on_mlps,
                    disable_half_rule=self.disable_half_rule,
                    ablation_mode="zero",
                    return_nodes_only=False,
                    focus_last_residual=False,
                    skip_attr_contrib=False,
                    ig_steps=self.ig_steps,
                    ig_mode=self.ig_mode,
                )
                nodes, edges = get_all_pairs_cl_ja_effects_with_attributions(
                    subject=self.model,
                    cis=cis,
                    config=config,
                    src_tokens=keep_pos,
                    tgt_tokens=[max(keep_pos) for _ in range(1)],
                    attention_masks=attention_masks,
                    focus_logits=focus_tokens,
                )
            elif "clso" in self.effect_method:
                # CLSO method uses individual parameters
                nodes, edges = get_all_pairs_cl_so_effects_with_attributions(
                    subject=self.model,  # this is a subject model
                    cis=cis,
                    attention_masks=attention_masks,
                    focus_logits=focus_tokens,
                    device=self.device,
                    verbose=self.verbose,
                    parent_threshold=None,
                    edge_threshold=self.edge_threshold,
                    node_attribution_threshold=None,
                    topk=None,
                    src_tokens=keep_pos,
                    tgt_tokens=[max(keep_pos) for _ in range(1)],
                    batch_aggregation="any",
                    topk_neurons=self.topk_neurons,
                    use_shapley_grad=self.use_shapley_grad,
                    use_relp_grad=self.use_relp_grad,
                    use_shapley_qk=False,
                    disable_stop_grad=self.disable_stop_grad,
                    use_stop_grad_on_mlps=self.use_stop_grad_on_mlps,
                    disable_half_rule=self.disable_half_rule,
                    ablation_mode="zero",
                    return_nodes_only=False,
                    focus_last_residual=False,
                    skip_attr_contrib=False,
                    ig_steps=self.ig_steps,
                    ig_mode=self.ig_mode,
                )
            else:
                raise ValueError(f"Invalid effect method: {self.effect_method}")
            running_nodes.append(nodes)
            running_edges.append(edges)

        # For distributed training, return raw results to be gathered
        if is_distributed:
            return {
                "running_nodes": running_nodes,
                "running_edges": running_edges,
                "min_length": len(keep_pos),
            }

        # For non-distributed, process immediately
        start_time = time.time()
        self.nodes_weight = enap_mlp_nodes_to_sparse_act_nodes(
            running_nodes,
            running_edges,
            self.model,
            batch_aggregation=self.aggregation,
            min_length=len(keep_pos),
            topk_edges=self.topk_edges,
            edge_weight_type=self.edge_weight_type,
        )
        elapsed_time = time.time() - start_time
        print(f"enap_mlp_nodes_to_sparse_act_nodes took {elapsed_time:.2f} seconds")

        return None

    def load(self, dump_dir, **kwargs):
        start_layer = kwargs["start_layer"]
        # remove modules before start_layer
        serialized_submodules = []
        if isinstance(self.submodules, list):
            serialized_submodules = self.submodules
        else:
            for submods in self.submodules:
                if submods is not None and len(submods) != 0:
                    for submod in submods:
                        if int(submod.name.split("_")[-1]) >= start_layer:
                            serialized_submodules.append(submod)

        self.submodules = serialized_submodules
        # Historically this initialized an IdentityDict per submodule; this method no longer
        # relies on neuron_dicts for ENAP, so we skip creating it.

        circuit = t.load(dump_dir, map_location=self.device)
        self.nodes = circuit["nodes_weight"] if "nodes_weight" in circuit else None
