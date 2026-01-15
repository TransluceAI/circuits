from abc import ABC, abstractmethod
from functools import partial

import torch as t
from circuits.utils.modeling_utils import SparseAct
from nnsight import LanguageModel
from tqdm import tqdm


def count_edges_from_nodes(nodes, nodes_dict, **kwargs):
    # TODO: implement this
    return (0, 0)


def absolute_logit_metric_fn(model, answer_idxs):
    return t.gather(
        model.output.logits[:, -1, :],
        dim=-1,
        index=answer_idxs.view(-1, 1),
    ).squeeze(-1)


def logit_difference_metric_fn(model, patch_answer_idxs, clean_answer_idxs):
    logits = model.output.logits[:, -1, :]
    return t.gather(
        logits,
        dim=-1,
        index=patch_answer_idxs.view(-1, 1),
    ).squeeze(-1) - t.gather(
        logits,
        dim=-1,
        index=clean_answer_idxs.view(-1, 1),
    ).squeeze(-1)


def prob_metric_fn(model, answer_idxs):
    logits = model.output.logits[:, -1, :]
    log_probs = logits.log_softmax(dim=-1)
    return (
        t.gather(
            log_probs,
            dim=-1,
            index=answer_idxs.view(-1, 1),
        )
        .squeeze(-1)
        .exp()
    )


def ablation_fn(x, ablation_type="mean"):
    if ablation_type == "resample":
        idxs = t.multinomial(t.ones(x.act.shape[0]), x.act.shape[0], replacement=True).to(
            x.act.device
        )
        return SparseAct(act=x.act[idxs], res=x.res[idxs])
    elif ablation_type == "zero":
        return x.zeros_like()
    else:  # mean ablation
        return x.mean(dim=0).expand_as(x)


def run_with_ablations(
    clean,  # clean inputs
    patch,  # patch inputs for use in computing ablation values
    model,  # a nnsight LanguageModel
    submodules,  # list of submodules
    dictionaries,  # dictionaries[submodule] is an autoencoder for submodule's output
    nodes,  # nodes[submodule] is a boolean SparseAct with True for the nodes to keep (or ablate if complement is True)
    metric_fn,  # metric_fn(model, **metric_kwargs) -> t.Tensor
    metric_kwargs=dict(),
    complement=False,  # if True, then use the complement of nodes
    test_ablation_type="mean",  # what to do to the patch hidden states to produce values for ablation, default mean ablation
    handle_errors="default",  # or 'remove' to zero ablate all; 'keep' to keep all
):
    if patch is None:
        patch = clean
    patch_states = {}
    with model.trace(patch if test_ablation_type != "zero" else clean), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            patch_states[submodule] = SparseAct(act=f, res=x - x_hat).save()
    patch_states = {k: ablation_fn(v.value, test_ablation_type) for k, v in patch_states.items()}

    # total_residual_error = []
    with model.trace(clean), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            submod_nodes = nodes[submodule]

            x = submodule.get_activation()
            f = dictionary.encode(x)
            res = x - dictionary(x)

            # ablate features
            if complement:
                submod_nodes = ~submod_nodes
            resc_shape_to_expand = submod_nodes.resc.shape[:-1] + (
                patch_states[submodule].res.shape[-1],
            )
            submod_nodes.resc = submod_nodes.resc.expand(resc_shape_to_expand)

            # if handle_errors == "remove":
            #     submod_nodes.resc = t.zeros_like(submod_nodes.resc).to(t.bool)
            # if handle_errors == "keep":
            #     submod_nodes.resc = t.ones_like(submod_nodes.resc).to(t.bool)

            f[..., ~submod_nodes.act] = patch_states[submodule].act[..., ~submod_nodes.act]

            # manually handle error modes
            # for some reason above code created nnsight proxies for the tensors
            # so we do this instead for the residual
            if handle_errors == "default":
                res[..., ~submod_nodes.resc] = patch_states[submodule].res[..., ~submod_nodes.resc]
            elif handle_errors == "remove":
                res = patch_states[submodule].res  # all get replaced
            elif handle_errors == "keep":
                pass  # nothing happens

            # total_residual_error.append(
            #     (x - (dictionary.decode(f) + res)).abs().sum().save()
            # )

            submodule.set_activation(dictionary.decode(f) + res)

        metric = metric_fn(model, **metric_kwargs).save()
    # print("total residual error", sum(total_residual_error).item())
    return metric.value


class BaseMethod(ABC):
    def __init__(self, model, args, mode, **kwargs):
        self.mode = mode
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = args.device
        self.dtype = kwargs["dtype"]
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.auc_test_random = args.auc_test_random
        self.auc_test_handle_errors = args.auc_test_handle_errors
        self.auc_test_ablation_type = args.auc_test_ablation_type
        self.use_transcoder = args.use_transcoder
        self.use_neurons = args.use_neurons
        self.include_embed = kwargs["include_embed"] if self.mode == "train" else False
        self.n_layers = kwargs["n_layers"]
        self.parallel_attn = kwargs["parallel_attn"]
        self.nopair = args.nopair
        self.nodes_only = args.nodes_only
        self.aggregation = args.aggregation
        self.node_threshold = args.node_threshold
        self.submodule_types = args.submodule_types
        self.use_mlp_acts = args.use_mlp_acts
        self.verbose = kwargs.get("verbose", False)
        self.width = args.width

        # Initialize attributes that will be set by subclasses
        self.submodules = None
        self.dictionaries = None
        self.nodes = None
        self.edges = None
        # ENAP-specific attributes
        self.nodes_final_attribution = None
        self.nodes_weight = None

        if isinstance(model, LanguageModel):
            self.module_dims = {
                "embed": model.config.vocab_size,
                "attn": model.config.hidden_size,
                "mlp": (
                    model.config.intermediate_size
                    if self.use_mlp_acts
                    else model.config.hidden_size
                ),
                "resid": model.config.hidden_size,
            }
        else:
            self.module_dims = {
                "embed": model.model.config.vocab_size,
                "attn": model.model.config.hidden_size,
                "mlp": (
                    model.model.config.intermediate_size
                    if self.use_mlp_acts
                    else model.model.config.hidden_size
                ),
                "resid": model.model.config.hidden_size,
            }

    def __str__(self):
        pass

    def make_dataloader(self, examples, **kwargs):
        pass

    @abstractmethod
    def train(self, examples, **kwargs):
        pass

    def inference_faithfulness_and_completeness(self, examples, thresholds, **kwargs):
        if kwargs["component"] == "nodes":
            return self.inference_faithfulness_and_completeness_with_nodes(
                examples, thresholds, **kwargs
            )
        elif kwargs["component"] == "edges":
            return self.inference_faithfulness_and_completeness_with_edges(
                examples, thresholds, **kwargs
            )
        else:
            raise ValueError(f"Invalid component: {kwargs['component']}")

    def inference_faithfulness_and_completeness_with_edges(
        self, examples, thresholds=None, **kwargs
    ):
        """
        Compute faithfulness and completeness by pruning edges instead of nodes.

        Args:
            examples: List of evaluation examples
            thresholds: Thresholds for pruning (unused, we use percentage-based thresholds)
            **kwargs: Additional arguments including:
                - edge_weight_type: "final_attr" or "weight" (default: "final_attr")
        """
        # Unused argument for API compatibility
        _ = thresholds

        kwargs.get("edge_weight_type", "final_attr")

        clean_inputs = [e["clean_prefix"] for e in examples]
        clean_answer_idxs = t.tensor(
            [self.tokenizer(e["clean_answer"]).input_ids[-1] for e in examples],
            dtype=t.long,
            device=self.device,
        )
        patch_inputs = [e["patch_prefix"] for e in examples]
        patch_answer_idxs = t.tensor(
            [self.tokenizer(e["patch_answer"]).input_ids[-1] for e in examples],
            dtype=t.long,
            device=self.device,
        )
        metric_fn = partial(
            logit_difference_metric_fn,
            # flipped to keep consistent with the original implementation
            patch_answer_idxs=clean_answer_idxs,
            clean_answer_idxs=patch_answer_idxs,
        )

        data = []

        # Compute faithfulness
        with t.no_grad():
            # Compute F(M)
            with self.model.trace(clean_inputs):
                metric = metric_fn(self.model).save()
            fm = metric.value.mean().item()

            # Compute F(∅)
            fempty = (
                run_with_ablations(
                    clean_inputs,
                    patch_inputs,
                    self.model,
                    self.submodules,
                    self.dictionaries,
                    nodes={
                        submod: SparseAct(
                            act=t.zeros(self.dictionaries[submod].dict_size, dtype=t.bool),
                            resc=t.zeros(1, dtype=t.bool),
                        ).to(self.device)
                        for submod in self.submodules
                    },
                    metric_fn=metric_fn,
                    test_ablation_type=self.auc_test_ablation_type,
                    handle_errors=self.auc_test_handle_errors,
                )
                .mean()
                .item()
            )

            # Collect all edge weights and sort them by magnitude
            all_edge_weights = []
            total_edges = 0

            for submod in self.submodules:
                sparse_act = self.nodes[submod.name]
                edge_weights_dict = getattr(sparse_act, "incoming_edge_weights", {})

                # Iterate through position -> neuron -> edge_weights
                for token_pos, neurons_dict in edge_weights_dict.items():
                    for neuron_idx, edge_weights_tensor in neurons_dict.items():
                        # edge_weights_tensor is now a tensor, so we can use tensor operations
                        abs_weights = edge_weights_tensor.abs()
                        all_edge_weights.extend(abs_weights.tolist())
                        total_edges += len(abs_weights)

            # Handle case where no edges are found
            if total_edges == 0:
                print("Warning: No edges found in the circuit. Returning empty results.")
                return []

            # Sort all edge weights by magnitude (descending)
            all_edge_weights_sorted = sorted(all_edge_weights, reverse=True)

            # Convert percentage thresholds to absolute thresholds
            percentage_thresholds = [
                0.0,
                0.2,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                20.0,
                50.0,
                70.0,
                100.0,
            ]
            absolute_thresholds = []

            for pct in percentage_thresholds:
                # Calculate how many edges to include (top pct%)
                edges_to_include = int((pct / 100.0) * total_edges)
                if edges_to_include > 0 and edges_to_include <= total_edges:
                    # Get the threshold value at this percentile
                    threshold_value = all_edge_weights_sorted[edges_to_include - 1]
                    absolute_thresholds.append(threshold_value)
                elif edges_to_include == 0:
                    # For very small percentages, use the maximum value
                    absolute_thresholds.append(all_edge_weights_sorted[0])
                else:
                    # For 100%, use a value smaller than the minimum
                    absolute_thresholds.append(all_edge_weights_sorted[-1] * 0.5)

            for i, threshold in enumerate(tqdm(absolute_thresholds)):
                # Prune edges and convert to node boolean masks
                nodes = self._prune_edges_and_get_node_masks(threshold)

                n_nodes = sum([n.act.sum() + n.resc.sum() for n in nodes.values()]).item()
                total_nodes = sum([n.act.numel() + n.resc.numel() for n in nodes.values()])
                p = n_nodes / total_nodes

                # Count remaining edges
                n_edges = self._count_remaining_edges(threshold)
                p_edges = n_edges / total_edges if total_edges > 0 else 0

                if self.auc_test_random:
                    for k in nodes:
                        nodes[k].act = (
                            t.bernoulli(t.ones_like(nodes[k].act, dtype=t.float) * p)
                            .to(self.device)
                            .to(dtype=t.bool)
                        )
                        nodes[k].resc = t.ones_like(nodes[k].resc, dtype=t.bool).to(self.device)

                # Compute F(C)
                fc = (
                    run_with_ablations(
                        clean_inputs,
                        patch_inputs,  # if not self.nopair else None,
                        self.model,
                        self.submodules,
                        self.dictionaries,
                        nodes,
                        metric_fn,
                        test_ablation_type=self.auc_test_ablation_type,
                        handle_errors=self.auc_test_handle_errors,
                    )
                    .mean()
                    .item()
                )

                fccomp = (
                    run_with_ablations(
                        clean_inputs,
                        patch_inputs,  # if not self.nopair else None,
                        self.model,
                        self.submodules,
                        self.dictionaries,
                        nodes,
                        metric_fn,
                        test_ablation_type=self.auc_test_ablation_type,
                        handle_errors=self.auc_test_handle_errors,
                        complement=True,
                    )
                    .mean()
                    .item()
                )

                # Use percentage threshold for reporting instead of absolute threshold
                percentage_threshold = (
                    percentage_thresholds[i] if i < len(percentage_thresholds) else threshold
                )

                data.append(
                    {
                        "threshold": percentage_threshold,
                        "absolute_threshold": threshold,
                        "n_nodes": n_nodes,
                        "p": p,
                        "n_edges": n_edges,
                        "p_edges": p_edges,
                        "total_edges": total_edges,
                        "fccomp": fccomp,
                        "fc": fc,
                        "fempty": fempty,
                        "fm": fm,
                        "faithfulness": (fc - fempty) / (fm - fempty),
                        "completeness": (fccomp - fempty) / (fm - fempty),
                    }
                )
                print(data[-1])
                print("=" * 50)

        return data

    def _prune_edges_and_get_node_masks(self, threshold):
        """
        Prune edges below threshold and convert to node boolean masks.

        Args:
            threshold: Absolute threshold for edge pruning

        Returns:
            Dict of submodule -> SparseAct with boolean masks for nodes to keep
        """
        # Track which nodes have at least one incoming and one outgoing edge above threshold
        nodes_with_incoming = {}
        nodes_with_outgoing = {}

        for submod in self.submodules:
            sparse_act = self.nodes[submod.name]
            layer_key = submod.name

            nodes_with_incoming[layer_key] = set()
            nodes_with_outgoing[layer_key] = set()

            # Check incoming edges
            incoming_weights_dict = getattr(sparse_act, "incoming_edge_weights", {})
            for token_pos, neurons_dict in incoming_weights_dict.items():
                for neuron_idx, edge_weights_tensor in neurons_dict.items():
                    # Check if any incoming edge is above threshold using tensor operations
                    if (edge_weights_tensor.abs() >= threshold).any():
                        nodes_with_incoming[layer_key].add((token_pos, neuron_idx))

            # Check outgoing edges
            outgoing_weights_dict = getattr(sparse_act, "outgoing_edge_weights", {})
            for token_pos, neurons_dict in outgoing_weights_dict.items():
                for neuron_idx, edge_weights_tensor in neurons_dict.items():
                    # Check if any outgoing edge is above threshold using tensor operations
                    if (edge_weights_tensor.abs() >= threshold).any():
                        nodes_with_outgoing[layer_key].add((token_pos, neuron_idx))

        # Create boolean masks for nodes that have both incoming AND outgoing edges
        nodes = {}
        for submod in self.submodules:
            layer_key = submod.name
            sparse_act = self.nodes[layer_key]

            # Initialize boolean masks
            act_mask = t.zeros_like(sparse_act.act, dtype=t.bool)
            resc_mask = t.zeros_like(sparse_act.resc, dtype=t.bool)

            # Set nodes to True if they have both incoming and outgoing edges
            nodes_to_keep = nodes_with_incoming[layer_key] & nodes_with_outgoing[layer_key]

            for token_pos, neuron_idx in nodes_to_keep:
                if token_pos < act_mask.shape[0] and neuron_idx < act_mask.shape[1]:
                    act_mask[token_pos, neuron_idx] = True

            # For resc (residual), we keep it if any node in that position is kept
            for token_pos in range(resc_mask.shape[0]):
                if any(act_mask[token_pos, :]):
                    resc_mask[token_pos, 0] = True

            nodes[submod] = SparseAct(act=act_mask, resc=resc_mask).to(self.device)

        return nodes

    def _count_remaining_edges(self, threshold):
        """
        Count the number of edges remaining after pruning.

        Args:
            threshold: Absolute threshold for edge pruning

        Returns:
            Number of edges above threshold
        """
        # Use incoming edges to avoid double counting
        n_edges = 0
        for submod in self.submodules:
            sparse_act = self.nodes[submod.name]
            edge_weights_dict = getattr(sparse_act, "incoming_edge_weights", {})

            # Count edges above threshold
            for token_pos, neurons_dict in edge_weights_dict.items():
                for neuron_idx, edge_weights_tensor in neurons_dict.items():
                    # Count edges above threshold using tensor operations
                    n_edges += (edge_weights_tensor.abs() >= threshold).sum().item()

        return n_edges

    def inference_faithfulness_and_completeness_with_nodes(self, examples, thresholds, **kwargs):
        clean_inputs = [e["clean_prefix"] for e in examples]
        clean_answer_idxs = t.tensor(
            [self.tokenizer(e["clean_answer"]).input_ids[-1] for e in examples],
            dtype=t.long,
            device=self.device,
        )
        patch_inputs = [e["patch_prefix"] for e in examples]
        patch_answer_idxs = t.tensor(
            [self.tokenizer(e["patch_answer"]).input_ids[-1] for e in examples],
            dtype=t.long,
            device=self.device,
        )
        metric_fn = partial(
            logit_difference_metric_fn,
            # flipped to keep consistent with the original implementation
            patch_answer_idxs=clean_answer_idxs,
            clean_answer_idxs=patch_answer_idxs,
        )

        # if self.nopair:
        #     patch_inputs = None
        #     patch_answer_idxs = None
        #     metric_fn = partial(
        #         absolute_logit_metric_fn,
        #         answer_idxs=clean_answer_idxs,
        #     )

        # out = {}
        data = []

        # Compute faithfulness
        with t.no_grad():
            # Compute F(M)
            with self.model.trace(clean_inputs):
                metric = metric_fn(self.model).save()
            fm = metric.value.mean().item()
            # out["fm"] = fm

            # Compute F(∅)
            fempty = (
                run_with_ablations(
                    clean_inputs,
                    patch_inputs,  # if not self.nopair else None -> eval always uses patch_inputs
                    self.model,
                    self.submodules,
                    self.dictionaries,
                    nodes={
                        submod: SparseAct(
                            act=t.zeros(self.dictionaries[submod].dict_size, dtype=t.bool),
                            resc=t.zeros(1, dtype=t.bool),
                        ).to(self.device)
                        for submod in self.submodules
                    },
                    metric_fn=metric_fn,
                    test_ablation_type=self.auc_test_ablation_type,
                    handle_errors=self.auc_test_handle_errors,
                )
                .mean()
                .item()
            )
            # out["fempty"] = fempty

            # Convert percentage thresholds to absolute thresholds based on node values
            # Collect all non-zero absolute values from self.nodes
            all_values = []
            for submod in self.submodules:
                sparse_act = self.nodes[submod.name]
                # Get non-zero values from both act and resc
                act_values = sparse_act.act.abs().flatten()
                resc_values = sparse_act.resc.abs().flatten()
                # Only keep non-zero values
                act_nonzero = act_values[act_values > 0]
                resc_nonzero = resc_values[resc_values > 0]
                all_values.extend([act_nonzero, resc_nonzero])

            # Concatenate all values and sort them
            all_values_tensor = t.cat(all_values)
            all_values_sorted, _ = t.sort(all_values_tensor, descending=True)
            total_nonzero_nodes = len(all_values_sorted)

            # Convert percentage thresholds to absolute thresholds
            percentage_thresholds = [
                0.0,
                0.2,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                20.0,
                50.0,
                70.0,
                100.0,
            ]
            absolute_thresholds = []

            for pct in percentage_thresholds:
                # Calculate how many nodes to include (top pct%)
                nodes_to_include = int((pct / 100.0) * total_nonzero_nodes)
                if pct == 0.0:
                    absolute_thresholds.append(all_values_sorted[0].item() + 1)
                elif nodes_to_include > 0 and nodes_to_include <= total_nonzero_nodes:
                    # Get the threshold value at this percentile
                    threshold_value = all_values_sorted[nodes_to_include - 1].item()
                    absolute_thresholds.append(threshold_value)
                elif nodes_to_include == 0:
                    # For very small percentages, use the maximum value
                    absolute_thresholds.append(all_values_sorted[0].item())
                else:
                    # For 100%, use a negative value, since abs values are all positive,
                    # this will include all nodes
                    absolute_thresholds.append(-1.0)

            # override if thresholds are provided
            if len(thresholds) != 0:
                absolute_thresholds = thresholds + [float("inf"), -1.0]

            for i, threshold in enumerate(tqdm(absolute_thresholds)):
                # out[threshold] = {}
                # recalculate nodes weights based on the edge weights
                nodes = {
                    submod: self.nodes[submod.name].abs() > threshold for submod in self.submodules
                }

                n_nodes = sum([n.act.sum() + n.resc.sum() for n in nodes.values()]).item()
                total_nodes = sum([n.act.numel() + n.resc.numel() for n in nodes.values()])
                p = n_nodes / total_nodes
                # print(i, threshold, f"{p:.2%}", n_nodes, total_nodes)
                # input()

                # Count edges if available
                total_edges, n_edges = count_edges_from_nodes(nodes, self.nodes)
                p_edges = n_edges / total_edges if total_edges > 0 else 0

                if self.auc_test_random:
                    for k in nodes:
                        nodes[k].act = (
                            t.bernoulli(t.ones_like(nodes[k].act, dtype=t.float) * p)
                            .to(self.device)
                            .to(dtype=t.bool)
                        )
                        nodes[k].resc = t.ones_like(nodes[k].resc, dtype=t.bool).to(self.device)
                    # out[threshold]["n_nodes"] = sum(
                    #     [n.act.sum() + n.resc.sum() for n in nodes.values()]
                    # ).item()
                else:
                    # out[threshold]["n_nodes"] = n_nodes
                    pass

                # Compute F(C)
                fc = (
                    run_with_ablations(
                        clean_inputs,
                        patch_inputs,  # if not self.nopair else None,
                        self.model,
                        self.submodules,
                        self.dictionaries,
                        nodes,
                        metric_fn,
                        test_ablation_type=self.auc_test_ablation_type,
                        handle_errors=self.auc_test_handle_errors,
                    )
                    .mean()
                    .item()
                )
                # out[threshold]["fc"] = fc

                fccomp = (
                    run_with_ablations(
                        clean_inputs,
                        patch_inputs,  # if not self.nopair else None,
                        self.model,
                        self.submodules,
                        self.dictionaries,
                        nodes,
                        metric_fn,
                        test_ablation_type=self.auc_test_ablation_type,
                        handle_errors=self.auc_test_handle_errors,
                        complement=True,
                    )
                    .mean()
                    .item()
                )

                # Use percentage threshold for reporting instead of absolute threshold
                percentage_threshold = (
                    percentage_thresholds[i] if i < len(percentage_thresholds) else threshold
                )

                data.append(
                    {
                        "threshold": percentage_threshold,
                        "absolute_threshold": threshold,
                        "n_nodes": n_nodes,
                        "p": p,
                        "n_edges": n_edges,
                        "p_edges": p_edges,
                        "total_edges": total_edges,
                        "fccomp": fccomp,
                        "fc": fc,
                        "fempty": fempty,
                        "fm": fm,
                        "faithfulness": (fc - fempty) / (fm - fempty),
                        "completeness": (fccomp - fempty) / (fm - fempty),
                    }
                )
                print(data[-1])
                print("=" * 50)
                # out[threshold]["fccomp"] = fccomp
                # out[threshold]["faithfulness"] = (out[threshold]["fc"] - out["fempty"]) / (
                #     out["fm"] - out["fempty"]
                # )
                # out[threshold]["completeness"] = (out[threshold]["fccomp"] - out["fempty"]) / (
                #     out["fm"] - out["fempty"]
                # )
        return data

    def inference_steering(self, examples, thresholds, **kwargs):
        """
        Run position-wise patching interventions similar to steer.py but using run_with_ablations.

        For each layer, position, and threshold, patches only the neurons that exceed the threshold
        from patch to clean examples and computes various metrics.

        Args:
            examples: List of examples with clean_prefix, clean_answer, patch_prefix, patch_answer
            thresholds: List of thresholds to test (patches neurons with |attribution| > threshold)
            **kwargs: Additional arguments

        Returns:
            List of dictionaries containing intervention results for each (layer, position, threshold)
        """
        clean_inputs = [e["clean_prefix"] for e in examples]
        clean_answer_idxs = t.tensor(
            [self.tokenizer(e["clean_answer"]).input_ids[-1] for e in examples],
            dtype=t.long,
            device=self.device,
        )
        [e["patch_prefix"] for e in examples]
        patch_answer_idxs = t.tensor(
            [self.tokenizer(e["patch_answer"]).input_ids[-1] for e in examples],
            dtype=t.long,
            device=self.device,
        )

        # Create metric function for logit differences
        # if self.nopair:
        #     metric_fn = partial(
        #         absolute_logit_metric_fn,
        #         answer_idxs=clean_answer_idxs,
        #     )
        # else:
        metric_fn = partial(
            logit_difference_metric_fn,
            patch_answer_idxs=clean_answer_idxs,
            clean_answer_idxs=patch_answer_idxs,
        )

        data = []

        # Compute baseline metrics
        with t.no_grad():
            # Compute clean baseline
            with self.model.trace(clean_inputs):
                logits = self.model.output.logits.save()
                clean_metric = metric_fn(self.model).save()
            clean_baseline = clean_metric.value.mean().item()
            logits = logits.value
            logits.shape[1]

        # Iterate over each layer
        for submodule in self.submodules:
            # Patch effect
            whole_module_nodes = {
                submod: (
                    (self.nodes[submod.name].ones_like_bool())
                    if submod.name == submodule.name
                    else (self.nodes[submod.name].zeros_like_bool())
                )
                for submod in self.submodules
            }

            # zero ablate the whole layer to see what it does to the metric
            ablate_layer_baseline = (
                run_with_ablations(
                    clean_inputs,
                    None,
                    self.model,
                    self.submodules,
                    self.dictionaries,
                    whole_module_nodes,
                    metric_fn,
                    complement=True,  # We want to patch the True positions
                    test_ablation_type="zero",
                    handle_errors=self.auc_test_handle_errors,
                )
                .mean()
                .item()
            )

            # Iterate over each threshold
            for threshold in tqdm(thresholds, desc="Processing thresholds"):
                # Create threshold-based node masks
                # 1 = zero ablate, 0 = keep
                # since we do complement=True
                threshold_nodes = {
                    submod: (
                        (self.nodes[submod.name].abs() < threshold)
                        if submod.name == submodule.name
                        else (self.nodes[submod.name].zeros_like_bool())
                    )
                    for submod in self.submodules
                }

                # Count nodes above threshold
                n_nodes = sum([n.act.sum() + n.resc.sum() for n in threshold_nodes.values()]).item()
                total_nodes = sum(
                    [n.act.numel() + n.resc.numel() for n in threshold_nodes.values()]
                )
                p = n_nodes / total_nodes

                # Count edges if available
                total_edges, n_edges = count_edges_from_nodes(threshold_nodes, self.nodes)
                p_edges = n_edges / total_edges if total_edges > 0 else 0

                # zero ablate nodes that were below the threshold
                intervened_metric = (
                    run_with_ablations(
                        clean_inputs,
                        None,
                        self.model,
                        self.submodules,
                        self.dictionaries,
                        threshold_nodes,
                        metric_fn,
                        complement=True,  # We want to patch the True positions
                        test_ablation_type="zero",
                        handle_errors=self.auc_test_handle_errors,
                    )
                    .mean()
                    .item()
                )

                data.append(
                    {
                        "threshold": threshold,
                        "n_nodes": n_nodes,
                        "p": p,
                        "n_edges": n_edges,
                        "p_edges": p_edges,
                        "total_edges": total_edges,
                        "submod": submodule.name,
                        "clean_baseline": clean_baseline,
                        "ablate_layer_baseline": ablate_layer_baseline,
                        "intervened_metric": intervened_metric,
                        "intervention_effect": intervened_metric - clean_baseline,
                        "patch_recovery": (
                            (intervened_metric - clean_baseline)
                            / (ablate_layer_baseline - clean_baseline)
                            if ablate_layer_baseline != clean_baseline
                            else 1.0
                        ),
                    }
                )

        return data

    def save(self, dump_dir, examples, **kwargs):
        if "ENAP" in self.__str__():
            save_dict = {
                "examples": examples,
                "nodes_final_attribution": self.nodes_final_attribution,
                "edges": self.edges,
                "nodes_weight": self.nodes_weight,
            }
        else:
            save_dict = {"examples": examples, "nodes": self.nodes, "edges": self.edges}
        with open(dump_dir, "wb") as outfile:
            t.save(save_dict, outfile)

    def load(self, dump_dir, **kwargs):
        pass
