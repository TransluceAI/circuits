from dataclasses import dataclass
from typing import Any, List, Literal

import torch
from circuits.core.core import (
    _get_global_important_neurons_mask,
    _get_grad_attributions_from_logits,
    _get_ig_attributions_from_logits,
    _get_neuron_attr_and_contrib,
    _get_neuron_attr_and_contrib_ig,
    _get_neuron_attr_and_contrib_with_stop_grad_on_mlps,
)
from circuits.core.grad import (
    layerwise_revert_stop_nonlinear_grad_for_llama,
    layerwise_stop_nonlinear_grad_for_llama,
    remove_forward_hooks,
    revert_stop_nonlinear_grad_for_llama,
    stop_nonlinear_grad_for_llama,
)
from circuits.core.utils import Edge, NeuronIdx, Node, collect_neuron_acts
from util.chat_input import IdsInput, ModelInput
from util.subject import Subject, construct_dataset

BLACKLISTED_NEURONS = [
    NeuronIdx(layer=23, token=-1, neuron=306),
    NeuronIdx(layer=20, token=-1, neuron=3972),
    NeuronIdx(layer=18, token=-1, neuron=7417),
    NeuronIdx(layer=16, token=-1, neuron=1241),
    NeuronIdx(layer=13, token=-1, neuron=4208),
    NeuronIdx(layer=11, token=-1, neuron=11321),
    NeuronIdx(layer=10, token=-1, neuron=11570),
    NeuronIdx(layer=9, token=-1, neuron=4255),
    NeuronIdx(layer=7, token=-1, neuron=6673),
    NeuronIdx(layer=6, token=-1, neuron=5866),
    NeuronIdx(layer=5, token=-1, neuron=7012),
    NeuronIdx(layer=2, token=-1, neuron=4786),
]


@dataclass
class ADAGConfig:
    """Configuration for ADAG circuit tracing."""

    # Basic settings
    device: str = "cuda:0"
    verbose: bool = False
    return_only_important_neurons: bool = False
    return_nodes_only: bool = False
    skip_attr_contrib: bool = False

    # Gradient settings
    use_shapley_grad: bool = False
    use_shapley_qk: bool = False
    use_relp_grad: bool = False
    disable_half_rule: bool = False
    disable_stop_grad: bool = False
    use_stop_grad_on_mlps: bool = False
    ablation_mode: Literal["zero", "mean"] = "zero"
    center_logits: bool = False

    # IG settings
    ig_steps: int | None = None
    ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs"

    # Edge pruning settings
    node_attribution_threshold: float | None = 1.0
    topk_neurons: int | None = None
    parent_threshold: float | None = None
    edge_threshold: float | None = None
    topk: int | None = None
    percentage_threshold: float | None = None
    batch_aggregation: Literal["mean", "max", "max_abs", "any"] = "mean"
    return_absolute: bool = False
    apply_blacklist: bool = False

    # Layer settings
    start_layer: int | None = None
    end_layer: int | None = None

    # Tracing settings
    focus_last_residual: bool = False


def get_all_pairs_cl_ja_effects_with_attributions(
    subject: Subject,
    cis: list[ModelInput],
    # config
    config: ADAGConfig,
    # where to trace from and to
    src_tokens: list[int],
    tgt_tokens: list[int],
    keep_tokens: list[int] | None = None,
    attention_masks: list[list[int]] | torch.Tensor | None = None,
    focus_positions: list[int] | None = None,
    focus_logits: list[list[int]] | list[int] | None = None,
) -> tuple[list[Node], list[Edge]] | tuple[torch.Tensor, torch.Tensor]:
    """
    Cross Layer Jacobian Attribution (CLJA) for circuit tracing.

    This function follows the exact same procedure as the original CLSO algorithm
    for finding important neurons, but uses jacobian computation for edge weights
    instead of the Cross Layer Second Order (CLSO) lens method.

    Args:
        Same as get_all_pairs_cl_so_effects_with_attributions in core.py

    Returns:
        tuple[list[Node], list[Edge]]: Circuit nodes and edges
        torch.Tensor: Only return the final attributions for the important neurons
    """
    ############
    # SETTINGS #
    ############
    device = config.device
    verbose = config.verbose
    return_only_important_neurons = config.return_only_important_neurons
    return_nodes_only = config.return_nodes_only
    use_shapley_grad = config.use_shapley_grad
    use_shapley_qk = config.use_shapley_qk
    use_relp_grad = config.use_relp_grad
    disable_half_rule = config.disable_half_rule
    disable_stop_grad = config.disable_stop_grad
    use_stop_grad_on_mlps = config.use_stop_grad_on_mlps
    ablation_mode = config.ablation_mode
    center_logits = config.center_logits
    ig_steps = config.ig_steps
    ig_mode = config.ig_mode
    # edge pruning settings
    node_attribution_threshold = config.node_attribution_threshold
    topk_neurons = config.topk_neurons
    parent_threshold = config.parent_threshold
    edge_threshold = config.edge_threshold
    topk = config.topk
    percentage_threshold = config.percentage_threshold
    batch_aggregation = config.batch_aggregation
    return_absolute = config.return_absolute
    apply_blacklist = config.apply_blacklist
    # more circuit settings
    focus_last_residual = config.focus_last_residual
    start_layer = config.start_layer
    end_layer = config.end_layer
    skip_attr_contrib = config.skip_attr_contrib

    #########
    # SETUP #
    #########

    if disable_stop_grad:
        if any([use_shapley_grad, use_shapley_qk, use_relp_grad, use_stop_grad_on_mlps]):
            print("warning: stop grad is disabled but some stop grad configurations are used")

    if focus_positions is None:
        focus_positions = tgt_tokens

    if focus_logits is None and not focus_last_residual:
        try:
            focus_logits = [cis[0].tokenize(subject)[0][pos_idx + 1] for pos_idx in focus_positions]
        except Exception as e:
            print(e)
            max_token_expected = max(focus_positions) + 1
            raise ValueError(f"failed to get labels for {max_token_expected} tokens.")

    if keep_tokens is None:
        keep_tokens = list(range(max(tgt_tokens) + 1))

    # start and end layer
    start_layer = -1 if start_layer is None else start_layer
    end_layer = subject.L if end_layer is None else end_layer

    # get input ids
    if isinstance(attention_masks, torch.Tensor):
        attention_masks = attention_masks.tolist()
    ds = construct_dataset(
        subject,
        [(ci, IdsInput(input_ids=[])) for ci in cis],
        shift_labels=False,
        prompt_attn_mask=attention_masks,
    )
    input_ids, attn_mask_final = (
        torch.tensor([x["input_ids"] for x in ds.to_list()], device=device),  # type: ignore
        torch.tensor([x["attention_mask"] for x in ds.to_list()], device=device),  # type: ignore
    )

    ########
    # CORE #
    ########

    # ensure model is on the correct device
    subject.model._model = subject.model._model.to(device)

    # core HF model has stop gradient replacement model
    if not disable_stop_grad:
        try:
            _ = revert_stop_nonlinear_grad_for_llama(subject.model._model)
        except Exception:
            pass
        model = stop_nonlinear_grad_for_llama(
            subject.model._model,
            use_shapley_grad,
            use_shapley_qk,
            use_relp_grad,
            not disable_half_rule,
        )
    else:
        model = subject.model._model

    # get attributions (same as original)
    if ig_steps is None:
        (
            mlp_final_attributions,
            embed_final_attributions,
            goal_value,
            mlp_final_acts,
            embed_final_acts,
        ) = _get_grad_attributions_from_logits(
            subject,
            model,
            input_ids,
            keep_tokens,
            focus_positions,
            focus_logits=focus_logits,
            focus_last_residual=focus_last_residual,
            attention_masks=attn_mask_final,
            ablation_mode=ablation_mode,
            disable_stop_grad=disable_stop_grad,
            center_logits=center_logits,
            verbose=verbose,
        )
    else:
        (
            mlp_final_attributions,
            embed_final_attributions,
            goal_value,
            mlp_final_acts,
            embed_final_acts,
        ) = _get_ig_attributions_from_logits(
            subject,
            model,
            input_ids,
            keep_tokens,
            focus_positions,
            focus_logits=focus_logits,
            focus_last_residual=focus_last_residual,
            attention_masks=attn_mask_final,
            disable_stop_grad=disable_stop_grad,
            center_logits=center_logits,
            verbose=verbose,
            ig_steps=ig_steps,
        )

    mlp_final_attributions = mlp_final_attributions.unsqueeze(-1)  # shape: (L, B, T, D_ff, 1)
    embed_final_attributions = embed_final_attributions.unsqueeze(-1)  # shape: (B, T, 1)
    if verbose:
        print("collected attributions for mlp", mlp_final_attributions.shape)
        print("collected attributions for embed", embed_final_attributions.shape)

    if apply_blacklist:
        for idx in BLACKLISTED_NEURONS:
            layer, neuron = idx.layer, idx.neuron
            mlp_final_attributions[layer, :, :, neuron, :] = 0

    # compute per-batch item absolute attribution thresholds
    absolute_attribution_threshold = None
    if percentage_threshold is not None:
        absolute_attribution_threshold = goal_value * percentage_threshold

    # before calculating anything, we get important neurons globally (same as original)
    global_important_neurons_mask = _get_global_important_neurons_mask(
        keep_tokens=keep_tokens,
        start_layer=start_layer,
        end_layer=end_layer,
        mlp_final_attributions=mlp_final_attributions,
        node_attribution_threshold=node_attribution_threshold,
        topk_neurons=topk_neurons,
        absolute_attribution_threshold=absolute_attribution_threshold,
        batch_aggregation=batch_aggregation,
        verbose=verbose,
    )
    if verbose:
        print("global important neurons mask", global_important_neurons_mask.shape)
        print("TOTAL NEURONS", global_important_neurons_mask.sum().item())
        print(f"GOAL VALUE {goal_value.item():.5f}")
        print(f"EMBED SUM {embed_final_attributions.sum().item():.5f}")
        for layer in range(len(model.model.layers)):
            print(f"LAYER {layer} ATTR {mlp_final_attributions[layer].sum().item():.5f}")
        if percentage_threshold is not None:
            print(f"PERCENTAGE THRESHOLD {percentage_threshold:.2%}")
        if absolute_attribution_threshold is not None:
            print(f"ABSOLUTE ATTRIBUTION THRESHOLD {absolute_attribution_threshold.item():.5f}")

    # get important neurons for each layer (same as original)
    neuron_cfg: dict[int, list[list[int]]] = {
        layer: global_important_neurons_mask[layer].nonzero(as_tuple=False).tolist()
        for layer in range(max(start_layer, 0), end_layer)
    }
    last_non_zero_layer = 0
    for layer, neurons in neuron_cfg.items():
        if verbose:
            print(f"Layer {layer} has {neurons} important neurons")
        if len(neurons) > 0:
            last_non_zero_layer = max(last_non_zero_layer, layer)
    end_layer = min(end_layer, last_non_zero_layer + 1)

    # if we only want to return the important neurons, we can do that now
    if return_only_important_neurons:
        model = revert_stop_nonlinear_grad_for_llama(model)
        return mlp_final_attributions, embed_final_attributions, mlp_final_acts, embed_final_acts

    # get attributions and contributions for important neurons (same as original)
    if not skip_attr_contrib:
        if ig_steps is None:
            attr, contrib, embed_grad_contrib, neuron_tags = _get_neuron_attr_and_contrib(
                model,
                neuron_cfg,
                input_ids,
                src_tokens,
                tgt_tokens,
                focus_positions,
                focus_logits,
                attn_mask_final,
                disable_stop_grad=disable_stop_grad,
                center_logits=center_logits,
                neuron_chunk_size=50,
                verbose=verbose,
            )
        else:
            attr, contrib, embed_grad_contrib, neuron_tags = _get_neuron_attr_and_contrib_ig(
                model,
                neuron_cfg,
                input_ids,
                src_tokens,
                tgt_tokens,
                focus_positions,
                focus_logits,
                attn_mask_final,
                disable_stop_grad=disable_stop_grad,
                center_logits=center_logits,
                ig_steps=ig_steps,
                ig_mode=ig_mode,
                neuron_chunk_size=20,  # Smaller chunk size for IG
                verbose=verbose,
            )

        if verbose:
            print("collecting attributions for", attr.shape)  # shape: (neurons, batch, src)
            print("collecting contributions for", contrib.shape)  # shape: (neurons, batch, tgt)
            print(
                "collecting embed contributions for", embed_grad_contrib.shape
            )  # shape: (src, batch, tgt)

    # store neuron attributions and contributions (keep on CPU to save GPU memory)
    neuron_attr_map: dict[NeuronIdx, torch.Tensor] = {}
    neuron_contrib_map: dict[NeuronIdx, torch.Tensor] = {}
    if not skip_attr_contrib:
        for neuron_count, neuron_idx in enumerate(neuron_tags):
            neuron_attr_map[neuron_idx] = attr[neuron_count].cpu()
            neuron_contrib_map[neuron_idx] = contrib[neuron_count].cpu()
        for src_token in src_tokens:
            neuron_contrib_map[NeuronIdx(layer=-1, token=src_token, neuron=0)] = embed_grad_contrib[
                src_token
            ].cpu()

        # Clean up GPU tensors after copying to CPU
        del attr, contrib, embed_grad_contrib
        torch.cuda.empty_cache()

    # we repopulate attr and contri maps if stop grad is on to get direct edge weights
    if not disable_stop_grad and use_stop_grad_on_mlps:
        (
            attr_with_stop_grad_on_mlps,
            contrib_with_stop_grad_on_mlps,
            embed_grad_contrib_with_stop_grad_on_mlps,
            neuron_tags_with_stop_grad_on_mlps,
        ) = _get_neuron_attr_and_contrib_with_stop_grad_on_mlps(
            model,
            neuron_cfg,
            input_ids,
            src_tokens,
            tgt_tokens,
            focus_positions,
            focus_logits,
            attn_mask_final,
            use_shapley_grad=use_shapley_grad,
            use_shapley_qk=use_shapley_qk,
            use_relp_grad=use_relp_grad,
            center_logits=center_logits,
            neuron_chunk_size=10,
            verbose=verbose,
        )
        # store neuron attributions and contributions (keep on CPU to save GPU memory)
        neuron_attr_map_with_stop_grad_on_mlps: dict[NeuronIdx, torch.Tensor] = {}
        neuron_contrib_map_with_stop_grad_on_mlps: dict[NeuronIdx, torch.Tensor] = {}
        for neuron_count, neuron_idx in enumerate(neuron_tags_with_stop_grad_on_mlps):
            neuron_attr_map_with_stop_grad_on_mlps[neuron_idx] = attr_with_stop_grad_on_mlps[
                neuron_count
            ].cpu()
            neuron_contrib_map_with_stop_grad_on_mlps[neuron_idx] = contrib_with_stop_grad_on_mlps[
                neuron_count
            ].cpu()
        for src_token in src_tokens:
            neuron_contrib_map_with_stop_grad_on_mlps[
                NeuronIdx(layer=-1, token=src_token, neuron=0)
            ] = embed_grad_contrib_with_stop_grad_on_mlps[src_token].cpu()

        # Clean up GPU tensors after copying to CPU
        del attr_with_stop_grad_on_mlps, contrib_with_stop_grad_on_mlps
        del embed_grad_contrib_with_stop_grad_on_mlps
        torch.cuda.empty_cache()
        # the model is reverted back already so no need to do it again
        # if not disable_stop_grad:
        #     model = revert_stop_nonlinear_grad_for_llama(model)
    else:
        neuron_attr_map_with_stop_grad_on_mlps = neuron_attr_map
        neuron_contrib_map_with_stop_grad_on_mlps = neuron_contrib_map
        # revert back the replacement model to the original HF model
        if not disable_stop_grad:
            model = revert_stop_nonlinear_grad_for_llama(model)

    # Additional cleanup before edge computation
    torch.cuda.empty_cache()

    if verbose:
        print(f"Global important neurons mask: {global_important_neurons_mask.sum()}")
        print(
            f"Global important (layer, token): "
            f"{len(global_important_neurons_mask.sum(dim=-1).nonzero())}"
        )

    # cross-layer jacobian edge tracing
    nodes, edges = _get_cl_ja_based_edges(
        subject,
        cis,
        mlp_final_attributions,
        embed_final_attributions,
        global_important_neurons_mask,
        neuron_attr_map,
        neuron_contrib_map,
        device,
        verbose,
        parent_threshold=parent_threshold,
        edge_threshold=edge_threshold,
        topk=topk,
        keep_tokens=keep_tokens,
        src_tokens=src_tokens,  # the token positions to trace from
        tgt_tokens=tgt_tokens,  # the token positions to trace to
        focus_positions=focus_positions,  # the logits to include
        focus_logits=focus_logits,  # the token vocab ids to trace logits
        start_layer=start_layer,
        end_layer=end_layer,
        attention_masks=attention_masks,
        ig_steps=ig_steps,
        ig_mode=ig_mode,
        return_nodes_only=return_nodes_only,
        return_absolute=return_absolute,
        # stop grad params
        use_shapley_grad=use_shapley_grad,
        use_shapley_qk=use_shapley_qk,
        use_relp_grad=use_relp_grad,
        disable_stop_grad=disable_stop_grad,
        use_stop_grad_on_mlps=use_stop_grad_on_mlps,
        neuron_attr_map_with_stop_grad_on_mlps=neuron_attr_map_with_stop_grad_on_mlps,
        neuron_contrib_map_with_stop_grad_on_mlps=neuron_contrib_map_with_stop_grad_on_mlps,
    )

    # final return
    return nodes, edges


def _get_cl_ja_based_edges(
    subject: Subject,
    cis: List[ModelInput],
    mlp_final_attributions: torch.Tensor,
    embed_final_attributions: torch.Tensor,
    global_important_neurons_mask: torch.Tensor,
    neuron_attr_map,
    neuron_contrib_map,
    device: str = "cuda:0",
    verbose: bool = False,
    parent_threshold: float | None = None,
    edge_threshold: float | None = None,
    topk: int | None = None,
    # we use these lists to trace circuit effects
    keep_tokens: List[int] | None = None,  # the token positions to trace circuits
    src_tokens: List[int] | None = None,  # the token positions to trace from
    tgt_tokens: List[int] | None = None,  # the token positions to trace to
    focus_positions: List[int] | None = None,  # the token positions to consider the logits effect
    focus_logits: List[int] | None = None,  # the token vocab ids to trace logits
    return_cross_edges_only: bool = True,  # only return cross tgt -> src token edges only
    return_absolute: bool = False,
    # we can skip early/late layers by setting these two
    start_layer: int | None = None,
    end_layer: int | None = None,
    keep_fo: bool = True,
    use_absolute: bool = False,
    batch_aggregation: Literal["mean", "max"] = "mean",
    include_error_node: bool = False,
    include_error_edges: bool = False,
    return_nodes_only: bool = False,
    max_buffers: int | None = None,
    return_all_edge_weights: bool = True,
    attention_masks: list[list[int]] | None = None,
    # stop grad params
    use_shapley_grad: bool = False,
    use_shapley_qk: bool = False,
    use_relp_grad: bool = False,
    disable_stop_grad: bool = False,
    use_stop_grad_on_mlps: bool = False,
    neuron_attr_map_with_stop_grad_on_mlps=None,
    neuron_contrib_map_with_stop_grad_on_mlps=None,
    # IG params
    ig_steps: int | None = None,
    ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs",
) -> tuple[Any, Any]:
    """
    Compute circuit nodes and edges using Cross Layer Jacobian Attribution (CLJA).

    This function computes edge weights between important neurons using jacobian computation
    with stop gradient on nonlinear components, as an alternative to CLSO lens methods.
    """
    nodes: list[Node] = []
    edges: list[Edge] = []

    # caching activations with subject model
    if verbose:
        print("Collecting acts... ", end="", flush=True)
    collect_layers = list(range(subject.L))
    (neurons_LBTI, resids_LBTD, tokens, output_norm_const_BTf11D) = collect_neuron_acts(
        subject,
        cis,
        attention_masks,
        collect_layers=collect_layers,
        keep_tokens=keep_tokens,
        device=device,
        verbose=verbose,
    )

    # key constants
    L = subject.L
    D = subject.D
    B = neurons_LBTI[0].size(0)
    neurons_LBTI[0].size(1)

    # creating final logits nodes
    if verbose:
        print("Creating final logits nodes and all incoming edges...")
    for target_idx, target_token_pos_idx in enumerate(focus_positions):
        focus_logits_target = torch.tensor(focus_logits)[:, target_idx]
        resid_final_BD = resids_LBTD[subject.L - 1][:, target_token_pos_idx].view(
            B, D
        ) * output_norm_const_BTf11D[:, target_token_pos_idx, 0].view(B, D)
        logits_BV = torch.einsum(
            "bd,vd->bv", resid_final_BD, subject.unembed.weight[focus_logits_target]
        ) * torch.eye(B, device=device)

        # add attr and contrib map
        for batch_idx in range(len(focus_logits_target)):
            logit_id = (
                subject.L,
                target_token_pos_idx,
                focus_logits_target[batch_idx].item(),
            )
            if logit_id not in neuron_attr_map:
                neuron_attr_map[logit_id] = torch.zeros((B, len(src_tokens)), device=device)
            if logit_id not in neuron_contrib_map:
                neuron_contrib_map[logit_id] = torch.zeros((B, len(tgt_tokens)), device=device)

            # contrib = logit of this tgt token
            neuron_contrib_map[logit_id][batch_idx, target_idx] += logits_BV[batch_idx, batch_idx]

            # attr = sum of contribs from src tokens
            for src_token in src_tokens:
                src_id = (-1, src_token, 0)
                contrib_from_src = neuron_contrib_map.get(
                    src_id, torch.zeros((B, len(tgt_tokens)))
                )[batch_idx, target_idx]
                neuron_attr_map[logit_id][batch_idx, src_token] += contrib_from_src

        # to add logit nodes
        neuron_indices_map = {
            i: focus_logits_target[i].item() for i in range(len(focus_logits_target))
        }
        for idx, neuron_idx in neuron_indices_map.items():
            final_attribution = torch.stack(
                [torch.diagflat(logits_BV[b]) for b in range(logits_BV.shape[0])]
            )[:, idx, :]
            nodes.append(
                Node(
                    layer=L,
                    token=target_token_pos_idx,
                    neuron=neuron_idx,
                    activation=logits_BV[:, idx].float().cpu(),
                    final_attribution=final_attribution.float().cpu(),
                    attr_map=neuron_attr_map.get((L, target_token_pos_idx, neuron_idx), None),
                    contrib_map=neuron_contrib_map.get((L, target_token_pos_idx, neuron_idx), None),
                )
            )

            if return_nodes_only:
                continue

            # creating edges pointing to the logit node (this uses stop grad if specified)
            for (
                source_key,
                source_contrib,
            ) in neuron_contrib_map_with_stop_grad_on_mlps.items():
                # skip incoming from last layer
                if return_nodes_only or source_key[0] == L:
                    continue
                target_key = NeuronIdx(layer=L, token=target_token_pos_idx, neuron=neuron_idx)

                # Move source_contrib to device if needed
                if source_contrib.device.type == "cpu":
                    source_contrib = source_contrib.to(device)

                edge_weight = torch.zeros(len(tokens), device=device)
                edge_weight[idx] = source_contrib[idx, target_idx]

                eps = logits_BV[idx, idx].abs().mean() * 1e-6
                edge_weight = edge_weight / (logits_BV[idx, idx] + eps)

                source_key = NeuronIdx(
                    layer=source_key[0],
                    token=source_key[1],
                    neuron=tokens[idx][source_key[1]] if source_key[0] == -1 else source_key[2],
                )

                # thresholding
                if edge_threshold is not None and edge_weight.abs().max() < edge_threshold:
                    continue
                if parent_threshold is not None and edge_weight.abs().max() < parent_threshold:
                    continue

                edges.append(
                    Edge(
                        src=source_key,
                        tgt=target_key,
                        weight=edge_weight.detach().float().cpu(),
                        final_attribution=(edge_weight[:, None] * final_attribution)
                        .detach()
                        .float()
                        .cpu(),
                    )
                )

    # creating MLP neuron nodes
    for layer in range(max(start_layer, 0), end_layer):
        important_positions = global_important_neurons_mask[layer].nonzero(as_tuple=False)
        for pos_neuron in important_positions:
            token_pos, neuron_idx = pos_neuron.tolist()
            if token_pos in keep_tokens:
                neuron_key = NeuronIdx(layer=layer, token=token_pos, neuron=neuron_idx)
                activation = neurons_LBTI[layer][:, token_pos, neuron_idx]  # shape: (batch,)
                # get attribution scores (already on CPU from earlier)
                attr_map = neuron_attr_map.get(neuron_key, None)
                contrib_map = neuron_contrib_map.get(neuron_key, None)
                final_attribution = mlp_final_attributions[layer, :, token_pos, neuron_idx, :]
                nodes.append(
                    Node(
                        layer=layer,
                        token=token_pos,
                        neuron=neuron_idx,
                        activation=activation.float().cpu(),
                        final_attribution=final_attribution.float().cpu(),
                        attr_map=attr_map if attr_map is not None else None,
                        contrib_map=contrib_map if contrib_map is not None else None,
                    )
                )

    # Clean up after creating MLP nodes
    del mlp_final_attributions
    torch.cuda.empty_cache()

    # creating embedding nodes
    if verbose:
        print("Creating embedding nodes and all outgoing edges...")
    for src_token in src_tokens:
        final_attributions = embed_final_attributions[:, src_token, :]  # (batch, logits)
        for token_type in set([tokens[t][src_token] for t in range(len(tokens))]):
            relevant_idxs = [
                batch_idx
                for batch_idx in range(len(tokens))
                if tokens[batch_idx][src_token] == token_type
            ]
            mask = torch.zeros(len(tokens), device=device)
            mask[relevant_idxs] = 1
            mask = mask.to(torch.bool)
            final_attribution = torch.where(mask[:, None], final_attributions, 0)
            attr_map = torch.zeros(
                (len(tokens), min(len(keep_tokens), len(tokens[0]))), device=device
            )
            attr_map[relevant_idxs, src_token] = 1
            contrib_map = neuron_contrib_map.get((-1, src_token, 0), None)
            activations = torch.ones(len(tokens), 1, device=device)

            # Move contrib_map to device if needed for computation, then back to CPU
            if contrib_map is not None:
                if contrib_map.device.type == "cpu":
                    contrib_map = contrib_map.to(device)
                contrib_map_final = (contrib_map * mask[:, None]).cpu()
            else:
                contrib_map_final = None

            nodes.append(
                Node(
                    layer=-1,
                    token=src_token,
                    neuron=token_type,
                    activation=torch.where(mask, activations[:, 0], 0).float().cpu(),
                    final_attribution=final_attribution.float().cpu(),
                    attr_map=attr_map.cpu(),
                    contrib_map=contrib_map_final,
                )
            )

            if return_nodes_only:
                continue
            # creating edges pointing from the embedding node to the neurons
            for (
                target_key,
                target_attr,
            ) in neuron_attr_map_with_stop_grad_on_mlps.items():
                if return_nodes_only or target_key[0] == -1 or target_key[0] == L:
                    continue
                source_key = NeuronIdx(layer=-1, token=src_token, neuron=token_type)

                target_key = NeuronIdx(
                    layer=target_key[0], token=target_key[1], neuron=target_key[2]
                )
                target_activation = neurons_LBTI[target_key.layer][
                    :, target_key.token, target_key.neuron
                ]  # shape: (B, )
                target_attribution = neuron_contrib_map.get(target_key, None)

                # Move target_attr to device for computation
                target_attr_device = (
                    target_attr.to(device) if target_attr.device.type == "cpu" else target_attr
                )

                # Move target_attribution to device if needed
                if target_attribution is not None and target_attribution.device.type == "cpu":
                    target_attribution = target_attribution.to(device)

                # Use adaptive epsilon for numerical stability (matching CLSO approach)
                eps = target_activation.abs().mean() * 1e-6
                edge_weight = target_attr_device[:, src_token] / (target_activation + eps)
                edge_weight = torch.where(mask, edge_weight, 0)
                # thresholding
                if edge_threshold is not None and edge_weight.abs().max() < edge_threshold:
                    continue
                if parent_threshold is not None and edge_weight.abs().max() < parent_threshold:
                    continue

                edges.append(
                    Edge(
                        src=source_key,
                        tgt=target_key,
                        weight=edge_weight.detach().float().cpu(),
                        final_attribution=(
                            (edge_weight[:, None] * target_attribution).detach().float().cpu()
                            if target_attribution is not None
                            else None
                        ),
                    )
                )

    # Clean up activation tensors after creating all embedding edges
    del neurons_LBTI, resids_LBTD, output_norm_const_BTf11D
    torch.cuda.empty_cache()

    if return_nodes_only:
        return nodes, edges

    # compute edges with raw stop grad HF models
    if verbose:
        print("Computing CLJA-based edges...")

    # get input ids
    if isinstance(attention_masks, torch.Tensor):
        attention_masks = attention_masks.tolist()
    ds = construct_dataset(
        subject,
        [(ci, IdsInput(input_ids=[])) for ci in cis],
        shift_labels=False,
        prompt_attn_mask=attention_masks,
    )
    input_ids, attn_mask_final = (
        torch.tensor([x["input_ids"] for x in ds.to_list()], device=device),  # type: ignore
        torch.tensor([x["attention_mask"] for x in ds.to_list()], device=device),  # type: ignore
    )

    # for any layer pair, we compute the jacobian-based edge weights
    for tgt_layer in range(end_layer - 1, start_layer + 1, -1):
        # if there is no important neurons in the target layer, skip
        if not global_important_neurons_mask[tgt_layer].any():
            continue
        # layers before the target layer only
        for src_layer in range(tgt_layer - 1, start_layer, -1):
            # Skip if no important neurons in source layer (or embeddings)
            if not global_important_neurons_mask[src_layer].any():
                continue

            # get the fixed neuron lists
            src_positions = global_important_neurons_mask[src_layer].nonzero(as_tuple=False)
            src_neuron_list = [
                (pos[0].item(), pos[1].item())
                for pos in src_positions
                if pos[0].item() in keep_tokens
            ]
            tgt_positions = global_important_neurons_mask[tgt_layer].nonzero(as_tuple=False)
            tgt_neuron_list = [
                (pos[0].item(), pos[1].item())
                for pos in tgt_positions
                if pos[0].item() in keep_tokens
            ]

            if verbose:
                print(f"Compute edge weights {tgt_layer} -> {src_layer}")

            # for other layer pairs, calculate the jacobian
            if ig_steps is None:
                relative_attribution = _compute_cl_ja_layer_jacobian(
                    subject.model._model,
                    input_ids,
                    attn_mask_final,
                    src_layer,
                    tgt_layer,
                    src_neuron_list,
                    tgt_neuron_list,
                    keep_tokens,
                    src_tokens if src_layer == -1 else None,
                    # stop grad params
                    use_shapley_grad,
                    use_shapley_qk,
                    use_relp_grad,
                    disable_stop_grad,
                    use_stop_grad_on_mlps,
                    device,
                    tgt_chunk_size=50,  # Can use larger chunk for non-IG
                    verbose=verbose,
                )  # shape: (batch, n_src, n_tgt)
            else:
                relative_attribution = _compute_cl_ja_layer_jacobian_ig(
                    subject.model._model,
                    input_ids,
                    attn_mask_final,
                    src_layer,
                    tgt_layer,
                    src_neuron_list,
                    tgt_neuron_list,
                    keep_tokens,
                    src_tokens if src_layer == -1 else None,
                    device,
                    ig_steps=ig_steps,
                    ig_mode=ig_mode,
                    tgt_chunk_size=20,  # Smaller chunk size for IG to save memory
                    verbose=verbose,
                )  # shape: (batch, n_src, n_tgt)

            # adding edges from every src neuron to every tgt neuron
            for i, (src_token, src_neuron) in enumerate(src_neuron_list):
                for j, (tgt_token, tgt_neuron) in enumerate(tgt_neuron_list):
                    edge_weight = relative_attribution[:, i, j]  # shape: (batch,)

                    # thresholding
                    if edge_threshold is not None and edge_weight.abs().max() < edge_threshold:
                        continue
                    if parent_threshold is not None and edge_weight.abs().max() < parent_threshold:
                        continue

                    target_attribution = neuron_contrib_map.get(
                        (tgt_layer, tgt_token, tgt_neuron), None
                    )  # shape: (B, logits)

                    # Move target_attribution to device if needed
                    if target_attribution is not None and target_attribution.device.type == "cpu":
                        target_attribution = target_attribution.to(device)

                    src_key = NeuronIdx(layer=src_layer, token=src_token, neuron=src_neuron)
                    tgt_key = NeuronIdx(layer=tgt_layer, token=tgt_token, neuron=tgt_neuron)

                    edges.append(
                        Edge(
                            src=src_key,
                            tgt=tgt_key,
                            weight=edge_weight.detach().float().cpu(),
                            final_attribution=(edge_weight[:, None] * target_attribution)
                            .detach()
                            .float()
                            .cpu(),
                        )
                    )

            # Clean up after processing this layer pair
            del relative_attribution
            torch.cuda.empty_cache()

    if verbose:
        print(f"# found nodes: {len(nodes)}")
        print(f"# found edges: {len(edges)}")

    return nodes, edges


def _compute_cl_ja_layer_jacobian(
    model,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    src_layer: int,
    tgt_layer: int,
    src_neuron_list,
    tgt_neuron_list,
    keep_tokens: list[int],
    src_tokens: list[int] | None,
    use_shapley_grad: bool,
    use_shapley_qk: bool,
    use_relp_grad: bool,
    disable_stop_grad: bool,
    use_stop_grad_on_mlps: bool,
    device: str,
    alpha: float | None = None,
    tgt_chunk_size: int = 50,
    verbose: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute CLJA edge weights between source and target layers using batched jacobian.

    Args:
        alpha: Optional scaling factor for integrated gradients. When set, returns
               (gradients, src_acts, tgt_acts) instead of final attributions.

    Returns:
        If alpha is None: relative_attribution tensor of shape (batch, n_src, n_tgt)
        If alpha is not None: tuple of (gradients, src_acts, tgt_acts)
    """

    for layer_idx in range(len(model.model.layers)):
        remove_forward_hooks(model.model.layers[layer_idx].mlp.down_proj)

    # stop grads
    if not disable_stop_grad:
        model = layerwise_stop_nonlinear_grad_for_llama(
            model,
            src_layer,
            tgt_layer,
            use_shapley_grad,
            use_shapley_qk,
            use_relp_grad,
            use_stop_grad_on_mlps,
        )
    model.zero_grad()
    # populate activation cache for getting jacobian
    embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()
    if alpha is not None:
        embeds = embeds * alpha
    activation_cache = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            activation_cache[layer_idx] = input[0]  # shape: (batch, seq, hidden)

        return hook

    handles = []
    if not disable_stop_grad:
        src_handle = model.model.layers[src_layer].mlp.mlp.down_proj.register_forward_hook(
            make_hook(src_layer)
        )
        tgt_handle = model.model.layers[tgt_layer].mlp.mlp.down_proj.register_forward_hook(
            make_hook(tgt_layer)
        )
    else:
        src_handle = model.model.layers[src_layer].mlp.down_proj.register_forward_hook(
            make_hook(src_layer)
        )
        tgt_handle = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(
            make_hook(tgt_layer)
        )
    handles.append(src_handle)
    handles.append(tgt_handle)
    _ = model(inputs_embeds=embeds, attention_mask=attention_masks)

    # get batch size
    batch = input_ids.shape[0]

    # Collect source activations for all neurons (needed later)
    src_acts_full = activation_cache[src_layer]  # (batch, seq, hidden)
    src_acts_list = []
    for src_pos, src_neuron in src_neuron_list:
        src_acts_list.append(src_acts_full[:, src_pos, src_neuron])  # (batch,)
    src_acts = torch.stack(src_acts_list, dim=-1)  # (batch, n_src)

    # Process target neurons in chunks to avoid OOM
    n_tgt = len(tgt_neuron_list)
    tgt_acts_full = activation_cache[tgt_layer]  # (batch, seq, hidden)

    # Collect all target activations
    tgt_acts_list = []
    for token, neuron in tgt_neuron_list:
        tgt_acts_list.append(tgt_acts_full[:, token, neuron])  # (batch,)
    tgt_activations = torch.stack(tgt_acts_list, dim=0)  # (n_tgt, batch)

    # Process in chunks
    src_ja_chunks = []
    for chunk_start in range(0, n_tgt, tgt_chunk_size):
        chunk_end = min(chunk_start + tgt_chunk_size, n_tgt)
        tgt_chunk = tgt_activations[chunk_start:chunk_end]  # (n_chunk, batch)
        n_chunk = tgt_chunk.shape[0]

        # prepare batched grads for this chunk
        grad_outputs = torch.eye(
            batch * n_chunk, device=device
        )  # (n_chunk * batch, n_chunk * batch)

        src_acts_full.grad = None
        chunk_src_ja = torch.autograd.grad(
            tgt_chunk.flatten(),  # (n_chunk * batch)
            src_acts_full,  # (batch, seq, d)
            grad_outputs=grad_outputs,  # (n_chunk * batch, n_chunk * batch)
            is_grads_batched=True,
            retain_graph=True,
        )[0]
        # shape: (n_chunk * batch, batch, seq, d)
        # convert back to (n_chunk, batch, batch, seq, d)
        chunk_src_ja = chunk_src_ja.reshape(
            n_chunk, batch, batch, chunk_src_ja.shape[-2], chunk_src_ja.shape[-1]
        )
        # get only identity along batch, batch
        chunk_src_ja = chunk_src_ja.diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
        # now (n_chunk, batch, seq, d)

        src_ja_chunks.append(chunk_src_ja)

        # Clean up memory after each chunk
        del grad_outputs, chunk_src_ja
        torch.cuda.empty_cache()

    # Concatenate all chunks
    src_ja = torch.cat(src_ja_chunks, dim=0)  # (n_tgt, batch, seq, d)
    del src_ja_chunks
    torch.cuda.empty_cache()

    if alpha is not None:
        # For IG: return gradients only, multiply by activations later
        grads = []
        for src_pos, src_neuron in src_neuron_list:
            grads.append(src_ja[:, :, src_pos, src_neuron])  # (n_tgt, batch)
        grads = torch.stack(grads, dim=-1)  # (n_tgt, batch, n_src)
        grads = grads.permute(1, 2, 0).contiguous()  # (batch, n_src, n_tgt)

        # Clean up src_ja
        del src_ja
        torch.cuda.empty_cache()

        # tgt_activations = (n_tgt, batch)
        tgt_acts = tgt_activations.detach().permute(1, 0)  # (batch, n_tgt)

        # remove all hooks
        for handle in handles:
            handle.remove()

        # revert stop grads
        if not disable_stop_grad:
            model = layerwise_revert_stop_nonlinear_grad_for_llama(
                model,
                src_layer,
                tgt_layer,
            )

        return (
            grads,
            src_acts,
            tgt_acts,
        )  # (batch, n_src, n_tgt), (batch, n_src), (batch, n_tgt)

    else:
        # For regular attribution: gradient * activation
        jvps = []
        for src_pos, src_neuron in src_neuron_list:
            jvps.append(
                src_ja[:, :, src_pos, src_neuron]
                * src_acts_full[:, src_pos, src_neuron][None, :].detach()
            )  # (n_tgt, batch) * (1, batch) -> (n_tgt, batch)
        jvps = torch.stack(jvps, dim=-1)  # (n_tgt, batch, n_src)
        jvps = jvps.permute(1, 2, 0).contiguous()  # (batch, n_src, n_tgt)

        # Clean up src_ja
        del src_ja
        torch.cuda.empty_cache()

        # tgt_activations = (n_tgt, batch)
        tgt_activations = tgt_activations.detach().permute(1, 0)  # (batch, n_tgt)
        # Use adaptive epsilon for numerical stability (matching CLSO approach)
        eps = tgt_activations.abs().mean() * 1e-6
        relative_attribution = jvps / (
            tgt_activations[:, None, :] + eps
        )  # (batch, n_src, n_tgt) / (batch, 1, n_tgt)

        # Clean up jvps
        del jvps
        torch.cuda.empty_cache()

        # remove all hooks
        for handle in handles:
            handle.remove()

        # revert stop grads
        if not disable_stop_grad:
            model = layerwise_revert_stop_nonlinear_grad_for_llama(
                model,
                src_layer,
                tgt_layer,
            )

        return relative_attribution  # shape: (batch, n_src, n_tgt)


def _compute_cl_ja_layer_jacobian_ig(
    model,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    src_layer: int,
    tgt_layer: int,
    src_neuron_list,
    tgt_neuron_list,
    keep_tokens: list[int],
    src_tokens: list[int] | None,
    device: str,
    ig_steps: int = 10,
    ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs",
    tgt_chunk_size: int = 20,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute CLJA edge weights using Integrated Gradients.

    Note: For IG, stop gradients are disabled (disable_stop_grad=True).

    Args:
        ig_steps: Number of steps for integrated gradients.
        ig_mode: Mode for integrated gradients aggregation ("ig-inputs" or "conductance").

    Returns:
        relative_attribution tensor of shape (batch, n_src, n_tgt)
    """
    # Collect step-wise gradients and activations
    grads_steps = []
    src_acts_steps = []
    tgt_acts_steps = []

    for step in range(0, ig_steps + 1):
        alpha = step / ig_steps

        grads, src_acts, tgt_acts = _compute_cl_ja_layer_jacobian(
            model=model,
            input_ids=input_ids,
            attention_masks=attention_masks,
            src_layer=src_layer,
            tgt_layer=tgt_layer,
            src_neuron_list=src_neuron_list,
            tgt_neuron_list=tgt_neuron_list,
            keep_tokens=keep_tokens,
            src_tokens=src_tokens,
            use_shapley_grad=False,  # Not used for IG
            use_shapley_qk=False,  # Not used for IG
            use_relp_grad=False,  # Not used for IG
            disable_stop_grad=True,  # Always disable stop grad for IG
            use_stop_grad_on_mlps=False,  # Not used for IG
            device=device,
            alpha=alpha,
            tgt_chunk_size=tgt_chunk_size,
            verbose=verbose,
        )

        # Immediately detach and move to CPU to save GPU memory
        grads_steps.append(grads.detach().cpu())
        src_acts_steps.append(src_acts.detach().cpu())
        tgt_acts_steps.append(tgt_acts.detach().cpu())

        # Clean up GPU tensors and computation graph
        del grads, src_acts, tgt_acts
        torch.cuda.empty_cache()

        # Force garbage collection of computation graph
        if step % 5 == 0:  # Every 5 steps
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    if ig_mode == "ig-inputs":
        # Riemann sum in IG (ignore step 0)
        # Average gradients across steps (move back to device for computation)
        grads_avg = torch.stack(grads_steps[1:]).mean(dim=0).to(device)  # (batch, n_src, n_tgt)

        # Compute activation differences
        src_acts_diff = (src_acts_steps[-1] - src_acts_steps[0]).to(device)  # (batch, n_src)
        tgt_acts_final = src_acts_steps[-1].to(device)  # (batch, n_tgt)

        # Apply IG formula: averaged_gradient * src_activation_diff
        # Shape: (batch, n_src, n_tgt) * (batch, n_src, 1) -> (batch, n_src, n_tgt)
        jvps = grads_avg * src_acts_diff[:, :, None]

        # Normalize by target activations (use final activations)
        tgt_acts_final = tgt_acts_steps[-1].to(device)  # (batch, n_tgt)
        eps = tgt_acts_final.abs().mean() * 1e-6
        relative_attribution = jvps / (tgt_acts_final[:, None, :] + eps)

    elif ig_mode == "conductance":
        # Stack all steps (excluding step 0) and move to device
        grads_all = torch.stack(grads_steps[1:]).to(device)  # (steps, batch, n_src, n_tgt)

        # Compute step-wise differences in activations
        src_acts_all = torch.stack(src_acts_steps).to(device)  # (steps+1, batch, n_src)

        src_acts_diffs = torch.diff(src_acts_all, dim=0)  # (steps, batch, n_src)

        # Apply conductance: sum over steps of (gradient * src_activation_diff)
        # Shape: (steps, batch, n_src, n_tgt) * (steps, batch, n_src, 1) -> sum over steps
        jvps = (grads_all * src_acts_diffs[:, :, :, None]).sum(dim=0)  # (batch, n_src, n_tgt)

        # Normalize by target activations (use final activations)
        tgt_acts_final = tgt_acts_steps[-1].to(device)  # (batch, n_tgt)
        eps = tgt_acts_final.abs().mean() * 1e-6
        relative_attribution = jvps / (tgt_acts_final[:, None, :] + eps)

    else:
        raise ValueError(f"Invalid IG mode: {ig_mode}")

    return relative_attribution  # shape: (batch, n_src, n_tgt)
