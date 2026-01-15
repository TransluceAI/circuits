from typing import Any, List, Literal

import torch
from circuits.core.attribution import EmbedBuffer, LogitsBuffer, MLPBuffer, attribute_lightweight
from circuits.core.grad import (
    layerwise_revert_stop_nonlinear_grad_for_llama,
    layerwise_stop_nonlinear_grad_for_llama,
    remove_forward_hooks,
    revert_stop_nonlinear_grad_for_llama,
    stop_nonlinear_grad_for_llama,
)
from circuits.core.utils import Edge, NeuronIdx, Node, collect_acts
from opt_einsum import contract
from tqdm import tqdm
from util.chat_input import IdsInput, ModelInput
from util.subject import Subject, construct_dataset


def precompute_cross_layer_headwise_OV(
    subject: Subject,
    layers: List[int],
    collect_layers: List[int],
    device: str = "cuda:0",
):
    """
    Precompute W_{OV} matrices for each head at each layer.
    Can run this on any device; defaults to 0.
    """

    attn_vs_LXD = {
        layer: subject.attn_vs[layer].weight.detach().to(device) for layer in collect_layers
    }
    attn_os_LDD = {
        layer: subject.attn_os[layer].weight.detach().to(device) for layer in collect_layers
    }
    K, Q, D, L = subject.K, subject.Q, subject.D, subject.L

    # Repeat W_V matrices into (D, D); originally (X, D) where X<D due to multi-query attention
    attn_vs_LDD = {
        layer: attn_vs_LXD[layer].view(K, D // Q, D).repeat_interleave(Q // K, dim=0).view(D, D)
        for layer in list(collect_layers)
    }

    # Compute W_{OV} matrices for each head
    attn_ovs_LQDD = {}
    for effect_layer in range(min(layers) + 1, max(collect_layers) + 1):  # Only to target_layer
        attn_o_DD, attn_v_DD = (
            attn_os_LDD[effect_layer],
            attn_vs_LDD[effect_layer],
        )

        # Split W_V, W_O matrices into heads
        attn_v_QHD = attn_v_DD.reshape(Q, D // Q, D)
        attn_o_DQH = attn_o_DD.reshape(D, Q, D // Q)
        # Compute W_{OV} matrix for each head
        attn_ovs_LQDD[effect_layer] = torch.bmm(attn_o_DQH.permute(1, 0, 2), attn_v_QHD)

        # Deallocate memory
        del attn_o_DD, attn_v_QHD, attn_o_DQH

    return attn_ovs_LQDD


def get_layer_headwise_OV(
    subject: Subject,
    layer: int,
    device: str = "cuda:0",
):
    """
    Without precomputing, we can compute W_{OV} for a single layer headwise.
    Can save memory by not precomputing and saving the OV matrices.
    """
    attn_vs = subject.attn_vs[layer].weight.detach()
    attn_o_DD = subject.attn_os[layer].weight.detach()
    K, Q, D, H = subject.K, subject.Q, subject.D, subject.H
    # check if head_dim * num_heads = D, if not, we handle differently, e.g., gemma2
    if D == H:
        attn_v_DD = attn_vs.view(K, D // Q, D).repeat_interleave(Q // K, dim=0).view(D, D)
        attn_v_QHD = attn_v_DD.reshape(Q, D // Q, D)
        attn_o_DQH = attn_o_DD.reshape(D, Q, D // Q)
    else:
        attn_v_DD = attn_vs.view(K, H, D).repeat_interleave(Q // K, dim=0).view(H * Q, D)
        attn_v_QHD = attn_v_DD.reshape(Q, H, D)
        attn_o_DQH = attn_o_DD.reshape(D, Q, H)
    attn_ov_result = torch.bmm(attn_o_DQH.permute(1, 0, 2), attn_v_QHD)
    return attn_ov_result


def _get_grad_attributions_from_logits(
    subject: Subject,
    model,
    input_ids,
    keep_tokens: list[int],
    focus_positions: list[list[int]] | list[int],
    focus_logits: list[list[int]] | list[int],
    focus_last_residual: bool = False,
    attention_masks: list[list[int]] | None = None,
    ablation_mode: Literal["zero", "mean", "activations"] = "zero",
    disable_stop_grad: bool = False,
    center_logits: bool = False,
    alpha: float | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[NeuronIdx], torch.Tensor]:
    """
    Get the grad attributions for all MLP neurons to the given tokens.

    Args:
        model: The model to get the grad attributions from.
        input_ids: The input id tokens to the model.
        focus_positions: The positions of the tokens to get the grad attributions to.
        focus_logits: The next-token logits of the tokens to get the grad attributions to.
        attention_masks: The attention masks for the input ids.
        ablation_mode: The ablation mode for the grad attributions.

    Returns:
        The grad attributions for all MLP neurons and embed tokens to the given tokens, under the given settings.
        - layer_grad_attr: (L, B, T, D_ff)
        - embed_grad_attr: (B, T)
    """

    for param in model.parameters():
        param.requires_grad = False

    # compute neuron attributions to source tokens and contributions to target tokens
    # clear hooks
    for i in range(len(model.model.layers)):
        if hasattr(model.model.layers[i].mlp, "mlp"):
            remove_forward_hooks(model.model.layers[i].mlp.mlp.down_proj)
        else:
            remove_forward_hooks(model.model.layers[i].mlp.down_proj)

    # differentiable embeds
    embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()
    if alpha is not None:
        embeds = embeds * alpha
    cache: dict[int, torch.Tensor] = {}
    for layer_index in range(len(model.model.layers)):

        def _hook(layer_index):
            def fn(_, input, output):
                cache[layer_index] = input[0]

            return fn

        if not disable_stop_grad:
            model.model.layers[layer_index].mlp.mlp.down_proj.register_forward_hook(
                _hook(layer_index)
            )
        else:
            model.model.layers[layer_index].mlp.down_proj.register_forward_hook(_hook(layer_index))

    # forward pass
    out = model(inputs_embeds=embeds, attention_mask=attention_masks)
    if focus_last_residual:
        goal = cache[len(model.model.layers) - 1][:, -1].sum()  # last residual stream vector
        goal_value = (
            cache[len(model.model.layers) - 1][:, -1].detach().sum(dim=-1)
        )  # shape: (batch,)
    else:
        logits = out.logits[:, focus_positions]  # shape: (batch, positions, vocab)
        foc_log = torch.tensor(focus_logits, device=logits.device)  # shape: (batch, positions)
        # Add an extra dimension so shapes line up
        selected_logits = torch.gather(
            logits,  # (B, P, V)
            dim=2,  # select along vocab axis
            index=foc_log.unsqueeze(-1),  # (B, P, 1)
        ).squeeze(
            -1
        )  # (B, P)
        # optionally subtract the mean of all logits
        if center_logits:
            selected_logits -= logits.mean(dim=-1)
        if verbose:
            print("selected_logits", selected_logits)
        goal = selected_logits.sum()  # only grab the associated logits for each batch item
        goal_value = selected_logits.detach().sum(dim=-1)  # shape: (batch,)

    # collect grad attributions
    layer_grad_attr = []
    layer_acts = []
    for layer_index in range(len(model.model.layers)):
        grad_attr = torch.autograd.grad(
            goal,
            cache[layer_index],
            retain_graph=True,
        )[0]
        if alpha is not None:
            attributions = grad_attr
        elif ablation_mode == "mean":
            # TODO: attention-based mean ablation is needed
            # take mean over batch
            attributions = grad_attr * (
                cache[layer_index] - cache[layer_index].mean(dim=0, keepdim=True)
            )
        elif ablation_mode == "zero":
            attributions = grad_attr * cache[layer_index]
        else:
            raise ValueError(f"Invalid ablation mode: {ablation_mode}")
        layer_grad_attr.append(attributions.detach())
        layer_acts.append(cache[layer_index].detach())

    # collect embed grad attr
    # shape: (B, T)
    if verbose:
        print("goal", goal.shape)
        print("embeds", embeds.shape)
    embed_attributions = torch.autograd.grad(
        goal,
        embeds,
        retain_graph=True,
    )[0]
    if alpha is not None:
        embed_grad_attr = embed_attributions.detach()
    elif ablation_mode == "mean":
        embed_attributions = embed_attributions * (embeds - embeds.mean(dim=0, keepdim=True))
        embed_grad_attr = embed_attributions.sum(dim=-1).detach()
    elif ablation_mode == "zero":
        embed_attributions = embed_attributions * embeds
        embed_grad_attr = embed_attributions.sum(dim=-1).detach()
    else:
        raise ValueError(f"Invalid ablation mode: {ablation_mode}")

    # the shape of layer_grad_attr is (L, B, T, D_ff)
    # we want to aggregate over the batch dimension
    layer_grad_attr = torch.stack(layer_grad_attr)
    layer_acts = torch.stack(layer_acts)

    # compute neuron attributions to source tokens and contributions to target tokens
    for i in range(len(model.model.layers)):
        if hasattr(model.model.layers[i].mlp, "mlp"):
            remove_forward_hooks(model.model.layers[i].mlp.mlp.down_proj)
        else:
            remove_forward_hooks(model.model.layers[i].mlp.down_proj)

    return layer_grad_attr, embed_grad_attr, goal_value, layer_acts, embeds


def _get_ig_attributions_from_logits(
    subject: Subject,
    model,
    input_ids,
    keep_tokens: list[int],
    focus_positions: list[list[int]] | list[int],
    focus_logits: list[list[int]] | list[int],
    focus_last_residual: bool = False,
    attention_masks: list[list[int]] | None = None,
    disable_stop_grad: bool = False,
    center_logits: bool = False,
    ig_steps: int = 10,
    ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs",
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the IG attributions for all MLP neurons to the given tokens.
    """
    # get step-wise grads
    mlp_final_attributions = []
    embed_final_attributions = []
    mlp_final_acts = []
    embed_final_acts = []
    goal_value = []
    for step in range(0, ig_steps + 1):
        alpha = step / ig_steps
        (
            mlp_step_attributions,
            embed_step_attributions,
            goal_value_step,
            mlp_step_acts,
            embed_step_acts,
        ) = _get_grad_attributions_from_logits(
            subject,
            model,
            input_ids,
            keep_tokens,
            focus_positions,
            focus_logits=focus_logits,
            focus_last_residual=focus_last_residual,
            attention_masks=attention_masks,
            ablation_mode="zero",
            disable_stop_grad=disable_stop_grad,
            center_logits=center_logits,
            verbose=verbose,
            alpha=alpha,
        )
        # Immediately detach and move to CPU to save GPU memory
        mlp_final_attributions.append(mlp_step_attributions.detach().cpu())
        embed_final_attributions.append(embed_step_attributions.detach().cpu())
        mlp_final_acts.append(mlp_step_acts.detach().cpu())
        embed_final_acts.append(embed_step_acts.detach().cpu())
        goal_value.append(goal_value_step.detach())

        # Clean up GPU tensors and computation graph
        del mlp_step_attributions, embed_step_attributions, mlp_step_acts, embed_step_acts
        torch.cuda.empty_cache()

        # Force garbage collection every few steps
        if step % 5 == 0:
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    # Get device from model
    device = input_ids.device

    if ig_mode == "ig-inputs":
        # WE WILL IGNORE STEP 0
        # reimann sum in IG (move back to device for computation)
        mlp_final_attributions = torch.stack(mlp_final_attributions[1:]).mean(dim=0).to(device)
        embed_final_attributions = torch.stack(embed_final_attributions[1:]).mean(dim=0).to(device)

        # multiply by act diff
        mlp_activation_diff = (mlp_final_acts[-1] - mlp_final_acts[0]).to(device)
        embed_activation_diff = (embed_final_acts[-1] - embed_final_acts[0]).to(device)

        # multiply and collapse embed dims (i.e. outside the integral in IG)
        mlp_final_attributions = mlp_final_attributions * mlp_activation_diff
        embed_final_attributions = embed_final_attributions * embed_activation_diff
        embed_final_attributions = embed_final_attributions.sum(dim=-1).detach()
    elif ig_mode == "conductance":
        # Move to device for computation
        mlp_final_attributions = torch.stack(mlp_final_attributions[1:]).to(device)
        embed_final_attributions = torch.stack(embed_final_attributions[1:]).to(device)

        # all difs
        mlp_final_acts = torch.stack(mlp_final_acts).to(device)
        embed_final_acts = torch.stack(embed_final_acts).to(device)
        mlp_activation_diffs = torch.diff(mlp_final_acts, dim=0)
        embed_activation_diffs = torch.diff(embed_final_acts, dim=0)

        # multiply gradients by diffs
        mlp_final_attributions = (mlp_final_attributions * mlp_activation_diffs).sum(dim=0)
        embed_final_attributions = (
            (embed_final_attributions * embed_activation_diffs).sum(dim=0).sum(dim=-1).detach()
        )
    else:
        raise ValueError(f"Invalid IG mode: {ig_mode}")

    # return
    return (
        mlp_final_attributions,
        embed_final_attributions,
        goal_value[-1],
        mlp_final_acts[-1],
        embed_final_acts[-1],
    )


@torch.no_grad()
def _get_global_important_neurons_mask(
    keep_tokens: list[int],
    start_layer: int,
    end_layer: int,
    mlp_final_attributions: torch.Tensor,
    node_attribution_threshold: float | None = 1.0,
    topk_neurons: int | None = None,
    absolute_attribution_threshold: torch.Tensor | None = None,
    batch_aggregation: Literal["mean", "max", "max_abs", "any"] = "max",
    verbose: bool = False,
) -> torch.Tensor:
    """
    Filter for the most important MLP neurons.

    Args:
        mlp_final_attributions: Tensor with shape allowing indexing as [layer, batch, token_pos, dim, focus_logits].
        node_attribution_threshold: Cumulative absolute attribution threshold.
        topk_neurons: Topk neurons to select.

    Returns:
        A mask of shape [num_layers, num_tokens, num_neurons] indicating which neurons are important.
    """
    # Only one strategy can be used at a time
    if node_attribution_threshold is not None and topk_neurons is not None:
        raise ValueError("Only one strategy can be used at a time")

    # first aggregate over batch
    if batch_aggregation == "mean":
        mlp_final_attributions = torch.mean(mlp_final_attributions, dim=1, keepdim=True)
    elif batch_aggregation == "max":
        mlp_final_attributions = torch.max(mlp_final_attributions, dim=1, keepdim=True).values
    elif batch_aggregation == "max_abs":
        mlp_final_attributions = torch.max(
            torch.abs(mlp_final_attributions), dim=1, keepdim=True
        ).values
    elif batch_aggregation == "any":
        pass  # keep batch dim

    # aggregate over the last dimension of the logits
    mlp_final_attributions = mlp_final_attributions.sum(dim=-1)

    # Stack all layers into a single tensor for efficient processing
    # Resulting shape: [batch, num_layers, num_tokens, dim_neurons]
    # batch is 1 if batch_aggregation is not "any"
    abs_attributions = mlp_final_attributions.abs().permute(1, 0, 2, 3)

    # if it is not in the keep tokens, set them to zeros
    token_mask = ~torch.isin(torch.arange(abs_attributions.shape[2]), torch.tensor(keep_tokens))
    abs_attributions[:, :, token_mask, :] = 0

    # if layer indices fall outside start_layer and end_layer, set them to zeros
    layer_indices = torch.arange(abs_attributions.shape[1])
    layer_mask = (layer_indices < start_layer) | (layer_indices >= end_layer)
    abs_attributions[:, layer_mask, :, :] = 0

    if verbose:
        print("abs_attributions", abs_attributions.shape)

    # Flatten to find threshold
    # keep batch dim
    flat_abs_attributions = abs_attributions.flatten(start_dim=1)
    if verbose:
        print("flat_abs_attributions", flat_abs_attributions.shape)
    nonzero_values = flat_abs_attributions[flat_abs_attributions > 0]
    if len(nonzero_values) == 0:
        if verbose:
            print("all zero values")
        return torch.zeros_like(abs_attributions)[0]

    # option 1: threshold
    if node_attribution_threshold is not None:
        # Sort without stable=True to save memory
        sorted_values = torch.sort(flat_abs_attributions, descending=True, dim=-1).values
        if verbose:
            print("sorted_values", sorted_values.shape)

        total_mass = sorted_values.sum(dim=-1, keepdim=True)
        cumulative_mass = torch.cumsum(sorted_values, dim=-1)
        cumulative_ratio = cumulative_mass / total_mass  # shape: [batch, num_vals]

        # Find threshold value
        thr = torch.full(
            cumulative_ratio.shape[:-1],  # [B]
            float(node_attribution_threshold),
            dtype=cumulative_ratio.dtype,
            device=cumulative_ratio.device,
        ).unsqueeze(-1)
        cutoff_idx = torch.searchsorted(cumulative_ratio, thr) + 1
        cutoff_idx = cutoff_idx.squeeze(-1)
        threshold_value = sorted_values[torch.arange(sorted_values.shape[0]), cutoff_idx - 1]

        # Create mask and get coordinates directly
        mask = abs_attributions >= threshold_value[:, None, None, None]

        # Clean up
        del sorted_values, cumulative_mass, cumulative_ratio
        torch.cuda.empty_cache()
    # option 2: get topk neurons - use topk instead of full sort
    elif topk_neurons is not None:
        # Use topk directly instead of full sort to save memory
        topk_values = torch.topk(
            flat_abs_attributions, k=min(topk_neurons, flat_abs_attributions.shape[-1]), dim=-1
        ).values
        threshold_value = topk_values[:, -1]  # Last value in topk is the threshold
        if verbose:
            print("threshold_value", threshold_value.shape)
        mask = abs_attributions >= threshold_value[:, None, None, None]

        # Clean up
        del topk_values
        torch.cuda.empty_cache()
    # option 3: get neurons with absolute attribution above a threshold
    elif absolute_attribution_threshold is not None:
        threshold_value = absolute_attribution_threshold
        mask = abs_attributions >= threshold_value[:, None, None, None]
    else:
        raise ValueError(
            "Either node_attribution_threshold, topk_neurons, or absolute_attribution_threshold must be provided"
        )

    if batch_aggregation != "any":
        mask = mask.squeeze()
    else:
        if verbose:
            print("threshold_value", threshold_value)
        mask = mask.any(dim=0)
    if verbose:
        print("mask", mask.shape)

    return mask


def _get_neuron_attr_and_contrib_with_stop_grad_on_mlps(
    model,
    neuron_cfg: dict[int, list[list[int]]],
    input_ids: torch.Tensor,
    src_tokens: list[int],
    tgt_tokens: list[int],
    focus_positions: list[int],
    focus_logits,
    attention_masks,
    use_shapley_grad: bool = False,
    use_shapley_qk: bool = False,
    use_relp_grad: bool = False,
    center_logits: bool = False,
    neuron_chunk_size: int = 50,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[NeuronIdx]]:
    """
    Compute neuron attributions from source tokens and contributions to target tokens
    with stop gradient on MLPs.
    """

    model = revert_stop_nonlinear_grad_for_llama(model)
    model.zero_grad()
    for i in range(len(model.model.layers)):
        remove_forward_hooks(model.model.layers[i].mlp.down_proj)

    device = input_ids.device

    # process neuron activations layer by layer for memory efficiency
    attr_list = []
    neuron_acts_list = []
    neuron_tags: list[NeuronIdx] = []

    for lid, pairs in tqdm(
        neuron_cfg.items(), desc="Computing neuron attributions", disable=not verbose
    ):
        if not pairs:  # skip empty layers
            continue

        # stop gradient on MLPs
        model = layerwise_stop_nonlinear_grad_for_llama(
            model,
            -1,
            lid,
            use_shapley_grad,
            use_shapley_qk,
            use_relp_grad,
            True,
        )

        cache = {}

        def _hook(lid):
            def fn(_, input, output):
                cache[lid] = input[0]

            return fn

        if hasattr(model.model.layers[lid].mlp, "mlp"):
            model.model.layers[lid].mlp.mlp.down_proj.register_forward_hook(_hook(lid))
        else:
            model.model.layers[lid].mlp.down_proj.register_forward_hook(_hook(lid))

        # differentiable embeds
        # shape: (batch, seq, d)
        embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()

        # forward pass
        out = model(inputs_embeds=embeds, attention_mask=attention_masks)
        logits = out.logits
        if center_logits:
            logits -= logits.mean(dim=-1)

        act = cache[lid]  # (batch, seq, d)

        # Process this layer's neurons in chunks
        all_pairs = list(pairs)
        layer_attr_chunks = []
        layer_neuron_acts_chunks = []

        for chunk_start in range(0, len(all_pairs), neuron_chunk_size):
            chunk_pairs = all_pairs[chunk_start : chunk_start + neuron_chunk_size]

            layer_neuron_acts = []
            layer_neuron_tags = []

            for pos, nid in chunk_pairs:
                layer_neuron_acts.append(act[:, pos, nid])
                layer_neuron_tags.append(NeuronIdx(layer=lid, token=pos, neuron=nid))

            if not layer_neuron_acts:  # skip if no neurons
                continue

            # get all activations
            layer_neuron_acts = torch.stack(layer_neuron_acts)  # (n_chunk, batch)
            batch = layer_neuron_acts.shape[1]
            n_chunk = layer_neuron_acts.shape[0]
            grad_outputs = torch.eye(
                n_chunk * batch, device=device
            )  # (n_chunk * batch, n_chunk * batch)

            # attribution to inputs for this layer chunk
            embeds.grad = None
            layer_grad_attr = torch.autograd.grad(
                layer_neuron_acts.flatten(),  # (n_chunk * batch)
                embeds,  # (batch, seq, d)
                grad_outputs=grad_outputs,  # (n_chunk * batch, n_chunk * batch)
                is_grads_batched=True,
                retain_graph=True,
            )[0]
            # shape: (n_chunk * batch, batch, seq, d)
            # convert back to (n_chunk, batch, batch, seq, d)
            layer_grad_attr = layer_grad_attr.reshape(
                n_chunk, batch, batch, layer_grad_attr.shape[-2], layer_grad_attr.shape[-1]
            )
            # get only identity along batch, batch
            layer_grad_attr = layer_grad_attr.diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
            # now (n_chunk, batch, seq, d)

            layer_attr = (
                (layer_grad_attr.to(torch.float32) * embeds[None, ...].to(torch.float32))
                .sum(-1)
                .to(embeds.dtype)
            )  # (n_chunk, batch, seq)
            layer_attr = layer_attr[:, :, src_tokens]  # (n_chunk, batch, src)

            layer_attr_chunks.append(layer_attr)
            layer_neuron_acts_chunks.append(layer_neuron_acts.detach())
            neuron_tags.extend(layer_neuron_tags)

            # Clean up memory after each chunk
            del layer_grad_attr, layer_attr, grad_outputs
            torch.cuda.empty_cache()

        # Concatenate chunks for this layer
        if layer_attr_chunks:
            attr_list.append(torch.cat(layer_attr_chunks, dim=0))
            neuron_acts_list.append(torch.cat(layer_neuron_acts_chunks, dim=0))

        # revert stop grad for the next attribution
        model = layerwise_revert_stop_nonlinear_grad_for_llama(
            model,
            -1,
            lid,
        )
        remove_forward_hooks(model.model.layers[lid].mlp.down_proj)

        # clean up memory
        del embeds, layer_attr_chunks, layer_neuron_acts_chunks
        torch.cuda.empty_cache()

    # concatenate results from all layers
    attr = torch.cat(attr_list, dim=0)  # (neurons, batch, seq)
    neuron_acts = torch.cat(neuron_acts_list, dim=0)  # (neurons, batch)

    # clean up layer-wise lists to free memory
    del attr_list, neuron_acts_list
    torch.cuda.empty_cache()

    # contribution to tgt tokens
    # stop gradient on MLPs
    model = layerwise_stop_nonlinear_grad_for_llama(
        model,
        -1,
        len(model.model.layers),
        use_shapley_grad,
        use_shapley_qk,
        use_relp_grad,
        True,
    )

    # differentiable embeds
    # shape: (batch, seq, d)
    embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()

    # forward pass
    out = model(inputs_embeds=embeds, attention_mask=attention_masks)
    logits = out.logits
    if center_logits:
        logits -= logits.mean(dim=-1)

    tgt_nodes = []
    for id, p in enumerate(focus_positions):
        tid = [focus_logit[id] for focus_logit in focus_logits]
        tgt_nodes.append(logits[torch.arange(logits.shape[0]), p, tid])
    tgt_vec = torch.stack(tgt_nodes)  # (t, batch)
    t = tgt_vec.size(0)
    grad_outputs = torch.eye(t * batch, device=device)  # (t * batch, t * batch)

    # compute embed grad contribs
    embeds.grad = None
    embed_grad_contrib_full = torch.autograd.grad(
        tgt_vec.flatten(),
        embeds,
        grad_outputs=grad_outputs,
        is_grads_batched=True,
        retain_graph=True,
    )[0]
    # shape (t * batch, batch, seq, d)
    # convert back to (t, batch, batch, seq, d)
    embed_grad_contrib = embed_grad_contrib_full.reshape(
        t, batch, batch, embed_grad_contrib_full.shape[-2], embed_grad_contrib_full.shape[-1]
    )
    del embed_grad_contrib_full  # Free immediately

    # get only identity along batch, batch
    embed_grad_contrib = embed_grad_contrib.diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
    # now (t, batch, seq, d)
    embed_grad_contrib = (
        embed_grad_contrib[:, :, src_tokens, :] * embeds[None, :, src_tokens, :]
    )  # (t, batch, src, d)
    embed_grad_contrib = embed_grad_contrib.sum(-1).permute(2, 1, 0)  # (src, batch, t)

    # revert stop grad for the next attribution
    model = layerwise_revert_stop_nonlinear_grad_for_llama(
        model,
        -1,
        len(model.model.layers),
    )
    del embeds, tgt_vec, grad_outputs
    torch.cuda.empty_cache()

    # compute neuron grad contribs
    grad_contrib = []
    for lid, pairs in tqdm(
        neuron_cfg.items(), desc="Computing neuron contributions", disable=not verbose
    ):
        # stop gradient on MLPs
        model = layerwise_stop_nonlinear_grad_for_llama(
            model,
            lid,
            len(model.model.layers),
            use_shapley_grad,
            use_shapley_qk,
            use_relp_grad,
            True,
        )

        cache = {}

        def _hook(lid):
            def fn(_, input, output):
                cache[lid] = input[0]

            return fn

        if hasattr(model.model.layers[lid].mlp, "mlp"):
            model.model.layers[lid].mlp.mlp.down_proj.register_forward_hook(_hook(lid))
        else:
            model.model.layers[lid].mlp.down_proj.register_forward_hook(_hook(lid))

        # differentiable embeds
        # shape: (batch, seq, d)
        embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()

        # forward pass
        out = model(inputs_embeds=embeds, attention_mask=attention_masks)
        logits = out.logits
        if center_logits:
            logits -= logits.mean(dim=-1)

        tgt_nodes = []
        for id, p in enumerate(focus_positions):
            tid = [focus_logit[id] for focus_logit in focus_logits]
            tgt_nodes.append(logits[torch.arange(logits.shape[0]), p, tid])
        tgt_vec = torch.stack(tgt_nodes)  # (t, batch)
        grad_outputs = torch.eye(t * batch, device=device)  # (t * batch, t * batch)

        layer_acts = cache[lid]  # (batch, seq, d)
        layer_acts.grad = None
        layer_grad_contrib = torch.autograd.grad(
            tgt_vec.flatten(),
            layer_acts,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=True,
        )[0]
        # shape: (t * batch, batch, seq, d)
        # convert back to (t, batch, batch, seq, d)
        layer_grad_contrib = layer_grad_contrib.reshape(
            t, batch, batch, layer_grad_contrib.shape[-2], layer_grad_contrib.shape[-1]
        )
        # get only identity along batch, batch
        layer_grad_contrib = layer_grad_contrib.diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
        # now (t, batch, seq, d)
        for pos, nid in pairs:
            grad_contrib.append(layer_grad_contrib[:, :, pos, nid])

        # revert stop grad for the next attribution
        model = layerwise_revert_stop_nonlinear_grad_for_llama(
            model,
            lid,
            len(model.model.layers),
        )
        remove_forward_hooks(model.model.layers[lid].mlp.down_proj)

        del layer_grad_contrib, layer_acts, embeds, tgt_vec, grad_outputs
        torch.cuda.empty_cache()

    # multiple by acts to get contributions
    grad_contrib = torch.stack(grad_contrib).permute(0, 2, 1)  # (neurons, batch, tgt)
    contrib = grad_contrib * neuron_acts.detach()[:, :, None]  # (neurons, batch, tgt)

    # Clean up
    del grad_contrib
    torch.cuda.empty_cache()

    # assert shapes
    if verbose:
        print("attr.shape", attr.shape)
        print("contrib.shape", contrib.shape)
    assert attr.shape == (len(neuron_tags), batch, len(src_tokens))
    assert contrib.shape == (len(neuron_tags), batch, len(tgt_tokens))

    # detach
    attr = attr.detach()
    contrib = contrib.detach()

    # return
    return attr, contrib, embed_grad_contrib, neuron_tags


def _get_neuron_attr_and_contrib(
    model,
    neuron_cfg: dict[int, list[list[int]]],
    input_ids: torch.Tensor,
    src_tokens: list[int],
    tgt_tokens: list[int],
    focus_positions: list[int],
    focus_logits,
    attention_masks,
    use_shapley_grad: bool = False,
    use_shapley_qk: bool = False,
    use_relp_grad: bool = False,
    disable_stop_grad: bool = False,
    center_logits: bool = False,
    alpha: float | None = None,
    neuron_chunk_size: int = 50,
    verbose: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[NeuronIdx]]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[NeuronIdx], torch.Tensor, torch.Tensor]
):
    """
    Compute neuron attributions from source tokens and contributions to target tokens.

    Args:
        model: The model to compute neuron attributions from.
        neuron_cfg: A dict mapping layer indices to lists of (position, neuron) tuples.
        input_ids: The input id tokens to the model.
        src_tokens: The token positions to trace from.
        tgt_tokens: The token positions to trace to.
        focus_positions: The token positions to consider the logits effect to.
        focus_logits: The token vocab ids to trace logits to.
        attention_masks: The attention masks for the input ids.
        alpha: Optional scaling factor for integrated gradients.

    Returns:
        If alpha is None:
            A tuple of (neuron attributions, neuron contributions, embed_grad_contrib, neuron tags).
        If alpha is not None:
            A tuple of (neuron attributions, neuron contributions, embed_grad_contrib, neuron tags, neuron_acts, embeds).
        - neuron attributions: Tensor with shape [neurons, batch, src].
        - neuron contributions: Tensor with shape [neurons, batch, tgt].
        - embed_grad_contrib: Tensor with shape [src, batch, tgt].
        - neuron tags: List of NeuronIdx objects indicating which neurons are important.
        - neuron_acts: Tensor with shape [neurons, batch] (only when alpha is not None).
        - embeds: Tensor with shape [batch, seq, d] (only when alpha is not None).
    """

    # compute neuron attributions to source tokens and contributions to target tokens
    # clear hooks
    for i in range(len(model.model.layers)):
        if hasattr(model.model.layers[i].mlp, "mlp"):
            remove_forward_hooks(model.model.layers[i].mlp.mlp.down_proj)
        else:
            remove_forward_hooks(model.model.layers[i].mlp.down_proj)

    device = input_ids.device

    # differentiable embeds
    # shape: (batch, seq, d)
    embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()
    if alpha is not None:
        embeds = embeds * alpha

    cache = {}
    for lid in neuron_cfg:

        def _hook(lid):
            def fn(_, input, output):
                cache[lid] = input[0]

            return fn

        if hasattr(model.model.layers[lid].mlp, "mlp"):
            model.model.layers[lid].mlp.mlp.down_proj.register_forward_hook(_hook(lid))
        else:
            model.model.layers[lid].mlp.down_proj.register_forward_hook(_hook(lid))

    # forward pass
    out = model(inputs_embeds=embeds, attention_mask=attention_masks)
    logits = out.logits
    if center_logits:
        logits -= logits.mean(dim=-1)

    # process neuron activations layer by layer for memory efficiency
    attr_list = []
    neuron_acts_list = []
    neuron_tags: list[NeuronIdx] = []

    # Process neurons in chunks to avoid OOM with large identity matrices
    chunk_size = neuron_chunk_size

    for lid, pairs in tqdm(
        neuron_cfg.items(), desc="Computing neuron attributions", disable=not verbose
    ):
        if not pairs:  # skip empty layers
            continue

        act = cache[lid]  # (batch, seq, d)

        # Process this layer's neurons in chunks
        all_pairs = list(pairs)
        layer_attr_chunks = []
        layer_neuron_acts_chunks = []

        for chunk_start in range(0, len(all_pairs), chunk_size):
            chunk_pairs = all_pairs[chunk_start : chunk_start + chunk_size]

            layer_neuron_acts = []
            layer_neuron_tags = []

            for pos, nid in chunk_pairs:
                layer_neuron_acts.append(act[:, pos, nid])
                layer_neuron_tags.append(NeuronIdx(layer=lid, token=pos, neuron=nid))

            if not layer_neuron_acts:  # skip if no neurons
                continue

            # get all activations
            layer_neuron_acts = torch.stack(layer_neuron_acts)  # (n_chunk, batch)
            batch = layer_neuron_acts.shape[1]
            n_chunk = layer_neuron_acts.shape[0]
            grad_outputs = torch.eye(
                n_chunk * batch, device=device
            )  # (n_chunk * batch, n_chunk * batch)

            # attribution to inputs for this layer chunk
            embeds.grad = None
            layer_grad_attr = torch.autograd.grad(
                layer_neuron_acts.flatten(),  # (n_chunk * batch)
                embeds,  # (batch, seq, d)
                grad_outputs=grad_outputs,  # (n_chunk * batch, n_chunk * batch)
                is_grads_batched=True,
                retain_graph=True,
            )[0]
            # shape: (n_chunk * batch, batch, seq, d)
            # convert back to (n_chunk, batch, batch, seq, d)
            layer_grad_attr = layer_grad_attr.reshape(
                n_chunk, batch, batch, layer_grad_attr.shape[-2], layer_grad_attr.shape[-1]
            )
            # get only identity along batch, batch
            layer_grad_attr = layer_grad_attr.diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
            # now (n_chunk, batch, seq, d)

            if alpha is not None:
                # For IG: return gradient only, multiply by activation later
                layer_attr = layer_grad_attr[:, :, src_tokens, :]  # (n_chunk, batch, src, d)
            else:
                # For regular attribution: gradient * activation
                layer_attr = (layer_grad_attr * embeds[None, ...]).sum(-1)  # (n_chunk, batch, seq)
                layer_attr = layer_attr[:, :, src_tokens]  # (n_chunk, batch, src)

            layer_attr_chunks.append(layer_attr)
            layer_neuron_acts_chunks.append(layer_neuron_acts.detach())
            neuron_tags.extend(layer_neuron_tags)

            # clean up memory after each chunk
            del layer_grad_attr, layer_attr, grad_outputs
            torch.cuda.empty_cache()

        # Concatenate chunks for this layer
        if layer_attr_chunks:
            attr_list.append(torch.cat(layer_attr_chunks, dim=0))
            neuron_acts_list.append(torch.cat(layer_neuron_acts_chunks, dim=0))

    # concatenate results from all layers
    attr = torch.cat(attr_list, dim=0)  # (neurons, batch, seq)
    neuron_acts = torch.cat(neuron_acts_list, dim=0)  # (neurons, batch)

    # clean up layer-wise lists to free memory
    del attr_list, neuron_acts_list
    torch.cuda.empty_cache()

    # contribution to tgt tokens
    tgt_nodes = []
    for id, p in enumerate(focus_positions):
        tid = [focus_logit[id] for focus_logit in focus_logits]
        tgt_nodes.append(logits[torch.arange(logits.shape[0]), p, tid])
    tgt_vec = torch.stack(tgt_nodes)  # (t, batch)
    t = tgt_vec.size(0)
    grad_outputs = torch.eye(t * batch, device=device)  # (t * batch, t * batch)

    # compute embed grad contribs
    embeds.grad = None
    embed_grad_contrib_full = torch.autograd.grad(
        tgt_vec.flatten(),
        embeds,
        grad_outputs=grad_outputs,
        is_grads_batched=True,
        retain_graph=True,
    )[0]
    # shape (t * batch, batch, seq, d)
    # convert back to (t, batch, batch, seq, d)
    embed_grad_contrib = embed_grad_contrib_full.reshape(
        t, batch, batch, embed_grad_contrib_full.shape[-2], embed_grad_contrib_full.shape[-1]
    )
    del embed_grad_contrib_full  # Free memory immediately

    # get only identity along batch, batch
    embed_grad_contrib = embed_grad_contrib.diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
    # now (t, batch, seq, d)

    if alpha is not None:
        # For IG: return gradient only, multiply by activation later
        embed_grad_contrib = embed_grad_contrib[:, :, src_tokens, :]  # (t, batch, src, d)
    else:
        # For regular contribution: gradient * activation
        embed_grad_contrib = (
            embed_grad_contrib[:, :, src_tokens, :] * embeds[None, :, src_tokens, :]
        )  # (t, batch, src, d)
        embed_grad_contrib = embed_grad_contrib.sum(-1).permute(2, 1, 0)  # (src, batch, t)

    torch.cuda.empty_cache()

    # compute neuron grad contribs
    grad_contrib = []
    for lid, pairs in tqdm(
        neuron_cfg.items(), desc="Computing neuron contributions", disable=not verbose
    ):

        layer_acts = cache[lid]  # (batch, seq, d)
        layer_acts.grad = None
        layer_grad_contrib = torch.autograd.grad(
            tgt_vec.flatten(),
            layer_acts,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=True,
        )[0]
        # shape: (t * batch, batch, seq, d)
        # convert back to (t, batch, batch, seq, d)
        layer_grad_contrib = layer_grad_contrib.reshape(
            t, batch, batch, layer_grad_contrib.shape[-2], layer_grad_contrib.shape[-1]
        )
        # get only identity along batch, batch
        layer_grad_contrib = layer_grad_contrib.diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
        # now (t, batch, seq, d)
        for pos, nid in pairs:
            grad_contrib.append(layer_grad_contrib[:, :, pos, nid])

        # Clean up memory after each layer
        del layer_grad_contrib
        torch.cuda.empty_cache()

    # multiple by acts to get contributions
    grad_contrib = torch.stack(grad_contrib).permute(0, 2, 1)  # (neurons, batch, tgt)
    if alpha is not None:
        # For IG: return gradient only
        contrib = grad_contrib
    else:
        # For regular contribution: gradient * activation
        contrib = grad_contrib * neuron_acts.detach()[:, :, None]  # (neurons, batch, tgt)

    # Clean up after contribution computation
    del grad_contrib, tgt_vec, grad_outputs
    torch.cuda.empty_cache()

    # assert shapes
    if verbose:
        print("attr.shape", attr.shape)
        print("contrib.shape", contrib.shape)

    if alpha is not None:
        # When alpha is set, shapes include the embedding dimension
        # attr: (neurons, batch, src, d), contrib: (neurons, batch, tgt)
        assert attr.shape[0] == len(neuron_tags)
        assert attr.shape[1] == batch
        assert attr.shape[2] == len(src_tokens)
        assert contrib.shape == (len(neuron_tags), batch, len(tgt_tokens))
    else:
        assert attr.shape == (len(neuron_tags), batch, len(src_tokens))
        assert contrib.shape == (len(neuron_tags), batch, len(tgt_tokens))

    # detach
    attr = attr.detach()
    contrib = contrib.detach()

    # return
    if alpha is not None:
        return attr, contrib, embed_grad_contrib, neuron_tags, neuron_acts, embeds
    return attr, contrib, embed_grad_contrib, neuron_tags


def _get_neuron_attr_and_contrib_ig(
    model,
    neuron_cfg: dict[int, list[list[int]]],
    input_ids: torch.Tensor,
    src_tokens: list[int],
    tgt_tokens: list[int],
    focus_positions: list[int],
    focus_logits,
    attention_masks,
    use_shapley_grad: bool = False,
    use_shapley_qk: bool = False,
    use_relp_grad: bool = False,
    disable_stop_grad: bool = False,
    center_logits: bool = False,
    ig_steps: int = 10,
    ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs",
    neuron_chunk_size: int = 50,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[NeuronIdx]]:
    """
    Compute neuron attributions and contributions using Integrated Gradients.

    Args:
        model: The model to compute neuron attributions from.
        neuron_cfg: A dict mapping layer indices to lists of (position, neuron) tuples.
        input_ids: The input id tokens to the model.
        src_tokens: The token positions to trace from.
        tgt_tokens: The token positions to trace to.
        focus_positions: The token positions to consider the logits effect to.
        focus_logits: The token vocab ids to trace logits to.
        attention_masks: The attention masks for the input ids.
        ig_steps: Number of steps for integrated gradients.
        ig_mode: Mode for integrated gradients aggregation ("ig-inputs" or "conductance").

    Returns:
        A tuple of (neuron attributions, neuron contributions, embed_grad_contrib, neuron tags).
        - neuron attributions: Tensor with shape [neurons, batch, src].
        - neuron contributions: Tensor with shape [neurons, batch, tgt].
        - embed_grad_contrib: Tensor with shape [src, batch, tgt].
        - neuron tags: List of NeuronIdx objects indicating which neurons are important.
    """
    # Collect step-wise attributions and contributions
    attr_steps = []
    contrib_steps = []
    embed_grad_contrib_steps = []
    neuron_acts_steps = []
    embeds_steps = []
    neuron_tags = None

    for step in range(0, ig_steps + 1):
        alpha = step / ig_steps

        attr, contrib, embed_grad_contrib, tags, neuron_acts, embeds = _get_neuron_attr_and_contrib(
            model=model,
            neuron_cfg=neuron_cfg,
            input_ids=input_ids,
            src_tokens=src_tokens,
            tgt_tokens=tgt_tokens,
            focus_positions=focus_positions,
            focus_logits=focus_logits,
            attention_masks=attention_masks,
            use_shapley_grad=use_shapley_grad,
            use_shapley_qk=use_shapley_qk,
            use_relp_grad=use_relp_grad,
            disable_stop_grad=disable_stop_grad,
            center_logits=center_logits,
            alpha=alpha,
            neuron_chunk_size=neuron_chunk_size,
            verbose=verbose,
        )

        if neuron_tags is None:
            neuron_tags = tags

        # Immediately detach and move to CPU to save GPU memory
        attr_steps.append(attr.detach().cpu())
        contrib_steps.append(contrib.detach().cpu())
        embed_grad_contrib_steps.append(embed_grad_contrib.detach().cpu())
        neuron_acts_steps.append(neuron_acts.detach().cpu())
        embeds_steps.append(embeds.detach().cpu())

        # Clean up GPU tensors and computation graph
        del attr, contrib, embed_grad_contrib, neuron_acts, embeds
        torch.cuda.empty_cache()

        # Force garbage collection every few steps
        if step % 5 == 0:
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    # Get device for moving tensors back
    device = input_ids.device

    if ig_mode == "ig-inputs":
        # Riemann sum in IG (ignore step 0)
        # Average gradients across steps (move back to device for computation)
        attr_grad_avg = (
            torch.stack(attr_steps[1:]).mean(dim=0).to(device)
        )  # (neurons, batch, src, d)
        contrib_grad_avg = (
            torch.stack(contrib_steps[1:]).mean(dim=0).to(device)
        )  # (neurons, batch, tgt)
        embed_grad_avg = (
            torch.stack(embed_grad_contrib_steps[1:]).mean(dim=0).to(device)
        )  # (t, batch, src, d)

        # Compute activation differences
        neuron_acts_diff = (neuron_acts_steps[-1] - neuron_acts_steps[0]).to(
            device
        )  # (neurons, batch)
        embeds_diff = (embeds_steps[-1] - embeds_steps[0]).to(device)  # (batch, seq, d)
        embeds_diff_src = embeds_diff[:, src_tokens, :]  # (batch, src, d)

        # Apply IG formula: averaged_gradient * (final_activation - initial_activation)
        # For attr: gradient is w.r.t. neuron acts projected to embeds
        # Shape: (neurons, batch, src, d) * (batch, src, d) -> sum over d -> (neurons, batch, src)
        attr_final = (attr_grad_avg * embeds_diff_src[None, :, :, :]).sum(dim=-1)

        # For contrib: gradient is w.r.t. neuron acts
        # Shape: (neurons, batch, tgt) * (neurons, batch, 1) -> (neurons, batch, tgt)
        contrib_final = contrib_grad_avg * neuron_acts_diff[:, :, None]

        # For embed_grad_contrib: gradient is w.r.t. embeds
        # Shape: (t, batch, src, d) * (batch, src, d) -> sum over d -> (t, batch, src) -> permute -> (src, batch, t)
        embed_grad_contrib_final = (embed_grad_avg * embeds_diff_src[None, :, :, :]).sum(dim=-1)
        embed_grad_contrib_final = embed_grad_contrib_final.permute(2, 1, 0)  # (src, batch, t)

    elif ig_mode == "conductance":
        # Stack all steps (excluding step 0) and move to device
        attr_grads = torch.stack(attr_steps[1:]).to(device)  # (steps, neurons, batch, src, d)
        contrib_grads = torch.stack(contrib_steps[1:]).to(device)  # (steps, neurons, batch, tgt)
        embed_grads = torch.stack(embed_grad_contrib_steps[1:]).to(
            device
        )  # (steps, t, batch, src, d)

        # Compute step-wise differences in activations
        neuron_acts_all = torch.stack(neuron_acts_steps).to(device)  # (steps+1, neurons, batch)
        embeds_all = torch.stack(embeds_steps).to(device)  # (steps+1, batch, seq, d)

        neuron_acts_diffs = torch.diff(neuron_acts_all, dim=0)  # (steps, neurons, batch)
        embeds_diffs = torch.diff(embeds_all, dim=0)  # (steps, batch, seq, d)
        embeds_diffs_src = embeds_diffs[:, :, src_tokens, :]  # (steps, batch, src, d)

        # Apply conductance: sum over steps of (gradient * activation_diff)
        # For attr: (steps, neurons, batch, src, d) * (steps, batch, src, d) -> sum over d and steps
        attr_final = (attr_grads * embeds_diffs_src[:, None, :, :, :]).sum(dim=-1).sum(dim=0)

        # For contrib: (steps, neurons, batch, tgt) * (steps, neurons, batch, 1)
        contrib_final = (contrib_grads * neuron_acts_diffs[:, :, :, None]).sum(dim=0)

        # For embed_grad_contrib: (steps, t, batch, src, d) * (steps, batch, src, d)
        embed_grad_contrib_final = (
            (embed_grads * embeds_diffs_src[:, None, :, :, :]).sum(dim=-1).sum(dim=0)
        )
        embed_grad_contrib_final = embed_grad_contrib_final.permute(2, 1, 0)  # (src, batch, t)
    else:
        raise ValueError(f"Invalid IG mode: {ig_mode}")

    return attr_final, contrib_final, embed_grad_contrib_final, neuron_tags


@torch.no_grad()
def _get_all_pairs_cl_so_effects_with_attributions(
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
) -> tuple[Any, Any]:
    """WIP. this is the dev version. it is always faster than the v1 version till completion."""

    if return_nodes_only:
        include_error_node, include_error_edges = False, False
    if include_error_edges:
        assert include_error_node, "Error edges require error node"
    assert tgt_tokens is not None, "tgt_tokens must be provided"

    ##############
    # Initialize #
    ##############

    nodes: list[Node] = []
    edges: list[Edge] = []

    #######################
    # Collect activations #
    #######################

    if verbose:
        print("Collecting acts... ", end="", flush=True)

    # collect acts
    collect_layers = list(range(subject.L))
    (
        resids_LBTD,
        attn_outs_LBTD,
        attn_maps_LBQTT,
        neurons_LBTI,
        mlp_gate_LBTI,
        w_outs_LDI,
        w_ins_LDI,
        input_embeddings_BTD,
        input_norm_consts_LB1TsDD,
        target_norm_consts_LBTf11D,
        output_norm_const_BTf11D,
        tokens,
    ) = collect_acts(
        subject,
        cis,
        attention_masks,
        collect_layers=collect_layers,
        keep_tokens=keep_tokens,
        device=device,
        verbose=verbose,
    )

    #######################
    # Determine constants #
    #######################

    # key constants
    L = subject.L
    D = subject.D
    B = resids_LBTD[0].size(0)
    resids_LBTD[0].size(0)
    Ts = resids_LBTD[0].size(1)
    resids_LBTD[0].size(1)

    start_layer = -1 if start_layer is None else start_layer
    end_layer = L if end_layer is None else end_layer
    keep_tokens = list(range(Ts)) if keep_tokens is None else sorted(keep_tokens)

    ##############
    # Compute
    # To get the interesting internal edges, we chop and head and tail off.
    ##############

    # we get tgt tokens and put them in the logits buffer
    buffer = {}
    for target_idx, target_token_pos_idx in enumerate(focus_positions):
        # get the logits for the vocab items we care about at this position
        focus_logits_target = torch.tensor(focus_logits)[:, target_idx]
        resid_final_BD = resids_LBTD[subject.L - 1][:, target_token_pos_idx].view(
            B, D
        ) * output_norm_const_BTf11D[:, target_token_pos_idx, 0].view(B, D)

        # since we never computed grads for off-diag batch x logits, we set them to 0
        logits_BV = torch.einsum(
            "bd,vd->bv", resid_final_BD, subject.unembed.weight[focus_logits_target]
        ) * torch.eye(B, device=device)
        final_logits = LogitsBuffer(
            layer=subject.L,
            token=target_token_pos_idx,
            ln_B1D=output_norm_const_BTf11D[:, target_token_pos_idx, 0],
            unembed_VD=subject.unembed.weight[focus_logits_target].detach().to(device),
            logits_BV=logits_BV,
            keep_indices=set(focus_logits_target.tolist()),
            final_attribution_BNsNf=torch.stack(
                [torch.diagflat(logits_BV[b]) for b in range(logits_BV.shape[0])]
            ),
            neuron_indices_map={
                i: focus_logits_target[i].item() for i in range(len(focus_logits_target))
            },
        )

        # add attr and contrib map
        for batch_idx in range(len(focus_logits_target)):
            logit_id = (subject.L, target_token_pos_idx, focus_logits_target[batch_idx].item())
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

        # add to buffer
        buffer[(subject.L, target_idx)] = final_logits

    # start the main search loop
    buffers = 0
    while len(buffer) > 0:
        # grab last buffer
        target_idx = max(buffer.keys())
        target = buffer[target_idx]
        del buffer[target_idx]
        buffers += 1

        # process this buffer's nodes
        activations = target.get_activations()
        if isinstance(target, MLPBuffer) or isinstance(target, LogitsBuffer):
            # skip if no important neurons or no outgoing edges
            if len(target.neuron_indices_map) == 0 or len(target.keep_indices) == 0:
                continue

            # add node to the graph
            for idx, neuron_idx in target.neuron_indices_map.items():
                nodes.append(
                    Node(
                        layer=target.layer,
                        token=target.token,
                        neuron=neuron_idx,
                        activation=activations[:, idx].float().cpu(),
                        final_attribution=target.final_attribution_BNsNf[:, idx, :].float().cpu(),
                        attr_map=neuron_attr_map.get(
                            (target.layer, target.token, neuron_idx), None
                        ),
                        contrib_map=neuron_contrib_map.get(
                            (target.layer, target.token, neuron_idx), None
                        ),
                    )
                )
        elif isinstance(target, EmbedBuffer):
            for token_type in set([tokens[t][target.token] for t in range(len(tokens))]):
                relevant_idxs = [
                    batch_idx
                    for batch_idx in range(len(tokens))
                    if tokens[batch_idx][target.token] == token_type
                ]
                mask = torch.zeros(len(tokens), device=activations.device)
                mask[relevant_idxs] = 1
                mask = mask.to(torch.bool)
                final_attribution = torch.where(
                    mask[:, None], target.final_attribution_BNsNf[:, 0, :], 0
                )
                attr_map = torch.zeros(
                    (len(tokens), min(len(keep_tokens), len(tokens[0]))), device=activations.device
                )
                attr_map[relevant_idxs, target.token] = 1
                contrib_map = neuron_contrib_map.get((target.layer, target.token, 0), None)
                nodes.append(
                    Node(
                        layer=target.layer,
                        token=target.token,
                        neuron=token_type,
                        activation=torch.where(mask, activations[:, 0], 0).float().cpu(),
                        final_attribution=final_attribution.float().cpu(),
                        attr_map=attr_map,
                        contrib_map=(
                            contrib_map * mask[:, None] if contrib_map is not None else None
                        ),
                    )
                )

        # print info
        if verbose:
            num_indices = len(target.neuron_indices_map)
            print(
                f"layer {target.layer} @ tok {target.token} ({subject.decode(cis[0].tokenize(subject)[target.token])}), neurons {num_indices}, type {type(target).__name__}"
            )

        # we only store attns to the target token
        attn_ov_sum_B1TsDD: torch.Tensor | None = None
        for layer in range(target.layer - 1, start_layer - 1, -1):
            ##############
            # OV suffix #
            ##############
            # no attn for last MLP -> logits
            if layer + 1 != L and not return_nodes_only:
                # print(f"applying attn ov for layer {layer + 1} (max: {L})")
                # add current layer's OV to the running sum
                attn_ov_QDD = get_layer_headwise_OV(subject, layer + 1)
                # attn_ov_QDD = attn_ovs_LQDD[layer + 1]

                # Linearly combine head W_{OV} matrices
                attn_BQTfT = attn_maps_LBQTT[layer + 1][:, :, target.token : target.token + 1, :]
                attn_BQTfTs = attn_BQTfT[:, :, :, :]  # Not sure why this is necessary but it is
                attn_ov_BTfTsDD = contract("bqfs,qxd->bfsxd", attn_BQTfTs, attn_ov_QDD)
                # Apply layernorms
                attn_ov_BTfTsDD *= (
                    input_norm_consts_LB1TsDD[layer + 1]
                    # * output_norm_const_BTf11D[:, :]
                )

                # Update running W_{OV} sum
                if attn_ov_sum_B1TsDD is None:
                    attn_ov_sum_B1TsDD = attn_ov_BTfTsDD
                else:
                    attn_ov_sum_B1TsDD += attn_ov_BTfTsDD

                # deallocate memory
                del attn_ov_BTfTsDD, attn_BQTfTs, attn_BQTfT, attn_ov_QDD
                torch.cuda.empty_cache()

            for source_token in keep_tokens[::-1]:

                if (layer, source_token) in buffer:
                    source = buffer[(layer, source_token)]
                if layer >= 0:
                    final_attributions = mlp_final_attributions[
                        layer, :, source_token, :, :
                    ]  # (batch, neurons, logits)
                    source = MLPBuffer(
                        layer=layer,
                        token=source_token,
                        ln_in_B1D=target_norm_consts_LBTf11D[layer][:, source_token, 0],
                        ln_out_B1D=None,
                        mlp_gate_BN=mlp_gate_LBTI[layer][:, source_token, :],
                        w_in_ND=w_ins_LDI[layer],
                        w_out_DN=w_outs_LDI[layer],
                        neurons_BN=neurons_LBTI[layer][:, source_token, :],
                        final_attribution_BNsNf=final_attributions,
                        frozen_final_attribution_BNsNf=final_attributions,
                    )
                    # indices are just the neurons we kept
                    indices = (
                        global_important_neurons_mask[layer, source_token, :]
                        .nonzero()
                        .flatten()
                        .sort()
                        .values
                    )
                    source.apply_mask(indices)
                else:
                    final_attributions = embed_final_attributions[
                        :, source_token, :
                    ]  # (batch, logits)
                    source = EmbedBuffer(
                        layer=layer,
                        token=source_token,
                        embed_B1D=input_embeddings_BTD[:, source_token : source_token + 1, :],
                        keep_indices=set([0]),
                        final_attribution_BNsNf=final_attributions.unsqueeze(1),
                        frozen_final_attribution_BNsNf=final_attributions.unsqueeze(1),
                    )

                # don't trace edges for nodes only
                if return_nodes_only:
                    # add to buffer if important, or have a
                    if (layer, source_token) not in buffer:
                        source.keep_indices.update(set(source.neuron_indices_map.keys()))
                        buffer[(layer, source_token)] = source
                    continue

                # attribute to target
                # shape: (B, N_source, N_target)
                # we use relative attribution to compute final_attribution of shape (B, N_source, N_logits)

                relative_attribution = attribute_lightweight(
                    subject,
                    attn_ov_sum_B1TsDD,
                    source,
                    target,
                    keep_fo=keep_fo,
                    return_absolute=return_absolute,
                )

                # modify given threshold
                # get max over the batch dimension, then get the indices to keep
                attribution_scores = relative_attribution.abs().mean(dim=0).max(dim=-1).values

                if parent_threshold is None:
                    source_indices = attribution_scores.nonzero().reshape(-1).cpu().tolist()
                else:
                    source_indices = (
                        (attribution_scores > parent_threshold).nonzero().reshape(-1).cpu().tolist()
                    )

                # use this to get rid of nodes with non-outgoing edges
                source.keep_indices.update(set(source_indices))

                # add to buffer if important, or have a
                if len(source.neuron_indices_map) == 0 or len(source.keep_indices) == 0:
                    continue
                if (layer, source_token) not in buffer:
                    buffer[(layer, source_token)] = source

                # then per neuron
                if not return_nodes_only:
                    if isinstance(source, MLPBuffer):
                        for source_idx, source_neuron in source.neuron_indices_map.items():
                            source_key = NeuronIdx(
                                layer=source.layer, token=source.token, neuron=source_neuron
                            )
                            for target_idx, target_neuron in target.neuron_indices_map.items():
                                target_key = NeuronIdx(
                                    layer=target.layer, token=target.token, neuron=target_neuron
                                )
                                # add edge to the graph
                                edge_weight = relative_attribution[
                                    :, source_idx, target_idx
                                ]  # shape: (B,)
                                if (
                                    edge_threshold is not None
                                    and edge_weight.abs().max() < edge_threshold
                                ):
                                    continue
                                target_attribution = target.final_attribution_BNsNf[
                                    :, target_idx, :
                                ]  # shape: (B, N_logits)
                                edges.append(
                                    Edge(
                                        src=source_key,
                                        tgt=target_key,
                                        weight=edge_weight.detach().float().cpu(),
                                        final_attribution=(
                                            edge_weight[:, None] * target_attribution
                                        )
                                        .detach()
                                        .float()
                                        .cpu(),
                                    )
                                )
                    elif isinstance(source, EmbedBuffer):
                        for token_type in set(
                            [tokens[t][source.token] for t in range(len(tokens))]
                        ):
                            relevant_idxs = [
                                batch_idx
                                for batch_idx in range(len(tokens))
                                if tokens[batch_idx][source.token] == token_type
                            ]
                            mask = torch.zeros(len(tokens), device=activations.device)
                            mask[relevant_idxs] = 1
                            mask = mask.to(torch.bool)
                            source_key = NeuronIdx(
                                layer=source.layer, token=source.token, neuron=token_type
                            )
                            for target_idx, target_neuron in target.neuron_indices_map.items():
                                target_key = NeuronIdx(
                                    layer=target.layer, token=target.token, neuron=target_neuron
                                )
                                edge_weight = relative_attribution[:, 0, target_idx]  # shape: (B,)
                                edge_weight = torch.where(mask, edge_weight, 0)
                                if (
                                    edge_threshold is not None
                                    and edge_weight.abs().max() < edge_threshold
                                ):
                                    continue
                                target_attribution = target.final_attribution_BNsNf[
                                    :, target_idx, :
                                ]  # shape: (B, N_logits)
                                edges.append(
                                    Edge(
                                        src=source_key,
                                        tgt=target_key,
                                        weight=edge_weight.detach().float().cpu(),
                                        final_attribution=(
                                            edge_weight[:, None] * target_attribution
                                        )
                                        .detach()
                                        .float()
                                        .cpu(),
                                    )
                                )

                # deallocate memory
                del relative_attribution
                torch.cuda.empty_cache()

    if verbose:
        print(f"# found nodes: {len(nodes)}")
        print(f"# found edges: {len(edges)}")

    return nodes, edges


def get_all_pairs_cl_so_effects_with_attributions(
    subject: Subject,
    cis: List[ModelInput],
    # where to trace from and to
    src_tokens: list[int],
    tgt_tokens: list[int],
    # algo settings
    device: str = "cuda:0",
    verbose: bool = False,
    attention_masks: list[list[int]] | torch.Tensor | None = None,
    return_only_important_neurons: bool = False,
    return_nodes_only: bool = False,
    use_shapley_grad: bool = False,
    use_shapley_qk: bool = False,
    use_relp_grad: bool = False,
    disable_stop_grad: bool = False,
    use_stop_grad_on_mlps: bool = False,
    disable_half_rule: bool = False,
    ablation_mode: Literal["zero", "mean"] = "zero",
    center_logits: bool = False,
    # edge pruning settings
    node_attribution_threshold: float | None = 1.0,
    topk_neurons: int | None = None,
    parent_threshold: float | None = None,
    edge_threshold: float | None = None,
    topk: int | None = None,
    batch_aggregation: Literal["mean", "max", "max_abs", "any"] = "mean",
    return_absolute: bool = False,
    # more circuit settings
    keep_tokens: List[int] | None = None,
    focus_positions: list[int] | None = None,
    focus_logits: list[list[int]] | list[int] | None = None,
    focus_last_residual: bool = False,
    start_layer: int | None = None,
    end_layer: int | None = None,
    skip_attr_contrib: bool = False,
    # IG settings
    ig_steps: int | None = None,
    ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs",
) -> dict[int, list[list[int]]] | tuple[list[Node], dict[tuple[NeuronIdx, NeuronIdx], float]]:
    #########
    # SETUP #
    #########

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

    # core HF model has stop gradient replacement model
    try:
        _ = revert_stop_nonlinear_grad_for_llama(subject.model._model)
    except Exception:
        pass
    model = stop_nonlinear_grad_for_llama(
        subject.model._model, use_shapley_grad, use_shapley_qk, use_relp_grad, not disable_half_rule
    )

    # get attributions
    if ig_steps is None:
        mlp_final_attributions, embed_final_attributions, _, _, _ = (
            _get_grad_attributions_from_logits(
                subject,
                model,
                input_ids,
                keep_tokens,
                focus_positions,
                focus_logits=focus_logits,
                focus_last_residual=focus_last_residual,
                attention_masks=attn_mask_final,
                ablation_mode=ablation_mode,
                center_logits=center_logits,
                verbose=verbose,
            )
        )
    else:
        mlp_final_attributions, embed_final_attributions, _, _, _ = (
            _get_ig_attributions_from_logits(
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
                ig_steps=ig_steps,
                ig_mode=ig_mode,
                verbose=verbose,
            )
        )
    mlp_final_attributions = mlp_final_attributions.unsqueeze(-1)  # shape: (L, B, T, D_ff, 1)
    embed_final_attributions = embed_final_attributions.unsqueeze(-1)  # shape: (B, T, 1)
    if verbose:
        print("collected attributions for mlp", mlp_final_attributions.shape)
        print("collected attributions for embed", embed_final_attributions.shape)

    # before calculating anything, we get important neurons globally
    global_important_neurons_mask = _get_global_important_neurons_mask(
        keep_tokens,
        start_layer,
        end_layer,
        mlp_final_attributions,
        node_attribution_threshold,
        topk_neurons,
        batch_aggregation,
        verbose,
    )
    if verbose:
        print("global important neurons mask", global_important_neurons_mask.shape)

    # get important neurons for each layer (that we care about)
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
        return neuron_cfg

    # get attributions and contributions for important neurons
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
                use_shapley_grad=use_shapley_grad,
                use_shapley_qk=use_shapley_qk,
                use_relp_grad=use_relp_grad,
                disable_stop_grad=disable_stop_grad,
                center_logits=center_logits,
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
                use_shapley_grad=use_shapley_grad,
                use_shapley_qk=use_shapley_qk,
                use_relp_grad=use_relp_grad,
                disable_stop_grad=disable_stop_grad,
                center_logits=center_logits,
                ig_steps=ig_steps,
                ig_mode=ig_mode,
                verbose=verbose,
            )

        if verbose:
            print("collecting attributions for", attr.shape)  # shape: (neurons, batch, src)
            print("collecting contributions for", contrib.shape)  # shape: (neurons, batch, tgt)
            print(
                "collecting embed contributions for", embed_grad_contrib.shape
            )  # shape: (src, batch, tgt)

    # store neuron attributions and contributions
    neuron_attr_map: dict[NeuronIdx, torch.Tensor] = {}
    neuron_contrib_map: dict[NeuronIdx, torch.Tensor] = {}
    if not skip_attr_contrib:
        for neuron_count, neuron_idx in enumerate(neuron_tags):
            neuron_attr_map[neuron_idx] = attr[neuron_count]
            neuron_contrib_map[neuron_idx] = contrib[neuron_count]
        for src_token in src_tokens:
            neuron_contrib_map[NeuronIdx(layer=-1, token=src_token, neuron=0)] = embed_grad_contrib[
                src_token
            ]

    if verbose:
        print(f"Global important neurons mask: {global_important_neurons_mask.sum()}")
        print(
            f"Global important (layer, token): "
            f"{len(global_important_neurons_mask.sum(dim=-1).nonzero())}"
        )

    # revert back the replacement model to the original HF model
    model = revert_stop_nonlinear_grad_for_llama(model)

    # now trace edges
    nodes, edges = _get_all_pairs_cl_so_effects_with_attributions(
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
        return_nodes_only=return_nodes_only,
        return_absolute=return_absolute,
    )

    # final return
    return nodes, edges
