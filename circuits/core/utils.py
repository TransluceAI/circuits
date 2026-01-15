"""
Utility functions/classes for CLSO.
"""

from enum import Enum
from typing import NamedTuple

import torch
from util.gpu import gpu_mem_str
from util.parallel import TensorDict
from util.subject import Subject


class Order(Enum):
    FIRST = "FO"
    SECOND = "SO"


class NeuronIdx(NamedTuple):
    layer: int
    token: int
    neuron: int


class EdgeIdx(NamedTuple):
    layer: str
    token: str
    neuron: str


class Node(NamedTuple):
    layer: int
    token: int
    neuron: int
    activation: torch.Tensor | None = None
    final_attribution: torch.Tensor | None = None
    attr_map: torch.Tensor | None = None
    contrib_map: torch.Tensor | None = None


class Edge(NamedTuple):
    src: NeuronIdx
    tgt: NeuronIdx
    weight: torch.Tensor | None = None
    final_attribution: torch.Tensor | None = None


def collect_acts(
    subject: Subject,
    cis,
    attention_masks,
    collect_layers: list[int],
    keep_tokens: list[int] | None = None,
    device: str = "cuda",
    verbose: bool = False,
) -> tuple[
    TensorDict,
    TensorDict,
    TensorDict,
    TensorDict,
    TensorDict,
    TensorDict,
    TensorDict,
    torch.Tensor,
    TensorDict,
    TensorDict,
    torch.Tensor,
    list[list[int]],
]:
    """
    Collect activations from the model.

    Args:
        subject: The subject to collect activations from.
        cis: The chat inputs to collect activations from.
        attention_masks: The attention masks to collect activations from.
        collect_layers: The layers to collect activations from.
    """

    # where to collect
    collect_modules = [
        "resid_BTD",
        "attn_out_BTD",
        "attn_map_BQTT",
        "neurons_BTI",
        "mlp_gate_BTD",
    ]

    with torch.no_grad():
        # collect acts
        acts_gpu = subject.collect_acts(
            cis,
            collect_layers,
            include=collect_modules,
            attention_masks=attention_masks,
        )

        # Put acts on the CPU
        acts = acts_gpu.to("cpu")

        # Clean memory
        del acts_gpu
        torch.cuda.empty_cache()

    # make tensor dicts on the same device as the acts
    resids_LBTD = TensorDict({layer: acts[layer].resid_BTD.to(device) for layer in collect_layers})
    attn_outs_LBTD = TensorDict(
        {layer: acts[layer].attn_out_BTD.to(device) for layer in collect_layers}
    )
    attn_maps_LBQTT = TensorDict(
        {layer: acts[layer].attn_map_BQTT.to(device) for layer in collect_layers}
    )
    neurons_LBTI = TensorDict(
        {layer: acts[layer].neurons_BTI[:, :, :].to(device) for layer in collect_layers}
    )
    mlp_gate_LBTI = TensorDict(
        {
            layer: subject.model.model.layers[layer].mlp.act_fn(
                acts[layer].mlp_gate_BTD[:, :, :].to(device)
            )
            for layer in collect_layers
        }
    )
    w_outs_LDI = TensorDict(
        {layer: subject.w_outs[layer].weight[:, :].detach().to(device) for layer in collect_layers}
    )
    w_ins_LDI = TensorDict(
        {layer: subject.w_ins[layer].weight[:, :].detach().to(device) for layer in collect_layers}
    )

    if verbose:
        print(f"After collecting acts: {gpu_mem_str()}")

        # print all shapes
        print(
            "resids_LBTD",
            resids_LBTD[0].shape,
            resids_LBTD[0].dtype,
            resids_LBTD[0].device,
        )
        print(
            "attn_outs_LBTD",
            attn_outs_LBTD[0].shape,
            attn_outs_LBTD[0].dtype,
            attn_outs_LBTD[0].device,
        )
        print(
            "attn_maps_LBQTT",
            attn_maps_LBQTT[0].shape,
            attn_maps_LBQTT[0].dtype,
            attn_maps_LBQTT[0].device,
        )
        print(
            "neurons_LBTI",
            neurons_LBTI[0].shape,
            neurons_LBTI[0].dtype,
            neurons_LBTI[0].device,
        )
        print(
            "mlp_gate_LBTI",
            mlp_gate_LBTI[0].shape,
            mlp_gate_LBTI[0].dtype,
            mlp_gate_LBTI[0].device,
        )
        print("w_outs_LDI", w_outs_LDI[0].shape, w_outs_LDI[0].dtype, w_outs_LDI[0].device)

    # key constants
    L = subject.L
    D = subject.D
    B = resids_LBTD[0].size(0)
    Ts = resids_LBTD[0].size(1)
    Tf = resids_LBTD[0].size(1)

    # now get the norms
    # compute embeddings over input tokens
    tokens = [ci.tokenize(subject) for ci in cis]
    input_embeddings_BTD: torch.Tensor = subject.model._model.model.embed_tokens(
        torch.tensor(tokens, device=device)
    ).view(B, Ts, D)

    if verbose:
        print(
            "input_embeddings_BTD",
            input_embeddings_BTD.shape,
            input_embeddings_BTD.dtype,
            input_embeddings_BTD.device,
        )

    # Compute normalization constants
    input_norm_weights_LD = TensorDict(
        {layer: subject.input_norms[layer].weight.detach() for layer in collect_layers}
    ).to(device)
    input_norm_variance_epsilons_L = {
        layer: subject.input_norms[layer].variance_epsilon for layer in collect_layers
    }
    input_norm_consts_LB1TsDD = TensorDict(
        {
            layer: subject.layernorm_fn(
                resids_LBTD[layer].new_ones((B * Ts, 1, 1), device=device),
                (
                    resids_LBTD[layer - 1][:, :].view(B * Ts, D)
                    if layer > 0
                    else input_embeddings_BTD[:, :, :].view(B * Ts, D)
                ),  # type
                input_norm_weights_LD[layer],
                input_norm_variance_epsilons_L[layer],
            ).view(B, 1, Ts, 1, D)
            for layer in collect_layers
        }
    )
    post_attention_norm_weight_D = TensorDict(
        {
            layer: subject.model.model.layers[layer]
            .post_attention_layernorm.weight.detach()
            .to(device)
            for layer in collect_layers
        }
    )
    post_attention_norm_variance_epsilon = {
        layer: subject.model.model.layers[layer].post_attention_layernorm.variance_epsilon
        for layer in collect_layers
    }
    target_norm_consts_LBTf11D = TensorDict(
        {
            layer: subject.layernorm_fn(
                (resids_LBTD[layer].new_ones((B * Tf, 1, 1), device=device)),
                (
                    (resids_LBTD[layer - 1][:, :].view(B * Tf, D))
                    if layer > 0
                    else (input_embeddings_BTD[:, :, :].view(B * Tf, D))
                    + attn_outs_LBTD[layer][:, :].view(B * Tf, D)
                ),  # type: ignore
                post_attention_norm_weight_D[layer],
                post_attention_norm_variance_epsilon[layer],
            ).view(B, Tf, 1, 1, D)
            for layer in collect_layers
        }
    )
    unembed_norm_weight_D = subject.unembed_norm.weight.detach().to(device)
    unembed_norm_variance_epsilon = subject.unembed_norm.variance_epsilon
    output_norm_const_BTf11D = subject.layernorm_fn(
        resids_LBTD[L - 1].new_ones((B * Tf, 1, 1), device=device),
        resids_LBTD[L - 1][:, :].view(B * Tf, D),
        unembed_norm_weight_D,
        unembed_norm_variance_epsilon,
    ).view(B, Tf, 1, 1, D)

    # we might overload these two terms
    del post_attention_norm_weight_D, post_attention_norm_variance_epsilon

    # return everything
    return (
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
    )


def collect_neuron_acts(
    subject: Subject,
    cis,
    attention_masks,
    collect_layers: list[int],
    keep_tokens: list[int] | None = None,
    device: str = "cuda",
    verbose: bool = False,
) -> tuple[TensorDict,]:
    """
    Collect MLP neuron activations from the model.
    """

    tokens = [ci.tokenize(subject) for ci in cis]

    # where to collect neuron acts
    collect_modules = [
        "resid_BTD",
        "neurons_BTI",
    ]

    with torch.no_grad():
        # collect acts
        acts_gpu = subject.collect_acts(
            cis,
            collect_layers,
            include=collect_modules,
            attention_masks=attention_masks,
        )

        # Put acts on the CPU
        acts = acts_gpu.to("cpu")

        # Clean memory
        del acts_gpu
        torch.cuda.empty_cache()

    # make tensor dicts on the same device as the acts
    resids_LBTD = TensorDict({layer: acts[layer].resid_BTD.to(device) for layer in collect_layers})
    neurons_LBTI = TensorDict(
        {layer: acts[layer].neurons_BTI[:, :, :].to(device) for layer in collect_layers}
    )

    # key constants
    L = subject.L
    D = subject.D
    B = resids_LBTD[0].size(0)
    resids_LBTD[0].size(1)
    Tf = resids_LBTD[0].size(1)

    # derive other constants
    unembed_norm_weight_D = subject.unembed_norm.weight.detach().to(device)
    unembed_norm_variance_epsilon = subject.unembed_norm.variance_epsilon
    output_norm_const_BTf11D = subject.layernorm_fn(
        resids_LBTD[L - 1].new_ones((B * Tf, 1, 1), device=device),
        resids_LBTD[L - 1][:, :].view(B * Tf, D),
        unembed_norm_weight_D,
        unembed_norm_variance_epsilon,
    ).view(B, Tf, 1, 1, D)

    if verbose:
        print(f"After collecting acts: {gpu_mem_str()}")
        print(
            "neurons_LBTI",
            neurons_LBTI[0].shape,
            neurons_LBTI[0].dtype,
            neurons_LBTI[0].device,
        )

    # return
    return (neurons_LBTI, resids_LBTD, tokens, output_norm_const_BTf11D)
