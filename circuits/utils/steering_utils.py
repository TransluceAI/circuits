import random
from collections import defaultdict
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import pyvene as pv
import torch
from util.chat_input import IdsInput
from util.subject import Subject

########################################################
# Intervention utilities
########################################################


class SubspaceZeroIntervention(
    pv.ConstantSourceIntervention, pv.LocalistRepresentationIntervention
):
    """Zero-out selected neurons, or scale them by a multiplier."""

    def __init__(
        self,
        intervene_subspaces: list[list[int]],
        intervene_pos: list[int],
        record_subspaces: list[list[int]],
        record_pos: list[int],
        special_layer: int,
        multiplier: float = 0.0,
        complement: bool = False,
        **kwargs,
    ):
        self.subspaces = intervene_subspaces
        self.pos = intervene_pos
        self.record_subspaces = record_subspaces
        self.record_pos = record_pos
        self.multiplier = multiplier
        self.layer = special_layer
        self.collected_activations = []
        self.complement = complement
        super().__init__(**kwargs)

    def forward(self, base, source=None, subspaces=None, **kwargs):
        # base shape: (batch_size, seq_len, hidden_size)
        result = base.clone()
        if not self.complement:
            for pos, subspaces in zip(self.pos, self.subspaces):
                result[:, pos, subspaces] *= self.multiplier
        else:
            result *= self.multiplier
            for pos, subspaces in zip(self.pos, self.subspaces):
                result[:, pos, subspaces] = base[:, pos, subspaces]  # restore the original values
        for pos, subspaces in zip(self.record_pos, self.record_subspaces):
            self.collected_activations.append(result[:, pos, subspaces].clone().detach())
        # print("ran zero intervention at layer", self.layer, ": ", result.to(torch.float32).sum().item())
        return result

    def __str__(self):
        return f"SubspaceZeroIntervention(intervene_subspaces={self.intervene_subspaces}, intervene_pos={self.intervene_pos}, record_subspaces={self.record_subspaces}, record_pos={self.record_pos}, multiplier={self.multiplier})"


def multiple_subspaces_config(
    subspaces: dict[tuple[int, int], list[int]],
    multiplier: float = 0.0,
    record_subspaces: dict[tuple[int, int], list[int]] | None = None,
    mode: Literal["serial", "parallel"] = "serial",
    num_layers: int = 32,
    complement: bool = False,
) -> list[dict[str, Any]]:
    config = []

    # merge subspaces and record subspaces
    all_params = defaultdict(lambda: defaultdict(list))
    for (layer, pos), subspace in sorted(list(subspaces.items())):
        all_params[layer]["intervene_subspaces"].append(subspace)
        all_params[layer]["intervene_pos"].append(pos)
    if record_subspaces is not None:
        for (layer, pos), subspace in sorted(list(record_subspaces.items())):
            all_params[layer]["record_subspaces"].append(subspace)
            all_params[layer]["record_pos"].append(pos)

    # where to intervene
    # for layer in range(num_layers):
    for layer in sorted(all_params.keys()) if not complement else range(num_layers):
        params = all_params[layer] if layer in all_params else defaultdict(list)
        config.append(
            {
                "layer": layer,
                "component": f"model.layers[{layer}].mlp.down_proj.input",
                "intervention": SubspaceZeroIntervention(
                    intervene_subspaces=params["intervene_subspaces"],
                    intervene_pos=params["intervene_pos"],
                    record_subspaces=params["record_subspaces"],
                    record_pos=params["record_pos"],
                    special_layer=layer,
                    multiplier=multiplier,
                    complement=complement,
                ),
                "unit": "pos",
                "group_key": len(config),
            }
        )

    return pv.IntervenableConfig(
        representations=config,
        mode=mode,
    )


########################################################
# Dataset utilities
########################################################


class Pair(NamedTuple):
    base: dict[str, torch.Tensor]
    source: dict[str, torch.Tensor]
    base_label: str
    source_label: str
    base_idx: int
    source_idx: int
    base_subspaces: dict[tuple[int, int], list[int]]
    source_subspaces: dict[tuple[int, int], list[int]]
    subspaces: dict[tuple[int, int], set[int]]
    record_subspaces: dict[tuple[int, int], set[int]] | None


def pad_left(input_tensor: torch.Tensor, offset: int) -> torch.Tensor:
    return torch.cat(
        [torch.zeros(offset, device=input_tensor.device, dtype=input_tensor.dtype), input_tensor]
    )


def get_subspaces(
    df_node: pd.DataFrame,
    id: str,
    offset: int = 0,
) -> dict[tuple[int, int], list[int]]:
    """Get subspaces from a dataframe of nodes."""
    nodes = df_node[df_node["label"] == id]
    subspaces = defaultdict(list)
    for _, node in nodes.iterrows():
        if node.attribution == 0:
            continue
        subspaces[(int(node.layer), int(node.token) + offset)].append(int(node.neuron))
    return subspaces


def combine_subspaces(
    all_subspaces: list[dict[tuple[int, int], list[int] | set[int]]], offsets: list[int]
) -> dict[tuple[int, int], set[int]]:
    subspaces = defaultdict(set)
    for i, subspace in enumerate(all_subspaces):
        if subspace is None:
            continue
        for layer, pos in subspace:
            key = (layer, pos)
            new_key = (layer, pos + offsets[i])
            subspaces[new_key].update(subspace[key])
    return subspaces


def batchify(
    pairs: list[Pair],
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    """
    Batchify pairs of circuits for interchange interventions. Assumes templatic prompts with overlapping tokens.
    """
    batches = []
    for i in range(0, len(pairs), batch_size):
        max_length = max(len(p.base["input_ids"]) for p in pairs[i : i + batch_size])
        offsets = [max_length - len(p.base["input_ids"]) for p in pairs[i : i + batch_size]]
        batches.append(
            {
                "base": {
                    "input_ids": torch.stack(
                        [
                            pad_left(p.base["input_ids"], offsets[j])
                            for j, p in enumerate(pairs[i : i + batch_size])
                        ]
                    ),
                    "attention_mask": torch.stack(
                        [
                            pad_left(p.base["attention_mask"], offsets[j])
                            for j, p in enumerate(pairs[i : i + batch_size])
                        ]
                    ),
                },
                "source": {
                    "input_ids": torch.stack(
                        [
                            pad_left(p.source["input_ids"], offsets[j])
                            for j, p in enumerate(pairs[i : i + batch_size])
                        ]
                    ),
                    "attention_mask": torch.stack(
                        [
                            pad_left(p.source["attention_mask"], offsets[j])
                            for j, p in enumerate(pairs[i : i + batch_size])
                        ]
                    ),
                },
                "base_class": [p.base_label for p in pairs[i : i + batch_size]],
                "source_class": [p.source_label for p in pairs[i : i + batch_size]],
                "base_idx": [p.base_idx for p in pairs[i : i + batch_size]],
                "source_idx": [p.source_idx for p in pairs[i : i + batch_size]],
                "base_subspaces": [p.base_subspaces for p in pairs[i : i + batch_size]],
                "source_subspaces": [p.source_subspaces for p in pairs[i : i + batch_size]],
                "subspaces": combine_subspaces(
                    [p.subspaces for p in pairs[i : i + batch_size]], offsets
                ),
                "record_subspaces": combine_subspaces(
                    [p.record_subspaces for p in pairs[i : i + batch_size]], offsets
                ),
            }
        )
    return batches


def prepare_circuits_for_interchange_interventions(
    subject: Subject,
    cis: list[IdsInput],
    attention_masks: list[torch.Tensor],
    labels: list[str],
    df_node: pd.DataFrame,
    needed_pairs: int,
    allow_same_label: bool = False,
    ignore_source: bool = False,
    df_node_record: pd.DataFrame | None = None,
):
    """
    Prepare pairs of circuits for interchange interventions. Assumes templatic prompts with overlapping tokens.
    """
    device = str(subject.model.device)
    pairs = []
    while len(pairs) < needed_pairs:
        base_idx = random.randint(0, len(cis) - 1)
        source_idx = random.randint(0, len(cis) - 1)
        base_label = labels[base_idx]
        source_label = labels[source_idx]

        if (base_label == source_label and not allow_same_label) and not ignore_source:
            continue
        if ignore_source:
            source_idx = base_idx
            source_label = base_label

        base_input_ids, base_attention_mask = (
            cis[base_idx].input_ids[::],
            attention_masks[base_idx][::],
        )
        source_input_ids, source_attention_mask = (
            cis[source_idx].input_ids[::],
            attention_masks[source_idx][::],
        )

        # pad to left
        max_length = max(len(base_input_ids), len(source_input_ids))
        offset_base = max_length - len(base_input_ids)
        offset_source = max_length - len(source_input_ids)

        # add to pairs
        base = {
            "input_ids": pad_left(torch.tensor(base_input_ids, device=device), offset_base),
            "attention_mask": pad_left(
                torch.tensor(base_attention_mask, device=device), offset_base
            ),
        }
        source = {
            "input_ids": pad_left(torch.tensor(source_input_ids, device=device), offset_source),
            "attention_mask": pad_left(
                torch.tensor(source_attention_mask, device=device), offset_source
            ),
        }

        # get subspaces
        base_subspaces = get_subspaces(df_node, base_label + f"___{base_idx}")
        source_subspaces = get_subspaces(df_node, source_label + f"___{source_idx}")

        # compute combined subspaces
        subspaces = combine_subspaces(
            [base_subspaces, source_subspaces], [offset_base, offset_source]
        )

        # record subspaces
        record_subspaces = (
            get_subspaces(df_node_record, base_label + f"___{base_idx}")
            if df_node_record is not None
            else None
        )

        # append both directions
        pairs.append(
            Pair(
                base,
                source,
                base_label,
                source_label,
                base_idx,
                source_idx,
                base_subspaces,
                source_subspaces,
                subspaces,
                record_subspaces,
            )
        )
        if not ignore_source:
            pairs.append(
                Pair(
                    source,
                    base,
                    source_label,
                    base_label,
                    source_idx,
                    base_idx,
                    source_subspaces,
                    base_subspaces,
                    subspaces,
                    record_subspaces,
                )
            )

    return pairs


########################################################
# Metric utilities
########################################################


def format_token(token, tokenizer) -> str:
    result = tokenizer.decode([token])
    result = result.replace(" ", "_")
    result = result.replace("\n", "\\n")
    return result


def compute_metrics(
    original_logits: torch.Tensor,
    intervened_logits: torch.Tensor,
    source_logits: torch.Tensor | None,
    base_cont: int,
    source_cont: int,
    tokenizer,
    include_logits_and_probs: bool = False,
):
    """
    Compute metrics for a list of batches of circuits.
    """
    original_probs = torch.softmax(original_logits, dim=-1)
    intervened_probs = torch.softmax(intervened_logits, dim=-1)
    original_base_prob = original_probs[base_cont].item()
    original_source_prob = original_probs[source_cont].item()
    intervened_base_prob = intervened_probs[base_cont].item()
    intervened_source_prob = intervened_probs[source_cont].item()
    top_10_original_probs = torch.topk(original_probs, k=10, dim=-1)
    top_10_intervened_probs = torch.topk(intervened_probs, k=10, dim=-1)
    result = {
        "base_prob": original_base_prob,
        "source_prob": original_source_prob,
        "intervened_base_prob": intervened_base_prob,
        "intervened_source_prob": intervened_source_prob,
        "log_odds_ratio": (
            np.log(original_base_prob)
            - np.log(original_source_prob)
            + np.log(intervened_source_prob)
            - np.log(intervened_base_prob)
        ).item(),
        "kl_div": torch.nn.functional.kl_div(
            torch.log(intervened_probs[:]),
            original_probs[:],
            reduction="sum",
        ).item(),
        "base_prob_diff": intervened_base_prob - original_base_prob,
        "source_prob_diff": intervened_source_prob - original_source_prob,
        "base_logit_diff": (intervened_logits[base_cont] - original_logits[base_cont]).item(),
        "source_logit_diff": (intervened_logits[source_cont] - original_logits[source_cont]).item(),
        "intervened_argmax_token": format_token(
            torch.argmax(intervened_probs[:]).item(), tokenizer
        ),
        "intervened_argmax_token_prob": intervened_probs[torch.argmax(intervened_probs[:])].item(),
        "original_top_10_tokens": [
            format_token(token, tokenizer) for token in top_10_original_probs.indices
        ],
        "original_top_10_tokens_probs": [
            original_probs[token].item() for token in top_10_original_probs.indices
        ],
        "intervened_top_10_tokens": [
            format_token(token, tokenizer) for token in top_10_intervened_probs.indices
        ],
        "intervened_top_10_tokens_probs": [
            intervened_probs[token].item() for token in top_10_intervened_probs.indices
        ],
        "most_promoted_token": format_token(
            torch.argmax(intervened_logits[:] - original_logits[:]).item(),
            tokenizer,
        ),
        "most_promoted_token_prob": intervened_probs[
            torch.argmax(intervened_logits[:] - original_logits[:])
        ].item(),
        "most_demoted_token": format_token(
            torch.argmin(intervened_logits[:] - original_logits[:]).item(),
            tokenizer,
        ),
        "most_demoted_token_prob": intervened_probs[
            torch.argmin(intervened_logits[:] - original_logits[:])
        ].item(),
    }
    if source_logits is not None:
        source_probs = torch.softmax(source_logits, dim=-1)
        source_base_prob = source_probs[base_cont].item()
        source_source_prob = source_probs[source_cont].item()
        result.update(
            {
                "source_base_prob": source_base_prob,
                "source_source_prob": source_source_prob,
                "log_odds_ratio_source": (
                    np.log(source_base_prob)
                    - np.log(source_source_prob)
                    + np.log(intervened_source_prob)
                    - np.log(intervened_base_prob)
                ).item(),
                "kl_div_source": torch.nn.functional.kl_div(
                    torch.log(intervened_probs[:]),
                    source_probs[:],
                    reduction="sum",
                ).item(),
            }
        )
        if include_logits_and_probs:
            result.update(
                {
                    "source_logits": source_logits.detach().cpu(),
                    "source_probs": source_probs.detach().cpu(),
                }
            )
    if include_logits_and_probs:
        result.update(
            {
                "original_logits": original_logits.detach().cpu(),
                "original_probs": original_probs.detach().cpu(),
                "intervened_logits": intervened_logits.detach().cpu(),
                "intervened_probs": intervened_probs.detach().cpu(),
            }
        )
    return result
