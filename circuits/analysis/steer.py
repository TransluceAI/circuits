import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyvene as pv
import torch
from circuits.analysis.process_circuits import convert_inputs_to_circuits
from circuits.utils.steering_utils import (
    SubspaceZeroIntervention,
    batchify,
    compute_metrics,
    format_token,
    multiple_subspaces_config,
    prepare_circuits_for_interchange_interventions,
)
from tqdm import tqdm
from user_modeling.datasets.wikipedia import get_wikipedia_dataset_by_split
from util.subject import Subject, llama31_8B_instruct_config

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def prepare_user_modeling_dataset(
    subject: Subject = Subject(llama31_8B_instruct_config),
    split: str = "gender",
    num_datapoints: int = 1,
    batch_size: int = 1,
    neurons: int = 100,
):
    """
    Prepare user modeling dataset for interchange interventions. Only collects neurons, not edges.
    """
    tokenizer = subject.tokenizer

    # prepare inputs
    datasets = get_wikipedia_dataset_by_split(subject, question_types=[f"{split}"], only_good=True)
    # datasets["train"].datapoints = (
    #     datasets["train"].datapoints + datasets["test"].datapoints + datasets["valid"].datapoints
    # )
    prompts = [datapoint.conversation[0]["content"] for datapoint in datasets["train"].datapoints][
        :num_datapoints
    ]
    labels = [datapoint.latent_attributes[split][0] for datapoint in datasets["train"].datapoints][
        :num_datapoints
    ]
    labels = [" " + l[0].upper() + l[1:] for l in labels]
    seed_responses = [f"{{{{Infobox person\n| {split} ="] * len(prompts)

    # prepare circuits
    df_node, df_edge, cis, attention_masks = convert_inputs_to_circuits(
        subject,
        tokenizer,
        prompts,
        seed_responses,
        labels,
        neurons=neurons,
        batch_size=batch_size,
        num_datapoints=num_datapoints,
        return_cis=True,
        return_nodes_only=True,
        ignore_bos=True,
        skip_attr_contrib=True,
        verbose=True,
    )

    return prompts, labels, seed_responses, df_node, df_edge, cis, attention_masks


def run_vanilla_interchange_intervention(
    subject: Subject,
    batches: list[dict[str, torch.Tensor]],
    batch_size: int,
    use_subspaces: bool = False,
):
    """
    Run vanilla interchange intervention on a list of batches of circuits. Assumes templatic prompts with overlapping tokens.

    Parameters
    ----------
    subject : Subject
        The subject model to intervene on.
    batches : list[dict[str, torch.Tensor]]
        The batches of paired circuits to intervene on.
    batch_size : int
        The batch size.
    use_subspaces : bool, optional
        Whether to use subspaces or just intervene on whole MLPs, by default False.

    Returns
    -------
    list[dict[str, Any]]
        The results of the intervention.
    """
    model = subject.model._model
    tokenizer = subject.tokenizer
    num_layers = int(model.config.num_hidden_layers)
    data = []
    cached_original_outputs, cached_source_outputs = [], []

    # intervene at each layer
    for layer in tqdm(range(num_layers), desc="Intervening at each layer"):
        # set up pyvene config
        config = {
            "layer": layer,
            "component": "mlp_activation",
            "intervention_type": pv.VanillaIntervention,
            "unit": "pos",
        }
        pv_model = pv.IntervenableModel(
            config,
            model=model,
        )

        # go through each pair
        for idx, batch in enumerate(batches):
            # get vars
            (
                base,
                source,
                base_class,
                source_class,
                base_idx,
                source_idx,
                base_subspaces,
                source_subspaces,
                subspaces,
            ) = (
                batch["base"],
                batch["source"],
                batch["base_class"],
                batch["source_class"],
                batch["base_idx"],
                batch["source_idx"],
                batch["base_subspaces"],
                batch["source_subspaces"],
                batch["subspaces"],
            )

            # get base and src conts
            base_conts = [tokenizer.encode(c)[1] for c in base_class]
            source_conts = [tokenizer.encode(c)[1] for c in source_class]

            # run the forward pass intervening at each token
            for pos in range(base["input_ids"].shape[-1]):
                # skip tokens that differ or are masked
                if (
                    base["input_ids"][0, pos] != source["input_ids"][0, pos]
                    or base["attention_mask"][0, pos] == 0
                ) and len(cached_original_outputs) > idx:
                    continue

                # intervene with or without subspaces
                if subspaces is None or not use_subspaces:
                    original_outputs, intervened_outputs = pv_model(
                        base,
                        [source],
                        {"sources->base": pos},
                        output_original_output=(
                            True if len(cached_original_outputs) == idx else False
                        ),
                    )
                    subspaces_len = None
                else:
                    if (layer, pos) not in subspaces:
                        continue
                    sub = list(subspaces[(layer, pos)])
                    subspaces_len = len(sub)
                    original_outputs, intervened_outputs = pv_model(
                        base,
                        [source],
                        {"sources->base": pos},
                        output_original_output=(
                            True if len(cached_original_outputs) == idx else False
                        ),
                        subspaces=sub,
                    )

                # cache the original outputs to compute diffs
                if len(cached_original_outputs) == idx:
                    cached_original_outputs.append(original_outputs)
                    source_outputs, _ = pv_model(
                        source,
                        [source],
                        {"sources->base": pos},
                        output_original_output=True,
                    )
                    cached_source_outputs.append(source_outputs)
                else:
                    original_outputs = cached_original_outputs[idx]
                    source_outputs = cached_source_outputs[idx]

                if (
                    base["input_ids"][0, pos] != source["input_ids"][0, pos]
                    or base["attention_mask"][0, pos] == 0
                ):
                    continue

                # get the probabilities for each continuation
                original_logits = original_outputs.logits[:, -1]
                source_logits = source_outputs.logits[:, -1]
                intervened_logits = intervened_outputs.logits[:, -1]

                # store the results
                for i in range(batch_size):
                    base_cont, source_cont = base_conts[i], source_conts[i]
                    # print(f"fxn {base_class[i]} <- {source_class[i]} (token {format_token(base['input_ids'][i, pos], tokenizer)} <- {format_token(source['input_ids'][i, pos], tokenizer)}), layer {layer}, pos {pos}: intervened_argmax_token {format_token(torch.argmax(intervened_probs[i, :]).item(), tokenizer)} with prob {intervened_probs[i, torch.argmax(intervened_probs[i, :])].item()}")
                    data.append(
                        {
                            "layer": layer,
                            "pos": pos,
                            "pos_neg": base["input_ids"].shape[-1] - pos,
                            "base_token": format_token(base["input_ids"][i, pos], tokenizer),
                            "source_token": format_token(source["input_ids"][i, pos], tokenizer),
                            "base_idx": base_idx[i],
                            "source_idx": source_idx[i],
                            "base_class": base_class[i],
                            "source_class": source_class[i],
                            "fxn": f"{base_class[i]} <- {source_class[i]}",
                            "subspaces": subspaces_len,
                            **compute_metrics(
                                original_logits[i, :],
                                intervened_logits[i, :],
                                source_logits[i, :],
                                base_cont,
                                source_cont,
                                tokenizer,
                            ),
                        }
                    )

        # deregister intervention
        pv_model._cleanup_states()
        torch.cuda.empty_cache()

    return data


def run_zero_intervention(
    subject: Subject,
    batches: list[dict[str, torch.Tensor]],
    batch_size: int,
    multiplier: float = 0.0,
    mode: Literal["parallel", "serial"] = "parallel",
    complement: bool = False,
):
    model = subject.model._model
    tokenizer = subject.tokenizer
    data = []

    # go through each pair
    for idx, batch in enumerate(batches):
        # get vars
        (
            base,
            source,
            base_class,
            source_class,
            base_idx,
            source_idx,
            base_subspaces,
            source_subspaces,
            subspaces,
            record_subspaces,
        ) = (
            batch["base"],
            batch["source"],
            batch["base_class"],
            batch["source_class"],
            batch["base_idx"],
            batch["source_idx"],
            batch["base_subspaces"],
            batch["source_subspaces"],
            batch["subspaces"],
            batch["record_subspaces"],
        )

        # set up config
        subspaces_without_ends = {
            k: list(v) for k, v in subspaces.items() if 0 <= k[0] < model.config.num_hidden_layers
        }
        record_subspaces_without_ends = {
            k: list(v)
            for k, v in record_subspaces.items()
            if 0 <= k[0] < model.config.num_hidden_layers
        }
        config = multiple_subspaces_config(
            subspaces_without_ends,
            multiplier,
            record_subspaces_without_ends,
            mode=mode,
            complement=complement,
        )
        if mode == "serial":
            num_interventions = max(x.group_key + 1 for x in config.representations)
        pv_model = pv.IntervenableModel(
            config,
            model=model,
        )

        if mode == "serial":
            pv_model.is_model_stateless = True

        # get base and src conts
        base_conts = [tokenizer.encode(c)[1] for c in base_class]
        source_conts = [tokenizer.encode(c)[1] for c in source_class]

        # run the forward pass
        original_outputs, intervened_outputs = pv_model(
            base,
            None if mode == "parallel" else [base],
            (
                {"base": list(range(base["input_ids"].shape[-1]))}
                if mode == "parallel"
                else {
                    "base": list(range(base["input_ids"].shape[-1])),
                    f"source_{num_interventions-1}->base": list(range(base["input_ids"].shape[-1])),
                    **{
                        f"source_{i}->source_{i+1}": list(range(base["input_ids"].shape[-1]))
                        for i in range(0, num_interventions - 1)
                    },
                }
            ),
            output_original_output=True,
        )

        # collect the record subspaces
        collected_subspaces = []
        for intervention in pv_model.interventions.values():
            if isinstance(intervention, SubspaceZeroIntervention):
                for pos_idx in range(len(intervention.record_pos)):
                    for batch_idx in range(intervention.collected_activations[pos_idx].shape[0]):
                        for subspace, activation in zip(
                            intervention.record_subspaces[pos_idx],
                            intervention.collected_activations[pos_idx][batch_idx].tolist(),
                        ):
                            collected_subspaces.append(
                                {
                                    "batch_idx": batch_idx,
                                    "layer": intervention.layer,
                                    "token": intervention.record_pos[pos_idx],
                                    "neuron": subspace,
                                    "activation": activation,
                                }
                            )

        for i in range(batch_size):
            base_cont, source_cont = int(base_conts[i]), int(source_conts[i])
            data.append(
                {
                    "base_tokens": [
                        format_token(base["input_ids"][i, j], tokenizer)
                        for j in range(base["input_ids"].shape[-1])
                    ],
                    "source_tokens": [
                        format_token(source["input_ids"][i, j], tokenizer)
                        for j in range(source["input_ids"].shape[-1])
                    ],
                    "base_idx": base_idx[i],
                    "source_idx": source_idx[i],
                    "base_class": base_class[i],
                    "source_class": source_class[i],
                    "fxn": f"{base_class[i]} <- {source_class[i]}",
                    **compute_metrics(
                        original_outputs.logits[i, -1],
                        intervened_outputs.logits[i, -1],
                        None,
                        base_cont,
                        source_cont,
                        tokenizer,
                        include_logits_and_probs=True,
                    ),
                }
            )

        # deregister intervention
        pv_model._cleanup_states()
        torch.cuda.empty_cache()

    return data, pd.DataFrame(collected_subspaces) if collected_subspaces else None


def get_cluster_steering_effects(
    subject: Subject,
    df_node: pd.DataFrame,
    df_edge: pd.DataFrame,
    df_node_embedded: pd.DataFrame,
    df_edge_embedded: pd.DataFrame,
    cis: list[torch.Tensor],
    attention_masks: list[torch.Tensor],
    labels: list[str],
    multiplier: float = 0.0,
    verbose: bool = False,
    record: bool = False,
    complement: bool = False,
    custom_neuron_ids: list[tuple[int, int, int]] | None = None,
):
    """
    Get the steering effects of each cluster.
    """
    if verbose:
        logger.info(
            "get_cluster_steering_effects: processing %d examples (multiplier=%.2f, complement=%s)",
            len(cis),
            multiplier,
            complement,
        )

    # define vars
    tokenizer = subject.tokenizer
    diversity_stats = []
    cluster_to_cluster = []
    cluster_to_output = []

    # run steering for each example in the dataset
    for batch_idx, (ci, attention_mask, label) in enumerate(zip(cis, attention_masks, labels)):
        # reorg as if only one example were in the dataset
        true_label = f"{label}___{batch_idx}"
        df_node_ex = df_node[df_node.label == true_label].copy()
        df_edge_ex = df_edge[df_edge.label == true_label].copy()
        df_node_ex.loc[:, "layer"] = df_node_ex.input_variable.apply(lambda x: x.layer)
        df_node_ex.loc[:, "token"] = df_node_ex.input_variable.apply(lambda x: x.token)
        df_node_ex.loc[:, "neuron"] = df_node_ex.input_variable.apply(lambda x: x.neuron)
        df_node_ex.loc[:, "label"] = f"{label}___0"
        df_edge_ex.loc[:, "label"] = f"{label}___0"

        top_output, top_2_output, top_10_output = [], [], []
        top_input, top_2_input, top_10_input = [], [], []
        node_to_cluster = {}

        # apply cluster data
        if custom_neuron_ids is None:
            for _, row in df_node_embedded.iterrows():
                node_to_cluster[
                    (row.input_variable.layer, row.input_variable.token, row.input_variable.neuron)
                ] = row["cluster"]
            clusters = sorted(df_node_embedded["cluster"].unique().tolist())
            all_clusters = ["none", "all"] + clusters
        else:
            for neuron_id in custom_neuron_ids:
                node_to_cluster[neuron_id] = "0"
            clusters = ["0"]
            all_clusters = ["0"]

        # apply cluster data to df_node_ex
        df_node_ex.loc[:, "cluster"] = df_node_ex.apply(
            lambda row: node_to_cluster.get(
                (row.layer, row.token if custom_neuron_ids is None else -1, row.neuron), None
            ),
            axis=1,
        )

        # run steering for each cluster + none + all
        num_clusters = 0
        none_result = None
        for cluster in all_clusters:
            # get subset df
            if cluster == "none":
                subset_df = df_node_ex[
                    (df_node_ex.cluster == cluster)
                    & (df_node_ex.layer != -1)
                    & (df_node_ex.layer != 32)
                ]
            elif cluster == "all":
                subset_df = df_node_ex
            else:
                subset_df = df_node_ex[
                    (df_node_ex.cluster == cluster)
                    & (df_node_ex.layer != -1)
                    & (df_node_ex.layer != 32)
                ]
                if len(subset_df) == 0:
                    continue

            # prepare circuits for intervention and run steering
            pairs = prepare_circuits_for_interchange_interventions(
                subject=subject,
                cis=[ci],
                attention_masks=[attention_mask],
                labels=[label],
                df_node=subset_df,
                needed_pairs=1,
                allow_same_label=True,
                ignore_source=True,
                df_node_record=df_node_ex if record else None,
            )
            batch_size = 1
            batches = batchify(pairs, batch_size)
            if verbose:
                logger.info(
                    "Example %s: running intervention with multiplier %.2f for cluster %s on %d nodes",
                    true_label,
                    multiplier,
                    cluster,
                    len(subset_df),
                )

            data, collected_subspaces = run_zero_intervention(
                subject,
                batches,
                batch_size,
                multiplier=multiplier,
                mode="parallel",
                complement=complement,
            )

            # record subspaces effects for downstream clusters
            if collected_subspaces is not None:
                collected_subspaces.rename(
                    columns={"activation": "steered_activation"}, inplace=True
                )
                intervened_df = pd.merge(
                    df_node_ex, collected_subspaces, on=["layer", "token", "neuron"], how="right"
                )
                intervened_df.dropna(inplace=True)
                intervened_df["effect"] = (intervened_df["steered_activation"]) / intervened_df[
                    "activation"
                ]
                if intervened_df.cluster.nunique() == len(intervened_df):
                    intervened_df["cluster"] = intervened_df.apply(
                        lambda row: f"{row.layer},{row.token},{row.neuron}", axis=1
                    )
                effect_df = intervened_df.groupby("cluster").effect.mean().to_dict()

            # collect stats
            for d in data:
                logit_diff = d["intervened_logits"] - d["original_logits"]
                top_logit_diffs = torch.topk(logit_diff, k=10)
                bottom_logit_diffs = torch.topk(logit_diff, k=10, largest=False)
                prob_str = ", ".join(
                    [
                        f"{token:>15}: {prob:>7.2%}"
                        for token, prob in zip(
                            d["intervened_top_10_tokens"], d["intervened_top_10_tokens_probs"]
                        )
                    ]
                )
                if verbose:
                    logger.info(
                        "%s (%3d %7.2f attribution): %s",
                        cluster if cluster is not None else "all",
                        len(subset_df),
                        subset_df.attribution.sum(),
                        prob_str,
                    )
                if cluster == "none":
                    none_result = d["intervened_probs"]
                cluster_to_output.extend(
                    [
                        {
                            "cluster": cluster,
                            "label": true_label,
                            "logits": d["intervened_logits"],
                            "probs": d["intervened_probs"],
                            "logprobs": torch.nn.functional.log_softmax(
                                d["intervened_logits"], dim=-1
                            ),
                            "top_tokens": d["intervened_top_10_tokens"],
                            "top_tokens_probs": d["intervened_top_10_tokens_probs"],
                            "top_logit_diffs": [
                                format_token(token, tokenizer) for token in top_logit_diffs.indices
                            ],
                            "top_logit_diffs_logits": top_logit_diffs.values.tolist(),
                            "bottom_logit_diffs": [
                                format_token(token, tokenizer)
                                for token in bottom_logit_diffs.indices
                            ],
                            "bottom_logit_diffs_logits": bottom_logit_diffs.values.tolist(),
                            "prob_diff": d["intervened_probs"] - d["original_probs"],
                            "logit_diff": d["intervened_logits"] - d["original_logits"],
                            "kl_div": (
                                torch.nn.functional.kl_div(
                                    d["intervened_probs"].log(),
                                    none_result,
                                    reduction="sum",
                                ).item()
                                if cluster != "none" and none_result is not None
                                else 0.0
                            ),
                        },
                    ]
                )

                # store data about downstream clusters
                if collected_subspaces is not None:
                    for k, v in effect_df.items():
                        cluster_to_cluster.append(
                            {
                                "label": true_label,
                                "source_cluster": str(cluster),
                                "target_cluster": str(k),
                                "effect": v,
                            }
                        )
                    for _, row in df_node_ex[df_node_ex.layer == 32].iterrows():
                        logit = d["intervened_logits"][row.neuron]
                        original_logit = d["original_logits"][row.neuron]
                        cluster_to_cluster.append(
                            {
                                "label": true_label,
                                "source_cluster": str(cluster),
                                "target_cluster": "Say " + format_token(row.neuron, tokenizer),
                                "effect": ((logit) / original_logit).item(),
                            }
                        )

                # outputs
                top_output.extend(d["intervened_top_10_tokens"][:1])
                top_2_output.extend(d["intervened_top_10_tokens"][:2])
                top_10_output.extend(d["intervened_top_10_tokens"])

                # inputs
                if len(subset_df) > 0:
                    attr_maps = np.array(
                        subset_df.attr_map.dropna().values.tolist()
                    )  # shape: (n_neurons, n_tokens)
                    attr_maps = attr_maps / attr_maps.sum(
                        axis=1, keepdims=True
                    )  # each neuron's attr map sums to 1
                    attr_maps = attr_maps.mean(axis=0)  # average over neurons, shape: (n_tokens)
                    top_10 = torch.topk(torch.tensor(attr_maps), k=10)
                    top_input.extend(top_10.indices[:1].tolist())
                    top_2_input.extend(top_10.indices[:2].tolist())
                    top_10_input.extend(top_10.indices.tolist())
            num_clusters += 1

        # compute diversity stats
        stats = {
            "top_inputs_unique": len(set(top_input)),
            "top_2_inputs_unique": len(set(top_2_input)),
            "top_10_inputs_unique": len(set(top_10_input)),
            "top_outputs_unique": len(set(top_output)),
            "top_2_outputs_unique": len(set(top_2_output)),
            "top_10_outputs_unique": len(set(top_10_output)),
            "label": true_label,
            "top_inputs": list(set(top_input)),
            "top_2_inputs": list(set(top_2_input)),
            "top_10_inputs": list(set(top_10_input)),
            "top_outputs": list(set(top_output)),
            "top_2_outputs": list(set(top_2_output)),
            "top_10_outputs": list(set(top_10_output)),
            "cluster_size": len(subset_df),
        }
        diversity_stats.append(stats)

    # return everything
    return (
        pd.DataFrame(diversity_stats),
        pd.DataFrame(cluster_to_cluster),
        pd.DataFrame(cluster_to_output),
    )


def export_cluster_data_to_json(df: pd.DataFrame, output_path: str) -> dict[str, Any]:
    import json

    """
    Export cluster DataFrame to efficient JSON format.

    Args:
        df: DataFrame with columns: cluster, label, top_tokens, top_tokens_probs,
            top_logit_diffs, top_logit_diffs_logits, bottom_logit_diffs,
            bottom_logit_diffs_logits, multiplier
        output_path: Path to save JSON file

    Returns:
        Dictionary with the exported data structure
    """

    def convert_to_serializable(obj):
        """Convert numpy arrays and other non-serializable objects to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, str):
            # Handle string representations of lists
            if obj.startswith("[") and obj.endswith("]"):
                try:
                    # Try to evaluate as a Python literal
                    import ast

                    return ast.literal_eval(obj)
                except (ValueError, SyntaxError):
                    # If it fails, split by comma and clean up
                    items = obj.strip("[]").split(", ")
                    return [item.strip().strip("'\"") for item in items]
            return obj
        return obj

    # Convert DataFrame to efficient structure
    clusters = []
    unique_labels = set()

    for _, row in df.iterrows():
        cluster_data = {
            "cluster_id": str(row["cluster"]),
            "label": str(row["label"]),
            "multiplier": float(row["multiplier"]),
            "top_tokens": convert_to_serializable(row["top_tokens"]),
            "top_tokens_probs": convert_to_serializable(row["top_tokens_probs"]),
            "top_logit_diffs": convert_to_serializable(row["top_logit_diffs"]),
            "top_logit_diffs_logits": convert_to_serializable(row["top_logit_diffs_logits"]),
            "bottom_logit_diffs": convert_to_serializable(row["bottom_logit_diffs"]),
            "bottom_logit_diffs_logits": convert_to_serializable(row["bottom_logit_diffs_logits"]),
        }
        clusters.append(cluster_data)
        unique_labels.add(str(row["label"]))

    # Create the final data structure
    export_data = {
        "clusters": clusters,
        "unique_labels": sorted(list(unique_labels)),
        "metadata": {"total_clusters": len(clusters), "export_format": "cluster_analysis_v1"},
    }

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    logger.info("Exported %d clusters to %s", len(clusters), output_path)
    logger.info("File size: %.1f KB", len(json.dumps(export_data)) / 1024)
