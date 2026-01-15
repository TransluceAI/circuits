"""
Feature-scoring utilities for `Circuit`.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast
from zipfile import Path

import numpy as np
import pandas as pd
from circuits.analysis.cluster import NeuronId, prepare_circuit_data  # type: ignore
from circuits.utils.descriptions import get_descriptions  # type: ignore
from tqdm import tqdm

if TYPE_CHECKING:
    from circuits.analysis.circuit_ops import Circuit


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_df_node_summed(circuit: "Circuit", verbose: bool = False) -> pd.DataFrame:
    """
    Aggregate node-level attributions by input variable and label, caching the result for reuse.
    """
    if not hasattr(circuit, "_df_node_summed_cache"):
        circuit._df_node_summed_cache = None  # type: ignore[attr-defined]
    cached = getattr(circuit, "_df_node_summed_cache", None)  # type: ignore[attr-defined]
    if cached is not None:
        if verbose:
            logger.info(
                "get_df_node_summed: returning cached dataframe with %d rows",
                len(cached),
            )
        return cached

    if verbose:
        logger.info(
            "get_df_node_summed: preprocessing %d node rows via prepare_circuit_data",
            len(circuit.df_node),
        )

    df_node_summed, _ = prepare_circuit_data(
        circuit.df_node.copy(),
        pd.DataFrame(columns=circuit.df_edge.columns),
        sum_over_tokens=True,
        verbose=verbose,
    )

    df_node_summed["label_index"] = (
        df_node_summed["label"].str.rsplit("___", n=1).str[-1].astype(int)
    )
    label_indices = df_node_summed["label_index"].to_numpy()
    if label_indices.size == 0:
        raise ValueError("No labels found in df_node; cannot score features.")

    # final
    circuit._df_node_summed_cache = df_node_summed  # type: ignore[attr-defined]
    if verbose:
        logger.info(
            "get_df_node_summed: aggregated to %d rows (unique features=%d)",
            len(df_node_summed),
            df_node_summed["input_variable"].nunique(),
        )
    return df_node_summed


def compute_auc(
    pos_values: np.ndarray,
    neg_values: np.ndarray,
    missing_pos: int,
    missing_neg: int,
    total_positive: int,
    total_negative: int,
) -> float:
    """
    Efficient computation of the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) score.
    """
    if total_positive == 0 or total_negative == 0:
        return np.nan

    all_unique = np.unique(
        np.concatenate([pos_values, neg_values, np.array([0.0], dtype=np.float64)])
    )
    pos_counts = np.zeros_like(all_unique, dtype=np.int64)
    neg_counts = np.zeros_like(all_unique, dtype=np.int64)

    if pos_values.size:
        pos_idx = np.searchsorted(all_unique, pos_values)
        np.add.at(pos_counts, pos_idx, 1)
    if neg_values.size:
        neg_idx = np.searchsorted(all_unique, neg_values)
        np.add.at(neg_counts, neg_idx, 1)

    zero_idx = np.searchsorted(all_unique, 0.0)
    pos_counts[zero_idx] += missing_pos
    neg_counts[zero_idx] += missing_neg

    cumulative_neg = 0
    rank_sum = 0.0
    for pos_count, neg_count in zip(pos_counts, neg_counts):
        if pos_count:
            rank_sum += pos_count * (cumulative_neg + 0.5 * neg_count)
        cumulative_neg += neg_count
    return rank_sum / (total_positive * total_negative)


def score_features(
    circuit: Circuit, example_labels: Sequence[bool], verbose: bool = False
) -> pd.DataFrame:
    """
    Compute ROC-AUC scores for aggregated neuron features across the dataset.
    """
    num_examples = len(circuit.cis)
    example_labels_array = np.asarray(example_labels, dtype=bool)
    if example_labels_array.ndim != 1:
        raise ValueError("example_labels must be a 1D sequence of booleans.")
    if example_labels_array.shape[0] != num_examples:
        raise ValueError(
            f"example_labels length ({example_labels_array.shape[0]}) must match number of circuit inputs ({num_examples})."
        )

    if verbose:
        logger.info(
            "score_features: scoring %d labels against %d examples",
            example_labels_array.shape[0],
            num_examples,
        )

    df_node_summed = get_df_node_summed(circuit, verbose=verbose)

    total_examples = example_labels_array.shape[0]
    class_array = example_labels_array.astype(bool)
    total_positive = int(class_array.sum())
    total_negative = int(total_examples - total_positive)

    if verbose:
        logger.info(
            "score_features: total_positive=%d total_negative=%d unique_features=%d",
            total_positive,
            total_negative,
            df_node_summed["input_variable"].nunique(),
        )

    records = []
    grouped = df_node_summed.groupby("input_variable", observed=True, sort=False)
    for feature, group in grouped:
        indices = group["label_index"].to_numpy(dtype=int)
        attributions = group["attribution"].to_numpy(dtype=np.float64)
        class_flags = class_array[indices]

        positive_mask = class_flags
        negative_mask = ~positive_mask

        pos_values = attributions[positive_mask]
        neg_values = attributions[negative_mask]

        observed_pos = int(positive_mask.sum())
        observed_neg = int(negative_mask.sum())
        missing_pos = max(total_positive - observed_pos, 0)
        missing_neg = max(total_negative - observed_neg, 0)

        auc_score = compute_auc(
            pos_values, neg_values, missing_pos, missing_neg, total_positive, total_negative
        )

        sum_in_class = pos_values.sum()
        sum_out_class = neg_values.sum()

        avg_in_class = sum_in_class / total_positive if total_positive > 0 else np.nan
        avg_out_class = sum_out_class / total_negative if total_negative > 0 else np.nan
        all_avg = (sum_in_class + sum_out_class) / total_examples if total_examples > 0 else np.nan

        records.append(
            {
                "input_variable": feature,
                "roc_auc_score": auc_score,
                "n_total": total_examples,
                "n_positive": total_positive,
                "n_negative": total_negative,
                "avg_attribution_in_class": avg_in_class,
                "avg_attribution_out_of_class": avg_out_class,
                "avg_attribution": all_avg,
                "n_zero_positive": missing_pos,
                "n_zero_negative": missing_neg,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "input_variable",
                "roc_auc_score",
                "n_total",
                "n_positive",
                "n_negative",
                "avg_attribution_in_class",
                "avg_attribution_out_of_class",
                "avg_attribution",
                "n_zero_positive",
                "n_zero_negative",
            ]
        )

    result_df = pd.DataFrame.from_records(records)
    result_df = result_df.sort_values("roc_auc_score", ascending=False, na_position="last")
    if verbose:
        logger.info(
            "score_features: produced %d scored features (max_auc=%0.2f, min_auc=%0.2f)",
            len(result_df),
            result_df["roc_auc_score"].max(),
            result_df["roc_auc_score"].min(),
        )
    return result_df


def score_features_multiclass(
    circuit: Circuit,
    example_labels: Sequence[Any],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute per-class ROC-AUC scores for multi-class labels by reducing each class to a
    binary classification task and delegating to `score_features`.
    """
    labels_array = np.asarray(example_labels)
    if labels_array.ndim != 1:
        raise ValueError("example_labels must be a 1D sequence.")
    if labels_array.shape[0] != len(circuit.cis):
        raise ValueError(
            f"example_labels length ({labels_array.shape[0]}) must match number of circuit inputs ({len(circuit.cis)})."
        )

    unique_labels = pd.Series(labels_array).unique()
    if unique_labels.size < 2:
        raise ValueError("score_features_multiclass requires at least two unique labels.")

    unique_labels = unique_labels.tolist()
    if verbose:
        logger.info(
            "score_features_multiclass: evaluating %d unique labels",
            len(unique_labels),
        )

    per_class_scores = []
    for label in tqdm(unique_labels, desc="Scoring feature AUROC per class"):
        binary_labels = (labels_array == label).tolist()
        scores = circuit.score_features(binary_labels, verbose=verbose).copy()
        scores.insert(0, "class_label", label)
        per_class_scores.append(scores)

    if not per_class_scores:
        return pd.DataFrame(
            columns=[
                "class_label",
                "input_variable",
                "roc_auc_score",
                "n_total",
                "n_positive",
                "n_negative",
                "avg_attribution_in_class",
                "avg_attribution_out_of_class",
                "avg_attribution",
                "n_zero_positive",
                "n_zero_negative",
            ]
        )

    combined = pd.concat(per_class_scores, ignore_index=True)
    return combined


async def cluster_with_hypotheses(
    circuit: Circuit,
    hypotheses: Mapping[str, Sequence[str]],
    above_threshold: float = 0.8,
    below_threshold: float = 0.2,
    in_class_attribution_threshold: float | None = None,
    unique_only: bool = False,
    subset_labels: Sequence[str] | None = None,
    cluster_kwargs: Mapping[str, Any] | None = None,
    max_layer: int = 32,
    verbose: bool = False,
) -> dict[str, int]:
    """
    Assign neurons to hypothesis-defined clusters by scoring features against the provided label sets.
    """
    if not hypotheses:
        raise ValueError("hypotheses must be a non-empty mapping.")
    if not 0 <= below_threshold <= above_threshold <= 1:
        raise ValueError("below_threshold <= above_threshold and both must be within [0, 1].")

    feature_to_tags: dict[NeuronId, set[str]] = defaultdict(set)
    scored_features: set[str] = set()

    if verbose:
        logger.info(
            "cluster_with_hypotheses: evaluating %d hypotheses (thresholds=%0.2f/%0.2f)",
            len(hypotheses),
            above_threshold,
            below_threshold,
        )

    for hypothesis_name, example_labels in hypotheses.items():
        labels_list = list(example_labels)
        if len(labels_list) != len(circuit.cis):
            raise ValueError(
                f"Hypothesis '{hypothesis_name}' has {len(labels_list)} labels but circuit has {len(circuit.cis)} examples."
            )

        if verbose:
            logger.info(
                "cluster_with_hypotheses: scoring hypothesis '%s'",
                hypothesis_name,
            )

        scores = circuit.score_features_multiclass(
            labels_list,
            hypothesis_name,
            verbose=verbose,
        )
        if scores.empty:
            continue

        scored_features.update(scores["input_variable"].dropna().astype(str))

        for class_label, class_scores in scores.groupby("class_label", sort=False):
            for polarity, mask in (
                ("+", class_scores["roc_auc_score"] >= above_threshold),
                ("-", class_scores["roc_auc_score"] <= below_threshold),
            ):
                selected = cast(
                    Sequence[NeuronId], class_scores.loc[mask, "input_variable"].dropna().unique()
                )
                if in_class_attribution_threshold is not None:
                    selected = selected[
                        class_scores["avg_attribution_in_class"].abs()
                        >= in_class_attribution_threshold
                    ]
                if selected.size == 0:
                    continue
                label_tag = f"{hypothesis_name}={class_label}:{polarity}"
                for feature in selected:
                    if feature.layer in [-1, max_layer]:
                        continue
                    feature_to_tags[feature].add(label_tag)

    if verbose:
        logger.info(
            "cluster_with_hypotheses: %d/%d features met AUROC thresholds",
            len(feature_to_tags),
            len(scored_features),
        )
    if not feature_to_tags:
        raise ValueError(
            "No features met the provided AUROC thresholds for any hypothesis/class combination."
        )

    manual_clusters: dict[NeuronId, str] = {}
    cluster_label_to_id: dict[str, int] = {}

    for feature, tags in feature_to_tags.items():
        if unique_only and len(tags) > 1:
            continue
        ordered_tags = ", ".join(sorted(tags))
        cluster_id = cluster_label_to_id.setdefault(ordered_tags, len(cluster_label_to_id))
        manual_clusters[feature] = str(cluster_id)

    if not manual_clusters:
        raise ValueError("No clusters could be constructed from the provided hypotheses.")

    if verbose:
        logger.info(
            "cluster_with_hypotheses: assigning %d neurons to manual clusters",
            len(manual_clusters),
        )

    # filter df_node and df_edge to only include the subset of labels now
    if subset_labels is not None:
        circuit.df_node = circuit.df_node[circuit.df_node["label"].isin(subset_labels)]
        if len(circuit.df_edge) > 0:
            circuit.df_edge = circuit.df_edge[circuit.df_edge["label"].isin(subset_labels)]
        if verbose:
            logger.info(
                "cluster_with_hypotheses: filtered df_node and df_edge to %d labels, resulting in %d nodes, %d edges",
                len(subset_labels),
                len(circuit.df_node),
                len(circuit.df_edge),
            )
        if len(circuit.df_node) == 0:
            raise ValueError("Filtered df_node is empty after subsetting.")

    circuit._manual_clusters_map = manual_clusters  # type: ignore[attr-defined]
    cluster_params = dict(cluster_kwargs or {})
    cluster_params.pop("manual_clusters", None)
    n_clusters = cluster_params.pop("n_clusters", 0)
    circuit.cluster(
        n_clusters=n_clusters,
        manual_clusters=manual_clusters,
        **cluster_params,
    )

    cluster_labels_override = list(circuit.data_circuit.get("cluster_labels", []) or [])
    max_cluster_id = max(cluster_label_to_id.values(), default=-1)
    if max_cluster_id >= 0:
        if len(cluster_labels_override) <= max_cluster_id:
            cluster_labels_override.extend(
                [
                    f"Cluster {idx}"
                    for idx in range(len(cluster_labels_override), max_cluster_id + 1)
                ]
            )
        for label_str, cluster_id in cluster_label_to_id.items():
            cluster_labels_override[cluster_id] = label_str
        circuit.data_circuit["cluster_labels"] = cluster_labels_override
        circuit._manual_cluster_labels = list(cluster_labels_override)  # type: ignore[attr-defined]

    if verbose:
        logger.info(
            "cluster_with_hypotheses: created %d unique cluster labels",
            len(cluster_label_to_id),
        )

    return cluster_label_to_id


def _parse_input_variable(value: str) -> NeuronId:
    """
    Convert the "layer, neuron, polarity" string into a NeuronId.
    """
    if isinstance(value, NeuronId):
        return NeuronId(
            layer=int(value.layer), token=-1, neuron=int(value.neuron), polarity=value.polarity
        )
    value_str = str(value)
    parts = value_str.split(", ")
    if len(parts) != 3:
        raise ValueError(f"Unexpected input_variable format: {value}")
    layer_str, neuron_str, polarity = parts
    return NeuronId(layer=int(layer_str), token=-1, neuron=int(neuron_str), polarity=polarity)


def export_hypothesis_score_jsons(
    circuit: Circuit,
    hypotheses: Mapping[str, Sequence[Any]],
    output_dir: Path,
    auc_threshold_high: float = 0.8,
    auc_threshold_low: float = 0.2,
    in_class_attribution_threshold: float | None = None,
) -> None:
    """
    Score every hypothesis against the circuit, filter by AUROC thresholds,
    and export the resulting feature metadata to JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for hypothesis_name, example_labels in hypotheses.items():
        print(f"Scoring hypothesis '{hypothesis_name}'...")
        scores = circuit.score_features_multiclass(example_labels, hypothesis_name, verbose=True)
        if in_class_attribution_threshold is not None:
            scores = scores[
                scores["avg_attribution_in_class"].abs() >= in_class_attribution_threshold
            ]
        scores = scores[
            scores["input_variable"].apply(lambda x: x.layer not in [-1, circuit.subject.L])
        ]
        if scores.empty:
            output_path = output_dir / f"{hypothesis_name}.json"
            with open(output_path, "w") as f:
                json.dump([], f, indent=2)
            print(f"No scores produced for {hypothesis_name}; wrote empty JSON to {output_path}")
            continue

        mask = (scores["roc_auc_score"] >= auc_threshold_high) | (
            scores["roc_auc_score"] <= auc_threshold_low
        )
        filtered = scores[mask].dropna(subset=["input_variable"]).copy()
        filtered, circuit.neuron_label_cache = get_descriptions(
            filtered,
            circuit.tokenizer,
            last_layer=circuit.subject.L,
            get_desc=True,
            verbose=True,
            neuron_label_cache=circuit.neuron_label_cache,
        )

        output_path = output_dir / f"{hypothesis_name}.json"
        if filtered.empty:
            with open(output_path, "w") as f:
                json.dump([], f, indent=2)
            print(
                f"No features crossed thresholds for {hypothesis_name}; wrote empty JSON to {output_path}"
            )
            continue

        parsed_inputs = filtered["input_variable"].apply(_parse_input_variable)
        filtered.loc[:, "layer"] = parsed_inputs.apply(lambda iv: int(iv.layer))
        filtered.loc[:, "neuron"] = parsed_inputs.apply(lambda iv: int(iv.neuron))
        filtered.loc[:, "polarity"] = parsed_inputs.apply(lambda iv: iv.polarity)
        filtered.rename(columns={"class_label": "target_variable"}, inplace=True)
        for col in [
            "roc_auc_score",
            "avg_attribution_in_class",
            "avg_attribution_out_of_class",
            "avg_attribution",
        ]:
            filtered.loc[:, col] = pd.to_numeric(filtered[col], errors="coerce").astype(float)

        export_columns = [
            "layer",
            "neuron",
            "polarity",
            "roc_auc_score",
            "target_variable",
            "description",
            "avg_attribution_in_class",
            "avg_attribution_out_of_class",
            "avg_attribution",
        ]
        export_records = filtered[export_columns].to_dict(orient="records")
        with open(output_path, "w") as f:
            json.dump(export_records, f, indent=2)
        print(f"Wrote {len(export_records)} records to {output_path}")


__all__ = [
    "cluster_with_hypotheses",
    "get_df_node_summed",
    "score_features",
    "score_features_multiclass",
    "export_hypothesis_score_jsons",
]
