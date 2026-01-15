"""
Utilities for clustering neurons in a CLSO circuit.
"""

import logging
import random
from collections import Counter
from typing import Any, Literal, Mapping, NamedTuple, cast

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from circuits.utils.descriptions import get_descriptions
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

NODE_COLUMNS = [
    "layer",
    "token",
    "neuron",
    "label",
    "attr_map",
    "contrib_map",
    "attribution",
    "activation",
]
EDGE_COLUMNS = [
    "layer",
    "token",
    "neuron",
    "label",
    "attribution",
    "weight",
]

EMBEDDING_MODES = Literal["attr", "contrib", "attr + contrib", "attr x contrib", "random"]

UNCLUSTERED_CLUSTER_ID = "__unclustered__"


class NeuronId(NamedTuple):
    """
    A neuron identifier.
    """

    layer: int
    token: int
    neuron: int
    polarity: Literal["+", "-"]


class EdgeId(NamedTuple):
    """
    An edge identifier.
    """

    source: NeuronId
    target: NeuronId


def df_sum_over_tokens(
    df_node: pd.DataFrame, df_edge: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sum over tokens for the nodes and edges.
    """
    df_node = (
        df_node.groupby(["layer", "neuron", "polarity", "label"])
        .agg(
            {
                "attr_map": "sum",
                "contrib_map": "sum",
                "activation": "mean",
                "attribution": "sum",
            }
        )
        .reset_index()
    )
    if len(df_edge) > 0:
        df_edge = (
            df_edge.groupby(["layer", "neuron", "polarity", "label"])
            .agg({"attribution": "sum", "weight": "mean"})
            .reset_index()
        )

    # clean up token column
    df_node.loc[:, "token"] = -1
    if len(df_edge) > 0:
        df_edge.loc[:, "token"] = "-1->-1"

    # return
    return df_node, df_edge


def prepare_circuit_data(
    df_node: pd.DataFrame,
    df_edge: pd.DataFrame,
    sum_over_tokens: bool = False,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the dataframes to be used for clustering. This function should set types and make sure
    everything is in order for later processing.

    Parameters
    ----------
    df_node: pd.DataFrame
        The dataframe of nodes.
    df_edge: pd.DataFrame
        The dataframe of edges.
    sum_over_tokens: bool
        Whether to sum over tokens.
    verbose: bool
        Whether to emit progress logging.

    Returns
    -------
    df_node: pd.DataFrame
        The preprocessed dataframe of nodes.
    df_edge: pd.DataFrame
        The preprocessed dataframe of edges.
    """

    if verbose:
        logger.info(
            "prepare_circuit_data: %d nodes, %d edges (sum_over_tokens=%s)",
            len(df_node),
            len(df_edge),
            sum_over_tokens,
        )

    # check columns match
    assert len(df_node) > 0, "df_node must have at least one row"
    if set(df_node.columns) != set(NODE_COLUMNS):
        raise ValueError(
            f"df_node columns do not match: {set(df_node.columns)} != {set(NODE_COLUMNS)}"
        )
    if len(df_edge) > 0 and set(df_edge.columns) != set(EDGE_COLUMNS):
        raise ValueError(
            f"df_edge columns do not match: {set(df_edge.columns)} != {set(EDGE_COLUMNS)}"
        )

    # convert maps/attributions to numpy arrays so downstream math is consistent
    def _to_numpy(value: Any) -> np.ndarray[Any, np.dtype[Any]]:
        """
        Ensure map/attribution entries are numpy arrays.
        """
        if isinstance(value, np.ndarray):
            return value  # type: ignore
        # Treat scalars as length-1 arrays to keep shapes predictable
        return np.asarray(value if isinstance(value, (list, tuple)) else [value], dtype=np.float32)

    df_node["polarity"] = df_node["activation"].apply(lambda x: "+" if x >= 0 else "-")
    if verbose:
        polarity_counts = Counter(df_node["polarity"])
        logger.info(
            "prepare_circuit_data: polarity counts %s",
            dict(polarity_counts),
        )
    df_node["attr_map"] = cast(list[NDArray[np.float32]], df_node["attr_map"].apply(_to_numpy))
    df_node["contrib_map"] = cast(
        list[NDArray[np.float32]], df_node["contrib_map"].apply(_to_numpy)
    )

    # compute polarity for edges
    layer_token_neuron_to_polarity = cast(
        Mapping[tuple[int, int, int], Literal["+", "-"]],
        df_node.groupby(["layer", "token", "neuron"])["polarity"].first().to_dict(),
    )
    if len(df_edge) > 0:
        df_edge["polarity"] = df_edge.apply(
            lambda x: layer_token_neuron_to_polarity.get(
                (x["layer"].split("->")[0], x["token"].split("->")[0], x["neuron"].split("->")[0]),
                "+",
            )
            + "->"
            + layer_token_neuron_to_polarity.get(
                (x["layer"].split("->")[1], x["token"].split("->")[1], x["neuron"].split("->")[1]),
                "+",
            ),
            axis=1,
        )

    # sum over tokens
    if sum_over_tokens:
        if verbose:
            logger.info("prepare_circuit_data: summing over tokens")
        df_node, df_edge = df_sum_over_tokens(df_node, df_edge)

    # convert various neuron/edge identifier columns to single types
    df_node["input_variable"] = df_node.apply(
        lambda x: NeuronId(
            layer=x["layer"], token=x["token"], neuron=x["neuron"], polarity=x["polarity"]
        ),
        axis=1,
    )

    # for edges, layer is source->target, same for token and neuron
    if len(df_edge) > 0:
        df_edge["input_variable"] = df_edge.apply(
            lambda x: EdgeId(
                source=NeuronId(
                    layer=x["layer"].split("->")[0],
                    token=x["token"].split("->")[0],
                    neuron=x["neuron"].split("->")[0],
                    polarity=x["polarity"].split("->")[0],
                ),
                target=NeuronId(
                    layer=x["layer"].split("->")[1],
                    token=x["token"].split("->")[1],
                    neuron=x["neuron"].split("->")[1],
                    polarity=x["polarity"].split("->")[1],
                ),
            ),
            axis=1,
        )

    # drop unnecessary columns
    df_node = df_node.drop(columns=["layer", "token", "neuron", "polarity"])
    if len(df_edge) > 0:
        df_edge = df_edge.drop(columns=["layer", "token", "neuron", "polarity"])

    if verbose:
        logger.info(
            "prepare_circuit_data: prepared %d node rows and %d edge rows",
            len(df_node),
            len(df_edge),
        )

    # return
    return df_node, df_edge


def remerge_df_node(
    df_node: pd.DataFrame,
    df_edge: pd.DataFrame,
    df_node_clustered: pd.DataFrame | None = None,
    manual_clusters: Mapping[NeuronId, str] | None = None,
    sum_over_tokens: bool = False,
    verbose: bool = False,
    last_layer: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remerge the dataframe of nodes so that each node gets an embedding and cluster assigned.

    Parameters
    ----------
    df_node: pd.DataFrame
        Raw node dataframe.
    df_edge: pd.DataFrame
        Raw edge dataframe.
    df_node_clustered: pd.DataFrame
        Node dataframe with embeddings/clusters assigned.
    sum_over_tokens: bool
        Whether token dimension is collapsed.
    verbose: bool
        Whether to emit progress logging.
    last_layer: int | None
        Explicit final layer index; inferred from ``df_node`` if ``None``.
    """
    if verbose:
        logger.info(
            "remerge_df_node: starting with %d nodes, %d edges, %d clustered rows",
            len(df_node),
            len(df_edge),
            len(df_node_clustered) if df_node_clustered is not None else -1,
        )

    effective_last_layer = (
        last_layer
        if last_layer is not None
        else (cast(int, df_node.layer.max()) if len(df_node) else -1)
    )

    # prepare df_node
    df_node_prepared, df_edge_prepared = prepare_circuit_data(
        df_node,
        df_edge,
        sum_over_tokens=False,
        verbose=verbose,
    )
    if verbose:
        logger.info(
            "remerge_df_node: prepared %d node rows, %d edge rows",
            len(df_node_prepared),
            len(df_edge_prepared),
        )

    # pivot by label
    labels = cast(list[str], df_node_prepared["label"].unique())
    if verbose:
        logger.info(
            "remerge_df_node: pivoting across %d labels",
            len(labels),
        )
    df_node_prepared = df_node_prepared.pivot(
        index="input_variable",
        columns="label",
        values="attribution",
    ).reset_index()
    if len(df_edge_prepared) > 0:
        df_edge_prepared = df_edge_prepared.pivot(
            index="input_variable",
            columns="label",
            values="attribution",
        ).reset_index()

    # fillna with 0s
    df_node_prepared = df_node_prepared.fillna(0)
    if len(df_edge_prepared) > 0:
        df_edge_prepared = df_edge_prepared.fillna(0)
    if verbose:
        logger.info(
            "remerge_df_node: filled NaNs; node shape %s, edge shape %s",
            df_node_prepared.shape,
            df_edge_prepared.shape if len(df_edge_prepared) > 0 else 0,
        )

    # compute average over labels
    df_node_prepared["Average"] = df_node_prepared.apply(
        lambda row: np.mean([row[label] for label in labels]), axis=1
    )
    if len(df_edge_prepared) > 0:
        df_edge_prepared["Average"] = df_edge_prepared.apply(
            lambda row: np.mean([row[label] for label in labels]), axis=1
        )

    # set up mapping from input variable to cluster
    if manual_clusters is None:
        input_variable_to_cluster = cast(
            Mapping[NeuronId, str],
            df_node_clustered.groupby("input_variable")["cluster"].first().to_dict(),
        )
        input_variable_to_embedding = cast(
            Mapping[NeuronId, NDArray[np.float32]],
            df_node_clustered.groupby("input_variable")["embedding"].first().to_dict(),
        )
    else:
        input_variable_to_cluster = manual_clusters
        input_variable_to_embedding = {}

    def input_variable_to_neuron_id(x: NeuronId) -> NeuronId:
        return NeuronId(
            layer=x.layer,
            token=x.token if (not sum_over_tokens and manual_clusters is None) else -1,
            neuron=x.neuron,
            polarity=x.polarity,
        )

    # remerge nodes
    df_node_prepared.loc[:, "cluster"] = df_node_prepared["input_variable"].apply(
        lambda x: input_variable_to_cluster.get(input_variable_to_neuron_id(x), "-1"),
    )

    manual_assignment_count: int | None = None
    total_nodes = len(df_node_prepared)
    if manual_clusters is not None and verbose:
        manual_assignment_count = int((df_node_prepared["cluster"] != "-1").sum())

    if manual_clusters is not None:

        def normalize_manual_cluster(neuron_id: NeuronId, cluster_value: str) -> str:
            if cluster_value != "-1":
                return cluster_value
            layer_val = getattr(neuron_id, "layer", None)
            if layer_val in (-1, effective_last_layer):
                return str(neuron_id)
            return UNCLUSTERED_CLUSTER_ID

        df_node_prepared.loc[:, "cluster"] = df_node_prepared.apply(
            lambda row: normalize_manual_cluster(
                cast(NeuronId, row["input_variable"]), cast(str, row["cluster"])
            ),
            axis=1,
        )
    if verbose and manual_clusters is not None:
        assigned_nodes = manual_assignment_count or 0
        logger.info(
            "remerge_df_node: manual clusters applied to %d/%d nodes across %d clusters",
            assigned_nodes,
            total_nodes,
            len(set(manual_clusters.values())),
        )
        if assigned_nodes < total_nodes:
            logger.info(
                "remerge_df_node: %d nodes were not covered by manual clusters",
                total_nodes - assigned_nodes,
            )
    df_node_prepared.loc[:, "embedding"] = df_node_prepared["input_variable"].apply(
        lambda x: input_variable_to_embedding.get(
            input_variable_to_neuron_id(x),
            np.zeros(1, dtype=np.float32),
        )
    )

    # add source and target clusters to edges
    if len(df_edge_prepared) > 0:
        df_edge_prepared.loc[:, "source_cluster"] = df_edge_prepared["input_variable"].apply(
            lambda x: input_variable_to_cluster.get(input_variable_to_neuron_id(x.source), "-1"),
        )
        df_edge_prepared.loc[:, "target_cluster"] = df_edge_prepared["input_variable"].apply(
            lambda x: input_variable_to_cluster.get(input_variable_to_neuron_id(x.target), "-1"),
        )
        if manual_clusters is not None:
            df_edge_prepared.loc[:, "source_cluster"] = df_edge_prepared.apply(
                lambda row: normalize_manual_cluster(
                    cast(NeuronId, row["input_variable"].source), cast(str, row["source_cluster"])
                ),
                axis=1,
            )
            df_edge_prepared.loc[:, "target_cluster"] = df_edge_prepared.apply(
                lambda row: normalize_manual_cluster(
                    cast(NeuronId, row["input_variable"].target), cast(str, row["target_cluster"])
                ),
                axis=1,
            )

    # return
    if verbose:
        logger.info(
            "remerge_df_node: completed with %d node rows, %d edge rows",
            len(df_node_prepared),
            len(df_edge_prepared),
        )
    return df_node_prepared, df_edge_prepared


def group_by_cluster(
    df_node: pd.DataFrame,
    df_edge: pd.DataFrame,
    group_unclustered: bool = False,
    last_layer: int | None = None,
):
    """
    Group the nodes and edges by cluster and compute all the summary statistics for
    the clusters.

    Parameters
    ----------
    df_node: pd.DataFrame
        The dataframe of nodes. Already embedded and clustered.
    df_edge: pd.DataFrame
        The dataframe of edges.
    group_unclustered: bool
        Whether to collapse leftover ``-1`` cluster ids (except for input/logit layers)
        into a shared "unclustered" bucket.
    last_layer: int | None
        Explicit final layer index; if ``None`` it is derived from ``df_node``.

    Returns
    -------
    df_node_clustered: pd.DataFrame
        The dataframe of nodes with the clusters grouped.
    df_edge_clustered: pd.DataFrame
        The dataframe of edges with the clusters grouped.
    """
    # define some stuff
    metric_cols = ["input_variable", "cluster", "embedding", "description"]
    label_cols = [col for col in df_node.columns if col not in metric_cols]

    # grab unclustered and clustered nodes
    # only the latter will be grouped
    df_node_clustered = df_node.copy()
    df_edge_clustered = df_edge.copy()
    effective_last_layer = (
        last_layer
        if last_layer is not None
        else (cast(int, df_node["layer"].max()) if len(df_node) else -1)
    )
    unclustered_cluster_id = UNCLUSTERED_CLUSTER_ID

    # replace "-1" with some input_variable string
    def normalize_node_cluster(row: pd.Series) -> str:
        if row.cluster != "-1":
            return cast(str, row.cluster)
        if not group_unclustered:
            return str(row.input_variable)
        layer_val = getattr(row.input_variable, "layer", row.get("layer", None))
        try:
            layer_int = int(layer_val) if layer_val is not None else None
        except (TypeError, ValueError):
            layer_int = None
        if layer_int in (-1, effective_last_layer):
            return str(row.input_variable)
        return unclustered_cluster_id

    df_node_clustered.loc[:, "cluster"] = df_node_clustered.apply(normalize_node_cluster, axis=1)

    if len(df_edge_clustered) > 0:

        def normalize_edge_cluster(cluster_value: str, neuron_id) -> str:
            if cluster_value != "-1":
                return cast(str, cluster_value)
            if not group_unclustered:
                return str(neuron_id)
            layer_val = getattr(neuron_id, "layer", None)
            try:
                layer_int = int(layer_val) if layer_val is not None else None
            except (TypeError, ValueError):
                layer_int = None
            if layer_int in (-1, effective_last_layer):
                return str(neuron_id)
            return unclustered_cluster_id

        df_edge_clustered.loc[:, "source_cluster"] = df_edge_clustered.apply(
            lambda x: normalize_edge_cluster(x.source_cluster, x.input_variable.source),
            axis=1,
        )
        df_edge_clustered.loc[:, "target_cluster"] = df_edge_clustered.apply(
            lambda x: normalize_edge_cluster(x.target_cluster, x.input_variable.target),
            axis=1,
        )

    # group by cluster
    df_node_clustered = cast(
        pd.DataFrame,
        df_node_clustered.groupby("cluster")
        .agg(
            {
                "input_variable": "first",
                "description": "first",
                "embedding": "mean",
                **{col: "sum" for col in label_cols},
            }
        )
        .reset_index(),
    )
    if len(df_edge_clustered) > 0:
        df_edge_clustered = cast(
            pd.DataFrame,
            df_edge_clustered.groupby(["source_cluster", "target_cluster"])
            .agg(
                {
                    "input_variable": "first",
                    **{col: "sum" for col in label_cols},
                }
            )
            .reset_index(),
        )

        # subtract self-edges from the cluster attribution scores in df_node_clustered
        for _, self_edge in df_edge_clustered[
            df_edge_clustered.source_cluster == df_edge_clustered.target_cluster
        ].iterrows():
            for col in label_cols:
                df_node_clustered.loc[
                    df_node_clustered.cluster == self_edge.source_cluster, col
                ] -= self_edge[col]

        # remove all self-edges from df_edge_clustered
        df_edge_clustered = df_edge_clustered[
            df_edge_clustered.source_cluster != df_edge_clustered.target_cluster
        ]

    # if clusters have commas, make them "-1" (unless we're preserving manual grouping)
    if not group_unclustered:

        def decommafy(x: str) -> str:
            return "-1" if "," in x else x

        df_node_clustered.loc[:, "cluster"] = df_node_clustered["cluster"].apply(decommafy)
        if len(df_edge_clustered) > 0:
            df_edge_clustered.loc[:, "source_cluster"] = df_edge_clustered["source_cluster"].apply(
                decommafy
            )
            df_edge_clustered.loc[:, "target_cluster"] = df_edge_clustered["target_cluster"].apply(
                decommafy
            )

    # rename metric cols
    df_node_clustered = df_node_clustered.rename(
        columns={col: "Clustered: " + col for col in label_cols},
    )
    if len(df_edge_clustered) > 0:
        df_edge_clustered = df_edge_clustered.rename(
            columns={col: "Clustered: " + col for col in label_cols},
        )

    return df_node_clustered, df_edge_clustered


def prepare_visualization_data(
    df_node: pd.DataFrame,
    df_edge: pd.DataFrame,
    df_node_clustered: pd.DataFrame | None = None,
    df_edge_clustered: pd.DataFrame | None = None,
    include_attr_contrib: bool = False,
    unclustered_label: str | None = None,
    last_layer: int | None = None,
) -> dict:
    # compute average over metrics
    index_cols = [
        "input_variable",
        "layer",
        "token",
        "neuron",
        "description",
        "embedding",
        "cluster",
        "source_cluster",
        "target_cluster",
        "clustered",
        "Average",
    ]
    # all attribution values should have been normalized already!
    # compute averages
    df_node.loc[:, "Average"] = df_node.apply(
        lambda row: np.mean([float(v) for k, v in row.items() if k not in index_cols]),
        axis=1,
    )
    df_edge.loc[:, "Average"] = df_edge.apply(
        lambda row: np.mean([float(v) for k, v in row.items() if k not in index_cols]),
        axis=1,
    )
    index_cols.remove("Average")

    # just append clustered dfs to df_node and df_edge
    if df_node_clustered is not None:
        df_node = pd.concat(
            [df_node.assign(clustered=False), df_node_clustered.assign(clustered=True)]
        )
        df_node = df_node.fillna(0)
    if df_edge_clustered is not None:
        df_edge = pd.concat(
            [df_edge.assign(clustered=False), df_edge_clustered.assign(clustered=True)]
        )
        df_edge = df_edge.fillna(0)

    # get metric columns in consistent order
    metric_cols = [c for c in df_node.columns if c not in index_cols]
    unclustered_cols = [
        c for c in df_node.columns if c not in index_cols and not c.startswith("Clustered:")
    ]
    effective_last_layer = (
        last_layer
        if last_layer is not None
        else (cast(int, df_node["layer"].max()) if len(df_node) else -1)
    )

    # format nodes - use lists instead of dicts for lighter data
    # Node format: [id, layer, token, neuron, desc, metrics_list, importance_list, cluster]
    node_to_cluster = {}
    cluster_to_nodes = {}
    node_has_metric = {}
    nodes = []
    for _, node in df_node.iterrows():
        vals = [float(v) for k, v in node.items() if k in unclustered_cols]
        mean_val = np.mean(vals)
        node_cluster = cast(str, node.cluster)

        # Extract metrics and importance as ordered lists
        metrics_list = [float(node[k]) for k in metric_cols]
        importance_list = [float(node[k]) - mean_val for k in metric_cols]
        embedding = (
            (node.embedding.tolist() if isinstance(node.embedding, np.ndarray) else [0.0])
            if include_attr_contrib
            else [0.0]
        )

        # rename
        layer, token, neuron = (
            cast(int, node.input_variable.layer),
            cast(int, node.input_variable.token),
            cast(int, node.input_variable.neuron),
        )
        input_variable = f"{layer},{token},{neuron}"

        node_data = [
            input_variable,  # 0: id
            layer,  # 1: layer
            token,  # 2: token
            neuron,  # 3: neuron
            node.description,  # 4: desc
            metrics_list,  # 5: metrics (list in metric_cols order)
            importance_list,  # 6: importance (list in metric_cols order)
            node_cluster,  # 7: cluster
            embedding,  # 8: embedding
        ]

        node_has_metric[input_variable] = metric_cols
        nodes.append(node_data)
        node_to_cluster[input_variable] = node_cluster
        if node_cluster not in cluster_to_nodes:
            cluster_to_nodes[node_cluster] = []
        cluster_to_nodes[node_cluster].append(node_data)

    # format edges - use lists instead of dicts
    # Edge format: [source, target, source_layer, source_token, source_neuron,
    #               target_layer, target_token, target_neuron, metrics_list, cluster]
    links = []
    for _, edge in df_edge.iterrows():
        source_layer, target_layer = cast(int, edge.input_variable.source.layer), cast(
            int, edge.input_variable.target.layer
        )
        source_neuron, target_neuron = cast(int, edge.input_variable.source.neuron), cast(
            int, edge.input_variable.target.neuron
        )
        source_token, target_token = cast(int, edge.input_variable.source.token), cast(
            int, edge.input_variable.target.token
        )
        source_id = f"{source_layer},{source_token},{source_neuron}"
        if source_id not in node_to_cluster:
            print(f"Source {source_id} not in node_to_cluster")
            continue
        target_id = f"{target_layer},{target_token},{target_neuron}"

        # Extract metrics as ordered list
        edge_metrics_list = [float(edge[k]) if k in edge else 0.0 for k in metric_cols]

        links.append(
            [
                source_id,  # 0: source
                target_id,  # 1: target
                source_layer,  # 2: source_layer
                source_token,  # 3: source_token
                source_neuron,  # 4: source_neuron
                target_layer,  # 5: target_layer
                target_token,  # 6: target_token
                target_neuron,  # 7: target_neuron
                edge_metrics_list,  # 8: metrics (list in metric_cols order)
                node_to_cluster[source_id],  # 9: cluster
            ]
        )

    # compute cluster labels, preserving descriptive ids and naming stragglers
    cluster_label_lookup: dict[str, str] = {}
    numeric_cluster_ids: set[int] = set()
    non_numeric_present = False

    for cluster_value, nodes_in_cluster in cluster_to_nodes.items():
        cluster_str = str(cluster_value)
        if cluster_str in cluster_label_lookup:
            continue

        sample_node = nodes_in_cluster[0] if nodes_in_cluster else None
        sample_layer = int(sample_node[1]) if sample_node is not None else None
        sample_desc = sample_node[4] if sample_node is not None else ""
        is_boundary_layer = (
            sample_layer in (-1, effective_last_layer) if sample_layer is not None else False
        )

        if sample_node is not None and is_boundary_layer:
            label = sample_desc if sample_desc else cluster_str
            cluster_label_lookup[cluster_str] = str(label)
            non_numeric_present = True
            continue

        if unclustered_label is not None and cluster_str in ("-1", UNCLUSTERED_CLUSTER_ID):
            cluster_label_lookup[cluster_str] = unclustered_label
            non_numeric_present = True
            continue

        if cluster_str.lstrip("-").isdigit():
            cluster_id = int(cluster_str)
            numeric_cluster_ids.add(cluster_id)
            cluster_label_lookup[cluster_str] = f"Cluster {cluster_id}"
        else:
            non_numeric_present = True
            cluster_label_lookup[cluster_str] = cluster_str

    if non_numeric_present:
        cluster_labels = cluster_label_lookup
    else:
        max_cluster_id = max(numeric_cluster_ids) if numeric_cluster_ids else -1
        cluster_labels = [
            cluster_label_lookup.get(str(idx), f"Cluster {idx}")
            for idx in range(max_cluster_id + 1)
        ]

    # remaining stuff
    cluster_largest_nodes = {}
    for node_id, cluster in node_to_cluster.items():
        if cluster not in cluster_largest_nodes:
            cluster_largest_nodes[cluster] = node_id

    # return data
    result = {
        "nodes": nodes,
        "links": links,
        "metric_keys": metric_cols,
        "cluster_labels": cluster_labels,
        "cluster_largest_nodes": cluster_largest_nodes,
    }
    return result


def build_circuit_visualization(
    df_node: pd.DataFrame,
    df_edge: pd.DataFrame,
    tokenizer,
    # description
    neuron_label_cache: dict = {},
    get_desc: bool = True,
    # prepare
    sum_over_tokens: bool = False,
    # clustering
    manual_clusters: Mapping[NeuronId, str] | None = None,
    # visualization
    include_attr_contrib: bool = False,
    # more settings
    verbose: bool = False,
    last_layer: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[Any, Any]]:
    """
    Build the circuit visualization data.

    Parameters
    ----------
    df_node: pd.DataFrame
        The dataframe of nodes.
    df_edge: pd.DataFrame
        The dataframe of edges.
    tokenizer: Tokenizer
        The tokenizer to use for descriptions.
    neuron_label_cache: dict
        The cache of neuron labels.
    get_desc: bool
        Whether to get descriptions.
    sum_over_tokens: bool
        Whether to sum over tokens.
    manual_clusters: Mapping[NeuronId, str] | None
        The manual clusters to use. These clusters are assumed to not have the token axis.
    include_attr_contrib: bool
        Whether to include attribute contributions.
    verbose: bool
        Whether to emit progress logging.

    Returns
    -------
    data: dict
        The visualization data.
    df_node: pd.DataFrame
        The dataframe of nodes.
    df_edge: pd.DataFrame
        The dataframe of edges.
    neuron_label_cache: dict
        The cache of neuron labels.
    """
    # prepare input data
    effective_last_layer = last_layer if last_layer is not None else cast(int, df_node.layer.max())
    if verbose:
        logger.info("Preparing circuit data...")
    df_node_summed, df_edge_summed = prepare_circuit_data(
        df_node.copy(),
        df_edge.copy(),
        sum_over_tokens=sum_over_tokens,
        verbose=verbose,
    )

    # embed neurons
    assert manual_clusters is not None
    if manual_clusters is None:
        pass
    else:
        if verbose:
            unique_manual_clusters = set(manual_clusters.values())
            logger.info(
                "Using manual clusters: %d neurons mapped across %d clusters",
                len(manual_clusters),
                len(unique_manual_clusters),
            )
        df_node, df_edge = remerge_df_node(
            df_node,
            df_edge,
            df_node_clustered=None,
            manual_clusters=manual_clusters,
            sum_over_tokens=False,
            verbose=verbose,
            last_layer=effective_last_layer,
        )

    # get descriptions
    if verbose:
        logger.info("Getting descriptions...")
    df_node, neuron_label_cache = get_descriptions(
        df_node,
        tokenizer,
        last_layer,
        get_desc=get_desc,
        verbose=verbose,
        neuron_label_cache=neuron_label_cache,
    )

    # group by cluster
    if verbose:
        logger.info("Grouping by cluster...")
    df_node_clustered, df_edge_clustered = group_by_cluster(
        df_node,
        df_edge,
        group_unclustered=manual_clusters is not None,
        last_layer=effective_last_layer,
    )

    # format for visualization
    if verbose:
        logger.info("Formatting for visualization...")
    unclustered_label = "Unclustered" if manual_clusters is not None else None
    data = prepare_visualization_data(
        df_node,
        df_edge,
        df_node_clustered,
        df_edge_clustered,
        include_attr_contrib=include_attr_contrib,
        unclustered_label=unclustered_label,
        last_layer=effective_last_layer,
    )

    # return
    if verbose:
        logger.info("Pipeline complete.")
    return data, df_node, df_edge, neuron_label_cache


def plot_neurons_umap(
    df_node: pd.DataFrame,
):
    X = np.stack(df_node.embedding.values)
    reducer = umap.UMAP(random_state=42, n_components=2)
    embedding = reducer.fit_transform(X)

    # Build DataFrame for plotting
    plot_df = pd.DataFrame(
        {
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
            "id": df_node["input_variable"].astype(str),
            "is_token": df_node["input_variable"].str.startswith("-"),
            "description": df_node["description"].astype(str),
            "cluster": df_node["cluster"].astype(str),
            "Average": df_node["Average"].astype(float).abs(),
        }
    )

    # Wrap description text for hover readability
    def wrap_text(s, width=50):
        return "<br>".join([s[i : i + width] for i in range(0, len(s), width)])

    plot_df["description_wrapped"] = plot_df["description"].apply(lambda x: wrap_text(x, 50))
    plot_df["Average"] = (plot_df["Average"] / plot_df["Average"].max()) * 10
    plot_df["Average"] = plot_df["Average"].apply(lambda x: np.sqrt(x))

    # Interactive scatter plot
    fig = px.scatter(
        plot_df,
        x="umap1",
        y="umap2",
        color="cluster",
        size="Average",
        symbol="is_token",
        hover_data={
            "id": True,
            "description_wrapped": True,
        },
        title="UMAP Projection of Neurons",
        height=500,
        width=800,
    )

    fig.show()


def export_neuron_graph(data: dict, out_html: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_path = os.path.join(current_dir, "visualization.html")
    with open(visualization_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__GRAPH_JSON__", json.dumps(data))
    html = html.replace("__DEFAULT_METRIC__", "Average")

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out_html} with {len(data['nodes'])} nodes and {len(data['links'])} edges.")
    if not df_node_clustered["cluster"].map(np.isscalar).all():
        raise ValueError("group_by_cluster expected scalar cluster ids; found non-scalar entries.")
