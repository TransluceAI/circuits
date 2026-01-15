import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotnine as p9
from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.cluster import NeuronId
from user_modelling import _prepare_split_examples
from util.subject import Subject, llama31_8B_instruct_config

from env_util import ENV
if ENV.ARTIFACTS_DIR is None:
    raise RuntimeError("ARTIFACTS_DIR must be set in the .env file")

ARTIFACTS_DIR = Path(ENV.ARTIFACTS_DIR) / "user_modelling_circuit_hypotheses"
RESULTS_DIR = Path(ENV.ARTIFACTS_DIR) / "results" / "case_studies" / "user_modelling"
CIRCUIT_PICKLE = RESULTS_DIR / "wikipedia_gender_circuit.pkl"
JAPAN_CIRCUIT_PICKLE = RESULTS_DIR / "japan_circuit.pkl"

p9.theme_set(
    p9.theme_bw(base_size=10)
    + p9.theme(
        text=p9.element_text(color="#000"),
        # figure_size=(2.5, 2.5),
        axis_title=p9.element_text(size=10),
        axis_text=p9.element_text(size=8),
        axis_text_x=p9.element_text(angle=45, hjust=0.5),
        # legend_position="bottom",
        legend_text=p9.element_text(size=8),
        legend_title=p9.element_text(size=9),
        panel_grid_major=p9.element_line(size=1, color="#dddddd"),
        panel_grid_minor=p9.element_blank(),
        # legend_justification_bottom=1,
        strip_background=p9.element_blank(),
        legend_margin=0,
    )
)


def export_full_node_subset(
    circuit: Circuit,
    hypotheses: dict[str, list[str]],
    output_name: str = "gender_full_nodes.json",
    num_labels: int = 10,
    topk_edges: int = 100_000,
) -> None:
    """
    Export all df_node rows as a website JSON artifact without subsampling.
    Also limits df_edge to the top-k entries by absolute attribution to keep files small.
    """
    if circuit.df_node.empty:
        print("No node data available; skipping export.")
        return
    subset_df_node = circuit.df_node.copy()

    # filter out edges
    subset_df_edge = circuit.df_edge.copy()
    if topk_edges is not None and topk_edges > 0 and len(subset_df_edge) > topk_edges:
        threshold = subset_df_edge["attribution"].abs().nlargest(topk_edges).min()
        subset_df_edge = subset_df_edge[
            (subset_df_edge.attribution >= threshold) | (subset_df_edge.attribution <= -threshold)
        ]
    circuit.df_edge = subset_df_edge

    unique_labels = circuit.df_node["label"].dropna().unique()
    if len(unique_labels) == 0:
        print("No labels available; skipping sampled export.")
        return

    sample_size = min(num_labels, len(unique_labels))
    seed = 0
    rng = np.random.default_rng(seed)
    sampled_labels = rng.choice(unique_labels, size=sample_size, replace=False)

    # prepare per-label hypotheses and cluster
    if hypotheses:
        circuit.cluster_with_hypotheses(
            hypotheses,
            above_threshold=0.8,
            below_threshold=0.2,
            in_class_attribution_threshold=0.025,
            unique_only=False,
            subset_labels=sampled_labels,
            cluster_kwargs={"include_attr_contrib": False, "verbose": True},
            verbose=True,
        )
    else:
        circuit.cluster(
            n_clusters=0,
            do_one_cluster_per_neuron=True,
            include_attr_contrib=False,
            verbose=True,
        )

    export_dir = ARTIFACTS_DIR / "full"
    export_dir.mkdir(parents=True, exist_ok=True)
    try:
        output_path = export_dir / output_name
        circuit.export_for_website(str(output_path))
        print(
            f"Exported full subset (website style)"
            f"({subset_df_node.shape[0]} rows) to {output_path}"
        )
    except Exception as exc:
        print(f"Failed to export full subset: {exc}")


def plot_features_by_layer(
    circuit: Circuit,
    hypotheses: dict[str, list[str]],
):
    layer_data = set()
    all_scores = []
    for hypothesis_name, example_labels in hypotheses.items():
        scores = circuit.score_features_multiclass(example_labels, hypothesis_name, verbose=True)
        scores = scores[
            scores["input_variable"].apply(lambda x: x.layer not in [-1, circuit.subject.L])
        ]
        all_scores.append(scores.assign(feature_type=hypothesis_name))

        # add to df
        scores = scores[(scores["roc_auc_score"] >= 0.9) | (scores["roc_auc_score"] <= 0.1)]
        for _, row in scores.iterrows():
            iv = cast(NeuronId, row["input_variable"])
            layer_data.add((iv.layer, iv.neuron, iv.polarity, hypothesis_name))

        # # print top scoring
        # scores = scores.sort_values(by="avg_attribution_in_class", ascending=False)
        # print(scores.head(10))
        # input()

    # plot histogram of roc_auc_score
    all_scores_df = pd.concat(all_scores)
    # all_scores_df["roc_auc_score"] = (all_scores_df["roc_auc_score"] - 0.5).abs() + 0.5
    all_scores_df["feature_type"] = all_scores_df["feature_type"].str.capitalize()
    print(
        all_scores_df[all_scores_df["avg_attribution_in_class"] > 0.05].head(30)[
            ["class_label", "input_variable", "avg_attribution_in_class", "roc_auc_score"]
        ]
    )
    print(all_scores_df.n_positive.value_counts())
    plot = (
        p9.ggplot(
            all_scores_df,
            p9.aes(x="roc_auc_score", y="avg_attribution_in_class", fill="class_label"),
        )
        + p9.geom_point()
        + p9.facet_wrap("~class_label", nrow=1)
        + p9.labs(x="Per-class AUROC", y="In-class Attribution")
        + p9.theme(
            figure_size=(6, 2.5),
            legend_position="none",
        )
    )
    path = RESULTS_DIR / "roc_auc_score_distribution.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plot.save(path, dpi=300)

    # make plot
    layer_data = pd.DataFrame(
        list(layer_data), columns=["layer", "neuron", "polarity", "feature_type"]
    )
    layer_data["feature_type"] = layer_data["feature_type"].str.capitalize()
    plot = (
        p9.ggplot(layer_data, p9.aes(x="layer", fill="feature_type"))
        + p9.geom_histogram(binwidth=1)
        + p9.labs(x="Layer", y="Count", fill="Feature Type", title="Features by Layer")
        + p9.facet_wrap("~feature_type", nrow=1, scales="free_y")
        + p9.theme(figure_size=(6, 2.5), legend_position="none")
    )
    path = RESULTS_DIR / "features_by_layer.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plot.save(path, dpi=300)
    print(f"Saved features by layer plot to {path}")


def main():
    japan_circuit = Circuit.load_from_pickle(JAPAN_CIRCUIT_PICKLE)
    japan_circuit.set_subject(Subject(llama31_8B_instruct_config))
    export_full_node_subset(
        japan_circuit, {}, output_name="japan_full_nodes.json", topk_edges=1_000
    )
    exit(0)

    circuit = Circuit.load_from_pickle(CIRCUIT_PICKLE)
    circuit.set_subject(Subject(llama31_8B_instruct_config))
    prompts, seed_responses, labels = _prepare_split_examples(
        Subject(llama31_8B_instruct_config), "gender"
    )
    hypotheses = {
        "gender": labels,
    }
    # plot_features_by_layer(circuit, hypotheses)
    export_full_node_subset(circuit, hypotheses)
    # circuit.export_hypothesis_score_jsons(hypotheses, ARTIFACTS_DIR, 0.8, 0.2, in_class_attribution_threshold=0.025)
    os.system("luce artifact upload aryaman/user_modelling_circuit_hypotheses --force")


if __name__ == "__main__":
    main()
