import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotnine as p9
from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.cluster import NeuronId
from multilingual import labels, word_to_axis, word_to_lang
from tqdm import tqdm
from util.subject import Subject, llama31_8B_instruct_config

from env_util import ENV
if ENV.ARTIFACTS_DIR is None:
    raise RuntimeError("ARTIFACTS_DIR must be set in the .env file")

ARTIFACTS_DIR = Path(ENV.ARTIFACTS_DIR) / "multilingual_circuit_hypotheses"

RESULTS_DIR = ARTIFACTS_DIR / "case_studies" / "multilingual"
CIRCUIT_PICKLE = ARTIFACTS_DIR / "case_studies" / "multilingual_circuit.pkl"

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

hypotheses = {
    "language": [label.split("___")[1] for label in labels],
    "concept": [label.split("___")[0] for label in labels],
    "attribute": [word_to_axis[label.split("___")[0]] for label in labels],
}


def export_full_node_subset(
    circuit: Circuit,
    output_name: str = "multilingual_full_nodes.json",
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
    circuit.cluster_with_hypotheses(
        hypotheses,
        above_threshold=0.90,
        below_threshold=0.10,
        unique_only=False,
        subset_labels=sampled_labels,
        cluster_kwargs={"include_attr_contrib": False, "verbose": True},
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
    all_scores_df["roc_auc_score"] = (all_scores_df["roc_auc_score"] - 0.5).abs() + 0.5
    all_scores_df["feature_type"] = all_scores_df["feature_type"].str.capitalize()
    plot = (
        p9.ggplot(all_scores_df, p9.aes(x="roc_auc_score", fill="feature_type"))
        + p9.geom_histogram(binwidth=0.02)
        + p9.facet_wrap("~feature_type", nrow=1)
        + p9.labs(x="ROC AUC Score (abs diff. from 0.5)", y="Count")
        + p9.scale_y_log10()
        + p9.geom_vline(xintercept=0.90, color="red", linetype="dashed")
        + p9.theme(
            figure_size=(6, 2.5),
            legend_position="none",
        )
    )
    path = ARTIFACTS_DIR / "case_studies" / "multilingual" / "roc_auc_score_distribution.png"
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
    path = ARTIFACTS_DIR / "case_studies" / "multilingual" / "features_by_layer.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plot.save(path, dpi=300)
    print(f"Saved features by layer plot to {path}")


def steering_analysis(
    circuit: Circuit,
):
    # create directory for results
    directory = ARTIFACTS_DIR / "case_studies" / "multilingual"
    directory.mkdir(parents=True, exist_ok=True)

    if False:
        pass
    else:
        # get the top neuron
        circuit.subject.generate(circuit.cis[0], max_new_tokens=10, verbose=True)

        # sum over tokens and get descriptions
        circuit.df_edge = pd.DataFrame(columns=circuit.df_edge.columns)
        circuit.cluster(
            n_clusters=0,
            include_attr_contrib=False,
            do_one_cluster_per_neuron=True,
            get_desc=False,
            verbose=True,
            sum_over_tokens=True,
        )

        # collect all results
        dfs = []
        for top_neuron_tuple in [(21, -1, 4920)]:
            circuit.clear_steering_results()
            # for multiplier in tqdm([0, 1], desc="Steering"):
            for multiplier in tqdm([-8, -4, -2, -1, 0, 1, 2, 4, 8], desc="Steering"):
                circuit.steer(
                    custom_neuron_ids=[top_neuron_tuple],
                    multiplier=multiplier,
                    verbose=True,
                    store_results=True,
                )

            # analyse
            cluster_to_output = circuit.cluster_to_output
            dfs.append(
                cluster_to_output.assign(neuron=f"L{top_neuron_tuple[0]}/N{top_neuron_tuple[2]}")
            )

        cluster_to_output = pd.concat(dfs)
        all_labels = cluster_to_output.label.unique().tolist()
        for label in labels:
            word, lang = label.split("___")
            translated_word = word_to_lang[word][lang]
            axis = word_to_axis[word]
            opposite_word_same_axis = [
                w for w, a in word_to_axis.items() if a == axis and w != word
            ][0]
            translated_opposite_word = word_to_lang[opposite_word_same_axis][lang]
            matches = [x for x in all_labels if label in x]
            if len(matches) == 0:
                continue
            true_label = matches[0]
            mask = cluster_to_output.label == true_label
            answer_token = circuit.tokenizer.encode(" " + translated_word)[1]
            answer_token_capital = circuit.tokenizer.encode(" " + translated_word.capitalize())[1]
            antonym_of_answer_token = circuit.tokenizer.encode(" " + translated_opposite_word)[1]
            antonym_of_answer_token_capital = circuit.tokenizer.encode(
                " " + translated_opposite_word.capitalize()
            )[1]
            answer_english_token = circuit.tokenizer.encode(" " + word)[1]
            antonym_of_answer_english_token = circuit.tokenizer.encode(
                " " + opposite_word_same_axis
            )[1]
            print(
                answer_token,
                antonym_of_answer_token,
                answer_english_token,
                antonym_of_answer_english_token,
            )
            cluster_to_output.loc[mask, "answer_token"] = answer_token
            cluster_to_output.loc[mask, "answer_token_capital"] = answer_token_capital
            cluster_to_output.loc[mask, "antonym_of_answer_token"] = antonym_of_answer_token
            cluster_to_output.loc[mask, "antonym_of_answer_token_capital"] = (
                antonym_of_answer_token_capital
            )
            cluster_to_output.loc[mask, "answer_english_token"] = answer_english_token
            cluster_to_output.loc[mask, "antonym_of_answer_english_token"] = (
                antonym_of_answer_english_token
            )

        # get probs
        for token_col in [
            "answer_token",
            "answer_token_capital",
            "antonym_of_answer_token",
            "antonym_of_answer_token_capital",
            "answer_english_token",
            "antonym_of_answer_english_token",
        ]:
            cluster_to_output[token_col + "_prob"] = cluster_to_output.apply(
                lambda x: (
                    x["probs"][int(x[token_col])].item() if not pd.isna(x[token_col]) else None
                ),
                axis=1,
            )

        # pivot into long form for plotting
        prob_cols = [
            token_col + "_prob"
            for token_col in [
                "answer_token",
                "answer_token_capital",
                "antonym_of_answer_token",
                "antonym_of_answer_token_capital",
                "answer_english_token",
                "antonym_of_answer_english_token",
            ]
        ]
        cluster_to_output_long = cluster_to_output.melt(
            id_vars=["multiplier", "label", "neuron"],
            value_vars=prob_cols,
            var_name="prob_type",
            value_name="probability",
        )
        prob_label_map = {
            "answer_token_prob": "p(Synonym)",
            "answer_token_capital_prob": "p(Capital(Synonym))",
            "antonym_of_answer_token_prob": "p(Antonym)",
            "antonym_of_answer_token_capital_prob": "p(Capital(Antonym))",
            "answer_english_token_prob": "p(English(Synonym))",
            "antonym_of_answer_english_token_prob": "p(English(Antonym))",
        }
        cluster_to_output_long["prob_type"] = cluster_to_output_long["prob_type"].map(
            prob_label_map
        )

        # save dataframe to csv
        cluster_to_output_long.to_csv(
            ARTIFACTS_DIR / "case_studies" / "multilingual" / "multilingual_label_probs_vs_multiplier.csv", index=False
        )

    # plot against multiplier for each probability type
    plot = (
        p9.ggplot(
            cluster_to_output_long,
            p9.aes(x="multiplier", y="probability"),
        )
        + p9.geom_line(p9.aes(group="label"), alpha=0.2)
        + p9.stat_summary(p9.aes(color="prob_type"), geom="line", alpha=1.0, size=2)
        + p9.scale_color_brewer(type="qual", palette="Set1")
        + p9.facet_grid("neuron~prob_type")
        + p9.theme(figure_size=(6, 3), legend_position="none")
        + p9.labs(x="Steering multiplier", y="Probability")
    )
    plot.save(ARTIFACTS_DIR / "case_studies" / "multilingual" / "multilingual_label_probs_vs_multiplier.png", dpi=300)


def main():
    circuit = Circuit.load_from_pickle(CIRCUIT_PICKLE)
    circuit.set_subject(Subject(llama31_8B_instruct_config))
    export_full_node_subset(circuit)
    plot_features_by_layer(circuit)
    # steering_analysis(circuit)
    circuit.export_hypothesis_score_jsons(hypotheses, ARTIFACTS_DIR, 0.9, 0.1)


if __name__ == "__main__":
    main()
