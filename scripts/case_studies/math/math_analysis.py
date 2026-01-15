"""
Analyze the math case study.

Currently tests the following labels:
- Sum mod 10
- Sum mod 3
- Sum mod 2
- Sum tens digit
Produces plots for the top features for each class in each label, as well as a JSON file
with the features with at >0.8 or <0.2 AUROC score.

Usage:
python math_analysis.py

Output:
- JSON files in $ARTIFACTS_DIR/math_circuit_hypotheses/
- Plots in $ARTIFACTS_DIR/math_circuit_hypotheses/
"""

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotnine as p9
from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.cluster import NeuronId  # type: ignore
from circuits.utils.descriptions import get_descriptions
from tqdm import tqdm
from util.subject import Subject, llama31_8B_instruct_config
from pathlib import Path

from env_util import ENV
if ENV.ARTIFACTS_DIR is None:
    raise RuntimeError("ARTIFACTS_DIR must be set in the .env file")

ARTIFACTS_DIR = Path(ENV.ARTIFACTS_DIR) / "math_circuit_hypotheses"

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

features = {
    "sum_mod_10": np.add.outer(np.arange(100), np.arange(100)).reshape(-1) % 10,
    "sum_mod_5": np.add.outer(np.arange(100), np.arange(100)).reshape(-1) % 5,
    "sum_mod_2": np.add.outer(np.arange(100), np.arange(100)).reshape(-1) % 2,
    "sum_tens": (np.add.outer(np.arange(100), np.arange(100)).reshape(-1) // 10),
    "sum_tens_mod_10": (np.add.outer(np.arange(100), np.arange(100)).reshape(-1) // 10) % 10,
    "sum_hundreds": (np.add.outer(np.arange(100), np.arange(100)).reshape(-1) // 100),
}

for add_n in range(10):
    # if either x or y mod 10 == add_n, then the value is 1, else 0
    # this is definitely inefficient but who cares
    matrix = np.zeros((100, 100))
    for x in range(100):
        for y in range(100):
            if (x % 10 == add_n) or (y % 10 == add_n):
                matrix[x, y] = 1
    features[f"add_{add_n}"] = matrix.reshape(-1)

prompts = [f"What is {x} + {y}?" for x in range(100) for y in range(100)]
seed_responses = ["Answer: "] * len(prompts)
labels = [f"{x} + {y} = {x + y}" for x in range(100) for y in range(100)]

RESULTS_DIR = ARTIFACTS_DIR / "case_studies" / "math"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _parse_input_variable(value: str) -> NeuronId:
    """
    Convert the "layer, neuron, polarity" string into a NeuronId.
    """
    if isinstance(value, NeuronId):
        return NeuronId(
            layer=int(value.layer),
            token=-1,
            neuron=int(value.neuron),
            polarity=value.polarity,
        )
    value_str = str(value)
    parts = value_str.split(", ")
    if len(parts) != 3:
        raise ValueError(f"Unexpected input_variable format: {value}")
    layer_str, neuron_str, polarity = parts
    return NeuronId(layer=int(layer_str), token=-1, neuron=int(neuron_str), polarity=polarity)


def _attach_descriptions_to_scores(df_scores: pd.DataFrame, circuit: Circuit) -> pd.DataFrame:
    """
    Fetch neuron descriptions for the scored features and attach them to the dataframe.
    """
    df_scores = df_scores.copy()
    if not hasattr(circuit, "neuron_label_cache") or circuit.neuron_label_cache is None:
        circuit.neuron_label_cache = {}
    if "input_variable" not in df_scores.columns:
        df_scores["description"] = ""
        return df_scores

    unique_inputs = pd.Series(df_scores["input_variable"].dropna().unique())
    if unique_inputs.empty:
        df_scores["description"] = ""
        return df_scores

    neuron_ids = unique_inputs.apply(_parse_input_variable)
    request_df = pd.DataFrame(
        {
            "input_variable": neuron_ids.tolist(),
            "input_variable_key": unique_inputs.tolist(),
        }
    )
    last_layer = int(pd.to_numeric(circuit.df_node["layer"], errors="coerce").max())
    described_nodes, circuit.neuron_label_cache = get_descriptions(
        request_df,
        circuit.tokenizer,
        last_layer=last_layer,
        get_desc=True,
        verbose=False,
        neuron_label_cache=circuit.neuron_label_cache,
    )
    desc_map = dict(zip(described_nodes["input_variable_key"], described_nodes["description"]))
    df_scores["description"] = df_scores["input_variable"].map(desc_map).fillna("")
    return df_scores


def export_hypothesis_score_jsons(
    circuit: Circuit,
    hypotheses: dict[str, np.ndarray] | None = None,
    output_dir: Path = ARTIFACTS_DIR,
    auc_threshold_high: float = 0.8,
    auc_threshold_low: float = 0.2,
) -> None:
    """
    Score every hypothesis against the circuit, filter by AUROC thresholds,
    and export the resulting feature metadata to JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hypotheses = hypotheses or features

    for hypothesis_name, example_labels in hypotheses.items():
        print(f"Scoring hypothesis '{hypothesis_name}'...")
        scores = circuit.score_features_multiclass(example_labels, verbose=True)
        if scores.empty:
            output_path = output_dir / f"{hypothesis_name}.json"
            with open(output_path, "w") as f:
                json.dump([], f, indent=2)
            print(f"No scores produced for {hypothesis_name}; wrote empty JSON to {output_path}")
            continue

        scores = _attach_descriptions_to_scores(scores, circuit)
        mask = (scores["roc_auc_score"] >= auc_threshold_high) | (
            scores["roc_auc_score"] <= auc_threshold_low
        )
        filtered = scores[mask].dropna(subset=["input_variable"]).copy()

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
        ]
        export_records = filtered[export_columns].to_dict(orient="records")
        with open(output_path, "w") as f:
            json.dump(export_records, f, indent=2)
        print(f"Wrote {len(export_records)} records to {output_path}")


def export_score_matrix(
    circuit: Circuit,
    neurons_and_descriptions: list[tuple[NeuronId, str]],
    output_name: str = "scores.json",
) -> None:
    circuit.df_edge = pd.DataFrame(columns=circuit.df_edge.columns)
    neurons = [neuron for neuron, _ in neurons_and_descriptions]
    node_mask = circuit.df_node.apply(
        lambda row: NeuronId(
            layer=row["layer"],
            token=-1,
            neuron=row["neuron"],
            polarity="+" if row["activation"] >= 0 else "-",
        )
        in neurons,
        axis=1,
    )
    circuit.df_node = cast(pd.DataFrame, circuit.df_node[node_mask])
    circuit.cluster(
        n_clusters=0,
        include_attr_contrib=False,
        do_one_cluster_per_neuron=True,
        get_desc=False,
        verbose=True,
        sum_over_tokens=True,
    )

    # get scores
    result: list[dict[str, str | int | list[float]]] = []
    for neuron, description in tqdm(neurons_and_descriptions):
        mask = circuit.df_node_embedded.input_variable.apply(
            lambda x: NeuronId(layer=x.layer, token=-1, neuron=x.neuron, polarity=x.polarity)
            == neuron
        )
        subset = circuit.df_node_embedded[mask]
        if len(subset) == 0:
            raise ValueError(f"No subset found for neuron {neuron} with description {description}")
        attribution = [0.0] * 10000
        for column in subset.columns:
            if "___" in column:
                idx = int(column.split("___")[-1])
                attribution[idx] = float(subset[column].sum())
        result.append(
            {
                "layer": neuron.layer,
                "token": neuron.token,
                "neuron": neuron.neuron,
                "polarity": neuron.polarity,
                "description": description,
                "attribution": attribution,
            }
        )

    # export to case studies
    export_path = ARTIFACTS_DIR / output_name
    with open(export_path, "w") as f:
        json.dump(result, f)


def plot_sum_mod_histograms(
    circuit: Circuit,
):
    hypotheses = {
        f"Sum mod {x}": np.add.outer(np.arange(100), np.arange(100)).reshape(-1) % x
        for x in range(2, 11)
    }
    if True:
        all_dfs = []
        for hypothesis_name, example_labels in hypotheses.items():
            scores = circuit.score_features_multiclass(example_labels, verbose=True)
            scores["roc_auc_score"] -= 0.5
            scores["roc_auc_score"] = scores["roc_auc_score"].abs() + 0.5
            scores = scores.groupby("input_variable").agg({"roc_auc_score": "max"}).reset_index()
            all_dfs.append(scores.assign(hypothesis=hypothesis_name))

        all_dfs = pd.concat(all_dfs)
        all_dfs.to_csv(ARTIFACTS_DIR / "sum_mod_histograms.csv", index=False)
    else:
        all_dfs = pd.read_csv(ARTIFACTS_DIR / "sum_mod_histograms.csv")

    plot = (
        p9.ggplot(all_dfs, p9.aes(x="roc_auc_score"))
        + p9.geom_histogram(bins=50)
        + p9.facet_wrap("~hypothesis", nrow=3)
        + p9.labs(x="ROC AUC Score (abs diff. from 0.5)", y="Count")
        + p9.scale_y_log10()
        + p9.theme(
            figure_size=(6, 4),
        )
        + p9.geom_vline(xintercept=0.8, color="red", linetype="dashed")
    )
    path = ARTIFACTS_DIR / "case_studies" / "math" / "sum_mod_histograms.png"
    plot.save(path, dpi=300)


def export_sampled_node_subset(
    circuit: Circuit,
    num_labels: int = 20,
    seed: int = 0,
    output_name: str = "math_sampled_nodes.json",
    cluster: bool = False,
    topk_edges: int | None = 100,
) -> None:
    """
    Sample df_node rows for `num_labels` labels and export via Circuit.save_to_json.
    """

    unique_labels = circuit.df_node["label"].dropna().unique()
    if len(unique_labels) == 0:
        print("No labels available; skipping sampled export.")
        return
    sample_size = min(num_labels, len(unique_labels))
    rng = np.random.default_rng(seed)
    sampled_labels = rng.choice(unique_labels, size=sample_size, replace=False)

    if cluster:
        math_hypotheses = {
            name: values for name, values in features.items() if name in ["sum_mod_10", "sum_mod_2"]
        }
        circuit.cluster_with_hypotheses(
            math_hypotheses,
            above_threshold=0.8,
            below_threshold=0.2,
            subset_labels=sampled_labels,
            cluster_kwargs={
                "include_attr_contrib": False,
                "get_desc": True,
                "verbose": True,
            },
            verbose=True,
        )
    else:
        manual_clusters_map = {}
        circuit.cluster(
            n_clusters=0,
            manual_clusters=manual_clusters_map,
            include_attr_contrib=False,
            verbose=True,
        )

    # filter out edges
    subset_df_edge = circuit.df_edge.copy()
    print(len(subset_df_edge))
    input()
    if topk_edges is not None and topk_edges > 0 and len(subset_df_edge) > topk_edges:
        threshold = subset_df_edge["attribution"].abs().nlargest(topk_edges).min()
        subset_df_edge = subset_df_edge[
            (subset_df_edge.attribution >= threshold) | (subset_df_edge.attribution <= -threshold)
        ]
    print(len(subset_df_edge))
    input()
    circuit.df_edge = subset_df_edge

    sample_dir = ARTIFACTS_DIR / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    output_path = sample_dir / output_name
    circuit.export_for_website(str(output_path))
    print(
        f"Exported sampled subset (website style) with {len(sampled_labels)} labels "
        f"({len(circuit.df_node)} rows) to {output_path}"
    )


def score_matrices(circuit: Circuit, experiment_name: str):
    if experiment_name == "sum_mod_10":
        export_score_matrix(
            circuit,
            neurons_and_descriptions=[
                (NeuronId(28, -1, 9319, "+"), "Sum ≡ 0 (mod 10)"),
                (NeuronId(21, -1, 10677, "-"), "Sum ≡ 1 (mod 10)"),
                (NeuronId(28, -1, 11237, "-"), "Sum ≡ 2 (mod 10)"),
                (NeuronId(28, -1, 3626, "-"), "Sum ≡ 3 (mod 10)"),
                (NeuronId(21, -1, 9891, "+"), "Sum ≡ 4 (mod 10)"),
                (NeuronId(28, -1, 10278, "-"), "Sum ≡ 5 (mod 10)"),
                (NeuronId(21, -1, 7106, "+"), "Sum ≡ 6 (mod 10)"),
                (NeuronId(23, -1, 9990, "-"), "Sum ≡ 7 (mod 10)"),
                (NeuronId(28, -1, 9154, "-"), "Sum ≡ 8 (mod 10)"),
                (NeuronId(22, -1, 6220, "-"), "Sum ≡ 9 (mod 10)"),
            ],
            output_name="scores_sum_mod_10.json",
        )
    elif experiment_name == "sum_mod_n":
        export_score_matrix(
            circuit,
            neurons_and_descriptions=[
                (NeuronId(21, -1, 9178, "-"), "Even/odd"),
                (NeuronId(21, -1, 9178, "+"), "Even/odd"),
                (NeuronId(31, -1, 9428, "-"), "Sum (mod 5)"),
            ],
            output_name="scores_sum_mod_n.json",
        )
    elif experiment_name == "sum_tens":
        export_score_matrix(
            circuit,
            neurons_and_descriptions=[
                (NeuronId(28, -1, 8649, "+"), "Sum // 10 ≡ 0 (mod 10)"),
                (NeuronId(28, -1, 9549, "-"), "Sum // 10 ≡ 1 (mod 10)"),
                (NeuronId(21, -1, 9679, "-"), "Sum // 10 ≡ 2 (mod 10)"),
                (NeuronId(20, -1, 11837, "-"), "Sum // 10 ≡ 3 (mod 10)"),
                (NeuronId(22, -1, 568, "-"), "Sum // 10 ≡ 4 (mod 10)"),
                (NeuronId(26, -1, 5789, "-"), "Sum // 10 ≡ 5 (mod 10)"),
                (NeuronId(21, -1, 13150, "-"), "Sum // 10 ≡ 6 (mod 10)"),
                (NeuronId(19, -1, 6862, "+"), "Sum // 10 ≡ 7 (mod 10)"),
                (NeuronId(22, -1, 7431, "+"), "Sum // 10 ≡ 8 (mod 10)"),
                (NeuronId(31, -1, 14328, "-"), "Sum // 10 ≡ 9 (mod 10)"),
            ],
            output_name="scores_sum_tens.json",
        )


def main():
    # save experiments
    math_circuit_path = ARTIFACTS_DIR / "case_studies" / "math_circuit.pkl"
    circuit = Circuit.load_from_pickle(math_circuit_path)
    circuit.set_subject(Subject(llama31_8B_instruct_config))
    score_matrices(circuit, experiment_name="sum_mod_10")
    # export_hypothesis_score_jsons(circuit)
    # plot_sum_mod_histograms(circuit)

    # 36_plus_59_circuit_path = ARTIFACTS_DIR / "case_studies" / "36_plus_59_circuit.pkl"
    # circuit = Circuit.load_from_pickle(36_plus_59_circuit_path)
    # circuit.set_subject(Subject(llama31_8B_instruct_config))
    # # export_score_data(circuit)
    # export_sampled_node_subset(
    #     circuit, num_labels=1, output_name="36_plus_59_nodes.json"
    # )


if __name__ == "__main__":
    main()
