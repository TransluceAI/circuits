"""
Utilities for exporting the capitals circuit without subsampling nodes.
"""

import os
from pathlib import Path

import pandas as pd
import plotnine as p9
from capitals import city_state_capital
from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.cluster import NeuronId
from tqdm import tqdm
from util.subject import Subject, llama31_8B_instruct_config

from env_util import ENV
if ENV.ARTIFACTS_DIR is None:
    raise RuntimeError("ARTIFACTS_DIR must be set in the .env file")

ARTIFACTS_DIR = Path(ENV.ARTIFACTS_DIR) / "capitals_circuit_hypotheses"
TEXAS_PICKLE = ARTIFACTS_DIR / "case_studies" / "texas_circuit.pkl"
CIRCUIT_PICKLE = ARTIFACTS_DIR / "case_studies" / "capitals_circuit.pkl"


"""
- **Capital**: [`L3/N14335-`](https://neurons.transluce.org/3/14335/-) (English-specific), [`L4/N13489-`](https://neurons.transluce.org/4/13489/-) (multilingual), [`L19/N2520-`](https://neurons.transluce.org/19/2520/-) (Washington, D.C.)
- **State**: [`L0/N9296-`](https://neurons.transluce.org/0/9296/-) (English-specific), [`L2/N5246+`](https://neurons.transluce.org/2/5246/+) (multilingual), [`L4/N604-`](https://neurons.transluce.org/4/604/-) (broader semantics), [`L19/N4478+`](https://neurons.transluce.org/19/4478/+) (statehood)
- **Dallas / cities in Texas**: [`L0/N12136-`](https://neurons.transluce.org/0/12136/-) (primarily Houston), [`L5/N8659+`](https://neurons.transluce.org/5/8659/+) (various Texas locations)
- **Texas**: [`L6/N10965+`](https://neurons.transluce.org/6/10965/+)
- **Say a capital**: [`L23/N8079-`](https://neurons.transluce.org/23/8079/-), [`L21/N4924-`](https://neurons.transluce.org/21/4924/-), [`L23/N2709-`](https://neurons.transluce.org/23/2709/-) (all specifically include "capital" in their description)
- **Say Austin**: [`L30/N8371+`](https://neurons.transluce.org/30/8371/+) (words ending in "un")
"""


manual_clusters = {
    "capital": [
        (3, 14335, "-"),
        (4, 13489, "-"),
        (19, 2520, "-"),
        (20, 3520, "+"),
        (16, 13326, "-"),
        (13, 4038, "+"),
    ],
    "state": [
        (0, 9296, "-"),
        (2, 5246, "+"),
        (4, 604, "-"),
        (19, 4478, "+"),
        (21, 5790, "-"),
        (21, 12118, "-"),
    ],
    "Dallas": [(0, 12136, "-"), (5, 8659, "+")],
    "Texas": [(6, 10965, "-"), (21, 3093, "+")],
    "say a capital": [
        (23, 8079, "-"),
        (21, 4924, "-"),
        (23, 2709, "-"),
        (17, 3663, "+"),
    ],
    "say Austin": [(30, 8371, "+"), (31, 4876, "+"), (31, 6705, "+")],
    "location": [
        (5, 7404, "-"),
        (23, 9355, "+"),
        (19, 4982, "-"),
        (17, 1276, "+"),
        (17, 9922, "-"),
    ],
    "say a location": [
        (27, 4319, "+"),
        (29, 1846, "+"),
        (29, 8838, "-"),
        (23, 2825, "+"),
        (22, 10506, "-"),
        (28, 2580, "-"),
        (28, 8928, "+"),
        (30, 13458, "+"),
        (30, 1644, "-"),
        (30, 3283, "+"),
        (29, 5785, "-"),
        (25, 13461, "+"),
        (24, 8483, "-"),
    ],
}


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


def steering_analysis(
    circuit: Circuit,
    texas_mode: bool = False,
):
    # create directory for results
    directory = ARTIFACTS_DIR / "case_studies" / "capitals"
    directory.mkdir(parents=True, exist_ok=True)

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
    if texas_mode:
        for name, neurons in manual_clusters.items():
            circuit.clear_steering_results()
            for multiplier in tqdm(
                [-4, -2, -1, 0, 1, 2, 4],
                desc="Steering",
            ):
                circuit.steer(
                    custom_neuron_ids=[(layer, -1, neuron) for layer, neuron, _ in neurons],
                    multiplier=multiplier,
                    verbose=False,
                    store_results=True,
                )

            # analyse
            cluster_to_output = circuit.cluster_to_output
            dfs.append(
                # cluster_to_output.assign(neuron=f"L{top_neuron_tuple[0]}/N{top_neuron_tuple[2]}")
                cluster_to_output.assign(neuron=name)
            )

            # print
            print(name)
            for idx, row in cluster_to_output.iterrows():
                print(f"{row['multiplier']:>3}")
                for token, prob in zip(row["top_tokens"][:3], row["top_tokens_probs"][:3]):
                    print(f"    {token:>15}: {prob:>7.2%}")
                # for token, logit in zip(
                #     row["top_logit_diffs"][:3], row["top_logit_diffs_logits"][:3]
                # ):
                #     print(f"    {token:>15}: {logit:>7.2f}")
            print("=" * 100)

    else:
        for top_neuron_tuple in [
            (23, -1, 8079),
        ]:
            circuit.clear_steering_results()
            # for multiplier in tqdm([0, 1], desc="Steering"):
            for multiplier in tqdm(
                [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                desc="Steering",
            ):
                circuit.steer(
                    custom_neuron_ids=[top_neuron_tuple],
                    multiplier=multiplier,
                    verbose=False,
                    store_results=True,
                )

            # analyse
            cluster_to_output = circuit.cluster_to_output
            dfs.append(
                cluster_to_output.assign(neuron=f"L{top_neuron_tuple[0]}/N{top_neuron_tuple[2]}")
            )

        cluster_to_output = pd.concat(dfs)
        all_labels = cluster_to_output.label.unique().tolist()
        for city, state, capital in city_state_capital:
            true_label = [x for x in all_labels if capital in x]
            if len(true_label) == 0:
                continue
            true_label = true_label[0]
            mask = cluster_to_output.label == true_label
            cluster_to_output.loc[mask, "state_token"] = circuit.tokenizer.encode(" " + state)[1]
            cluster_to_output.loc[mask, "city_token"] = circuit.tokenizer.encode(" " + city)[1]
            cluster_to_output.loc[mask, "capital_token"] = circuit.tokenizer.encode(" " + capital)[
                1
            ]

        # get probs
        cluster_to_output["state_prob"] = cluster_to_output.apply(
            lambda x: x["probs"][int(x["state_token"])].item(), axis=1
        )
        cluster_to_output["city_prob"] = cluster_to_output.apply(
            lambda x: x["probs"][int(x["city_token"])].item(), axis=1
        )
        cluster_to_output["capital_prob"] = cluster_to_output.apply(
            lambda x: x["probs"][int(x["capital_token"])].item(), axis=1
        )

        # pivot into long form for plotting
        prob_cols = ["state_prob", "city_prob", "capital_prob"]
        cluster_to_output_long = cluster_to_output.melt(
            id_vars=["multiplier", "label", "neuron"],
            value_vars=prob_cols,
            var_name="prob_type",
            value_name="probability",
        )
        prob_label_map = {
            "state_prob": "p(State)",
            "city_prob": "p(City)",
            "capital_prob": "p(Capital)",
        }
        cluster_to_output_long["prob_type"] = cluster_to_output_long["prob_type"].map(
            prob_label_map
        )

        # save dataframe to csv
        cluster_to_output_long.to_csv(
            directory / "capitals_label_probs_vs_multiplier.csv", index=False
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
            + p9.theme(figure_size=(6, 2.5), legend_position="none")
            + p9.labs(x="Steering multiplier", y="Probability")
        )
        plot.save(directory / "capitals_label_probs_vs_multiplier.png", dpi=300)


def export_full_node_subset(
    circuit: Circuit,
    output_name: str = "capitals_full_nodes.json",
    topk_edges: int | None = 10_000,
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

    # prepare per-label hypotheses and cluster
    hypotheses = {"state": [label for label in circuit.labels]}
    # circuit.cluster(
    #     n_clusters=10,
    #     include_attr_contrib=False,
    #     verbose=True,
    # )
    # for multiplier in [-2, -1, 0, 1, 2]:
    #     circuit.steer(
    #         multiplier=multiplier,
    #         verbose=True,
    #         store_results=True,
    #     )
    # circuit.label_clusters(
    #     use_steering_results=True,
    # )
    manual_clusters_map = {}
    for cluster, nodes in manual_clusters.items():
        for layer, neuron, polarity in nodes:
            manual_clusters_map[NeuronId(layer, -1, neuron, polarity)] = cluster

    circuit.cluster(
        n_clusters=0,
        manual_clusters=manual_clusters_map,
        include_attr_contrib=False,
        verbose=True,
    )
    # circuit.cluster_with_hypotheses(
    #     hypotheses,
    #     above_threshold=1.0,
    #     below_threshold=0.0,
    #     unique_only=True,
    #     cluster_kwargs={"include_attr_contrib": False, "verbose": True},
    #     verbose=True,
    # )

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


def main() -> None:
    if not TEXAS_PICKLE.exists():
        print(f"Missing Texas circuit pickle at {TEXAS_PICKLE}")
        return

    circuit = Circuit.load_from_pickle(str(TEXAS_PICKLE))
    circuit.set_subject(Subject(llama31_8B_instruct_config))
    circuit.subject.generate(circuit.cis[0], max_new_tokens=10, verbose=True)
    # steering_analysis(circuit, texas_mode=True)
    export_full_node_subset(circuit, topk_edges=1_000)

    # if not CIRCUIT_PICKLE.exists():
    #     print(f"Missing capitals circuit pickle at {CIRCUIT_PICKLE}")
    #     return

    # circuit = Circuit.load_from_pickle(str(CIRCUIT_PICKLE))
    # circuit.set_subject(Subject(llama31_8B_instruct_config))
    # circuit.subject.generate(circuit.cis[0], max_new_tokens=10, verbose=True)
    # steering_analysis(circuit, texas_mode=False)


if __name__ == "__main__":
    main()
