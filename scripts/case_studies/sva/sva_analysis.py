import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotnine as p9
import torch
from circuits.analysis.cluster import NeuronId
from circuits.utils.descriptions import get_descriptions
from util.subject import Subject, llama31_8B_instruct_config

from env_util import ENV
if ENV.ARTIFACTS_DIR is None:
    raise RuntimeError("ARTIFACTS_DIR must be set in the .env file")

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


_NEURON_LABEL_CACHE: Dict[Tuple[int, int, int], str] = {}


def analyse_subfolder(
    subfolder: str, subject: Subject, top_k: int = 500, label: str = "ig"
) -> None:
    model_path = Path(subfolder) / "train.pt"
    if not model_path.exists():
        print(f"Missing model checkpoint at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location="cpu")
    tensor = torch.stack([sparse_act.act for sparse_act in checkpoint["nodes"].values()])
    assert tensor is not None

    tensor = tensor.detach().float()
    if tensor.ndim != 3:
        print(f"Unexpected attribution tensor shape {tuple(tensor.shape)} in {model_path}")
        return

    abs_flat = tensor.abs().reshape(-1)
    k = min(top_k, abs_flat.numel())
    if k == 0:
        print(f"No attribution values available in {model_path}")
        return

    topk = torch.topk(abs_flat, k)
    indices = topk.indices
    abs_values = topk.values
    layers, tokens, neurons = torch.unravel_index(indices, tensor.shape)

    layer_list = list(map(int, layers.tolist()))
    token_list = list(map(int, tokens.tolist()))
    neuron_list = list(map(int, neurons.tolist()))
    input_variables = [
        NeuronId(layer=layer, token=token, neuron=neuron, polarity="0")
        for layer, token, neuron in zip(layer_list, token_list, neuron_list)
    ]
    abs_list = abs_values.tolist()

    request_df = pd.DataFrame(
        {
            "input_variable": input_variables,
            "activation": [0] * len(layer_list),
        }
    )

    global _NEURON_LABEL_CACHE
    desc_df, _NEURON_LABEL_CACHE = get_descriptions(
        request_df,
        subject.tokenizer,
        last_layer=subject.L,
        get_desc=True,
        verbose=False,
        neuron_label_cache=_NEURON_LABEL_CACHE,
    )
    desc_df = desc_df.drop_duplicates(["input_variable"], keep="first")
    desc_df.loc[:, "layer"] = desc_df["input_variable"].apply(lambda x: x.layer)
    desc_df.loc[:, "token"] = desc_df["input_variable"].apply(lambda x: x.token)
    desc_df.loc[:, "neuron"] = desc_df["input_variable"].apply(lambda x: x.neuron)
    desc_map = desc_df.set_index(["layer", "token", "neuron"])["description"].to_dict()

    metric_keys = ["Attribution"]
    nodes = []
    for layer, token, neuron, abs_val in zip(layer_list, token_list, neuron_list, abs_list):
        value = tensor[layer, token, neuron].item()
        node_id = f"L{layer}_T{token}_N{neuron}"
        description = desc_map.get(
            (layer, token, neuron), f"Layer {layer}, Token {token}, Neuron {neuron}"
        )
        nodes.append(
            [
                node_id,
                int(layer),
                int(token),
                int(neuron),
                description,
                [value],
                [abs_val],
                0,
                [],
            ]
        )

    nodes.sort(key=lambda node: abs(node[5][0]), reverse=True)

    artifacts_dir = Path(ENV.ARTIFACTS_DIR) / "sva_circuit_hypotheses"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    subfolder_path = Path(subfolder.rstrip("/"))
    experiment_name = label
    dataset_name = subfolder_path.parent.name or "dataset"
    json_path = artifacts_dir / f"sva_{dataset_name}_{experiment_name}.json"

    payload = {
        "nodes": nodes,
        "links": [],
        "metric_keys": metric_keys,
        "cluster_labels": {0: "Top attributions"},
        "contrib_dim_to_label": {},
        "metadata": {
            "dataset": dataset_name,
            "experiment": experiment_name,
            "tensor_shape": list(tensor.shape),
            "top_k": k,
        },
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    os.system("luce artifact upload aryaman/sva_circuit_hypotheses --force")


def plot_scores_for_methods(
    subfolders: list[str],
    labels: list[str],
    datasets: list[str] | str | None = None,
    output_name: str = "",
) -> None:
    """Make histograms for the scores for all neurons in each method across datasets."""
    if len(subfolders) != len(labels):
        raise ValueError("`subfolders` and `labels` must have the same length.")

    if datasets is None:
        datasets_list = [Path(subfolder).parent.name for subfolder in subfolders]
    elif isinstance(datasets, str):
        datasets_list = [datasets] * len(subfolders)
    else:
        if len(datasets) != len(subfolders):
            raise ValueError("`datasets` must be None or match the length of `subfolders`.")
        datasets_list = list(datasets)

    method_scores: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for subfolder, label, dataset_name in zip(subfolders, labels, datasets_list):
        train_path = Path(subfolder) / "train.pt"
        if not train_path.exists():
            print(f"Missing model checkpoint at {train_path}")
            continue
        checkpoint = torch.load(train_path, map_location="cpu")
        tensor = torch.stack([sparse_act.act for sparse_act in checkpoint["nodes"].values()])
        if tensor is None:
            print(f"'all_neuron_attributions' not found in {train_path}")
            continue
        tensor = tensor.detach().float()
        if tensor.ndim != 3:
            print(f"Unexpected attribution tensor shape {tuple(tensor.shape)} in {train_path}")
            continue
        flat_scores = tensor.flatten().detach().cpu().numpy()
        layer_indices = (
            torch.arange(tensor.shape[0], dtype=torch.int64)
            .view(-1, 1, 1)
            .expand_as(tensor)
            .flatten()
            .detach()
            .cpu()
            .numpy()
        )
        method_scores.append((label, dataset_name, flat_scores, layer_indices))

    if not method_scores:
        print("No score data available to plot.")
        return

    output_dir = Path(ENV.ARTIFACTS_DIR) / "results" / "sva"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_order: list[str] = []
    for _, dataset_name, _, _ in method_scores:
        if dataset_name not in dataset_order:
            dataset_order.append(dataset_name)
    method_order: list[str] = []
    for label in labels:
        if label not in method_order:
            method_order.append(label)

    if len(dataset_order) == 1:
        figure_suffix = dataset_order[0]
    else:
        figure_suffix = "all_datasets"
    figure_path_scores = output_dir / f"{output_name}method_score_histograms_{figure_suffix}.png"
    figure_path_layers = output_dir / f"{output_name}method_layer_histograms_{figure_suffix}.png"

    frames = []
    for label, dataset_name, scores, layers in method_scores:
        frames.append(
            pd.DataFrame(
                {
                    "score": scores,
                    "method": label,
                    "dataset": dataset_name,
                    "layer": layers,
                }
            )
        )
    plot_df = pd.concat(frames, ignore_index=True)
    plot_df["dataset"] = pd.Categorical(plot_df["dataset"], categories=dataset_order, ordered=True)
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=method_order, ordered=True)
    plot_df["representation"] = plot_df["method"].apply(lambda x: x[4:] if "SAE" in x else x)
    plot_df["layer"] = plot_df["layer"].astype(int)

    score_plot = (
        p9.ggplot(plot_df, p9.aes(x="score", fill="representation"))
        + p9.geom_histogram(bins=40, alpha=0.9)
        + p9.facet_grid("dataset ~ method", scales="free_y")
        + p9.scale_y_log10()
        + p9.labs(
            title="Neuron Attribution Score Distributions",
            x="Score",
            y="Count (log scale)",
        )
        + p9.theme(
            legend_position="none",
        )
        + p9.scale_fill_brewer(type="qual", palette="Set1")
    )

    score_plot.save(filename=str(figure_path_scores), dpi=320, verbose=False)
    print(f"Saved histogram figure to {figure_path_scores}")

    top_count_levels = [
        ("Top 10,000", 10_000),
        ("Top 1,000", 1_000),
        ("Top 100", 100),
    ]

    layer_frames: list[pd.DataFrame] = []
    for dataset_name in dataset_order:
        for method_name in method_order:
            group = plot_df[
                (plot_df["dataset"] == dataset_name) & (plot_df["method"] == method_name)
            ]
            if group.empty:
                continue
            ranked_group = group.assign(_abs_score=group["score"].abs()).sort_values(
                "_abs_score", ascending=False
            )
            for band_label, top_n in top_count_levels:
                subset = (
                    ranked_group.head(min(top_n, len(ranked_group)))
                    .drop(columns="_abs_score")
                    .copy()
                )
                if subset.empty:
                    continue
                subset["top_band"] = band_label
                layer_frames.append(subset)

    if layer_frames:
        layer_plot_df = pd.concat(layer_frames, ignore_index=True)
        layer_plot_df["top_band"] = pd.Categorical(
            layer_plot_df["top_band"],
            categories=[band for band, _ in top_count_levels],
            ordered=True,
        )
        layer_plot_df["layer"] = layer_plot_df["layer"].astype(int)
    else:
        fallback_label = top_count_levels[-1][0]
        layer_plot_df = plot_df.assign(top_band=fallback_label)
        layer_plot_df["top_band"] = pd.Categorical(
            layer_plot_df["top_band"],
            categories=[band for band, _ in top_count_levels],
            ordered=True,
        )
        layer_plot_df["layer"] = layer_plot_df["layer"].astype(int)

    layer_plot = (
        p9.ggplot(layer_plot_df, p9.aes(x="layer", fill="top_band"))
        + p9.geom_histogram(binwidth=1, boundary=-0.5, position="identity")
        + p9.facet_grid("dataset ~ method", scales="free_y")
        + p9.scale_y_log10()
        + p9.labs(
            title="Neuron Attribution Layer Distributions",
            x="Layer",
            y="Count",
            fill="Top-N",
        )
        + p9.theme(
            legend_position="right",
        )
        + p9.scale_fill_brewer(type="seq")
        + p9.scale_y_continuous(trans="log1p", breaks=[10**i for i in range(10)])
    )

    layer_plot.save(filename=str(figure_path_layers), dpi=320, verbose=False)
    print(f"Saved layer histogram figure to {figure_path_layers}")


def plot_subfolder_scatter(
    subfolder_a: str,
    subfolder_b: str,
    max_points: int = 250_000,
    output_path: str | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
) -> None:
    """Scatter plot of attribution tensors from two experiment folders."""
    path_a = Path(subfolder_a) / "train.pt"
    path_b = Path(subfolder_b) / "train.pt"
    if not path_a.exists() or not path_b.exists():
        print(f"Missing checkpoints: {path_a} or {path_b}")
        return

    checkpoint_a = torch.load(path_a, map_location="cpu")
    checkpoint_b = torch.load(path_b, map_location="cpu")

    tensor_a = torch.stack([sparse_act.act for sparse_act in checkpoint_a["nodes"].values()])
    tensor_b = torch.stack([sparse_act.act for sparse_act in checkpoint_b["nodes"].values()])

    tensor_a = tensor_a.detach().float()
    tensor_b = tensor_b.detach().float()
    if tensor_a.shape != tensor_b.shape:
        print(f"Shape mismatch: {tensor_a.shape} vs {tensor_b.shape}")
        return

    values_a = tensor_a.flatten().abs().cpu().numpy()
    values_b = tensor_b.flatten().abs().cpu().numpy()

    correlation = float(np.corrcoef(values_a, values_b)[0, 1])
    slope, intercept = np.polyfit(values_a, values_b, deg=1)

    df = pd.DataFrame({"value_a": values_a, "value_b": values_b})
    name_a = label_a or Path(subfolder_a).name
    name_b = label_b or Path(subfolder_b).name
    title = f"{name_a} vs {name_b} (r={correlation:.3f})"

    scatter_plot = (
        p9.ggplot(df, p9.aes(x="value_a", y="value_b"))
        + p9.geom_point(alpha=0.2, size=0.5)
        + p9.geom_abline(intercept=intercept, slope=slope, color="#d62728", linetype="dashed")
        + p9.labs(
            title=title,
            x=f"{name_a} values",
            y=f"{name_b} values",
        )
    )

    output_dir = Path(output_path) if output_path else Path(ENV.ARTIFACTS_DIR) / "results" / "sva"
    output_dir.mkdir(parents=True, exist_ok=True)
    slug_a = name_a.replace(" ", "_")
    slug_b = name_b.replace(" ", "_")
    fig_path = output_dir / f"scatter_{slug_a}_vs_{slug_b}.png"
    scatter_plot.save(filename=str(fig_path), dpi=320, verbose=False)
    print(f"Saved scatter plot to {fig_path}")


def main() -> None:
    subject = Subject(llama31_8B_instruct_config)
    results_base = str(Path(ENV.ARTIFACTS_DIR) / "results" / "sva")
    for dataset in []:
        plot_subfolder_scatter(
            f"{results_base}/pair/{dataset}/"
            f"Llama-3.1-8B_{dataset}_N300_AGGmean_Mnap_USE_NEURONS_USE_MLPACTS_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100",
            f"{results_base}/nopair/{dataset}/"
            f"Llama-3.1-8B_{dataset}_N300_AGGmean_Mnap_USE_NEURONS_USE_MLPACTS_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100",
            label_a="SVA paired",
            label_b="SVA unpaired",
        )
        analyse_subfolder(
            f"{results_base}/pair/{dataset}/"
            f"Llama-3.1-8B_{dataset}_N300_AGGmean_Mnap_USE_NEURONS_USE_MLPACTS_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100",
            subject=subject,
            label="mlpacts_ig",
            top_k=100,
        )
        analyse_subfolder(
            f"{results_base}/nopair/{dataset}/"
            f"Llama-3.1-8B_{dataset}_N300_AGGmean_Mnap_USE_NEURONS_USE_MLPACTS_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100",
            subject=subject,
            label="mlpacts_ig_nopair",
            top_k=100,
        )

    datasets = ["nounpp", "rc", "simple", "within_rc"]
    method_labels = [
        "Attn.",
        "MLP acts.",
        "MLP outs.",
        "Resid.",
        "SAE MLP outs.",
    ]  # , "SAE Resid.", "SAE Attn."]

    all_subfolders: list[str] = []
    all_labels: list[str] = []
    all_datasets: list[str] = []

    for mode in ["pair", "nopair"]:
        for dataset in datasets:
            prefix = f"{results_base}/{mode}/{dataset}/"
            mlpacts_ig_path = f"{prefix}Llama-3.1-8B_{dataset}_N300_AGGmean_Mnap_USE_NEURONS_USE_MLPACTS_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100"
            mlp_ig_path = f"{prefix}Llama-3.1-8B_{dataset}_N300_AGGmean_Mnap_USE_NEURONS_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100"
            resid_ig_path = f"{prefix}Llama-3.1-8B_{dataset}_N300_AGGmean_Mnap_USE_NEURONS_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100_SUBMODULES_resid"
            attn_ig_path = f"{prefix}Llama-3.1-8B_{dataset}_N300_AGGmean_Mnap_USE_NEURONS_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100_SUBMODULES_attn"
            f"{prefix}Llama-3.1-8B_{dataset}_N300_AGGmean_Mnap_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100"
            f"{prefix}Llama-3.1-8B_{dataset}_N300_AGGmean_Mnap_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100_SUBMODULES_resid"
            f"{prefix}Llama-3.1-8B_{dataset}_N300_AGGmean_Mnap_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100_SUBMODULES_attn"
            dataset_subfolders = [
                attn_ig_path,
                mlpacts_ig_path,
                mlp_ig_path,
                resid_ig_path,
                # sae_mlp_ig_path,
                # sae_resid_ig_path,
                # sae_attn_ig_path,
            ]

            all_subfolders.extend(dataset_subfolders)
            all_labels.extend(method_labels[: len(dataset_subfolders)])
            all_datasets.extend([dataset] * len(dataset_subfolders))

        plot_scores_for_methods(all_subfolders, all_labels, all_datasets, output_name=f"{mode}_")


if __name__ == "__main__":
    main()
