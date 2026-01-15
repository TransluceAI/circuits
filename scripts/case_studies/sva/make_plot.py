import glob
import logging
import os
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9

from env_util import ENV
if ENV.ARTIFACTS_DIR is None:
    raise RuntimeError("ARTIFACTS_DIR must be set in the .env file")

# suppress all warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# Default paths based on ENV.ARTIFACTS_DIR
_DEFAULT_RESULTS_PREFIX = str(Path(ENV.ARTIFACTS_DIR) / "results")
_DEFAULT_ENAP_PREFIX = str(Path(ENV.ARTIFACTS_DIR) / "enap_results")


def log_plot_saved(description: str, output_path: str | Path) -> None:
    """
    Log where a plot has been saved for easier discovery when running sweeps.
    """
    logger.info("%s saved to %s", description, output_path)


def infer_module_target(run_name: str) -> str:
    name = run_name.lower()
    module_tokens = []
    if "_submodules_attn" in name or name.endswith("_attn"):
        module_tokens.append("attn")
    if "_submodules_resid" in name or name.endswith("_resid"):
        module_tokens.append("resid")
    if "use_transcoder" in name or "transcoder" in name:
        module_tokens.append("transcoder")
    if "use_mlpacts" in name or "mlpacts" in name:
        module_tokens.append("mlp acts")
    if "use_neurons" in name or "mlp" in name:
        module_tokens.append("mlp")

    if not module_tokens:
        return "unspecified"
    # deduplicate while preserving order
    seen = set()
    unique_tokens = []
    for token in module_tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    if len(unique_tokens) == 1:
        return unique_tokens[0]
    return "+".join(unique_tokens)


def simplify_method_label(run_name: str, dataset_name: str) -> str:
    # label = run_name
    # pattern = re.escape(dataset_name) + "_"
    # label = re.sub(pattern, "", label, count=1)
    # label = re.sub(r"_N\d+_AGG.*", "", label)  # remove common run suffixes
    # label = label.replace("_", " ")
    # label = re.sub(r"\s+", " ", label).strip()
    return run_name


def plot_selected_methods_for_nodes(
    subfolder: str = "sva/pair",
    model_name: str = "Llama-3.1-8B",
    agg: str = "sum",
    prefix: str | None = None,
    datasets: list[str] = ["rc", "within_rc", "nounpp", "simple"],
    width: str = "",
):
    if prefix is None:
        prefix = _DEFAULT_RESULTS_PREFIX
    print(f"PROCESSING {subfolder} | {model_name} | {agg} | {prefix} | width={width}")
    pattern_prefix = prefix + subfolder + "/"
    model = model_name
    agg_mode = agg
    width_suffix = f"_WIDTH{width}" if width else ""
    save_path_prefix = f"{prefix}{subfolder.replace('/', '_')}_{model_name}_{agg}{width_suffix}"
    os.makedirs(save_path_prefix, exist_ok=True)

    def get_file(dataset: str = "rc", opts: str = "USE_NEURONS_USE_MLPACTS", suffix: str = ""):
        if opts != "":
            opts = "_" + opts
        pattern = f"{pattern_prefix}/{dataset}/{model}_{dataset}_N300_AGG{agg_mode}_Mnap{opts}_EDGE_THRESHOLD0.02_TOPK_NEURONS100{suffix}{width_suffix}/faithfulness_and_completeness/*.json"
        files = list(glob.glob(pattern))
        if len(files) == 0:
            res = pd.DataFrame()
        else:
            res = pd.read_json(files[0])
        return res

    dfs = []
    for dataset in datasets:
        for use_mlp_acts in [True, False]:
            df_sae_keep = get_file(dataset, "HANDLE_KEEP") if not use_mlp_acts else pd.DataFrame()
            df_sae_remove = (
                get_file(dataset, "HANDLE_REMOVE") if not use_mlp_acts else pd.DataFrame()
            )
            df_sae_default = (
                get_file(dataset, "DISABLE_STOP_GRAD") if not use_mlp_acts else pd.DataFrame()
            )
            df_sae_transcoder = (
                get_file(dataset, "DISABLE_STOP_GRAD", suffix="_USE_TRANSCODER")
                if not use_mlp_acts
                else pd.DataFrame()
            )
            df_sae_resid = (
                get_file(dataset, "DISABLE_STOP_GRAD", suffix="_SUBMODULES_resid")
                if not use_mlp_acts
                else pd.DataFrame()
            )
            df_sae_attn = (
                get_file(dataset, "DISABLE_STOP_GRAD", suffix="_SUBMODULES_attn")
                if not use_mlp_acts
                else pd.DataFrame()
            )
            df_sae_relp = (
                get_file(dataset, "USE_STOP_GRAD_USE_RELP_GRAD")
                if not use_mlp_acts
                else pd.DataFrame()
            )
            df_sae_relp_resid = (
                get_file(
                    dataset,
                    "USE_STOP_GRAD_USE_RELP_GRAD",
                    suffix="_SUBMODULES_resid",
                )
                if not use_mlp_acts
                else pd.DataFrame()
            )
            middle = "_USE_MLPACTS" if use_mlp_acts else ""
            df_ig = get_file(dataset, f"USE_NEURONS{middle}_DISABLE_STOP_GRAD")
            df_ig_resid = get_file(
                dataset,
                f"USE_NEURONS{middle}_DISABLE_STOP_GRAD",
                suffix="_SUBMODULES_resid",
            )
            df_ig_attn = get_file(
                dataset,
                f"USE_NEURONS{middle}_DISABLE_STOP_GRAD",
                suffix="_SUBMODULES_attn",
            )
            df_stop_grad = get_file(dataset, f"USE_NEURONS{middle}_USE_STOP_GRAD")
            df_stop_shapley = get_file(
                dataset,
                f"USE_NEURONS{middle}_USE_SHAPLEY_GRAD_USE_STOP_GRAD",
            )
            # df_stop_qk      = get_file(dataset, f"USE_NEURONS{middle}_USE_SHAPLEY_GRAD_USE_SHAPLEY_QK_USE_STOP_GRAD")
            df_stop_relp = get_file(
                dataset,
                f"USE_NEURONS{middle}_USE_STOP_GRAD_USE_RELP_GRAD",
            )
            df_stop_relp_ig = get_file(
                dataset,
                f"USE_NEURONS{middle}_USE_STOP_GRAD_USE_RELP_GRAD",
                suffix="_STEPS10",
            )
            df_stop_relp_resid = get_file(
                dataset,
                f"USE_NEURONS{middle}_USE_STOP_GRAD_USE_RELP_GRAD",
                suffix="_SUBMODULES_resid",
            )
            df_ig_inputs = get_file(
                dataset, f"EFFECTIG-INPUTS_USE_NEURONS{middle}_DISABLE_STOP_GRAD"
            )
            df_random = get_file(dataset, f"EFFECTRANDOM_USE_NEURONS{middle}")
            df_delta = get_file(dataset, f"EFFECTDELTA_USE_NEURONS{middle}")
            df_stop_relp_half_rule = get_file(
                dataset,
                f"USE_NEURONS{middle}_USE_STOP_GRAD_USE_RELP_GRAD_DISABLE_HALF_RULE",
            )

            df = pd.concat(
                [
                    df_sae_keep.assign(method="SAE IG (keep)"),
                    df_sae_remove.assign(method="SAE IG (remove)"),
                    df_sae_default.assign(method="SAE IG"),
                    df_sae_transcoder.assign(method="SAE IG (transcoder)"),
                    df_sae_resid.assign(method="SAE IG (resid.)"),
                    df_sae_attn.assign(method="SAE IG (attn.)"),
                    df_sae_relp.assign(method="SAE + Linear + ST"),
                    df_sae_relp_resid.assign(method="SAE + Linear + ST (resid.)"),
                    df_ig.assign(method="IG"),
                    df_ig_resid.assign(method="IG (resid.)"),
                    df_ig_attn.assign(method="IG (attn.)"),
                    df_stop_grad.assign(method="Linear"),
                    df_stop_shapley.assign(method="Linear + Shap."),
                    # df_stop_qk.assign(method="Linear + Shapley + QK"),
                    df_stop_relp.assign(method="Linear + ST"),
                    df_stop_relp_resid.assign(method="Linear + ST (resid.)"),
                    df_ig_inputs.assign(method="IG-inputs"),
                    df_random.assign(method="Random"),
                    df_delta.assign(method="Act. delta"),
                    df_stop_relp_half_rule.assign(method="Linear + ST (minus half rule)"),
                    df_stop_relp_ig.assign(method="Linear + ST + IG"),
                ],
                ignore_index=True,
            )
            if len(df) == 0:
                continue

            # add identifiers
            df["dataset"] = dataset
            df["component"] = "MLP activations" if use_mlp_acts else "MLP outputs"

            dfs.append(df)

    # ----- unify & clean -----
    if len(dfs) == 0:
        print("No data found for the specified configuration.")
        return
    df = pd.concat(dfs, ignore_index=True)

    # force numeric types; anything non-numeric becomes NaN and is dropped
    for c in ["n_nodes", "faithfulness", "completeness"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["n_nodes", "faithfulness", "completeness"])

    # if multiple rows share the same (dataset, component, method, n_nodes) (e.g., seeds),
    # aggregate by mean instead of taking .first()
    df = df.groupby(["dataset", "component", "method", "n_nodes"], as_index=False).agg(
        {"faithfulness": "mean", "completeness": "mean"}
    )

    # --- interpolation function (will be applied per plot type) ---
    all_nodes = np.sort(df["n_nodes"].unique())

    def interpolate_group(group: pd.DataFrame, use_log_interp: bool = False) -> pd.DataFrame:
        # dedupe by n_nodes within this group and sort
        g = (
            group.groupby("n_nodes", as_index=False)[["faithfulness", "completeness"]]
            .mean()
            .sort_values(by="n_nodes")
        )
        x = g.dropna()["n_nodes"].to_numpy()

        out = {"n_nodes": all_nodes}
        for col in ["faithfulness", "completeness"]:
            y = g[col].astype(float).to_numpy()
            if len(x) >= 2:
                if False:
                    # Interpolate in log space for log plots
                    log_x = np.log10(x)
                    log_all_nodes = np.log10(all_nodes)
                    out[col] = np.interp(log_all_nodes, log_x, y)
                else:
                    # Linear interpolation for linear plots
                    out[col] = np.interp(all_nodes, x, y)
            else:
                # only one x -> repeat the constant value
                const_val = float(y[0]) if len(y) else np.nan
                out[col] = np.full_like(all_nodes, const_val, dtype=float)

        result = pd.DataFrame(out)
        # carry identifiers
        result["dataset"] = group["dataset"].iloc[0]
        result["component"] = group["component"].iloc[0]
        result["method"] = group["method"].iloc[0]
        return result

    # Don't interpolate here - do it per plot type
    df_base = df.copy()

    # optional sanity checks (uncomment if you want quick diagnostics)
    # print(df.groupby(["dataset","component","method","metric"])["value"].agg(['min','max','mean']).round(4))
    # print(df.dtypes)

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

    keep_methods = {
        "Linear + ST @ MLP activations": "RelP (acts.)",
        "Linear + ST @ MLP outputs": "RelP (outs.)",
        "Linear + ST (resid.) @ MLP outputs": "RelP (resid.)",
        "SAE + Linear + ST @ MLP outputs": "SAE + RelP (outs.)",
        "SAE + Linear + ST (resid.) @ MLP outputs": "SAE + RelP (resid.)",
        "Act. delta @ MLP activations": "Delta (acts.)",
        # "Random @ MLP activations": "Random (acts.)",
        "IG @ MLP activations": "IG (acts.)",
        "IG @ MLP outputs": "IG (outs.)",
        "IG (resid.) @ MLP outputs": "IG (resid.)",
        "IG (attn.) @ MLP outputs": "IG (attn.)",
        "SAE IG (resid.) @ MLP outputs": "SAE + IG (resid.)",
        "SAE IG (attn.) @ MLP outputs": "SAE + IG (attn.)",
        "SAE IG (transcoder) @ MLP outputs": "SAE + IG (transcoder)",
        "SAE IG @ MLP outputs": "SAE + IG (outs.)",
        "Linear @ MLP activations": "RelP, no gate (acts.)",
        "Linear + ST (minus half rule) @ MLP activations": "RelP, no half rule (acts.)",
        "Linear + ST + IG @ MLP activations": "RelP + IG (acts.)",
        "IG-inputs @ MLP activations": "IG-inputs (acts.)",
    }

    # add all remaining methods to keep_methods
    for method in df_base.method.unique():
        if method not in keep_methods:
            keep_methods[method] = method

    for use_log in [True, False]:
        # Apply appropriate interpolation based on plot type
        df_interpolated = (
            df_base.groupby(["dataset", "component", "method"], group_keys=False)
            .apply(lambda x: interpolate_group(x, use_log_interp=use_log))
            .reset_index(drop=True)
        )

        # melt to long form
        df_interpolated = df_interpolated.melt(
            id_vars=["dataset", "component", "method", "n_nodes"],
            value_vars=["faithfulness", "completeness"],
            var_name="metric",
            value_name="value",
        )

        # tidy labels
        df_interpolated["metric"] = df_interpolated["metric"].str.capitalize()
        df_interpolated["metric"] = pd.Categorical(
            df_interpolated["metric"],
            categories=["Faithfulness", "Completeness"],
            ordered=True,
        )
        df_interpolated["id"] = df_interpolated.apply(
            lambda x: f"{x['dataset']}_{x['component']}_{x['method']}", axis=1
        )

        df_temp = df_interpolated.copy()
        # df_temp["baseline"] = df_temp.method.isin(["Act. delta", "Random"])
        df_temp.method = df_temp.apply(lambda x: x.method + " @ " + x.component, axis=1)
        df_temp["style"] = df_temp.method.str.contains("SAE").apply(
            lambda x: "SAE" if x else "Neurons"
        )
        df_temp = df_temp[df_temp.method.isin(list(keep_methods.keys()))]
        df_temp.method = df_temp.method.map(keep_methods)
        df_avg = (
            df_temp.groupby(["method", "dataset", "metric", "n_nodes", "style"])["value"]
            .mean()
            .reset_index()
            .dropna()
        )
        # df_avg = df_avg
        plot = (
            p9.ggplot(
                df_temp,
                p9.aes(x="n_nodes", y="value", group="id", color="method", linetype="style"),
            )
            # + p9.stat_summary(p9.aes(group="method"), geom="line", size=0.8)
            # + p9.geom_line(size=0.5, alpha=0.2)
            + p9.geom_line(
                mapping=p9.aes(x="n_nodes", y="value", group="method", color="method"),
                size=1,
                alpha=0.8,
            )
            + p9.scale_color_brewer(type="qual", palette="Set1")
            # + p9.geom_point(size=2, stroke=0, alpha=0.9)
            + p9.facet_grid("dataset ~ metric")
            + p9.theme(figure_size=(6, 6), legend_position="right")
            + p9.labs(x="# of neurons patched", y="", color="Method", linetype="Basis")
        )
        if use_log:
            plot = plot + (p9.scale_x_log10(breaks=[10**i for i in range(10)]))
        else:
            plot = plot + p9.scale_x_continuous(breaks=[i for i in range(0, 400 + 1, 100)])
            plot = plot + p9.coord_cartesian(xlim=(0, 400), ylim=(-0.3, 2))
        try:
            plot_path = f"{save_path_prefix}/fc_{'log' if use_log else 'linear'}.png"
            plot.save(plot_path, dpi=300)
            log_plot_saved("Base plot", plot_path)
        except Exception as e:
            print(f"Error saving base plot: {e}")
        df_avg_old = df_avg.copy()

        # plot comparing bases
        methods = {
            # "RelP (acts.)": "MLP acts",
            # "RelP (outs.)": "MLP outs",
            "IG (acts.)": "MLP acts.",
            "IG (outs.)": "MLP outs.",
            "IG (resid.)": "Resid.",
            "IG (attn.)": "Attn.",
            "SAE + IG (resid.)": "Resid.",
            "SAE + IG (attn.)": "Attn.",
            # "IG-inputs (acts.)": "MLP acts",
            "SAE + IG (outs.)": "MLP outs.",
            "SAE + IG (transcoder)": "Transcoder",
            # "SAE + RelP (outs.)": "SAE",
        }
        df_avg = (
            df_avg.groupby(["method", "metric", "n_nodes"]).agg({"value": "mean"}).reset_index()
        )
        df_avg = df_avg[df_avg.method.isin(list(methods.keys()))]
        df_avg["style"] = df_avg.method.str.contains("SAE").apply(
            lambda x: "SAE" if x else "Neurons"
        )
        df_avg["method"] = df_avg.method.map(methods)
        df_avg["method"] = pd.Categorical(
            df_avg["method"], categories=["MLP acts.", "MLP outs.", "Resid."], ordered=True
        )
        df_avg["group"] = df_avg.apply(lambda x: f"{x.method}_{x.style}", axis=1)
        plot = (
            p9.ggplot(
                df_avg,
                p9.aes(
                    x="n_nodes",
                    y="value",
                    group="group",
                    color="method",
                    linetype="style",
                ),
            )
            + p9.geom_line(size=1, alpha=0.8)
            + p9.scale_color_brewer(type="qual", palette="Set1")
            + p9.facet_grid("~ metric")
            + p9.theme(figure_size=(6, 2.5), legend_position="right")
            + p9.labs(x="Circuit size", y="", color="Representation", linetype="Basis")
            + p9.ylim(-0.3, 1.7)
        )
        if use_log:
            plot = plot + (p9.scale_x_log10(breaks=[10**i for i in range(10)]))
        else:
            plot = plot + p9.scale_x_continuous(breaks=[i for i in range(0, 400 + 1, 100)])
            plot = plot + p9.coord_cartesian(xlim=(0, 400), ylim=(-0.3, 2))
        try:
            plot_path = f"{save_path_prefix}/fc_{'log' if use_log else 'linear'}_bases.png"
            plot.save(plot_path, dpi=300)
            log_plot_saved("Basis plot", plot_path)
        except Exception as e:
            print(f"Error saving basis plot: {e}")

        # plot comparing RelP vs. IG
        methods = {
            "RelP (acts.)": "MLP acts.",
            "RelP (outs.)": "MLP outs.",
            "RelP (resid.)": "Resid.",
            "IG (acts.)": "MLP acts.",
            "IG (outs.)": "MLP outs.",
            "IG (resid.)": "Resid.",
            "SAE + IG (outs.)": "MLP outs.",
            "SAE + IG (resid.)": "Resid.",
            "SAE + IG (transcoder)": "Transcoder",
            "SAE + RelP (outs.)": "MLP outs.",
            "SAE + RelP (resid.)": "Resid.",
            "SAE + RelP (transcoder)": "Transcoder",
        }
        df_avg = (
            df_avg_old.copy()
            .groupby(["method", "metric", "n_nodes"])
            .agg({"value": "mean"})
            .reset_index()
        )
        df_avg = df_avg[df_avg.method.isin(list(methods.keys()))]
        df_avg["style"] = df_avg.method.str.contains("SAE").apply(
            lambda x: "SAE" if x else "Neurons"
        )
        df_avg["linetype"] = df_avg.method.str.contains("IG").apply(lambda x: "IG" if x else "RelP")
        df_avg["method"] = df_avg.method.map(methods)
        df_avg["group"] = df_avg.apply(
            lambda x: f"{x['method']}_{x['style']}_{x['linetype']}", axis=1
        )
        plot = (
            p9.ggplot(
                df_avg,
                p9.aes(
                    x="n_nodes",
                    y="value",
                    group="group",
                    color="method",
                    linetype="style",
                ),
            )
            + p9.geom_line(size=1, alpha=0.8)
            + p9.scale_color_brewer(type="qual", palette="Set1")
            + p9.facet_grid("linetype ~ metric")
            + p9.theme(figure_size=(6, 4), legend_position="right")
            + p9.labs(x="Circuit size", y="", color="Representation", linetype="Basis")
            + p9.ylim(-0.3, 1.7)
        )
        if use_log:
            plot = plot + (p9.scale_x_log10(breaks=[10**i for i in range(10)]))
        else:
            plot = plot + p9.scale_x_continuous(breaks=[i for i in range(0, 400 + 1, 100)])
            plot = plot + p9.coord_cartesian(xlim=(0, 400), ylim=(-0.3, 2))
        try:
            plot_path = f"{save_path_prefix}/fc_{'log' if use_log else 'linear'}_adag_ig.png"
            plot.save(plot_path, dpi=300)
            log_plot_saved("RelP vs. IG plot", plot_path)
        except Exception as e:
            print(f"Error saving RelP vs. IG plot: {e}")

        # ablations on RelP
        methods = {
            "RelP (acts.)": "RelP",
            "RelP, no gate (acts.)": "RelP, no gate",
            "RelP, no half rule (acts.)": "RelP, no half rule",
            "SAE + RelP (outs.)": "RelP",
        }
        df_avg = (
            df_avg_old.copy()
            .groupby(["method", "metric", "n_nodes"])
            .agg({"value": "mean"})
            .reset_index()
        )
        df_avg = df_avg[df_avg.method.isin(list(methods.keys()))]
        df_avg["style"] = df_avg.method.str.contains("SAE").apply(
            lambda x: "SAE" if x else "Neurons"
        )
        df_avg["method"] = df_avg.method.map(methods)
        plot = (
            p9.ggplot(df_avg, p9.aes(x="n_nodes", y="value", group="method", color="method"))
            + p9.geom_line(size=1, alpha=0.8)
            + p9.scale_color_brewer(type="qual", palette="Set1")
            + p9.facet_grid("style ~ metric")
            + p9.theme(figure_size=(6, 4), legend_position="right")
            + p9.labs(x="Circuit size", y="", color="Attribution")
            + p9.ylim(-0.3, 1.7)
        )
        if use_log:
            plot = plot + (p9.scale_x_log10(breaks=[10**i for i in range(10)]))
        else:
            plot = plot + p9.scale_x_continuous(breaks=[i for i in range(0, 400 + 1, 100)])
            plot = plot + p9.coord_cartesian(xlim=(0, 400), ylim=(-0.3, 2))
        try:
            plot_path = f"{save_path_prefix}/fc_{'log' if use_log else 'linear'}_adag_ablations.png"
            plot.save(plot_path, dpi=300)
            log_plot_saved("RelP ablations plot", plot_path)
        except Exception as e:
            print(f"Error saving RelP ablations plot: {e}")

    df_base.to_csv(f"{save_path_prefix}/fc_base.txt", index=False)


def plot_all_methods_for_nodes(
    base_dir: str | None = None,
    output_path: str | None = None,
) -> None:
    if base_dir is None:
        base_dir = _DEFAULT_RESULTS_PREFIX
    """
    Sweep every run beneath `base_dir` and save one scatter plot per dataset.

    Each subplot corresponds to a method (folder name), showing
    completeness vs. faithfulness with marker size encoding circuit size.
    """
    base_path = Path(base_dir)
    if output_path is None:
        output_path = str(base_path / "fc_all_methods")
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    json_paths = glob.glob(
        str(base_path / "**" / "faithfulness_and_completeness" / "*.json"),
        recursive=True,
    )
    frames: dict[str, list[pd.DataFrame]] = {}

    for json_path in json_paths:
        try:
            df = pd.read_json(json_path)
        except ValueError:
            continue
        if df.empty:
            continue

        run_dir = Path(json_path).parents[1]
        dataset_dir = run_dir.parent
        mode_dir = dataset_dir.parent if dataset_dir is not None else None
        group_dir = mode_dir.parent if mode_dir is not None else None

        run_name = run_dir.name
        dataset_name = dataset_dir.name if dataset_dir is not None else "unknown"
        mode_name = mode_dir.name if mode_dir is not None else "unknown_mode"
        group_name = group_dir.name if group_dir is not None else "unknown_group"
        dataset_key = f"{group_name}/{mode_name}/{dataset_name}"

        frame = df.copy()
        frame["dataset"] = dataset_name
        frame["group"] = group_name
        frame["mode"] = mode_name
        frame["method"] = run_name
        frame["subfolder"] = dataset_key
        frames.setdefault(dataset_key, []).append(frame)

    if not frames:
        print(f"No faithfulness/completeness files found under {base_dir}")
        return

    for dataset_key, dataset_frames in frames.items():
        if not dataset_frames:
            continue
        data = pd.concat(dataset_frames, ignore_index=True)

        # ensure numeric columns
        for col in ["n_nodes", "faithfulness", "completeness"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["n_nodes", "faithfulness", "completeness"])

        # aggregate repeated measurements (e.g., multiple seeds) by mean
        data = (
            data.groupby(["method", "n_nodes"], as_index=False)[["faithfulness", "completeness"]]
            .mean()
            .sort_values(["method", "n_nodes"])
        )

        if data.empty:
            continue

        method_order = sorted(data["method"].astype(str).unique())
        data["method"] = pd.Categorical(data["method"], categories=method_order, ordered=True)

        dataset_name = data["dataset"].iloc[0] if "dataset" in data.columns else dataset_key
        label_order: list[str] = []
        label_map: dict[str, str] = {}
        for m in method_order:
            simplified = simplify_method_label(m, dataset_name)
            wrapped = textwrap.fill(simplified, width=36)
            label_map[m] = wrapped
            if wrapped not in label_order:
                label_order.append(wrapped)
        data["method_label"] = data["method"].apply(lambda m: label_map[str(m)])
        data["method_label"] = pd.Categorical(
            data["method_label"], categories=label_order, ordered=True
        )

        data = data.sort_values(["method", "n_nodes"])  # ensure ordering for lines
        data["module"] = data["method"].apply(lambda m: infer_module_target(str(m)))
        module_order = [
            module
            for module in [
                "mlp",
                "mlp acts",
                "attn",
                "resid",
                "transcoder",
                "mlp+attn",
                "mlp+resid",
            ]
            if module in set(data["module"])
        ]
        for module in sorted(data["module"].unique()):
            if module not in module_order:
                module_order.append(module)
        data["module"] = pd.Categorical(data["module"], categories=module_order, ordered=True)

        num_methods = len(method_order)
        num_cols = max(1, min(6, int(np.ceil(np.sqrt(num_methods)))))
        num_rows = int(np.ceil(num_methods / num_cols))
        figure_width = min(25, max(6, 4 * num_cols))
        figure_height = min(25, max(4, 4 * num_rows))

        for metric in ["faithfulness", "completeness"]:
            plot = (
                p9.ggplot(
                    data,
                    p9.aes(x="n_nodes", y=metric, color="module"),
                )
                + p9.geom_line(p9.aes(group="method"), alpha=0.6)
                + p9.geom_point(alpha=0.85)
                + p9.facet_wrap("~ method_label", scales="free", ncol=num_cols)
                + p9.theme_bw(base_size=10)
                + p9.geom_hline(yintercept=0.0, linetype="dashed", color="#bbbbbb", size=0.5)
                + p9.geom_hline(yintercept=1.0, linetype="dashed", color="#bbbbbb", size=0.5)
                + p9.theme(
                    figure_size=(figure_width, figure_height),
                    legend_position="bottom",
                    legend_direction="horizontal",
                    legend_title=p9.element_text(size=9),
                    legend_text=p9.element_text(size=8),
                )
                + p9.scale_color_brewer(type="qual", palette="Set2", name="Module")
                + p9.labs(title=f"{dataset_key}", x="Circuit size", y=metric.capitalize())
                + p9.scale_x_log10(breaks=[10**i for i in range(10)])
            )

            dataset_slug = dataset_key.replace("/", "_")
            dataset_output = output_root / f"fc_all_methods_{dataset_slug}_{metric}.png"
            plot.save(str(dataset_output), dpi=300)
            log_plot_saved("All-methods node plot", dataset_output)


def plot_selected_methods_for_edges(
    model_name: str = "Llama-3.1-8B",
    agg: str = "mean",
    prefix: str | None = None,
    datasets: list[str] | None = None,
    topk_edges_filter: int | None = 500000,
):
    if prefix is None:
        prefix = _DEFAULT_ENAP_PREFIX
    if datasets is None:
        datasets = ["within_rc_300", "nounpp_300", "simple_300"]

    print(f"PROCESSING edges | {model_name} | {agg} | {prefix} | datasets: {datasets}")
    base_path = Path(prefix)
    dataset_str = "_".join([d.replace("_300", "") for d in datasets])
    save_path_prefix = str(base_path / f"plots_{model_name}_{agg}_{dataset_str}")
    os.makedirs(save_path_prefix, exist_ok=True)
    print(f"Saving plots to: {save_path_prefix}")

    dataset_map = {
        "rc_300": "rc",
        "within_rc_300": "within_rc",
        "nounpp_300": "nounpp",
        "simple_300": "simple",
    }

    def get_method_folders():
        def normalize_method_name(folder_name: str, dataset_name: str) -> str:
            dataset_short = dataset_name.replace("_300", "")
            pattern_to_remove = f"_{dataset_short}_"
            if pattern_to_remove in folder_name:
                normalized = folder_name.replace(pattern_to_remove, "_DATASET_", 1)
            else:
                normalized = folder_name
            return normalized

        method_data = {}
        folder_mapping = {}

        for dataset_folder in datasets:
            dataset_path = base_path / dataset_folder
            if not dataset_path.exists():
                continue
            for method_folder in dataset_path.iterdir():
                if not method_folder.is_dir():
                    continue
                full_method_name = method_folder.name
                normalized_name = normalize_method_name(full_method_name, dataset_folder)

                if normalized_name not in method_data:
                    method_data[normalized_name] = {}
                    folder_mapping[normalized_name] = {}

                method_data[normalized_name][dataset_folder] = method_folder
                folder_mapping[normalized_name][dataset_folder] = full_method_name

        complete_methods = {}
        for method_name, dataset_folders in method_data.items():
            if len(dataset_folders) == len(datasets):
                complete_methods[method_name] = dataset_folders
            else:
                missing = [d for d in datasets if d not in dataset_folders]
                example_folder = list(folder_mapping[method_name].values())[0]
                print(
                    f"Skipping {method_name} (e.g., {example_folder}): missing data for {missing}"
                )

        return complete_methods

    method_folders = get_method_folders()
    if not method_folders:
        print("No complete methods found across all datasets.")
        return

    dfs = []
    for method_name, dataset_folders in method_folders.items():
        for dataset_folder, folder_path in dataset_folders.items():
            json_pattern = folder_path / "faithfulness_and_completeness" / "*.json"
            json_files = list(glob.glob(str(json_pattern)))

            if not json_files:
                continue

            try:
                df = pd.read_json(json_files[0])
            except ValueError:
                continue

            if df.empty:
                continue

            dataset_name = dataset_map[dataset_folder]
            df["dataset"] = dataset_name
            df["method"] = method_name
            dfs.append(df)

    if len(dfs) == 0:
        print("No data found for the specified configuration.")
        return

    df = pd.concat(dfs, ignore_index=True)

    for c in ["n_edges", "faithfulness", "completeness"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["n_edges", "faithfulness", "completeness"])

    df = df.groupby(["dataset", "method", "n_edges"], as_index=False).agg(
        {"faithfulness": "mean", "completeness": "mean"}
    )

    all_edges = np.sort(df["n_edges"].unique())

    def interpolate_group(group: pd.DataFrame, use_log_interp: bool = False) -> pd.DataFrame:
        g = (
            group.groupby("n_edges", as_index=False)[["faithfulness", "completeness"]]
            .mean()
            .sort_values(by="n_edges")
        )
        x = g["n_edges"].to_numpy()

        out = {"n_edges": all_edges}
        for col in ["faithfulness", "completeness"]:
            y = g[col].astype(float).to_numpy()
            if len(x) >= 2:
                out[col] = np.interp(all_edges, x, y)
            else:
                const_val = float(y[0]) if len(y) else np.nan
                out[col] = np.full_like(all_edges, const_val, dtype=float)

        result = pd.DataFrame(out)
        result["dataset"] = group["dataset"].iloc[0]
        result["method"] = group["method"].iloc[0]
        result["method_label"] = group["method_label"].iloc[0]
        return result

    df_base = df.copy()

    keep_methods = {}
    for method in df_base.method.unique():
        method_lower = method.lower()

        if topk_edges_filter is not None:
            edges_str = f"topk_edges{topk_edges_filter}"
            if edges_str not in method_lower:
                continue

        if "jvp-ig-inputs" in method_lower:
            keep_methods[method] = "IG-inp."
        elif "jvp" in method_lower and "use_stop_grad_on_mlps" in method_lower:
            keep_methods[method] = "RelP"
        elif "jvp" in method_lower:
            keep_methods[method] = "RelP (no stop grad on MLPs)"
        else:
            keep_methods[method] = method

    if not keep_methods:
        print("No methods matched the filter criteria. Skipping plot generation.")
        return

    df_base = df_base[df_base.method.isin(list(keep_methods.keys()))]
    df_base["method_label"] = df_base.method.map(keep_methods)

    p9.theme_set(
        p9.theme_bw(base_size=10)
        + p9.theme(
            text=p9.element_text(color="#000"),
            axis_title=p9.element_text(size=10),
            axis_text=p9.element_text(size=8),
            axis_text_x=p9.element_text(angle=45, hjust=0.5),
            legend_text=p9.element_text(size=8),
            legend_title=p9.element_text(size=9),
            panel_grid_major=p9.element_line(size=1, color="#dddddd"),
            panel_grid_minor=p9.element_blank(),
            strip_background=p9.element_blank(),
            legend_margin=0,
        )
    )

    for use_log in [True, False]:
        df_interpolated = (
            df_base.groupby(["dataset", "method", "method_label"], group_keys=False)
            .apply(lambda x: interpolate_group(x, use_log_interp=use_log))
            .reset_index(drop=True)
        )

        df_interpolated = df_interpolated.melt(
            id_vars=["dataset", "method", "method_label", "n_edges"],
            value_vars=["faithfulness", "completeness"],
            var_name="metric",
            value_name="value",
        )

        df_interpolated["metric"] = df_interpolated["metric"].str.capitalize()
        df_interpolated["metric"] = pd.Categorical(
            df_interpolated["metric"],
            categories=["Faithfulness", "Completeness"],
            ordered=True,
        )
        df_interpolated["id"] = df_interpolated.apply(
            lambda x: f"{x['dataset']}_{x['method_label']}", axis=1
        )

        df_temp = df_interpolated.copy()
        df_avg = (
            df_temp.groupby(["method_label", "dataset", "metric", "n_edges"])["value"]
            .mean()
            .reset_index()
            .dropna()
        )

        plot = (
            p9.ggplot(
                df_temp,
                p9.aes(x="n_edges", y="value", group="id", color="method_label"),
            )
            + p9.geom_line(
                data=df_avg,
                mapping=p9.aes(x="n_edges", y="value", group="method_label", color="method_label"),
                size=1,
                alpha=0.8,
            )
            + p9.scale_color_brewer(type="qual", palette="Set1")
            + p9.facet_grid("dataset ~ metric")
            + p9.theme(figure_size=(6, 2.5), legend_position="right")
            + p9.labs(x="# of edges patched", y="", color="Method")
        )
        if use_log:
            plot = plot + (p9.scale_x_log10(breaks=[10**i for i in range(10)]))
        else:
            plot = plot + p9.scale_x_continuous()

        try:
            plot_path = (
                f"{save_path_prefix}/fc_edges_by_dataset_{'log' if use_log else 'linear'}.png"
            )
            plot.save(plot_path, dpi=300)
            log_plot_saved(f"Edge plot by dataset ({'log' if use_log else 'linear'})", plot_path)
        except Exception as e:
            print(f"Error saving edge plot by dataset: {e}")

        df_aggregated = (
            df_temp.groupby(["method_label", "metric", "n_edges"])["value"]
            .mean()
            .reset_index()
            .dropna()
        )

        def format_scientific(x):
            labels = []
            for val in x:
                if val == 0:
                    labels.append("0")
                else:
                    exponent = int(np.log10(val))
                    mantissa = val / (10**exponent)
                    if mantissa == 1.0:
                        labels.append(f"1e{exponent}")
                    else:
                        labels.append(f"{mantissa:.0f}e{exponent}")
            return labels

        plot_agg = (
            p9.ggplot(
                df_aggregated,
                p9.aes(x="n_edges", y="value", group="method_label", color="method_label"),
            )
            + p9.geom_line(size=1, alpha=0.8)
            + p9.scale_color_brewer(type="qual", palette="Set1")
            + p9.facet_wrap("~ metric", nrow=1)
            + p9.theme(figure_size=(6, 2.5), legend_position="right")
            + p9.labs(x="Circuit size (num. of edges)", y="", color="Edge attribution method")
        )
        if use_log:
            plot_agg = plot_agg + p9.scale_x_log10(labels=format_scientific)
        else:
            plot_agg = plot_agg + p9.scale_x_continuous(labels=format_scientific)

        try:
            plot_path = (
                f"{save_path_prefix}/fc_edges_aggregated_{'log' if use_log else 'linear'}.png"
            )
            plot_agg.save(plot_path, dpi=300)
            log_plot_saved(f"Aggregated edge plot ({'log' if use_log else 'linear'})", plot_path)
        except Exception as e:
            print(f"Error saving aggregated edge plot: {e}")

    df_base.to_csv(f"{save_path_prefix}/fc_edges_base.txt", index=False)


def plot_all_methods_for_edges(
    base_dir: str | None = None,
    output_path: str | None = None,
) -> None:
    if base_dir is None:
        base_dir = _DEFAULT_ENAP_PREFIX
    base_path = Path(base_dir)
    if output_path is None:
        output_path = str(base_path / "fc_all_methods_edges")
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    json_paths = glob.glob(
        str(base_path / "**" / "faithfulness_and_completeness" / "*.json"),
        recursive=True,
    )
    frames: dict[str, list[pd.DataFrame]] = {}

    for json_path in json_paths:
        try:
            df = pd.read_json(json_path)
        except ValueError:
            continue
        if df.empty:
            continue

        run_dir = Path(json_path).parents[1]
        dataset_dir = run_dir.parent

        run_name = run_dir.name
        dataset_name = dataset_dir.name if dataset_dir is not None else "unknown"
        dataset_key = dataset_name

        frame = df.copy()
        frame["dataset"] = dataset_name
        frame["method"] = run_name
        frames.setdefault(dataset_key, []).append(frame)

    if not frames:
        print(f"No faithfulness/completeness files found under {base_dir}")
        return

    for dataset_key, dataset_frames in frames.items():
        if not dataset_frames:
            continue
        data = pd.concat(dataset_frames, ignore_index=True)

        for col in ["n_edges", "faithfulness", "completeness"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["n_edges", "faithfulness", "completeness"])

        data = (
            data.groupby(["method", "n_edges"], as_index=False)[["faithfulness", "completeness"]]
            .mean()
            .sort_values(["method", "n_edges"])
        )

        if data.empty:
            continue

        method_order = sorted(data["method"].astype(str).unique())
        data["method"] = pd.Categorical(data["method"], categories=method_order, ordered=True)

        dataset_name = data["dataset"].iloc[0] if "dataset" in data.columns else dataset_key
        label_order: list[str] = []
        label_map: dict[str, str] = {}
        for m in method_order:
            simplified = simplify_method_label(m, dataset_name)
            wrapped = textwrap.fill(simplified, width=36)
            label_map[m] = wrapped
            if wrapped not in label_order:
                label_order.append(wrapped)
        data["method_label"] = data["method"].apply(lambda m: label_map[str(m)])
        data["method_label"] = pd.Categorical(
            data["method_label"], categories=label_order, ordered=True
        )

        data = data.sort_values(["method", "n_edges"])

        num_methods = len(method_order)
        num_cols = max(1, min(6, int(np.ceil(np.sqrt(num_methods)))))
        num_rows = int(np.ceil(num_methods / num_cols))
        figure_width = min(25, max(6, 4 * num_cols))
        figure_height = min(25, max(4, 4 * num_rows))

        for metric in ["faithfulness", "completeness"]:
            plot = (
                p9.ggplot(
                    data,
                    p9.aes(x="n_edges", y=metric),
                )
                + p9.geom_line(p9.aes(group="method"), alpha=0.6)
                + p9.geom_point(alpha=0.85)
                + p9.facet_wrap("~ method_label", scales="free", ncol=num_cols)
                + p9.theme_bw(base_size=10)
                + p9.geom_hline(yintercept=0.0, linetype="dashed", color="#bbbbbb", size=0.5)
                + p9.geom_hline(yintercept=1.0, linetype="dashed", color="#bbbbbb", size=0.5)
                + p9.theme(
                    figure_size=(figure_width, figure_height),
                    legend_position="bottom",
                    legend_direction="horizontal",
                    legend_title=p9.element_text(size=9),
                    legend_text=p9.element_text(size=8),
                )
                + p9.labs(
                    title=f"{dataset_key}",
                    x="Circuit size (edges)",
                    y=metric.capitalize(),
                )
                + p9.scale_x_log10(breaks=[10**i for i in range(10)])
            )

            dataset_slug = dataset_key.replace("/", "_")
            dataset_output = output_root / f"fc_all_methods_edges_{dataset_slug}_{metric}.png"
            plot.save(str(dataset_output), dpi=300)
            log_plot_saved("All-methods edge plot", dataset_output)


if __name__ == "__main__":
    plot_selected_methods_for_nodes(subfolder="sva/pair", agg="mean")
    exit(0)
    plot_selected_methods_for_nodes(subfolder="sva/nopair", model_name="Llama-3.1-8B", agg="mean")
    plot_selected_methods_for_nodes(subfolder="sva/nopair", model_name="Llama-3.1-8B", agg="sum")
    plot_selected_methods_for_nodes(
        subfolder="sva/nopair",
        model_name="Llama-3.1-8B",
        agg="mean",
        prefix=str(Path(ENV.ARTIFACTS_DIR) / "results-mean-abl"),
    )
    plot_selected_methods_for_nodes(
        subfolder="sva/nopair",
        model_name="Llama-3.1-8B",
        agg="mean",
        prefix=str(Path(ENV.ARTIFACTS_DIR) / "results-mean-abl2"),
    )

    plot_selected_methods_for_edges(
        model_name="Llama-3.1-8B",
        agg="mean",
        datasets=["within_rc_300", "nounpp_300", "simple_300", "rc_300"],
        topk_edges_filter=500000,
    )
