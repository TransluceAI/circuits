import pandas as pd
import plotnine as p9


def auc(results, save_path):
    thresholds = [k for k in results.keys() if isinstance(k, float)]

    data = []
    for threshold in thresholds:
        metrics = results[threshold]
        data.append(
            {
                "threshold": threshold,
                "n_nodes": metrics["n_nodes"],
                "faithfulness": metrics["faithfulness"],
                "completeness": metrics["completeness"],
            }
        )

    df = pd.DataFrame(data)
    df_melted = pd.melt(
        df,
        id_vars=["threshold", "n_nodes"],
        value_vars=["faithfulness", "completeness"],
        var_name="metric",
        value_name="score",
    )

    plot = (
        p9.ggplot(df_melted, p9.aes(x="n_nodes", y="score"))
        + p9.geom_line(size=1, color="#472d7b")
        + p9.geom_point(size=2, color="#472d7b")
        + p9.facet_wrap("~metric", nrow=1, scales="free_y")
        + p9.labs(x="Number of Nodes", y="Score")
        + p9.theme(
            strip_background=p9.element_rect(fill="lightgray", color="black", size=0.8),
            strip_text=p9.element_text(size=12),
            plot_title=p9.element_text(size=0),
            axis_text=p9.element_text(size=10),
            axis_title=p9.element_text(size=12),
            legend_position="none",
            panel_grid_major=p9.element_line(color="lightgray", size=0.5),
            panel_grid_minor=p9.element_line(color="lightgray", size=0.3, linetype="dashed"),
            panel_background=p9.element_rect(fill="white"),
            panel_border=p9.element_rect(color="black", fill=None, size=1),
            figure_size=(8, 4),
        )
    )

    plot.save(save_path, width=8, height=4)
