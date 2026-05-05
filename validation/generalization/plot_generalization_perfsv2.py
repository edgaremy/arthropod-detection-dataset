import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypalettes import load_cmap


def _apply_style():
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
    plt.rcParams["font.size"] = 20

    ratio = "0.15"
    plt.rcParams["text.color"] = ratio
    plt.rcParams["xtick.color"] = ratio
    plt.rcParams["ytick.color"] = ratio
    plt.rcParams["axes.labelcolor"] = ratio


def _format_axes(ax):
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, axis="y")
    ax.set_axisbelow(True)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(which="both", direction="in", length=0)
    ax.tick_params(axis="both", which="major", pad=10)


def _annotate_bars(ax, fontsize=15):
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.008,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )


def _get_2x2_df(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["model_name"] == "arthro_mosaic2x2"].copy()

    dataset_order = ["same_species", "same_genus", "other_genus", "other_families"]
    dataset_labels = {
        "same_species": "Same species",
        "same_genus": "Same genus",
        "other_genus": "Other genus",
        "other_families": "Other families",
    }

    df["test_dataset"] = pd.Categorical(df["test_dataset"], categories=dataset_order, ordered=True)
    df = df.sort_values("test_dataset")
    df["test_dataset_label"] = df["test_dataset"].map(dataset_labels)

    return df


# def plot_2x2_by_dataset(csv_path, output, figsize=(14, 8), bar_width=0.36):
#     """
#     Plot ArthroNat 2x2 performances with test datasets on the x-axis.
#     Bars in each group are F1-score and mean IoU.
#     """
#     _apply_style()
#     df = _get_2x2_df(csv_path)

#     x_labels = df["test_dataset_label"].tolist()
#     f1_values = df["avg_F1"].to_numpy()
#     iou_values = df["avg_mean_IoU"].to_numpy()

#     x = np.arange(len(x_labels))

#     colors = load_cmap("Egypt").colors
#     metric_colors = [colors[1], colors[2]]

#     fig, ax = plt.subplots(figsize=figsize)

#     ax.bar(x - bar_width / 2, f1_values, width=bar_width, color=metric_colors[0], label="F1-score")
#     ax.bar(x + bar_width / 2, iou_values, width=bar_width, color=metric_colors[1], label="mean IoU")

#     _annotate_bars(ax)

#     ax.set_title("ArthroNat 2x2 across generalization test sets")
#     ax.set_ylim(0.5, 1.0)
#     ax.set_xlabel("Test dataset")
#     ax.set_xticks(x)
#     ax.set_xticklabels(x_labels, rotation=20, ha="right")
#     _format_axes(ax)

#     legend = ax.legend(loc="lower left", title="Metric")
#     legend.get_frame().set_alpha(0.9)

#     plt.tight_layout()
#     plt.savefig(output, dpi=300, bbox_inches="tight")
#     plt.close()


def plot_2x2_by_metric(csv_path, output, figsize=(12, 8), bar_width=0.18):
    """
    Plot ArthroNat 2x2 performances with metrics on the x-axis.
    Bars in each group are test datasets.
    """
    _apply_style()
    df = _get_2x2_df(csv_path)

    metrics = ["F1-score", "Precision", "Recall", "mean IoU"]
    dataset_labels = df["test_dataset_label"].tolist()

    values = {
        label: [
            row["avg_F1"],
            row["avg_precision"],
            row["avg_recall"],
            row["avg_mean_IoU"],
        ]
        for label, (_, row) in zip(dataset_labels, df.iterrows())
    }

    x = np.arange(len(metrics))

    colors = load_cmap("Egypt").colors
    dataset_colors = [colors[i] for i in [1, 2, 3, 0]]

    fig, ax = plt.subplots(figsize=figsize)

    offset_start = -((len(dataset_labels) - 1) / 2) * bar_width
    for i, label in enumerate(dataset_labels):
        offsets = x + offset_start + i * bar_width
        ax.bar(offsets, values[label], width=bar_width, color=dataset_colors[i], label=label)

    _annotate_bars(ax)

    # ax.set_title("Baseline model metrics accross generalization test sets")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    _format_axes(ax)

    legend = ax.legend(loc="lower left", title="Test dataset")
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    csv_path = "validation/generalization/model_comparison.csv"
    out_metric = "validation/generalization/generalization_2x2_by_metric.png"

    plot_2x2_by_metric(csv_path, out_metric)
    print(f"Generated plot: {out_metric}")
