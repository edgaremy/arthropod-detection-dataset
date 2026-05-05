from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import io
from PIL import Image


INPUT_CSVS_BY_NAME: dict[str, Path] = {
    "OOD": Path("validation/fine-tuning/finetuning_oodsplit_test_metrics.csv"),
    "Lepinoc": Path("validation/fine-tuning/finetuning_lepinoc_test_metrics.csv"),
}
# Per run/case text offset parameters used for trend value labels.
# y_offset is computed as max(y_min, y_span_multiplier * y_span).
# x_offset is expressed in points (display-space horizontal shift).
# If a run name is missing here, DEFAULT_OFFSET_PARAMS is used.
OFFSET_PARAMS_BY_NAME: dict[str, dict[str, float]] = {
    "OOD": {"y_min": 0.01, "y_span_multiplier": 0.05, "x": -30.0},
    "Lepinoc": {"y_min": 0.01, "y_span_multiplier": 0.025, "x": -30.0},
}
DEFAULT_OFFSET_PARAMS: dict[str, float] = {"y_min": 0.045, "y_span_multiplier": 0.12, "x": 0.0}
OUTPUT_DIR = Path("validation/fine-tuning")

F1_COLUMN = "avg_F1"
MIOU_COLUMN = "avg_mean_IoU"
X_AXIS_POWER = 0.8


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return slug or "run"


def _read_points_by_scenario(csv_path: Path) -> dict[str, dict[int, dict[str, list[float]]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows_by_scenario: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {F1_COLUMN: [], MIOU_COLUMN: []})
    )

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                scenario = str(row.get("scenario", "unknown")).strip() or "unknown"
                n_images = int(row["n_images"])
                _ = int(row.get("fold", "-1"))
                f1 = float(row[F1_COLUMN])
                mean_iou = float(row[MIOU_COLUMN])
            except (ValueError, KeyError):
                continue
            rows_by_scenario[scenario][n_images][F1_COLUMN].append(f1)
            rows_by_scenario[scenario][n_images][MIOU_COLUMN].append(mean_iou)

    if not rows_by_scenario:
        raise ValueError("No valid rows found in input CSV.")

    return {
        scenario: {n_images: dict(metrics) for n_images, metrics in by_size.items()}
        for scenario, by_size in rows_by_scenario.items()
    }


def _plot_metric(
    rows_by_scenario: dict[str, dict[int, dict[str, list[float]]]],
    metric_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    y_offset_min: float,
    y_offset_span_multiplier: float,
    x_offset: float,
    save: bool = True,
) -> None:
    fig = plt.figure(figsize=(11, 6.5))
    title_fs = 18
    label_fs = 16
    tick_fs = 13
    legend_fs = 15

    color_by_scenario = {
        "arthro+flatbug": "tab:blue",
        "fromscratch": "tab:orange",
    }
    label_by_scenario = {
        "arthro+flatbug": "ArthroNat+flatbug",
        "fromscratch": "Ultralytics pre-trained (COCO)",
    }

    scenarios_in_order = ["arthro+flatbug", "fromscratch"]
    for scenario in sorted(rows_by_scenario):
        if scenario not in scenarios_in_order:
            scenarios_in_order.append(scenario)

    all_sizes: set[int] = set()
    for by_size in rows_by_scenario.values():
        all_sizes.update(by_size.keys())
    x_sizes = sorted(all_sizes)

    def _x_transform(x_val: int) -> float:
        return float(x_val) ** X_AXIS_POWER

    def _blend_with_white(color: str, base_alpha: float) -> tuple[float, float, float, float]:
        r, g, b = mcolors.to_rgb(color)
        w = 1.0 - base_alpha
        return (base_alpha * r + w, base_alpha * g + w, base_alpha * b + w, 1.0)

    group_positions = np.array([_x_transform(x) for x in x_sizes], dtype=float)
    min_gap = float(np.min(np.diff(group_positions))) if len(group_positions) > 1 else 1.0
    box_width = max(0.26 * min_gap, 2.3)

    scenario_payloads: list[dict[str, object]] = []

    for s_idx, scenario in enumerate(scenarios_in_order):
        by_size = rows_by_scenario.get(scenario, {})
        color = color_by_scenario.get(scenario, "tab:gray")

        data: list[list[float]] = []
        positions: list[float] = []
        # Keep scenarios side by side around each x tick.
        offset = (s_idx - (len(scenarios_in_order) - 1) / 2.0) * box_width * 0.55

        x_line: list[float] = []
        y_line: list[float] = []

        for x_idx, n_images in enumerate(x_sizes):
            values = by_size.get(n_images, {}).get(metric_key, [])
            if not values:
                continue
            box_x = float(group_positions[x_idx] + offset)
            data.append(values)
            positions.append(box_x)
            x_line.append(box_x)
            y_line.append(float(np.mean(values)))

        if not data:
            continue

        scenario_payloads.append(
            {
                "color": color,
                "data": data,
                "positions": positions,
                "x_line": x_line,
                "y_line": y_line,
            }
        )

    # Draw all trend lines first so boxes can sit on top of every line.
    for payload in scenario_payloads:
        plt.plot(
            payload["x_line"],
            payload["y_line"],
            linewidth=2.5,
            color=payload["color"],
            alpha=0.35,
            marker=None,
            zorder=1,
        )

    # Draw boxes after lines so they remain visible above all line segments.
    for payload in scenario_payloads:
        color = str(payload["color"])
        data = payload["data"]
        positions = payload["positions"]
        x_line = payload["x_line"]
        y_line = payload["y_line"]

        boxplot = plt.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showmeans=False,
            meanline=False,
            manage_ticks=False,
            flierprops={
                "marker": "o",
                "markerfacecolor": color,
                "markeredgecolor": color,
                "markersize": 4,
                "alpha": 0.75,
            },
        )

        for patch in boxplot["boxes"]:
            patch.set_facecolor(_blend_with_white(color, 0.45))
            patch.set_edgecolor(mcolors.to_rgba(color, 0.95))
            patch.set_linewidth(1.8)
            patch.set_zorder(4)
        for whisker in boxplot["whiskers"]:
            whisker.set_color(color)
            whisker.set_alpha(0.95)
            whisker.set_zorder(4)
        for cap in boxplot["caps"]:
            cap.set_color(color)
            cap.set_alpha(0.95)
            cap.set_zorder(4)
        for median in boxplot["medians"]:
            median.set_color(color)
            median.set_linewidth(2.0)
            median.set_zorder(5)
        for flier in boxplot["fliers"]:
            flier.set_zorder(5)

        # Show y-value for each trend point.
        y_span = max(y_line) - min(y_line) if len(y_line) > 1 else 0.0
        y_offset = max(y_offset_min, y_offset_span_multiplier * y_span)
        for x_val, y_val in zip(x_line, y_line):
            plt.text(
                x_val,
                y_val + y_offset,
                f"{y_val:.3f}",
                color=color,
                fontsize=12,
                ha="center",
                va="bottom",
                zorder=6,
                transform=plt.gca().transData + plt.matplotlib.transforms.ScaledTranslation(
                    x_offset / 72.0,
                    0.0,
                    plt.gcf().dpi_scale_trans,
                ),
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.65,
                    "boxstyle": "round,pad=0.18",
                },
            )

    plt.xlabel("Number of images used for fine-tuning", fontsize=label_fs)
    plt.ylabel(ylabel, fontsize=label_fs)
    plt.grid(True, alpha=0.3)
    plt.xticks(group_positions, [str(x) for x in x_sizes], rotation=0, ha="center", fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color=color_by_scenario.get(s, "tab:gray"), lw=8, alpha=0.5)
        for s in scenarios_in_order
    ]
    legend_labels = [label_by_scenario.get(s, s) for s in scenarios_in_order]
    plt.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        fontsize=legend_fs,
        frameon=True,
        facecolor="#f5f5f5",
        edgecolor="#d9d9d9",
    )

    if metric_key == F1_COLUMN:
        bottom_prefix = "(a) "
    elif metric_key == MIOU_COLUMN:
        bottom_prefix = "(b) "
    else:
        bottom_prefix = ""

    ax = plt.gca()
    label_text = bottom_prefix.strip()
    if label_text:
        ax.text(
            0.01,
            1.02,
            label_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=title_fs,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
            zorder=10,
        )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    if save:
        plt.savefig(out_path, dpi=200)
        plt.close()
        return None
    else:
        return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    if not INPUT_CSVS_BY_NAME:
        raise ValueError("INPUT_CSVS_BY_NAME is empty. Add at least one named CSV input.")

    for run_name, input_csv in INPUT_CSVS_BY_NAME.items():
        rows_by_scenario = _read_points_by_scenario(input_csv)
        offset_params = {**DEFAULT_OFFSET_PARAMS, **OFFSET_PARAMS_BY_NAME.get(run_name, {})}
        y_offset_min = float(offset_params["y_min"])
        y_offset_span_multiplier = float(offset_params["y_span_multiplier"])
        x_offset = float(offset_params["x"])

        run_slug = _slugify_name(run_name)
        run_plots_dir = OUTPUT_DIR / "plots"
        run_plots_dir.mkdir(parents=True, exist_ok=True)

        f1_plot = run_plots_dir / f"{run_slug}_f1_vs_finetuning_images.png"
        miou_plot = run_plots_dir / f"{run_slug}_mean_iou_vs_finetuning_images.png"

        _plot_metric(
            rows_by_scenario,
            metric_key=F1_COLUMN,
            ylabel="F1-score",
            title=f"F1-score vs Fine-tuning Set Size (on {run_name})",
            out_path=f1_plot,
            y_offset_min=y_offset_min,
            y_offset_span_multiplier=y_offset_span_multiplier,
            x_offset=x_offset,
        )

        _plot_metric(
            rows_by_scenario,
            metric_key=MIOU_COLUMN,
            ylabel="Mean IoU",
            title=f"Mean IoU vs Fine-tuning Set Size (on {run_name})",
            out_path=miou_plot,
            y_offset_min=y_offset_min,
            y_offset_span_multiplier=y_offset_span_multiplier,
            x_offset=x_offset,
        )

        print(f"[{run_name}] Saved: {f1_plot}")
        print(f"[{run_name}] Saved: {miou_plot}")
        # --- Combined side-by-side export (F1 | Mean IoU) -----------------
        try:
            fig_f1 = _plot_metric(
                rows_by_scenario,
                metric_key=F1_COLUMN,
                ylabel="F1-score",
                title=f"F1-score vs Fine-tuning Set Size (on {run_name})",
                out_path=f1_plot,
                y_offset_min=y_offset_min,
                y_offset_span_multiplier=y_offset_span_multiplier,
                x_offset=x_offset,
                save=False,
            )

            fig_miou = _plot_metric(
                rows_by_scenario,
                metric_key=MIOU_COLUMN,
                ylabel="Mean IoU",
                title=f"Mean IoU vs Fine-tuning Set Size (on {run_name})",
                out_path=miou_plot,
                y_offset_min=y_offset_min,
                y_offset_span_multiplier=y_offset_span_multiplier,
                x_offset=x_offset,
                save=False,
            )

            # Render figs to PNGs in-memory and stitch horizontally using PIL
            buf1 = io.BytesIO()
            buf2 = io.BytesIO()
            fig_f1.savefig(buf1, format="png", dpi=200)
            fig_miou.savefig(buf2, format="png", dpi=200)
            buf1.seek(0)
            buf2.seek(0)
            img1 = Image.open(buf1).convert("RGBA")
            img2 = Image.open(buf2).convert("RGBA")

            total_w = img1.width + img2.width
            max_h = max(img1.height, img2.height)
            combined = Image.new("RGBA", (total_w, max_h), (255, 255, 255, 255))
            combined.paste(img1, (0, 0))
            combined.paste(img2, (img1.width, 0))

            combined_path = run_plots_dir / f"{run_slug}_f1_mean_iou_sidebyside.png"
            combined.convert("RGB").save(combined_path, format="PNG", dpi=(200, 200))
            plt.close(fig_f1)
            plt.close(fig_miou)
            print(f"[{run_name}] Saved: {combined_path}")
        except Exception as e:
            print(f"[{run_name}] Failed to create combined image: {e}")


if __name__ == "__main__":
    main()
