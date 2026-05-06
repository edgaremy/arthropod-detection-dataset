"""
Build a dataset-level bounding-box statistics table for selected datasets.

The script computes, for each dataset:
- number of images (based on label files for the selected split)
- number of bounding boxes
- bbox relative size stats (mean/min/max/std)
- bbox count per image stats (mean/min/max/std)

It then:
1) saves the table as CSV
2) prints a LaTeX table (booktabs style) to stdout
"""

import argparse
import os

import numpy as np
import pandas as pd


def _list_label_files(dataset_path, split):
    labels_dir = os.path.join(dataset_path, "labels")

    if split == "all":
        if not os.path.isdir(labels_dir):
            raise ValueError(f"Labels directory not found: {labels_dir}")

        split_dirs = sorted(
            d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d))
        )
        if not split_dirs:
            raise ValueError(f"No split subdirectories found in: {labels_dir}")

        label_files = [
            os.path.join(labels_dir, s, f)
            for s in split_dirs
            for f in os.listdir(os.path.join(labels_dir, s))
            if f.endswith(".txt")
        ]
    else:
        split_dir = os.path.join(labels_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Labels directory not found: {split_dir}")

        label_files = [
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".txt")
        ]

    return label_files


def _safe_stat(arr, reducer):
    if len(arr) == 0:
        return np.nan
    return float(reducer(arr))


def compute_dataset_bbox_stats(dataset_path, split):
    label_files = _list_label_files(dataset_path, split)

    bbox_counts = []
    bbox_sizes = []

    for label_file in label_files:
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        bbox_counts.append(len(lines))

        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                width = float(parts[3])
                height = float(parts[4])
                bbox_sizes.append(width * height)

    bbox_counts = np.array(bbox_counts, dtype=float)
    bbox_sizes = np.array(bbox_sizes, dtype=float)

    return {
        "num_images": int(len(label_files)),
        "num_bboxes": int(len(bbox_sizes)),
        "bbox_size_mean": _safe_stat(bbox_sizes, np.mean),
        "bbox_size_min": _safe_stat(bbox_sizes, np.min),
        "bbox_size_max": _safe_stat(bbox_sizes, np.max),
        "bbox_size_std": _safe_stat(bbox_sizes, np.std),
        "bboxes_per_image_mean": _safe_stat(bbox_counts, np.mean),
        "bboxes_per_image_min": _safe_stat(bbox_counts, np.min),
        "bboxes_per_image_max": _safe_stat(bbox_counts, np.max),
        "bboxes_per_image_std": _safe_stat(bbox_counts, np.std),
    }


def _format_float(value, decimals):
    if pd.isna(value):
        return ""
    # Use thousands separators with fixed decimal precision.
    return f"{float(value):,.{decimals}f}"


def _format_int(value):
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def _format_min_apparent_size(value, sig_digits=1):
    if pd.isna(value):
        return ""
    numeric = float(value)
    if numeric == 0:
        return r"$0$"

    sci = f"{numeric:.{sig_digits}e}"
    mantissa, exponent = sci.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = int(exponent)
    return f"${mantissa}e{exponent}$"


def dataframe_to_latex(df, caption="Bounding-box statistics across datasets", label="tab:other_datasets_bbox_stats"):
    latex = []
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(r"\footnotesize")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    latex.append(r"\begin{tabular}{lrr!{\color{gray!35}\vrule}rrrr!{\color{gray!35}\vrule}rrrr}")
    latex.append(r"\toprule")
    latex.append(
        r"\multirow[c]{2}{*}{\textbf{Dataset}} & "
        r"\multirow[c]{2}{*}{\textbf{\# test images}} & "
        r"\multirow[c]{2}{*}{\textbf{\# bboxes}} & "
        r"\multicolumn{4}{c!{\color{gray!35}\vrule}}{\textbf{bbox apparent size (\% of image)}} & "
        r"\multicolumn{4}{c}{\textbf{\# bbox per image}} \\" 
    )
    latex.append(r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}")
    latex.append(
        r" &  &  & "
        r"\textbf{mean} & "
        r"\textbf{std} & "
        r"\textbf{min} & "
        r"\textbf{max} & "
        r"\textbf{mean} & "
        r"\textbf{std} & "
        r"\textbf{min} & "
        r"\textbf{max} \\" 
    )
    latex.append(r"\midrule")

    for idx, row in df.iterrows():
        parts = [
            str(row["dataset"]),
            _format_int(row["num_images"]),
            _format_int(row["num_bboxes"]),
            _format_float(row["bbox_size_mean"], 2),
            _format_float(row["bbox_size_std"], 2),
            _format_min_apparent_size(row["bbox_size_min"], 1),
            _format_float(row["bbox_size_max"], 2),
            _format_float(row["bboxes_per_image_mean"], 2),
            _format_float(row["bboxes_per_image_std"], 2),
            _format_int(row["bboxes_per_image_min"]),
            _format_int(row["bboxes_per_image_max"]),
        ]
        row_content = " & ".join(parts) + r" \\" 
        if idx % 2 == 0:
            row_content = r"\rowcolor{gray!10} " + row_content
        latex.append(row_content)

        if str(row["dataset"]).strip().lower() == "flatbug":
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def build_other_datasets_table(flatbug_path, ood_path, lepinoc_path):
    datasets_config = [
        {"dataset": "ArthroNat", "path": "dataset", "split": "test"},
        {"dataset": "flatbug", "path": flatbug_path, "split": "test"},
        {"dataset": "SPIPOLL", "path": "datasets(others)/SPIPOLL", "split": "test"},
        {"dataset": "OOD", "path": ood_path, "split": "test"},
        {"dataset": "Lepinoc", "path": lepinoc_path, "split": "test"},
    ]

    rows = []
    for cfg in datasets_config:
        dataset_name = cfg["dataset"]
        dataset_path = cfg["path"]
        split = cfg["split"]

        if not os.path.exists(dataset_path):
            print(f"Warning: dataset path not found for {dataset_name}: {dataset_path}")
            row = {
                "dataset": dataset_name,
                "split": split,
                "num_images": np.nan,
                "num_bboxes": np.nan,
                "bbox_size_mean": np.nan,
                "bbox_size_min": np.nan,
                "bbox_size_max": np.nan,
                "bbox_size_std": np.nan,
                "bboxes_per_image_mean": np.nan,
                "bboxes_per_image_min": np.nan,
                "bboxes_per_image_max": np.nan,
                "bboxes_per_image_std": np.nan,
            }
            rows.append(row)
            continue

        try:
            stats = compute_dataset_bbox_stats(dataset_path, split)
            row = {"dataset": dataset_name, "split": split}
            row.update(stats)
            rows.append(row)
        except ValueError as exc:
            print(f"Warning: could not analyze {dataset_name} ({dataset_path}): {exc}")
            row = {
                "dataset": dataset_name,
                "split": split,
                "num_images": np.nan,
                "num_bboxes": np.nan,
                "bbox_size_mean": np.nan,
                "bbox_size_min": np.nan,
                "bbox_size_max": np.nan,
                "bbox_size_std": np.nan,
                "bboxes_per_image_mean": np.nan,
                "bboxes_per_image_min": np.nan,
                "bboxes_per_image_max": np.nan,
                "bboxes_per_image_std": np.nan,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a CSV + LaTeX table with bbox stats for selected datasets."
    )
    parser.add_argument(
        "--output_csv",
        default="stats/other_datasets_stats_table.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--flatbug_path",
        default="datasets(others)/flatbug-yolo-split",
        help="Path to the flatbug dataset root.",
    )
    parser.add_argument(
        "--ood_path",
        default="datasets(others)/OOD-split",
        help="Path to the OOD dataset root.",
    )
    parser.add_argument(
        "--lepinoc_path",
        default="datasets(others)/Lepinoc-split",
        help="Path to the Lepinoc dataset root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = build_other_datasets_table(
        flatbug_path=args.flatbug_path,
        ood_path=args.ood_path,
        lepinoc_path=args.lepinoc_path,
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"CSV saved to: {args.output_csv}")

    latex_code = dataframe_to_latex(
        df,
        caption="Bounding-box statistics across ArthroNat, flatbug, OOD, Lepinoc, and SPIPOLL.",
        label="tab:other_datasets_bbox_stats",
    )
    print("\nLaTeX table:\n")
    print(latex_code)


if __name__ == "__main__":
    main()