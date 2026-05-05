import os
import sys

import pandas as pd


def _pick_scenario_column(df, metric_suffix, scenario_name="ArthroNat"):
    """Pick the metric column for the selected scenario."""
    exact_col = f"{scenario_name}_{metric_suffix}"
    if exact_col in df.columns:
        return exact_col

    # Common alias used by users for the default ArthroNat setup.
    if scenario_name.lower().replace(" ", "") in {"arthronat2x2", "arthronat"}:
        fallback_col = f"ArthroNat_{metric_suffix}"
        if fallback_col in df.columns:
            return fallback_col

    available = [c for c in df.columns if c.endswith(f"_{metric_suffix}")]
    raise ValueError(
        f"Could not find scenario column '{exact_col}'. "
        f"Available {metric_suffix} columns: {available}"
    )


def _build_level_table(level, comparison_dir, scenario_name="ArthroNat"):
    """Load and merge F1 + mean IoU metrics for one taxonomic level."""
    level = level.lower()
    level_col = level

    f1_path = os.path.join(comparison_dir, f"comparison_{level}_F1.csv")
    iou_path = os.path.join(comparison_dir, f"comparison_{level}_mean_IoU.csv")

    f1_df = pd.read_csv(f1_path)
    iou_df = pd.read_csv(iou_path)

    f1_col = _pick_scenario_column(f1_df, "F1", scenario_name=scenario_name)
    iou_col = _pick_scenario_column(iou_df, "mean_IoU", scenario_name=scenario_name)

    f1_subset = f1_df[[level_col, "number_of_images", f1_col]].rename(columns={f1_col: "F1-score"})
    iou_subset = iou_df[[level_col, "number_of_images", iou_col]].rename(columns={iou_col: "mean IoU"})

    merged = pd.merge(
        f1_subset,
        iou_subset,
        on=[level_col, "number_of_images"],
        how="inner",
        validate="one_to_one",
    )

    merged = merged.sort_values("number_of_images", ascending=False).reset_index(drop=True)
    return merged


def generate_latex_table(level, comparison_dir="validation/hierarchical_metrics/comparisons", output_path=None, scenario_name="ArthroNat", max_display_rows=20):
    """
    Generate one LaTeX table for a given taxonomic level using a single scenario.

    Args:
        level: 'class' or 'order'
        comparison_dir: Directory containing comparison CSV files
        output_path: Optional path to save the LaTeX table
        scenario_name: Scenario to select (default: ArthroNat)
        max_display_rows: Maximum number of rows to display. If None, display all. (default: 20)
    """
    level = level.lower()
    if level not in {"class", "order"}:
        raise ValueError("level must be either 'class' or 'order'")

    df = _build_level_table(level, comparison_dir, scenario_name=scenario_name)

    total_rows = len(df)
    if max_display_rows is not None and total_rows > max_display_rows:
        display_df = df.head(max_display_rows).copy()
        caption_suffix = f" (top {max_display_rows} with most images out of {total_rows})"
    else:
        display_df = df.copy()
        caption_suffix = ""

    level_header = level.capitalize()

    latex_lines = []
    latex_lines.append(r"\begin{table}[H]")
    latex_lines.append(r"\centering")
    latex_lines.append(
        f"\\caption{{{level_header}-level F1-score and mean IoU on the ArthroNat test set ({scenario_name}){caption_suffix}}}"
    )
    latex_lines.append(f"\\label{{tab:{level}_single_scenario}}")
    latex_lines.append(r"\footnotesize")
    latex_lines.append(r"\begin{tabular}{lrcc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(
        f"\\thead{{\\textbf{{{level_header}}}}} & "
        r"\thead{\textbf{\# images}} & "
        r"\thead{\textbf{F1-score}} & "
        r"\thead{\textbf{mean IoU}} \\" 
    )
    latex_lines.append(r"\midrule")

    for idx, row in display_df.iterrows():
        row_content = (
            f"{row[level]} & {int(row['number_of_images'])} & "
            f"{row['F1-score']:.2f} & {row['mean IoU']:.2f} \\\\" 
        )
        if idx % 2 == 0:
            row_content = r"\rowcolor{gray!10} " + row_content
        latex_lines.append(row_content)

    latex_lines.append(r"\midrule")

    level_plural = "classes" if level == "class" else "orders"
    avg_f1 = df["F1-score"].mean()
    avg_iou = df["mean IoU"].mean()
    latex_lines.append(
        f"Average on all {total_rows} {level_plural} &  & {avg_f1:.2f} & {avg_iou:.2f} \\\\" 
    )

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_code = "\n".join(latex_lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_code)
        print(f"LaTeX table saved to: {output_path}")
    else:
        print(latex_code)

    return latex_code


def generate_two_tables(comparison_dir="validation/hierarchical_metrics/comparisons", output_dir=None, scenario_name="ArthroNat"):
    """
    Generate three LaTeX tables: class-level (top 20), order-level (top 20), and order-level (full).

    Args:
        comparison_dir: Directory containing comparison CSV files
        output_dir: Optional directory to save .tex files
        scenario_name: Scenario to select (default: ArthroNat)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("GENERATING 3 LATEX TABLES (CLASS + ORDER TOP 20 + ORDER FULL)")
    print(f"{'=' * 70}\n")

    # Table 1: Class-level (top 20)
    print("Processing class-level table (top 20)...")
    if output_dir:
        output_path = os.path.join(output_dir, f"single_class_F1_mean_IoU.tex")
    else:
        output_path = None
    generate_latex_table(
        level="class",
        comparison_dir=comparison_dir,
        output_path=output_path,
        scenario_name=scenario_name,
        max_display_rows=20,
    )
    print()

    # Table 2: Order-level (top 20)
    print("Processing order-level table (top 20)...")
    if output_dir:
        output_path = os.path.join(output_dir, f"single_order_F1_mean_IoU.tex")
    else:
        output_path = None
    generate_latex_table(
        level="order",
        comparison_dir=comparison_dir,
        output_path=output_path,
        scenario_name=scenario_name,
        max_display_rows=20,
    )
    print()

    # Table 3: Order-level (full, un-cropped)
    print("Processing order-level table (full, un-cropped)...")
    if output_dir:
        output_path = os.path.join(output_dir, f"single_order_F1_mean_IoU_full.tex")
    else:
        output_path = None
    generate_latex_table(
        level="order",
        comparison_dir=comparison_dir,
        output_path=output_path,
        scenario_name=scenario_name,
        max_display_rows=None,
    )
    print()

    print(f"{'=' * 70}")
    if output_dir:
        print(f"Generated 3 LaTeX tables in {output_dir}")
    else:
        print("Generated 3 LaTeX tables")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    # Usage:
    #   python print_LaTeX_table_codev2.py
    #   python print_LaTeX_table_codev2.py <output_dir>
    #   python print_LaTeX_table_codev2.py <output_dir> <scenario_name>
    output_dir = sys.argv[1] if len(sys.argv) > 1 else None
    scenario_name = sys.argv[2] if len(sys.argv) > 2 else "ArthroNat"

    generate_two_tables(output_dir=output_dir, scenario_name=scenario_name)
