import pandas as pd
import os
import math


def _display_metric_name(col_name: str) -> str:
    """Convert column name like 'avg_F1' or 'avg_mean_IoU' to a pretty header.
    Examples:
      'avg_F1' -> 'Avg F1'
      'avg_mean_IoU' -> 'Avg mean IoU'
      'overall_F1' -> 'Overall F1'
    """
    parts = col_name.split('_')
    parts = [p.capitalize() if i == 0 else p for i, p in enumerate(parts)]
    return ' '.join(parts)


def _generate_subtable(df_filtered, model_display_names, model_keys, metric_display_names, 
                      metric_keys, test_dataset_display_names, dataset_key, float_fmt):
    """Helper function to generate a subtable for juxtaposed layout."""
    
    # Filter rows by model_names while preserving the requested order
    lookup = {m: i for i, m in enumerate(model_keys)}
    df_filtered = df_filtered[df_filtered['model_name'].isin(model_keys)].copy()
    # Keep rows in the order of model_names
    df_filtered['__order'] = df_filtered['model_name'].map(lookup)
    df_filtered = df_filtered.sort_values('__order').reset_index(drop=True)
    df_filtered.drop(columns='__order', inplace=True)

    if df_filtered.empty:
        return ""

    # Validate metrics
    for m in metric_keys:
        if m not in df_filtered.columns:
            raise ValueError(f"Metric column '{m}' not found in CSV")

    # Compute best per metric (for bolding). Use numeric comparison; ignore NaNs
    best_values = {}
    for m in metric_keys:
        col = pd.to_numeric(df_filtered[m], errors='coerce')
        if col.isnull().all():
            best_values[m] = None
        else:
            best_values[m] = col.max()

    # Build LaTeX lines for subtable
    latex = []
    
    # Subtable title
    dataset_display = test_dataset_display_names.get(dataset_key, dataset_key)
    latex.append(f"\\textbf{{{dataset_display}}}")
    latex.append(r"\vspace{0.2em}")
    
    # Column spec: left for model name, then centered for metrics
    col_spec = 'l' + 'c' * len(metric_keys)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")

    # Header
    headers = [r"\thead{\textbf{Scenario}}"]
    for m in metric_keys:
        headers.append(rf"\thead{{\textbf{{{metric_display_names[m]}}}}}")
    latex.append(' & '.join(headers) + r" \\")
    latex.append(r"\midrule")

    # Rows
    for idx, row in df_filtered.iterrows():
        # Add row color for alternating rows (every other row starting from the first data row)
        row_color = r"\rowcolor{gray!10} " if idx % 2 == 0 else ""
        
        parts = []
        parts.append(model_display_names[row['model_name']])

        for m in metric_keys:
            val = row[m]
            # Try numeric
            try:
                num = float(val)
                if math.isnan(num):
                    s = ''
                else:
                    s = f"{num:.{float_fmt}f}"
                    # Bold if equals best (within tolerance)
                    if best_values[m] is not None and abs(num - best_values[m]) < 1e-6:
                        s = rf"\textbf{{{s}}}"
            except Exception:
                s = str(val)
            parts.append(s)

        latex.append(row_color + ' & '.join(parts) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")

    return '\n'.join(latex)


def _generate_single_table(df_filtered, model_display_names, model_keys, metric_display_names, 
                          metric_keys, test_dataset_display_names, caption, label, dataset_key, 
                          float_fmt, csv_path, table_index):
    """Helper function to generate a single table for split dataset mode."""
    
    # Filter rows by model_names while preserving the requested order
    lookup = {m: i for i, m in enumerate(model_keys)}
    df_filtered = df_filtered[df_filtered['model_name'].isin(model_keys)].copy()
    # Keep rows in the order of model_names
    df_filtered['__order'] = df_filtered['model_name'].map(lookup)
    df_filtered = df_filtered.sort_values('__order').reset_index(drop=True)
    df_filtered.drop(columns='__order', inplace=True)

    if df_filtered.empty:
        return ""

    # Validate metrics
    for m in metric_keys:
        if m not in df_filtered.columns:
            raise ValueError(f"Metric column '{m}' not found in CSV")

    # Compute best per metric (for bolding). Use numeric comparison; ignore NaNs
    best_values = {}
    for m in metric_keys:
        col = pd.to_numeric(df_filtered[m], errors='coerce')
        if col.isnull().all():
            best_values[m] = None
        else:
            best_values[m] = col.max()

    # Build LaTeX lines
    latex = []
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")

    # Build caption and label for this specific dataset
    if caption is None:
        dataset_display = test_dataset_display_names.get(dataset_key, dataset_key)
        table_caption = f"Model comparison on {dataset_display}"
    else:
        table_caption = caption
    
    if label is None:
        base = os.path.basename(csv_path).replace('.csv', '') if csv_path else "comparison"
        table_label = f"tab:{base}_{dataset_key}"
    else:
        table_label = f"{label}_{dataset_key}"

    latex.append(f"\\caption{{{table_caption}}}")
    latex.append(f"\\label{{{table_label}}}")
    latex.append(r"\footnotesize")

    # Column spec: left for model name, then centered for metrics (no dataset column for split tables)
    col_spec = 'l' + 'c' * len(metric_keys)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")

    # Header (no dataset column for split tables)
    headers = [r"\thead{\textbf{Scenario}}"]
    for m in metric_keys:
        headers.append(rf"\thead{{\textbf{{{metric_display_names[m]}}}}}")
    latex.append(' & '.join(headers) + r" \\")
    latex.append(r"\midrule")

    # Rows
    for idx, row in df_filtered.iterrows():
        # Add row color for alternating rows (every other row starting from the first data row)
        row_color = r"\rowcolor{gray!10} " if idx % 2 == 0 else ""
        
        parts = []
        parts.append(model_display_names[row['model_name']])
        # No dataset column for split tables

        for m in metric_keys:
            val = row[m]
            # Try numeric
            try:
                num = float(val)
                if math.isnan(num):
                    s = ''
                else:
                    s = f"{num:.{float_fmt}f}"
                    # Bold if equals best (within tolerance)
                    if best_values[m] is not None and abs(num - best_values[m]) < 1e-6:
                        s = rf"\textbf{{{s}}}"
            except Exception:
                s = str(val)
            parts.append(s)

        latex.append(row_color + ' & '.join(parts) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return '\n'.join(latex)


def generate_model_comparison_table(csv_path, model_names, metrics, test_datasets=None, output_path=None, caption=None, label=None, float_fmt=2, split_by_dataset=False, combined=False):
    """
    Generate a LaTeX table comparing models (scenarios) from a metrics CSV.

    Args:
        csv_path (str): Path to the CSV file containing metrics.
        model_names (list[str] or dict[str,str]): List of model_name values (rows) to include, order preserved.
                                                  If dict, keys are CSV column values, values are display names.
        metrics (list[str] or dict[str,str]): List of metric column names to include (must exist in CSV).
                                              If dict, keys are CSV column names, values are display names.
        test_datasets (list[str] or dict[str,str] or None): List/dict of test_dataset values to include. If None, all are included.
                                                           If dict, keys are CSV column values, values are display names.
        output_path (str|None): If provided, the generated LaTeX code is saved to this file.
                                If None, the LaTeX is printed to stdout and returned as string.
        caption (str|None): Optional caption (if None a default is built).
        label (str|None): Optional LaTeX label (if None a default is built).
        float_fmt (int): Number of decimals for formatting metric values (default 2).
        split_by_dataset (bool): If True and multiple test datasets are provided, generate separate tables for each dataset.
        combined (bool): If True with split_by_dataset, create one table with grouped columns instead of completely separate tables.

    Returns:
        str: The LaTeX table code.
    """
    df = pd.read_csv(csv_path)

    # Handle model_names parameter (list or dict)
    if isinstance(model_names, dict):
        model_display_names = model_names
        model_keys = list(model_names.keys())
    else:
        model_keys = model_names
        model_display_names = {name: name for name in model_names}
    
    # Handle metrics parameter (list or dict)
    if isinstance(metrics, dict):
        metric_display_names = metrics
        metric_keys = list(metrics.keys())
    else:
        metric_keys = metrics
        metric_display_names = {metric: _display_metric_name(metric) for metric in metrics}

    # Handle test_datasets parameter (list or dict)
    if test_datasets is not None:
        if isinstance(test_datasets, dict):
            test_dataset_display_names = test_datasets
            test_dataset_keys = list(test_datasets.keys())
        else:
            test_dataset_keys = test_datasets
            test_dataset_display_names = {ds: ds for ds in test_datasets}
    else:
        test_dataset_keys = None
        test_dataset_display_names = {}

    # Filter rows by test_datasets if specified
    if test_dataset_keys is not None:
        if 'test_dataset' not in df.columns:
            raise ValueError("test_datasets parameter provided but 'test_dataset' column not found in CSV")
        df = df[df['test_dataset'].isin(test_dataset_keys)].copy()
        if df.empty:
            raise ValueError('No rows matched the provided test_datasets')

    # If split_by_dataset is True and we have multiple test datasets, generate separate tables
    if split_by_dataset and test_dataset_keys is not None and len(test_dataset_keys) > 1:
        if combined:
            # Generate one big table with grouped columns for each dataset
            # Filter and prepare data for all datasets
            dataset_data = {}
            for dataset_key in test_dataset_keys:
                df_single_dataset = df[df['test_dataset'] == dataset_key].copy()
                if not df_single_dataset.empty:
                    # Filter rows by model_names while preserving the requested order
                    lookup = {m: i for i, m in enumerate(model_keys)}
                    df_filtered = df_single_dataset[df_single_dataset['model_name'].isin(model_keys)].copy()
                    df_filtered['__order'] = df_filtered['model_name'].map(lookup)
                    df_filtered = df_filtered.sort_values('__order').reset_index(drop=True)
                    df_filtered.drop(columns='__order', inplace=True)
                    dataset_data[dataset_key] = df_filtered
            
            if not dataset_data:
                raise ValueError('No valid data for any dataset')
            
            # Compute best values per metric per dataset (for bolding)
            best_values_per_dataset = {}
            for dataset_key, df_data in dataset_data.items():
                best_values = {}
                for m in metric_keys:
                    if m in df_data.columns:
                        col = pd.to_numeric(df_data[m], errors='coerce')
                        if col.isnull().all():
                            best_values[m] = None
                        else:
                            best_values[m] = col.max()
                best_values_per_dataset[dataset_key] = best_values
            
            # Build the main table with grouped columns
            latex = []
            latex.append(r"\begin{table}[H]")
            latex.append(r"\centering")
            
            # Caption and label
            if caption is None:
                dataset_display_list = [test_dataset_display_names.get(ds, ds) for ds in test_dataset_keys]
                caption = f"Comparison of the model's average performance on the {' and '.join(dataset_display_list)} test sets, depending on training scenario. The mean IoU is computed for each image and then averaged."
            if label is None:
                base = os.path.basename(csv_path).replace('.csv', '')
                label = f"tab:{base}_comparison"
            
            latex.append(f"\\caption{{{caption}}}")
            latex.append(f"\\label{{{label}}}")
            latex.append(r"\footnotesize")
            
            # Column spec: left for model name, then centered columns for each dataset's metrics with vertical separators
            num_datasets = len(test_dataset_keys)
            num_metrics = len(metric_keys)
            # Add vertical separators between column groups
            col_spec = 'l'
            for i in range(num_datasets):
                if i > 0:
                    col_spec += '|'  # Add vertical separator before each new dataset group
                col_spec += 'c' * num_metrics
            latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
            latex.append(r"\toprule")
            
            # First header row: dataset names spanning their metric columns
            header_row1 = [r"\multirow{2}{*}{\thead{\textbf{Scenario}}}"]
            for dataset_key in test_dataset_keys:
                dataset_display = test_dataset_display_names.get(dataset_key, dataset_key)
                header_row1.append(f"\\multicolumn{{{num_metrics}}}{{c}}{{\\textbf{{{dataset_display}}}}}")
            latex.append(' & '.join(header_row1) + r" \\")
            
            # Second header row: metric names for each dataset
            header_row2 = [""]  # Empty cell under Scenario column (multirow continues)
            for dataset_key in test_dataset_keys:
                for m in metric_keys:
                    header_row2.append(rf"\thead{{\textbf{{{metric_display_names[m]}}}}}")
            latex.append(' & '.join(header_row2) + r" \\")
            latex.append(r"\midrule")
            
            # Data rows
            for idx, model_key in enumerate(model_keys):
                # Add row color for alternating rows
                row_color = r"\rowcolor{gray!10} " if idx % 2 == 0 else ""
                
                parts = [model_display_names[model_key]]
                
                # Add metric values for each dataset
                for dataset_key in test_dataset_keys:
                    df_data = dataset_data.get(dataset_key)
                    best_values = best_values_per_dataset.get(dataset_key, {})
                    
                    # Find the row for this model in this dataset
                    model_row = df_data[df_data['model_name'] == model_key] if df_data is not None else None
                    
                    for m in metric_keys:
                        if model_row is not None and not model_row.empty and m in model_row.columns:
                            val = model_row.iloc[0][m]
                            try:
                                num = float(val)
                                if math.isnan(num):
                                    s = ''
                                else:
                                    s = f"{num:.{float_fmt}f}"
                                    # Bold if equals best within this dataset
                                    if best_values.get(m) is not None and abs(num - best_values[m]) < 1e-6:
                                        s = rf"\textbf{{{s}}}"
                            except Exception:
                                s = str(val) if not pd.isna(val) else ''
                        else:
                            s = ''  # No data for this model-dataset-metric combination
                        parts.append(s)
                
                latex.append(row_color + ' & '.join(parts) + r" \\")
            
            latex.append(r"\bottomrule")
            latex.append(r"\end{tabular}")
            latex.append(r"\end{table}")
            
            final_latex = '\n'.join(latex)
        else:
            # Generate completely separate tables
            all_latex_tables = []
            for i, dataset_key in enumerate(test_dataset_keys):
                # Create single-dataset subset
                df_single_dataset = df[df['test_dataset'] == dataset_key].copy()
                if df_single_dataset.empty:
                    continue
                    
                # Generate table for this dataset
                all_latex_tables.append(_generate_single_table(
                    df_single_dataset, model_display_names, model_keys, metric_display_names, 
                    metric_keys, test_dataset_display_names, caption, label, dataset_key, 
                    float_fmt, csv_path, i
                ))
            
            final_latex = '\n\n'.join(all_latex_tables)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(final_latex)
            print(f"Wrote LaTeX tables to {output_path}")
        else:
            print(final_latex)
        return final_latex

    # Filter rows by model_names while preserving the requested order
    lookup = {m: i for i, m in enumerate(model_keys)}
    df_filtered = df[df['model_name'].isin(model_keys)].copy()
    # Keep rows in the order of model_names
    df_filtered['__order'] = df_filtered['model_name'].map(lookup)
    df_filtered = df_filtered.sort_values('__order').reset_index(drop=True)
    df_filtered.drop(columns='__order', inplace=True)

    if df_filtered.empty:
        raise ValueError('No rows matched the provided model_names')

    # Validate metrics
    for m in metric_keys:
        if m not in df_filtered.columns:
            raise ValueError(f"Metric column '{m}' not found in CSV")

    # Compute best per metric (for bolding). Use numeric comparison; ignore NaNs
    best_values = {}
    for m in metric_keys:
        col = pd.to_numeric(df_filtered[m], errors='coerce')
        if col.isnull().all():
            best_values[m] = None
        else:
            best_values[m] = col.max()

    # Build LaTeX lines
    latex = []
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")

    # Default caption/label
    if caption is None:
        # try to include dataset info if present
        datasets = df_filtered['test_dataset'].unique() if 'test_dataset' in df_filtered.columns else []
        if len(datasets) > 0:
            # Use display names for datasets in caption
            dataset_display_list = [test_dataset_display_names.get(ds, ds) for ds in datasets]
            ds_text = f" on {', '.join(dataset_display_list)}"
        else:
            ds_text = ''
        caption = f"Model comparison{ds_text}"
    if label is None:
        base = os.path.basename(csv_path).replace('.csv', '')
        label = f"tab:{base}"

    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(r"\footnotesize")

    # Determine if we should show test_dataset column
    # Only show as column if there are multiple test datasets
    datasets = df_filtered['test_dataset'].unique() if 'test_dataset' in df_filtered.columns else []
    show_dataset_column = len(datasets) > 1

    # Column spec: left for model name, left for dataset (if multiple datasets), then centered for metrics
    col_spec = 'l' + ('l' if show_dataset_column else '') + 'c' * len(metric_keys)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")

    # Header
    headers = [r"\thead{\textbf{Scenario}}"]
    if show_dataset_column:
        headers.append(r"\thead{\textbf{Test dataset}}")
    for m in metric_keys:
        headers.append(rf"\thead{{\textbf{{{metric_display_names[m]}}}}}")
    latex.append(' & '.join(headers) + r" \\")
    latex.append(r"\midrule")

    # Rows
    for idx, row in df_filtered.iterrows():
        # Add row color for alternating rows (every other row starting from the first data row)
        row_color = r"\rowcolor{gray!10} " if idx % 2 == 0 else ""
        
        parts = []
        parts.append(model_display_names[row['model_name']])
        if show_dataset_column:
            # Use display name for dataset if available
            dataset_value = row['test_dataset']
            dataset_display = test_dataset_display_names.get(dataset_value, dataset_value)
            parts.append(str(dataset_display))

        for m in metric_keys:
            val = row[m]
            # Try numeric
            try:
                num = float(val)
                if math.isnan(num):
                    s = ''
                else:
                    s = f"{num:.{float_fmt}f}"
                    # Bold if equals best (within tolerance)
                    if best_values[m] is not None and abs(num - best_values[m]) < 1e-6:
                        s = rf"\textbf{{{s}}}"
            except Exception:
                s = str(val)
            parts.append(s)

        latex.append(row_color + ' & '.join(parts) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_code = '\n'.join(latex)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_code)
        print(f"Wrote LaTeX table to {output_path}")
    else:
        print(latex_code)

    return latex_code


if __name__ == '__main__':
    # Example usage using the attached sample CSV
    csv_path = 'validation/metrics/model_comparison(all).csv'
    
    # Example 1: Using lists (original behavior)
    scenarios_list = [
        'arthro_mosaic2x2',
        'flatbug',
        'arthro+flatbug',
        'arthro_mosaic3x3',
        'arthro_mosaic4x4',
        'arthro_no_mosaic',
        'arthro+flatbug_mosaic4x4',
        'arthro_mosaic6x6'
    ]
    metrics_list = ['avg_F1', 'avg_precision', 'avg_recall', 'avg_mean_IoU']
    
    # Example 2: Using dictionaries with custom display names
    scenarios_dict = {
        'arthro_mosaic2x2': 'ArthroNat',
        'flatbug': 'flatbug',
        'arthro+flatbug': 'ArthroNat+flatbug',
        'arthro_mosaic3x3': 'ArthroNat 3x3 Mosaic',
        'arthro_mosaic4x4': 'ArthroNat 4x4 Mosaic',
        'arthro_no_mosaic': 'ArthroNat no Mosaic',
    }
    
    metrics_dict = {
        'avg_F1': 'F1-Score',
        'avg_precision': 'Precision',
        'avg_recall': 'Recall',
        'avg_mean_IoU': 'Mean IoU'
    }
    
    # Test datasets with custom display names
    test_datasets_multiple = {
        'arthro': 'ArthroNat',
        'flatbug': 'flatbug'
    }

    print("\n\n=== Completely separate tables ===")
    generate_model_comparison_table(csv_path, scenarios_dict, metrics_dict, 
                                  test_datasets=test_datasets_multiple, 
                                  split_by_dataset=True, combined=False)
    
    print("\n\n=== Combined table with grouped columns ===")
    generate_model_comparison_table(csv_path, scenarios_dict, metrics_dict, 
                                  test_datasets=test_datasets_multiple, 
                                  split_by_dataset=True, combined=True)
