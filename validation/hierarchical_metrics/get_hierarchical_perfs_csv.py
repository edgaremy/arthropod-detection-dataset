import pandas as pd
import ast
import os

def calculate_metrics(group):
    TP = group['TP'].sum()
    FP = group['FP'].sum()
    FN = group['FN'].sum()
    group['IoUs'] = group['IoUs'].apply(lambda x: ast.literal_eval(x))
    IoUs = group['IoUs'].sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    mean_IoU = sum(IoUs) / len(IoUs) if len(IoUs) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return pd.Series({
        'precision': precision,
        'recall': recall,
        'mean_IoU': mean_IoU,
        'F1': F1
    })

def hierarchical_benchmark(csv_path, blacklist=None):
    df = pd.read_csv(csv_path)
    
    # Remove blacklisted entries for all levels
    if blacklist is not None:
        for level, items in blacklist.items():
            df = df[~df[level].isin(items)]

    results = {}
    for level in ['class', 'order', 'family', 'genus', 'specie']:
        grouped = df.groupby(level, group_keys=False).apply(calculate_metrics, include_groups=False).reset_index()
        grouped['count'] = df.groupby(level).size().values
        results[level] = grouped
    
    return results

def export_metrics_to_csv(results, level, output_path, sort_by='count', ascending=False):
    """
    Export metrics for a specific taxonomic level to a CSV file.
    
    Args:
        results: Dictionary containing metrics for all levels
        level: Taxonomic level to export ('class', 'order', 'family', 'genus', 'specie')
        output_path: Path to save the CSV file
        sort_by: Column to sort by (default: 'count')
        ascending: Sort order (default: False for descending)
    """
    data = results[level].copy()
    
    # Reorder columns for better readability first
    column_order = [level, 'count', 'F1', 'precision', 'recall', 'mean_IoU']
    data = data[column_order]
    
    # Round numeric values for cleaner output
    numeric_columns = ['F1', 'precision', 'recall', 'mean_IoU']
    data[numeric_columns] = data[numeric_columns].round(4)
    
    # Sort the data by count (number of images) in descending order - do this LAST
    data = data.sort_values(by='count', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    data.to_csv(output_path, index=False)
    print(f"Exported {level} metrics to: {output_path}")
    
    # Print summary statistics
    print(f"\n{level.upper()} LEVEL SUMMARY:")
    print(f"  Total entries: {len(data)}")
    print(f"  Total images: {data['count'].sum()}")
    print(f"  Mean F1: {data['F1'].mean():.4f}")
    print(f"  Mean Precision: {data['precision'].mean():.4f}")
    print(f"  Mean Recall: {data['recall'].mean():.4f}")
    print(f"  Mean IoU: {data['mean_IoU'].mean():.4f}")
    print()

def export_all_metrics(results, output_dir, model_name='model'):
    """
    Export metrics for all taxonomic levels to separate CSV files.
    
    Args:
        results: Dictionary containing metrics for all levels
        output_dir: Directory to save CSV files
        model_name: Name to include in output filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EXPORTING HIERARCHICAL METRICS TO CSV")
    print(f"{'='*60}\n")
    
    # Export each taxonomic level (we only need class and order)
    for level in ['class', 'order']:
        output_path = os.path.join(output_dir, f'metrics_{level}_{model_name}.csv')
        export_metrics_to_csv(results, level, output_path, sort_by='count', ascending=False)
    
    # Create a combined summary CSV with overall statistics
    summary_data = []
    for level in ['class', 'order', 'family', 'genus', 'specie']:
        data = results[level]
        summary_data.append({
            'taxonomic_level': level,
            'n_categories': len(data),
            'total_images': data['count'].sum(),
            'mean_F1': data['F1'].mean(),
            'std_F1': data['F1'].std(),
            'min_F1': data['F1'].min(),
            'max_F1': data['F1'].max(),
            'mean_precision': data['precision'].mean(),
            'mean_recall': data['recall'].mean(),
            'mean_IoU': data['mean_IoU'].mean()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(4)
    summary_path = os.path.join(output_dir, f'summary_{model_name}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Exported summary to: {summary_path}")
    
    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}\n")

# Example usage
scenarios = [
    'arthro',
    'arthro_mosaic_33',
    'arthro_mosaic_44',
    'arthro_nomosaic',
    'arthro_and_flatbug',
    'flatbug',
]

for model_name in scenarios:
    output_dir = 'validation/hierarchical_metrics/csv_tables/' + model_name
    validation_csv = 'validation/metrics/validation_' + model_name + '.csv'
    blacklist = None # or e.g. blacklist = {'class': ['Pauropoda', 'Ostracoda', 'Ichthyostraca']}
    print(f"Loading validation data from: {validation_csv}")
    results = hierarchical_benchmark(validation_csv, blacklist=blacklist)
    export_all_metrics(results, output_dir, model_name=model_name)
