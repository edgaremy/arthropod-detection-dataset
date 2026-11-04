import pandas as pd
import ast
import os

def calculate_metrics(group):
    # Per-image averaging: use pre-computed per-image metrics from validation CSV
    metrics = {
        'F1': group['F1'].mean(),
        'precision': group['precision'].mean(),
        'recall': group['recall'].mean(),
        'mean_IoU': group['mean_IoU'].mean(),
    }
    
    # Add uncertainty: standard deviation across images
    metrics.update({
        'std_F1': group['F1'].std(),
        'std_precision': group['precision'].std(),
        'std_recall': group['recall'].std(),
        'std_mean_IoU': group['mean_IoU'].std(),
    })
    
    # Sample size (number of images in this taxonomic group)
    metrics['count'] = len(group)
    
    return pd.Series(metrics)

def hierarchical_benchmark(csv_path, blacklist=None):
    df = pd.read_csv(csv_path)
    
    # Remove blacklisted entries for all levels
    if blacklist is not None:
        for level, items in blacklist.items():
            df = df[~df[level].isin(items)]

    results = {}
    for level in ['class', 'order', 'family', 'genus', 'specie']:
        grouped = df.groupby(level, group_keys=False).apply(calculate_metrics, include_groups=False).reset_index()
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
    
    # Reorder columns for better readability
    column_order = [
        level, 'count', 
        'F1', 'std_F1', 
        'precision', 'std_precision', 
        'recall', 'std_recall', 
        'mean_IoU', 'std_mean_IoU'
    ]
    
    # Only include columns that exist
    column_order = [c for c in column_order if c in data.columns]
    data = data[column_order]
    
    # Fill NaN std with 0.00 (for groups with only 1 image)
    std_columns = ['std_F1', 'std_precision', 'std_recall', 'std_mean_IoU']
    for col in std_columns:
        if col in data.columns:
            data[col] = data[col].fillna(0.00)
    
    # Round numeric values for cleaner output
    numeric_columns = ['F1', 'std_F1', 'precision', 'std_precision', 
                       'recall', 'std_recall', 'mean_IoU', 'std_mean_IoU']
    numeric_columns = [c for c in numeric_columns if c in data.columns]
    data[numeric_columns] = data[numeric_columns].round(4)
    
    # Sort the data by count (number of images) in descending order
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
    
    # Export each taxonomic level (class, order, and all others for supplementary)
    for level in ['class', 'order', 'family', 'genus', 'specie']:
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
