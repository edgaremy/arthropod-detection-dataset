import pandas as pd
import os
from pathlib import Path

def compare_scenarios(scenarios, csv_folder_base, level='class', metric='F1', output_path=None):
    """
    Compare multiple scenarios across a specific taxonomic level for a given metric.
    
    Args:
        scenarios: List of scenario names OR dict mapping folder names to display names
                  e.g., ['arthro', 'flatbug'] or {'arthro': 'Arthropod', 'flatbug': 'Flatbug'}
        csv_folder_base: Base directory containing scenario subfolders with CSV files
        level: Taxonomic level to compare ('class', 'order', 'family', 'genus', 'specie')
        metric: Metric to compare ('F1', 'precision', 'recall', 'mean_IoU')
        output_path: Path to save the comparison CSV (optional)
    
    Returns:
        DataFrame with comparison results
    """
    
    # Convert list to dict if needed (folder_name -> display_name)
    if isinstance(scenarios, list):
        scenarios = {s: s for s in scenarios}
    
    # Validate inputs
    valid_levels = ['class', 'order', 'family', 'genus', 'specie']
    valid_metrics = ['F1', 'precision', 'recall', 'mean_IoU']
    
    if level not in valid_levels:
        raise ValueError(f"Level must be one of {valid_levels}")
    if metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics}")
    
    print(f"\n{'='*70}")
    print(f"COMPARING SCENARIOS AT {level.upper()} LEVEL FOR {metric.upper()}")
    print(f"{'='*70}\n")
    
    # Load all scenario data
    scenario_data = {}
    count_data = None  # Store count data from first scenario
    for folder_name, display_name in scenarios.items():
        csv_path = os.path.join(csv_folder_base, folder_name, f'metrics_{level}_{folder_name}.csv')
        csv_path = os.path.join(csv_folder_base, folder_name, f'metrics_{level}_{folder_name}.csv')
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found for scenario '{display_name}' (folder: {folder_name}) at {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        # Store only the taxon name and the metric of interest, using display name as key
        scenario_data[display_name] = df[[level, metric]].set_index(level)
        
        # Store count data from the first scenario
        if count_data is None:
            count_data = df[[level, 'count']].set_index(level)
        
        print(f"Loaded {len(df)} {level}s for scenario: {display_name} (folder: {folder_name})")
    
    if not scenario_data:
        print("Error: No valid scenario data found!")
        return None
    
    # Merge all scenarios on the taxonomic level
    comparison_df = pd.concat(scenario_data.values(), axis=1, keys=scenario_data.keys())
    
    # Flatten column names (scenario_metric format)
    comparison_df.columns = [f'{scenario}_{metric}' for scenario in scenario_data.keys()]
    
    # Reset index to make the taxonomic level a column
    comparison_df = comparison_df.reset_index()
    
    # Add number_of_images column from count data
    if count_data is not None:
        count_data_reset = count_data.reset_index()
        count_data_reset = count_data_reset.rename(columns={'count': 'number_of_images'})
        comparison_df = pd.merge(comparison_df, count_data_reset, on=level, how='left')
    
    # Add summary statistics
    metric_columns = [col for col in comparison_df.columns if col != level and col != 'number_of_images']
    comparison_df['mean'] = comparison_df[metric_columns].mean(axis=1)
    comparison_df['std'] = comparison_df[metric_columns].std(axis=1)
    comparison_df['min'] = comparison_df[metric_columns].min(axis=1)
    comparison_df['max'] = comparison_df[metric_columns].max(axis=1)
    comparison_df['range'] = comparison_df['max'] - comparison_df['min']
    
    # Find best scenario for each taxon
    comparison_df['best_scenario'] = comparison_df[metric_columns].idxmax(axis=1).str.replace(f'_{metric}', '')
    comparison_df['best_value'] = comparison_df[metric_columns].max(axis=1)
    
    # Round numeric values
    numeric_cols = [col for col in comparison_df.columns if col != level and col != 'best_scenario' and col != 'number_of_images']
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(4)
    
    # Reorder columns to put number_of_images right after the taxonomic level
    if 'number_of_images' in comparison_df.columns:
        cols = [level, 'number_of_images'] + [col for col in comparison_df.columns if col not in [level, 'number_of_images']]
        comparison_df = comparison_df[cols]
    
    # Sort by number_of_images (descending) - do this last
    if 'number_of_images' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('number_of_images', ascending=False).reset_index(drop=True)
    
    # Save to CSV if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        print(f"\nComparison saved to: {output_path}")
    
    # Print summary
    print(f"\n{level.upper()} LEVEL COMPARISON SUMMARY:")
    print(f"  Total {level}s compared: {len(comparison_df)}")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Metric: {metric}")
    print(f"\nBest scenario counts:")
    best_counts = comparison_df['best_scenario'].value_counts()
    for scenario, count in best_counts.items():
        print(f"  {scenario}: {count} {level}s ({count/len(comparison_df)*100:.1f}%)")
    
    print(f"\nOverall statistics across all {level}s:")
    for scenario in scenario_data.keys():
        col_name = f'{scenario}_{metric}'
        if col_name in comparison_df.columns:
            mean_val = comparison_df[col_name].mean()
            print(f"  {scenario}: mean {metric} = {mean_val:.4f}")
    
    print(f"\n{'='*70}\n")
    
    return comparison_df

def compare_multiple_levels(scenarios, csv_folder_base, levels=None, metrics=None, output_dir='validation/hierarchical_metrics/comparisons'):
    """
    Compare scenarios across multiple taxonomic levels and metrics.
    
    Args:
        scenarios: List of scenario names OR dict mapping folder names to display names
                  e.g., ['arthro', 'flatbug'] or {'arthro': 'Arthropod', 'flatbug': 'Flatbug'}
        csv_folder_base: Base directory containing scenario subfolders
        levels: List of taxonomic levels to compare (default: ['class', 'order'])
        metrics: List of metrics to compare (default: ['F1', 'precision', 'recall', 'mean_IoU'])
        output_dir: Directory to save comparison CSV files
    """
    
    # Convert list to dict if needed
    if isinstance(scenarios, list):
        scenarios = {s: s for s in scenarios}
    
    if levels is None:
        levels = ['class', 'order']
    if metrics is None:
        metrics = ['F1', 'precision', 'recall', 'mean_IoU']
    
    print(f"\n{'#'*70}")
    print(f"MULTI-LEVEL SCENARIO COMPARISON")
    print(f"Scenarios: {', '.join(scenarios.values())}")
    print(f"Levels: {', '.join(levels)}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"{'#'*70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for level in levels:
        for metric in metrics:
            output_path = os.path.join(output_dir, f'comparison_{level}_{metric}.csv')
            
            comparison_df = compare_scenarios(
                scenarios=scenarios,
                csv_folder_base=csv_folder_base,
                level=level,
                metric=metric,
                output_path=output_path
            )
            
            if comparison_df is not None:
                results[f'{level}_{metric}'] = comparison_df
    
    # Create a summary of summaries
    summary_rows = []
    for key, df in results.items():
        level, metric = key.split('_', 1)
        
        # Get display names from scenarios dict
        display_names = scenarios.values() if isinstance(scenarios, dict) else scenarios
        for display_name in display_names:
            col_name = f'{display_name}_{metric}'
            if col_name in df.columns:
                summary_rows.append({
                    'level': level,
                    'metric': metric,
                    'scenario': display_name,
                    'mean': df[col_name].mean(),
                    'std': df[col_name].std(),
                    'min': df[col_name].min(),
                    'max': df[col_name].max(),
                    'n_taxa': len(df)
                })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.round(4)
        summary_path = os.path.join(output_dir, 'overall_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*70}")
        print(f"Overall summary saved to: {summary_path}")
        print(f"{'='*70}\n")
    
    return results

# Example usage
scenarios = {
    'arthro': 'ArthroNat',
    'flatbug': 'flatbug',
    'arthro_and_flatbug': 'ArthroNat+flatbug',
    'arthro_mosaic_33': 'ArthroNat mosaic3x3',
    'arthro_mosaic_44': 'ArthroNat mosaic4x4',
    'arthro_nomosaic': 'ArthroNat no mosaic',
}

# Alternative: simple list format (folder names = display names)
# scenarios = [
#     'arthro',
#     'flatbug',
#     'arthro_and_flatbug',
#     'arthro_mosaic_33',
#     'arthro_mosaic_44',
#     'arthro_nomosaic',
# ]

# Base directory containing scenario subfolders
csv_folder_base = 'validation/hierarchical_metrics/csv_tables'

# Output directory for comparison results
output_dir = 'validation/hierarchical_metrics/comparisons'

# Compare across multiple levels and metrics
results = compare_multiple_levels(
    scenarios=scenarios,
    csv_folder_base=csv_folder_base,
    levels=['class', 'order'],
    metrics=['F1', 'mean_IoU'],
    output_dir=output_dir
)
    