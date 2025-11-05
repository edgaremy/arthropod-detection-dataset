import pandas as pd
import matplotlib.pyplot as plt
from pypalettes import load_cmap


def plot_model_comparison(csv_path, metric, output, model_selection=None, figsize=(15, 10), bar_width=0.8):
    """
    Plot model comparison across taxonomic hierarchies for a single metric.
    
    Args:
        csv_path (str): Path to the CSV file with model comparison results
        metric (dict): Dictionary with 'csv_name' (column name in CSV) and 'display_name' (for plot title)
        output (str): Output file path for the plot
        model_selection (dict): Dictionary mapping CSV model names to display names. If None, use all models.
        figsize (tuple): Figure size
        bar_width (float): Width of the bars
    """
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
    plt.rcParams['font.size'] = 24
    # slightly less black text:
    ratio = '0.15'
    plt.rcParams['text.color'] = ratio 
    plt.rcParams['xtick.color'] = ratio
    plt.rcParams['ytick.color'] = ratio
    plt.rcParams['axes.labelcolor'] = ratio

    # Load the CSV data
    df = pd.read_csv(csv_path)
    
    # Filter models if model_selection is provided
    if model_selection is not None:
        df = df[df['model_name'].isin(model_selection.keys())]
        # Replace model names with display names
        df['model_name'] = df['model_name'].map(model_selection)
    
    # Create a pivot table with models as columns and test datasets as rows
    pivot_df = df.pivot(index='test_dataset', columns='model_name', values=metric['csv_name'])
    
    # Reorder the datasets to match taxonomic hierarchy
    dataset_order = ['same_species', 'same_genus', 'other_genus', 'other_families']
    dataset_labels = ['Same species', 'Same genus', 'Other genus', 'Other families']
    
    # Filter and reorder the dataframe
    pivot_df = pivot_df.reindex(dataset_order)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom color palette
    colors = load_cmap("Egypt").colors
    # Reorder colors from r b g y to b y g r (indices 1, 3, 2, 0)
    colors = [colors[i] for i in [1, 3, 2, 0]]
    
    # Plot the bar chart
    bars = pivot_df.T.plot(kind='bar', ax=ax, width=bar_width, color=colors[:len(pivot_df.columns)])
    
    # Add value labels on top of each bar
    for i, container in enumerate(ax.containers):
        # Get the values for this test dataset (column in pivot_df)
        dataset_values = pivot_df.iloc[i, :]
        best_value_idx = dataset_values.argmax()
        
        # Apply the labels with individual formatting
        for j, bar in enumerate(container):
            label_text = f'{bar.get_height():.2f}'[2:]
            if j == best_value_idx: # Highlight the best value
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       label_text, ha='center', va='bottom', fontsize=21, weight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       label_text, ha='center', va='bottom', fontsize=21)
    
    ax.set_title(f'{metric["display_name"]} across generalization test sets')
    ax.set_ylim(0.5, 1)
    ax.set_xlabel('Scenario')
    # ax.set_ylabel(metric["display_name"])
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha='right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Set y-axis ticks at 0.1 increments

    # Remove the black rectangle around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Remove small ticks next to the numbers
    ax.tick_params(which='both', direction='in', length=0)
    # Number further from the axis
    ax.tick_params(axis='both', which='major', pad=10)

    plt.tight_layout()
    legend = plt.legend(dataset_labels, loc='lower left', title='Test datasets')
    legend.get_frame().set_alpha(0.9)  # Set higher alpha value for legend background

    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()


# Example usage:
if __name__ == "__main__":
    csv_path = 'validation/generalization/model_comparison.csv'
    
    # Define which models to include and their display names
    model_selection = {
        'arthro_mosaic2x2': 'ArthroNat 2x2',
        'flatbug': 'flatbug',
        'arthro+flatbug': 'ArthroNat+flatbug',
        'arthro_mosaic3x3': 'ArthroNat 3x3',
        'arthro_mosaic4x4': 'ArthroNat 4x4',
        'arthro_no_mosaic': 'ArthroNat No Mosaic',
    }
    
    # Define metrics with their CSV column names and display names
    metrics = {
        'f1': {'csv_name': 'avg_F1', 'display_name': 'Mean F1-Score'},
        'precision': {'csv_name': 'avg_precision', 'display_name': ' Mean Precision'},
        'recall': {'csv_name': 'avg_recall', 'display_name': 'Mean Recall'},
        'iou': {'csv_name': 'avg_mean_IoU', 'display_name': 'Mean IoU'}
    }
    
    # Generate plots for each metric
    for metric_key, metric_info in metrics.items():
        output_path = f'validation/generalization/generalization_{metric_key}.png'
        plot_model_comparison(csv_path, metric_info, output_path, model_selection=model_selection)
        print(f"Generated plot: {output_path}")