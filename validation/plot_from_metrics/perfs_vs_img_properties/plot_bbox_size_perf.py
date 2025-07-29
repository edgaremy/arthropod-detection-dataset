import pandas as pd
import numpy as np
import ast
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
matplotlib.style.use('ggplot')
from matplotlib.patches import Patch

def plot_bbox_metrics(csv_path, output, log_scale=False, metric='IoU'):
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
    plt.rcParams['font.size'] = 18
    # slightly less black text:
    ratio = '0.2'
    plt.rcParams['text.color'] = ratio 
    plt.rcParams['xtick.color'] = ratio
    plt.rcParams['ytick.color'] = ratio
    plt.rcParams['axes.labelcolor'] = ratio

    df = pd.read_csv(csv_path)
    IoUs_with_zeros = df['IoUs_with_zeros'].map(ast.literal_eval)
    bbox_relative_sizes = df['bbox_sizes'].map(ast.literal_eval)
    F1 = df['F1']

    # Handle metrics differently
    if metric == 'IoU':
        # Flatten the lists
        IoUs_with_zeros = [item for sublist in IoUs_with_zeros for item in sublist]
        bbox_relative_sizes = [item for sublist in bbox_relative_sizes for item in sublist]
        metric_values = IoUs_with_zeros
    elif metric == 'F1':
        # Use average bbox size per image
        bbox_relative_sizes = [np.mean(sublist) if len(sublist) > 0 else 0 for sublist in bbox_relative_sizes]
        metric_values = F1
    else:
        raise ValueError("Metric must be either 'IoU' or 'F1'")

    # Sort based on the size of the bounding box
    sorted_indices = np.argsort(bbox_relative_sizes)
    metric_values = np.array(metric_values)[sorted_indices]
    bbox_relative_sizes = np.array(bbox_relative_sizes)[sorted_indices]

    # Plot IoU performance depending on the size of the bounding box
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a DataFrame for seaborn
    plot_df = pd.DataFrame({
        'bbox_size': bbox_relative_sizes,
        'metric': metric_values
    })
    
    # Plot the scatter points
    ax.scatter(bbox_relative_sizes, metric_values, alpha=0.5, s=25, color='#3b5998')  # Jean blue color
    
    # Calculate Spearman correlation
    res = pg.corr(x=bbox_relative_sizes, y=metric_values, method='spearman')
    print(res)
    rho = res['r'].iloc[0]
    p_value = res['p-val'].iloc[0]
    
    # Plot regression line with confidence interval using seaborn
    sns.regplot(x='bbox_size', y='metric', data=plot_df, 
                scatter=False, ci=95, line_kws={'color': 'red', 'linewidth': 2.5},
                ax=ax, label=f'Spearman rho = {rho:.2f}, p-value = {p_value:.2f}')
    
    if metric == 'IoU':
        ax.set_xlabel('Arthropod relative size')
        ax.set_ylabel('IoU')
    elif metric == 'F1':
        ax.set_xlabel('Arthropod relative size (averaged per image)')
        ax.set_ylabel('F1-score')

    # Remove borders of the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    if log_scale:
        ax.set_xscale('log')
        # replace 10^2 with 1%, 10^1 with 10%, 10^0 with 100%
        ax.set_xticks([0.01, 0.1, 1])
        ax.set_xticklabels(['1%', '10%', '100%'])
    else:
        ax.set_xscale('linear')
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Make the plot look nicer
    ax.set_facecolor('#f0f0f0') # grey background
    ax.grid(which='major', linestyle='-', linewidth='2.2', color='white') # thicker white grid lines
    ax.set_axisbelow(True) # grid lines are behind the plot
    ax.tick_params(which='both', direction='in', length=0) # remove small ticks next to the numbers
    ax.tick_params(axis='both', which='major', pad=10) # number further from the axis
    
    # Add the red line, CI patch, and correlation text to the legend
    line_patch = plt.Line2D([0], [0], color='red', linestyle='-', linewidth=2.5, label='Regression line')
    ci_patch = Patch(color='red', alpha=0.2, label='95% CI')
    ax.legend([line_patch, ci_patch, plt.Line2D([0], [0], linestyle='none')], 
              ['Regression line', '95% CI', f'Spearman rho = {rho:.2f}\np-value = {p_value:.2f}'],
              loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output)

# Example usage

# YOLO11l:
plot_bbox_metrics(csv_path = 'validation/metrics/validation_conf0.413yolo11l.csv',
                  output = 'validation/plot_from_metrics/perfs_vs_img_properties/plots/bbox_size_IoU_11l.png',
                  log_scale = False, metric='IoU')
plot_bbox_metrics(csv_path = 'validation/metrics/validation_conf0.413yolo11l.csv',
                  output = 'validation/plot_from_metrics/perfs_vs_img_properties/plots/bbox_size_F1_11l.png',
                  log_scale = False, metric='F1')

# YOLO11n:
plot_bbox_metrics(csv_path = 'validation/metrics/validation_conf0.437yolo11n.csv',
                  output = 'validation/plot_from_metrics/perfs_vs_img_properties/plots/bbox_size_IoU_11n.png',
                  log_scale = False, metric='IoU')
plot_bbox_metrics(csv_path = 'validation/metrics/validation_conf0.437yolo11n.csv',
                  output = 'validation/plot_from_metrics/perfs_vs_img_properties/plots/bbox_size_F1_11n.png',
                  log_scale = False, metric='F1')

