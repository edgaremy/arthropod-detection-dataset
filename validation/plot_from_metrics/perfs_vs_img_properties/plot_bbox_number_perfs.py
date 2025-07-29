import pandas as pd
import numpy as np
import ast
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.style.use('ggplot')
from matplotlib.patches import Patch

def plot_bbox_number_metrics(csv_path, output, log_scale=False, metric='IoU'):
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
    plt.rcParams['font.size'] = 18
    # slightly less black text:
    ratio = '0.2'
    plt.rcParams['text.color'] = ratio 
    plt.rcParams['xtick.color'] = ratio
    plt.rcParams['ytick.color'] = ratio
    plt.rcParams['axes.labelcolor'] = ratio

    df = pd.read_csv(csv_path)
    
    metrics_with_zeros = df['IoUs_with_zeros'].map(ast.literal_eval)
    if metric == 'IoU':
        ylabel = 'Average IoU'
        y_column = 'avg_IoUs'
    elif metric == 'F1':
        ylabel = 'F1-score'
        y_column = 'F1'
    else:
        raise ValueError("metric must be either 'IoU' or 'F1'")
    
    # Calculate number of bboxes per image
    num_bboxes = [len(values) for values in metrics_with_zeros]
    
    # Calculate average metric per image
    if metric == 'IoU':
        avg_metrics = [sum(values)/len(values) if len(values) > 0 else 0 for values in metrics_with_zeros]
    elif metric == 'F1':
        avg_metrics = df['F1']

    # Create a DataFrame for seaborn
    plot_df = pd.DataFrame({
        'num_bboxes': num_bboxes,
        y_column: avg_metrics
    })

    # Plot performance depending on the number of bounding boxes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # SCATTER PLOT
    # Count frequency of each point
    point_counts = {}
    for x, y in zip(num_bboxes, avg_metrics):
        point_counts[(x, y)] = point_counts.get((x, y), 0) + 1
    
    # Scatter with size based on frequency:
    # Apply a logarithmic scaling to prevent extremely large points
    sizes = [20 * np.log2(point_counts.get((x, y), 1) + 1) for x, y in zip(num_bboxes, avg_metrics)]
    ax.scatter(num_bboxes, avg_metrics, alpha=0.5, s=sizes, color='#3b5998')  # Jean blue color
    
    # Calculate Spearman correlation
    X = np.array(num_bboxes)
    y = np.array(avg_metrics)

    # Filter out any NaN values
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]

    # pingouin approach for Spearman correlation
    import pingouin as pg
    res = pg.corr(x=X, y=y, method='spearman')
    print(res)
    rho = res['r'].iloc[0]
    p_value = res['p-val'].iloc[0]

    # Use seaborn for regression with 95% CI
    sns.regplot(
        x='num_bboxes', 
        y=y_column, 
        data=plot_df, 
        scatter=False,
        ax=ax,
        line_kws={'color': 'red', 'linewidth': 2.5},
        ci=95,
        color='red'
    )
    
    ax.set_xlabel('Number of arthropods per image')
    ax.set_ylabel(ylabel)

    # Remove borders of the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    if log_scale:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')
    
    ax.set_ylim(0, 1)

    # Make the plot look nicer
    ax.set_facecolor('#f0f0f0')  # grey background
    ax.grid(which='major', linestyle='-', linewidth='2.2', color='white')  # thicker white grid lines
    ax.set_axisbelow(True)  # grid lines are behind the plot
    ax.tick_params(which='both', direction='in', length=0)  # remove small ticks next to the numbers
    ax.tick_params(axis='both', which='major', pad=10)  # number further from the axis
    
    # Add the red line, CI patch, and correlation text to the legend
    line_patch = plt.Line2D([0], [0], color='red', linestyle='-', linewidth=2.5, label='Regression line')
    ci_patch = Patch(color='red', alpha=0.2, label='95% CI')
    ax.legend([line_patch, ci_patch, plt.Line2D([0], [0], linestyle='none')], 
              ['Regression line', '95% CI', f'Spearman rho = {rho:.2f}\np-value = {p_value:.2f}'],
              loc='upper right')

    plt.tight_layout()
    plt.savefig(output)

# Usage examples
# yolo11n
plot_bbox_number_metrics(csv_path = 'validation/metrics/validation_conf0.437yolo11n.csv',
                  output = 'validation/plot_from_metrics/perfs_vs_img_properties/plots/bbox_number_IoU_log_11n.png',
                  log_scale = False, metric='IoU')
plot_bbox_number_metrics(csv_path = 'validation/metrics/validation_conf0.437yolo11n.csv',
                  output = 'validation/plot_from_metrics/perfs_vs_img_properties/plots/bbox_number_F1_11n.png',
                  log_scale = False, metric='F1')

# yolo11l
plot_bbox_number_metrics(csv_path = 'validation/metrics/validation_conf0.413yolo11l.csv',
                  output = 'validation/plot_from_metrics/perfs_vs_img_properties/plots/bbox_number_IoU_11l.png',
                  log_scale = False, metric='IoU')
plot_bbox_number_metrics(csv_path = 'validation/metrics/validation_conf0.413yolo11l.csv',
                  output = 'validation/plot_from_metrics/perfs_vs_img_properties/plots/bbox_number_F1_11l.png',
                  log_scale = False, metric='F1')