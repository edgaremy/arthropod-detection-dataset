from matplotlib.ticker import MaxNLocator
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import numpy as np

plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
plt.rcParams['font.size'] = 12
# slightly less black text:
ratio = '0.2'
plt.rcParams['text.color'] = ratio 
plt.rcParams['xtick.color'] = ratio
plt.rcParams['ytick.color'] = ratio
plt.rcParams['axes.labelcolor'] = ratio

def plot_bootstrap_metrics(metrics_csv: str, show_map_fitness=True, output_path=None):
    # Read the CSV into a dataframe
    df = pd.read_csv(metrics_csv)

    # Extract wave numbers from 'model_path' using the splitting logic
    limit = df['model_path'].to_list()
    wave_numbers = []
    for lim in limit:
        if lim.split('/')[0][6] == '_':
            wave_numbers.append(int(lim.split("/")[0][5]))
        else:
            wave_numbers.append(int(lim.split("/")[0][5:7]))
    
    df['wave'] = wave_numbers

    # Bootstrap function to calculate confidence intervals
    def bootstrap_ci(data, n_bootstrap=1000, ci=95):
        bootstrapped_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrapped_means.append(np.mean(sample))
        lower_bound = np.percentile(bootstrapped_means, (100 - ci) / 2)
        upper_bound = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
        return lower_bound, upper_bound

    # Calculate statistics for each wave
    grouped_df = df.groupby('wave').agg({
        'metrics/precision(B)': ['mean', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), 'min', 'max'],
        'metrics/recall(B)': ['mean', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), 'min', 'max'],
        'metrics/mAP50(B)': ['mean', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), 'min', 'max'],
        'metrics/mAP50-95(B)': ['mean', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), 'min', 'max'],
        'fitness': ['mean', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), 'min', 'max'],
        'mean_IoU': ['mean', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), 'min', 'max'],
        'F1-score': ['mean', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), 'min', 'max']
    })

    # Rename columns for easier access
    grouped_df.columns = grouped_df.columns.map(lambda x: f"{x[0]}_{x[1]}" if isinstance(x, tuple) else x)
    wave_numbers = grouped_df.index

    # Create a 2x3 grid for subplots
    subplot_columns = 3 if show_map_fitness else 2
    plot_width = 24 if show_map_fitness else 16
    fig, axs = plt.subplots(2, subplot_columns, figsize=(plot_width, 10))

    ymin, ymax = 0.5, 1.0
    xmin, xmax = min(wave_numbers) - 0.5, max(wave_numbers) + 0.5

    # Define the metrics to plot and their positions
    metrics = [
        ('F1-score', 'F1-score', (0, 0)),
        ('Precision', 'metrics/precision(B)', (0, 1)),
        ('Recall', 'metrics/recall(B)', (1, 0)),
        ('Mean IoU', 'mean_IoU', (1, 1))        
    ]

    if show_map_fitness:
        metrics.extend([
            ('mAP', ['metrics/mAP50(B)', 'metrics/mAP50-95(B)'], (0, 2)),
            ('Fitness', 'fitness', (1, 2))
        ])

    better_labels = {
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall',
        'metrics/mAP50(B)': 'mAP@50',
        'metrics/mAP50-95(B)': 'mAP@50-95',
        'fitness': 'Fitness',
        'mean_IoU': 'Average IoU',
        'F1-score': 'F1-score'
    }

    for title, columns, position in metrics:
        ax = axs[position[0], position[1]]

        color = "#536fc3"
        color_background = "#956fec"


        if isinstance(columns, list):  # Handle multiple curves (e.g., mAP)
            for col in columns:
                # Get better label for the metric
                metric_label = better_labels.get(col, col.split("/")[-1])
                
                # Plot the mean and median
                ax.plot(wave_numbers, grouped_df[f'{col}_mean'], label=f'Mean {metric_label}', marker='o', markersize=9, linestyle='-', color=color)

                # Add bootstrap confidence intervals
                ci_lower, ci_upper = zip(*[bootstrap_ci(df[df['wave'] == wave][col]) for wave in wave_numbers])
                ax.fill_between(wave_numbers, ci_lower, ci_upper, alpha=0.3, label='95% CI', color=color_background)
        else:
            # Get better label for the metric
            metric_label = better_labels.get(columns, columns.split("/")[-1])
            
            # Plot the mean and median
            ax.plot(wave_numbers, grouped_df[f'{columns}_mean'], label=f'Mean {metric_label}', marker='o', markersize=9, linestyle='-', color=color)

            # Add bootstrap confidence intervals
            ci_lower, ci_upper = zip(*[bootstrap_ci(df[df['wave'] == wave][columns]) for wave in wave_numbers])
            ax.fill_between(wave_numbers, ci_lower, ci_upper, alpha=0.3, label='95% CI', color=color_background)

        ax.tick_params(which='both', direction='in', length=0) # remove small ticks next to the numbers
        ax.tick_params(axis='both', which='major', pad=8) # number further from the axis

        # Change x axis tick labels
        n_img_wave = 14500 / 20
        x_labels = [n_img_wave * i for i in range(1, len(wave_numbers) + 1)]
        ticks_labels = [700, 5000, 10000, 14500]
        max_label = 14500
        ticks = [label / max_label * max(wave_numbers) for label in ticks_labels]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{label/1000:.1f}' for label in ticks_labels])

        ax.set_facecolor('#f0f0f0') # grey background
        ax.set_xlabel('Image quantity (in thousands)')
        # ax.set_ylabel(title)
        ax.set_title(f'{title}')
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        ax.legend()

    plt.tight_layout()
    
    # Print statistics for the last wave (highest training data quantity)
    last_wave = max(wave_numbers)
    last_wave_data = df[df['wave'] == last_wave]
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (Wave {last_wave} - Maximum Training Data)")
    print(f"{'='*60}")
    
    # Define all metrics to report
    all_metrics = [
        ('F1-score', 'F1-score'),
        ('Precision', 'metrics/precision(B)'),
        ('Recall', 'metrics/recall(B)'),
        ('Mean IoU', 'mean_IoU'),
        ('mAP@50', 'metrics/mAP50(B)'),
        ('mAP@50-95', 'metrics/mAP50-95(B)'),
        ('Fitness', 'fitness')
    ]
    
    for metric_name, metric_col in all_metrics:
        if metric_col in last_wave_data.columns:
            mean_val = last_wave_data[metric_col].mean()
            best_val = last_wave_data[metric_col].max()
            std_val = last_wave_data[metric_col].std()
            print(f"{metric_name:12} | Mean: {mean_val:.4f} | Best: {best_val:.4f} | Std: {std_val:.4f}")
    
    print(f"{'='*60}")
    print(f"Sample size: {len(last_wave_data)} models")
    print(f"{'='*60}\n")
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

# Example usage
plot_bootstrap_metrics('validation/cross-validation/cross-validation_metrics.csv',
                       show_map_fitness=False,
                       output_path='validation/cross-validation/cross_validation_results.png')