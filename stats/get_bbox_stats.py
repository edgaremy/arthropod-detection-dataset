import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyze_bbox_statistics(dataset_path, split='train', output_dir='stats', plot_title='Dataset Statistics'):
    """
    Analyze and plot statistics about bounding boxes in a YOLO dataset.
    
    Args:
        dataset_path (str): Path to the YOLO dataset root directory
        split (str): Dataset split to analyze ('train', 'val', 'test', or 'all'
                     to aggregate across all available splits)
        output_dir (str): Directory to save the plots
        plot_title (str): Title for the plots
    
    Returns:
        dict: Dictionary containing computed statistics
    """
    
    labels_dir = os.path.join(dataset_path, 'labels')
    
    if split == 'all':
        # Collect label files from every subfolder of labels/
        if not os.path.exists(labels_dir):
            raise ValueError(f"Labels directory not found: {labels_dir}")
        available_splits = sorted(
            d for d in os.listdir(labels_dir)
            if os.path.isdir(os.path.join(labels_dir, d))
        )
        if not available_splits:
            raise ValueError(f"No split subdirectories found in: {labels_dir}")
        label_files = [
            os.path.join(labels_dir, s, f)
            for s in available_splits
            for f in os.listdir(os.path.join(labels_dir, s))
            if f.endswith('.txt')
        ]
        split_label = f"All ({', '.join(available_splits)})"
    else:
        labels_path = os.path.join(labels_dir, split)
        if not os.path.exists(labels_path):
            raise ValueError(f"Labels directory not found: {labels_path}")
        label_files = [
            os.path.join(labels_path, f)
            for f in os.listdir(labels_path)
            if f.endswith('.txt')
        ]
        split_label = split.capitalize()
    
    # Lists to store statistics
    bbox_counts = []  # Number of bboxes per image
    bbox_sizes = []   # Relative sizes of all bboxes
    
    # Process each label file
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Count bboxes for this image
        num_bboxes = len(lines)
        bbox_counts.append(num_bboxes)
        
        # Extract bbox sizes for this image
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center width height (normalized)
                width = float(parts[3])
                height = float(parts[4])
                bbox_size = width * height  # Relative area
                bbox_sizes.append(bbox_size)
    
    # Convert to numpy arrays for easier computation
    bbox_counts = np.array(bbox_counts)
    bbox_sizes = np.array(bbox_sizes)
    
    # Compute statistics for bbox counts
    count_stats = {
        'mean': np.mean(bbox_counts),
        'min': np.min(bbox_counts),
        'max': np.max(bbox_counts),
        'std': np.std(bbox_counts),
        'median': np.median(bbox_counts)
    }
    
    # Compute statistics for bbox sizes
    size_stats = {
        'mean': np.mean(bbox_sizes),
        'min': np.min(bbox_sizes),
        'max': np.max(bbox_sizes),
        'std': np.std(bbox_sizes),
        'median': np.median(bbox_sizes)
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up matplotlib style
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
    plt.rcParams['font.size'] = 12
    plt.rcParams['text.color'] = '0.15'
    plt.rcParams['xtick.color'] = '0.15'
    plt.rcParams['ytick.color'] = '0.15'
    plt.rcParams['axes.labelcolor'] = '0.15'
    
    # Create figure with subplots (1 row x 2 columns: density histograms only)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{plot_title}', fontsize=16, fontweight='bold')
    
    # Plot 1: Histogram of bbox counts per image
    ax1.hist(bbox_counts, bins=range(int(np.max(bbox_counts)) + 2), alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Number of bounding boxes per image')
    ax1.set_ylabel('Number of images')
    ax1.set_title('Distribution of Bounding Box Counts')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add statistics text
    stats_text1 = f'Mean: {count_stats["mean"]:.2f}\nMedian: {count_stats["median"]:.2f}\nStd: {count_stats["std"]:.2f}\nMin: {count_stats["min"]}\nMax: {count_stats["max"]}'
    ax1.text(0.98, 0.98, stats_text1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Histogram of bbox relative sizes
    ax2.hist(bbox_sizes, bins=50, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Bounding box relative size (width × height)')
    ax2.set_ylabel('Number of bounding boxes')
    ax2.set_title('Distribution of Bounding Box Sizes')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add statistics text
    stats_text2 = f'Mean: {size_stats["mean"]:.4f}\nMedian: {size_stats["median"]:.4f}\nStd: {size_stats["std"]:.4f}\nMin: {size_stats["min"]:.4f}\nMax: {size_stats["max"]:.4f}'
    ax2.text(0.98, 0.98, stats_text2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot with dataset name to avoid conflicts
    # Extract dataset name from plot_title for unique filenames
    dataset_name_clean = plot_title.replace('Dataset', '').replace('Generalization - ', '').replace(' - ', '_').replace(' ', '_').strip('_').lower()
    # Use the split_label for the filename (strip parentheses/commas for 'all')
    split_slug = split_label.lower().replace('(', '').replace(')', '').replace(', ', '_').replace(' ', '_')
    if dataset_name_clean:
        output_filename = f'bbox_statistics_{dataset_name_clean}_{split_slug}.png'
    else:
        output_filename = f'bbox_statistics_{split_slug}.png'
    
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\n=== {plot_title} - {split_label} Split Statistics ===")
    print(f"Total images analyzed: {len(bbox_counts)}")
    print(f"Total bounding boxes: {len(bbox_sizes)}")
    print("\nBounding Box Counts per Image:")
    print(f"  Mean: {count_stats['mean']:.2f}")
    print(f"  Median: {count_stats['median']:.2f}")
    print(f"  Std Dev: {count_stats['std']:.2f}")
    print(f"  Min: {count_stats['min']}")
    print(f"  Max: {count_stats['max']}")
    
    print("\nBounding Box Relative Sizes:")
    print(f"  Mean: {size_stats['mean']:.4f}")
    print(f"  Median: {size_stats['median']:.4f}")
    print(f"  Std Dev: {size_stats['std']:.4f}")
    print(f"  Min: {size_stats['min']:.4f}")
    print(f"  Max: {size_stats['max']:.4f}")
    
    print(f"\nPlot saved to: {output_path}")
    
    # Return comprehensive statistics
    return {
        'total_images': len(bbox_counts),
        'total_bboxes': len(bbox_sizes),
        'bbox_counts': bbox_counts.tolist(),
        'bbox_sizes': bbox_sizes.tolist(),
        'count_statistics': count_stats,
        'size_statistics': size_stats,
        'output_path': output_path
    }

def compare_dataset_splits(dataset_path, splits=['train', 'val', 'test'], output_dir='stats', dataset_name='Dataset'):
    """
    Compare statistics across different dataset splits.
    
    Args:
        dataset_path (str): Path to the YOLO dataset root directory
        splits (list): List of splits to compare
        output_dir (str): Directory to save the plots
        dataset_name (str): Name of the dataset for plot titles
    """
    
    all_stats = {}
    
    # Analyze each split
    for split in splits:
        try:
            stats = analyze_bbox_statistics(dataset_path, split, output_dir, f"{dataset_name}")
            all_stats[split] = stats
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    if not all_stats:
        print("No valid splits found to analyze.")
        return
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dataset_name} - Split Comparison', fontsize=16, fontweight='bold')
    
    splits_available = list(all_stats.keys())
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(splits_available)]
    
    # Plot 1: Mean bbox counts comparison
    means_counts = [all_stats[split]['count_statistics']['mean'] for split in splits_available]
    stds_counts = [all_stats[split]['count_statistics']['std'] for split in splits_available]
    
    bars1 = ax1.bar(splits_available, means_counts, yerr=stds_counts, capsize=5, 
                   color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean number of bboxes per image')
    ax1.set_title('Mean Bounding Box Counts by Split')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars1, means_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean_val:.2f}', ha='center', va='bottom')
    
    # Plot 2: Mean bbox sizes comparison
    means_sizes = [all_stats[split]['size_statistics']['mean'] for split in splits_available]
    stds_sizes = [all_stats[split]['size_statistics']['std'] for split in splits_available]
    
    bars2 = ax2.bar(splits_available, means_sizes, yerr=stds_sizes, capsize=5,
                   color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Mean relative bbox size')
    ax2.set_title('Mean Bounding Box Sizes by Split')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars2, means_sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mean_val:.4f}', ha='center', va='bottom')
    
    # Plot 3: Total images and bboxes
    total_images = [all_stats[split]['total_images'] for split in splits_available]
    total_bboxes = [all_stats[split]['total_bboxes'] for split in splits_available]
    
    x_pos = np.arange(len(splits_available))
    width = 0.35
    
    ax3.bar(x_pos - width/2, total_images, width, label='Images', color='lightblue', alpha=0.7)
    ax3.bar(x_pos + width/2, total_bboxes, width, label='Bounding Boxes', color='lightcoral', alpha=0.7)
    ax3.set_ylabel('Count')
    ax3.set_title('Total Images and Bounding Boxes by Split')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(splits_available)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Split', 'Images', 'Bboxes', 'Mean Count', 'Mean Size']
    
    for split in splits_available:
        stats = all_stats[split]
        row = [
            split.capitalize(),
            str(stats['total_images']),
            str(stats['total_bboxes']),
            f"{stats['count_statistics']['mean']:.2f}",
            f"{stats['size_statistics']['mean']:.4f}"
        ]
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_output = os.path.join(output_dir, f'{dataset_name.lower().replace(" ", "_")}_splits_comparison.png')
    plt.savefig(comparison_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to: {comparison_output}")
    
    return all_stats

# Example usage
if __name__ == "__main__":

    # Analyze main arthropod dataset - test split only
    dataset_path = "dataset"
    stats = analyze_bbox_statistics(dataset_path, split='all', output_dir='stats/ArthroNat', 
                                  plot_title='ArthroNat Dataset')
    
    # Compare all splits for main dataset
    compare_dataset_splits(dataset_path, splits=['train', 'val', 'test'], 
                         output_dir='stats/ArthroNat', dataset_name='ArthroNat')
    
    # Analyze same_species generalization dataset (using main dataset test split)
    print("\n" + "="*50)
    print("Analyzing Same Species generalization...")
    analyze_bbox_statistics(dataset_path, split='test', output_dir='stats/generalization_sets',
                            plot_title='Generalization - Same Species')
    
    # Example for other generalization datasets
    generalization_datasets = [
        'same_genus',
        'other_genus', 
        'other_families',
    ]

    for gen_dataset in generalization_datasets:
        gen_path = f"datasets(generalization)/{gen_dataset}"
        if os.path.exists(gen_path):
            print("\n" + "="*50)
            print(f"Analyzing {gen_dataset.replace('_', ' ').title()} generalization...")
            analyze_bbox_statistics(gen_path, split='train', output_dir='stats/generalization_sets',
                                    plot_title=f'Generalization - {gen_dataset.replace("_", " ").title()}')
        else:
            print(f"Warning: generalization dataset path not found: {gen_path}")
    
    # Analyze flatbug dataset
    flatbug_path = "datasets(others)/flatbug-yolo-split"
    if os.path.exists(flatbug_path):
        print("\n" + "="*50)
        print("Analyzing Flatbug dataset...")
        
        # Analyze individual split
        analyze_bbox_statistics(flatbug_path, split='all', output_dir='stats/flatbug',
                                plot_title='Flatbug Dataset')
        
        # Compare all splits for flatbug dataset
        compare_dataset_splits(flatbug_path, splits=['train', 'val', 'test'], 
                             output_dir='stats/flatbug', dataset_name='Flatbug')
    else:
        print(f"Warning: Flatbug dataset path not found: {flatbug_path}")

    # Analyse OOD-split dataset
    ood_path = "datasets(others)/OOD-split"
    if os.path.exists(ood_path):
        print("\n" + "="*50)
        print("Analyzing OOD-split dataset...")
        
        analyze_bbox_statistics(ood_path, split='all', output_dir='stats/OOD',
                                plot_title='OOD Dataset')
        compare_dataset_splits(ood_path, splits=['train', 'val', 'test'], 
                             output_dir='stats/OOD', dataset_name='OOD Dataset')
    else:
        print(f"Warning: OOD dataset path not found: {ood_path}")

    # Analyze Lepinoc dataset
    lepinoc_path= "datasets(others)/Lepinoc"
    if os.path.exists(lepinoc_path):
        print("\n" + "="*50)
        print("Analyzing Lepinoc dataset...")
        
        analyze_bbox_statistics(lepinoc_path, split='test', output_dir='stats/Lepinoc',
                                plot_title='Lepinoc Dataset')
        
    # Analyse SPIPOLL dataset
    spipoll_path = "datasets(others)/SPIPOLL"
    if os.path.exists(spipoll_path):
        print("\n" + "="*50)
        print("Analyzing SPIPOLL dataset...")
        
        analyze_bbox_statistics(spipoll_path, split='test', output_dir='stats/SPIPOLL',
                                plot_title='SPIPOLL Dataset')