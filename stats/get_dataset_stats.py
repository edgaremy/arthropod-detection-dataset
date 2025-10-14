"""
Dataset Statistics Script

This script analyzes the YOLO dataset to count taxonomic diversity by:
1. Scanning all images in train/val/test directories
2. Extracting taxon_id from image filenames (first part before '_')
3. Matching taxon_ids with the hierarchy CSV
4. Counting unique classes, orders, and families represented

Usage:
    python get_dataset_stats.py [--dataset_path PATH] [--hierarchy_path PATH]
"""

import os
import pandas as pd
from pathlib import Path
from collections import Counter
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def get_image_files(dataset_images_path):
    """
    Get all image files from train/val/test directories.
    
    Args:
        dataset_images_path (str): Path to the dataset/images directory
        
    Returns:
        list: List of all image file paths
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', 
                       '*.JPG', '*.JPEG', '*.PNG', '*.TIFF', '*.TIF']
    image_files = []
    
    # Check each split directory
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_images_path, split)
        if os.path.exists(split_path):
            print(f"Scanning {split} directory...")
            for ext in image_extensions:
                pattern = os.path.join(split_path, ext)
                files = glob.glob(pattern)
                image_files.extend([(f, split) for f in files])
            print(f"  Found {len([f for f, s in image_files if s == split])} images in {split}")
    
    return image_files

def extract_taxon_ids(image_files):
    """
    Extract taxon_ids from image filenames.
    
    Args:
        image_files (list): List of (filepath, split) tuples
        
    Returns:
        pandas.DataFrame: DataFrame with columns ['taxon_id', 'filename', 'split', 'full_path']
    """
    taxon_data = []
    
    for filepath, split in image_files:
        filename = os.path.basename(filepath)
        # Extract taxon_id (first part before '_')
        taxon_id = filename.split('_')[0]
        
        # Try to convert to integer (should be numeric)
        try:
            taxon_id = int(taxon_id)
            taxon_data.append({
                'taxon_id': taxon_id,
                'filename': filename,
                'split': split,
                'full_path': filepath
            })
        except ValueError:
            print(f"Warning: Could not extract numeric taxon_id from {filename}")
    
    return pd.DataFrame(taxon_data)

def load_hierarchy(hierarchy_path):
    """
    Load the taxonomic hierarchy CSV.
    
    Args:
        hierarchy_path (str): Path to the hierarchy CSV file
        
    Returns:
        pandas.DataFrame: Hierarchy data
    """
    try:
        hierarchy = pd.read_csv(hierarchy_path)
        print(f"Loaded hierarchy with {len(hierarchy)} entries")
        print(f"Hierarchy columns: {list(hierarchy.columns)}")
        return hierarchy
    except Exception as e:
        print(f"Error loading hierarchy file: {e}")
        return None

def merge_with_hierarchy(taxon_df, hierarchy_df):
    """
    Merge taxon data with hierarchy information.
    
    Args:
        taxon_df (pandas.DataFrame): DataFrame with taxon_ids from images
        hierarchy_df (pandas.DataFrame): Hierarchy data
        
    Returns:
        pandas.DataFrame: Merged data with taxonomic information
    """
    # Merge on taxon_id
    merged = pd.merge(taxon_df, hierarchy_df, on='taxon_id', how='left')
    
    # Check for missing matches
    missing_count = merged['class'].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} taxon_ids from images not found in hierarchy")
        print("Missing taxon_ids:")
        missing_ids = merged[merged['class'].isna()]['taxon_id'].unique()
        print(f"  {list(missing_ids[:10])}..." if len(missing_ids) > 10 else f"  {list(missing_ids)}")
    
    return merged

def generate_histograms(clean_df, stats_folder='stats'):
    """
    Generate and save histograms for taxonomic distributions.
    
    Args:
        clean_df (pandas.DataFrame): Clean data with taxonomic information
        stats_folder (str): Folder to save histogram files
    """
    # Set style for better looking plots
    plt.style.use('seaborn-v0_8')
    
    # Define consistent color palette
    colors = {
        'class': '#2E86AB',      # Deep blue
        'order': '#A23B72',      # Deep magenta
        'family': '#F18F01',     # Orange
        'stats_box': '#F5F5DC',  # Beige for stats boxes
    }
    
    # Create stats folder if it doesn't exist
    os.makedirs(stats_folder, exist_ok=True)
    
    # 1. Class distribution histogram
    plt.figure(figsize=(12, 8))
    class_counts = clean_df['class'].value_counts()
    
    ax = class_counts.plot(kind='bar', color=colors['class'], edgecolor='white', linewidth=0.8, alpha=0.8)
    
    # Calculate statistics for legend
    class_mean = class_counts.mean()
    class_std = class_counts.std()
    class_min = class_counts.min()
    class_max = class_counts.max()
    
    plt.title(f'Distribution by Class\n({len(class_counts)} unique classes)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Class', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistical info as text box
    stats_text = f'Mean: {class_mean:.1f}\nStd: {class_std:.1f}\nMin: {class_min}\nMax: {class_max}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='top', horizontalalignment='right', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='gray'))
    
    # Add value labels on bars with better styling
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + max(class_counts.values) * 0.01, f'{v:,}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_folder, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Order distribution histogram (all orders)
    plt.figure(figsize=(20, 12))
    order_counts = clean_df['order'].value_counts()
    
    ax = order_counts.plot(kind='bar', color=colors['order'], edgecolor='white', linewidth=0.8, alpha=0.8)
    
    # Calculate statistics for legend
    order_mean = order_counts.mean()
    order_std = order_counts.std()
    order_min = order_counts.min()
    order_max = order_counts.max()
    
    plt.title(f'Complete Orders Distribution\n({clean_df["order"].nunique()} unique orders)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Order', fontsize=15, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistical info as text box
    stats_text = f'Mean: {order_mean:.1f}\nStd: {order_std:.1f}\nMin: {order_min}\nMax: {order_max}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=16,
             verticalalignment='top', horizontalalignment='right', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='gray'))
    
    # Add value labels on bars (only for bars with significant height to avoid clutter)
    max_val = max(order_counts.values)
    for i, v in enumerate(order_counts.values):
        if v > max_val * 0.02:  # Only show labels for bars > 2% of max value
            plt.text(i, v + max_val * 0.01, f'{v:,}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_folder, 'order_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Family distribution histogram (all families)
    plt.figure(figsize=(25, 15))
    family_counts = clean_df['family'].value_counts()
    
    ax = family_counts.plot(kind='bar', color=colors['family'], edgecolor='white', linewidth=0.5, alpha=0.8)
    
    # Calculate statistics for legend
    family_mean = family_counts.mean()
    family_std = family_counts.std()
    family_min = family_counts.min()
    family_max = family_counts.max()
    
    plt.title(f'Complete Families Distribution\n({clean_df["family"].nunique()} unique families)', 
              fontsize=20, fontweight='bold', pad=25)
    plt.xlabel('Families (ordered by frequency)', fontsize=17, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=17, fontweight='bold')
    plt.xticks([])  # Remove x-axis labels since there are too many families
    plt.yticks(fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistical info as text box
    stats_text = f'Mean: {family_mean:.1f}\nStd: {family_std:.1f}\nMin: {family_min}\nMax: {family_max}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=18,
             verticalalignment='top', horizontalalignment='right', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_folder, 'family_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Split distribution pie chart
    plt.figure(figsize=(10, 8))
    split_counts = clean_df['split'].value_counts()
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.pie(split_counts.values, labels=split_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title('Dataset Split Distribution', fontsize=16, fontweight='bold')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_folder, 'split_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Taxonomic diversity overview
    plt.figure(figsize=(12, 8))
    
    diversity_data = {
        'Classes': clean_df['class'].nunique(),
        'Orders': clean_df['order'].nunique(),
        'Families': clean_df['family'].nunique(),
        'Genus': clean_df['genus'].nunique(),
        'Species': clean_df['specie'].nunique()
    }
    
    bars = plt.bar(diversity_data.keys(), diversity_data.values(), 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], 
                   edgecolor='black', linewidth=1.5)
    
    plt.title('Taxonomic Diversity Overview', fontsize=16, fontweight='bold')
    plt.xlabel('Taxonomic Level', fontsize=14)
    plt.ylabel('Number of Unique Taxa', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, diversity_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(diversity_data.values()) * 0.01, 
                str(value), ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_folder, 'taxonomic_diversity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histograms saved to {stats_folder}/ folder:")
    print("  - class_distribution.png")
    print("  - order_distribution.png") 
    print("  - family_distribution.png")
    print("  - split_distribution.png")
    print("  - taxonomic_diversity.png")

def generate_statistics(merged_df):
    """
    Generate and print taxonomic statistics.
    
    Args:
        merged_df (pandas.DataFrame): Merged data with taxonomic information
    """
    # Remove rows with missing taxonomic info
    clean_df = merged_df.dropna(subset=['class', 'order', 'family'])
    
    # Prepare output lines for both console and file
    output_lines = []
    
    output_lines.append("="*50)
    output_lines.append("DATASET TAXONOMIC STATISTICS")
    output_lines.append("="*50)
    
    # Overall statistics
    total_images = len(merged_df)
    matched_images = len(clean_df)
    output_lines.append(f"Total images: {total_images:,}")
    output_lines.append(f"Images with taxonomic info: {matched_images:,}")
    output_lines.append(f"Coverage: {matched_images/total_images*100:.1f}%")
    
    # Unique taxonomic counts
    unique_classes = clean_df['class'].nunique()
    unique_orders = clean_df['order'].nunique()
    unique_families = clean_df['family'].nunique()
    unique_genera = clean_df['genus'].nunique()
    unique_species = clean_df['specie'].nunique()
    unique_taxon_ids = clean_df['taxon_id'].nunique()
    
    output_lines.append(f"\nTaxonomic Diversity:")
    output_lines.append(f"  Classes:   {unique_classes:,}")
    output_lines.append(f"  Orders:    {unique_orders:,}")
    output_lines.append(f"  Families:  {unique_families:,}")
    output_lines.append(f"  Genera:    {unique_genera:,}")
    output_lines.append(f"  Species:   {unique_species:,}")
    output_lines.append(f"  Taxon IDs: {unique_taxon_ids:,}")
    
    # Distribution by split
    output_lines.append(f"\nDistribution by Split:")
    split_stats = clean_df.groupby('split').agg({
        'taxon_id': 'count',
        'class': 'nunique',
        'order': 'nunique',
        'family': 'nunique'
    }).round(2)
    split_stats.columns = ['Images', 'Classes', 'Orders', 'Families']
    
    # Convert split_stats to string format
    output_lines.append("       Images  Classes  Orders  Families")
    output_lines.append("split                                   ")
    for split_name, row in split_stats.iterrows():
        output_lines.append(f"{split_name:<6} {row['Images']:>6.0f} {row['Classes']:>8.0f} {row['Orders']:>7.0f} {row['Families']:>9.0f}")
    
    # Most represented taxonomic groups
    output_lines.append(f"\nTop 10 Most Represented Classes:")
    class_counts = clean_df['class'].value_counts().head(10)
    for class_name, count in class_counts.items():
        output_lines.append(f"  {class_name}: {count:,} images")
    
    # Statistical measures for classes
    class_stats = clean_df['class'].value_counts()
    output_lines.append(f"\nClass Statistics:")
    output_lines.append(f"  Mean images per class: {class_stats.mean():.1f}")
    output_lines.append(f"  Standard deviation: {class_stats.std():.1f}")
    output_lines.append(f"  Min images per class: {class_stats.min()}")
    output_lines.append(f"  Max images per class: {class_stats.max()}")

    output_lines.append(f"\nTop 10 Most Represented Orders:")
    order_counts = clean_df['order'].value_counts().head(10)
    for order_name, count in order_counts.items():
        output_lines.append(f"  {order_name}: {count:,} images")
    
    # Statistical measures for orders
    order_stats = clean_df['order'].value_counts()
    output_lines.append(f"\nOrder Statistics:")
    output_lines.append(f"  Mean images per order: {order_stats.mean():.1f}")
    output_lines.append(f"  Standard deviation: {order_stats.std():.1f}")
    output_lines.append(f"  Min images per order: {order_stats.min()}")
    output_lines.append(f"  Max images per order: {order_stats.max()}")

    output_lines.append(f"\nTop 10 Most Represented Families:")
    family_counts = clean_df['family'].value_counts().head(10)
    for family_name, count in family_counts.items():
        output_lines.append(f"  {family_name}: {count:,} images")
    
    # Statistical measures for families
    family_stats = clean_df['family'].value_counts()
    output_lines.append(f"\nFamily Statistics:")
    output_lines.append(f"  Mean images per family: {family_stats.mean():.1f}")
    output_lines.append(f"  Standard deviation: {family_stats.std():.1f}")
    output_lines.append(f"  Min images per family: {family_stats.min()}")
    output_lines.append(f"  Max images per family: {family_stats.max()}")    # Print to console
    for line in output_lines:
        print(line)
    
    # Generate histograms
    print(f"\nGenerating histograms...")
    generate_histograms(clean_df)
    
    # Save to summary.txt file
    summary_path = 'stats/summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\nSummary saved to: {summary_path}")
    
    return clean_df

def analyze_dataset(dataset_path='./dataset/images', hierarchy_path='./src/arthro_dataset_hierarchy.csv'):
    """
    Analyze taxonomic diversity in YOLO dataset.
    
    Args:
        dataset_path (str): Path to dataset images directory
        hierarchy_path (str): Path to hierarchy CSV file
    
    Returns:
        pandas.DataFrame: Clean dataframe with taxonomic information
    """
    # Resolve paths
    dataset_path = os.path.abspath(dataset_path)
    hierarchy_path = os.path.abspath(hierarchy_path)
    
    print("Dataset Taxonomic Analysis")
    print("Dataset path: [DATASET_IMAGES_PATH]")  # Hide absolute path
    print("Hierarchy path: [HIERARCHY_CSV_PATH]")  # Hide absolute path
    print("-" * 50)
    
    # Check if paths exist
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist")
        return None
    
    if not os.path.exists(hierarchy_path):
        print(f"Error: Hierarchy file does not exist")
        return None
    
    # Step 1: Get all image files
    print("Step 1: Scanning image files...")
    image_files = get_image_files(dataset_path)
    print(f"Total images found: {len(image_files):,}")
    
    if not image_files:
        print("No image files found!")
        return None
    
    # Step 2: Extract taxon IDs
    print("\nStep 2: Extracting taxon IDs from filenames...")
    taxon_df = extract_taxon_ids(image_files)
    print(f"Successfully extracted {len(taxon_df):,} taxon IDs")
    print(f"Unique taxon IDs: {taxon_df['taxon_id'].nunique():,}")
    
    # Step 3: Load hierarchy
    print("\nStep 3: Loading taxonomic hierarchy...")
    hierarchy_df = load_hierarchy(hierarchy_path)
    if hierarchy_df is None:
        return None
    
    # Step 4: Merge with hierarchy
    print("\nStep 4: Matching with hierarchy data...")
    merged_df = merge_with_hierarchy(taxon_df, hierarchy_df)
    
    # Step 5: Generate statistics
    print("\nStep 5: Generating statistics...")
    clean_df = generate_statistics(merged_df)
    
    print("\nAnalysis complete!")
    return clean_df

if __name__ == "__main__":
    analyze_dataset()
