# Dataset Statistics

Scripts for analyzing dataset composition and characteristics.

## Scripts

**`get_dataset_stats.py`** - Analyzes taxonomic diversity by counting unique classes, orders, and families across train/val/test splits. Generates distribution plots and summary statistics.

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/stats/order_distribution.png?raw=true" width="300">

**`get_bbox_stats.py`** - Analyzes bounding box statistics including bbox counts per image, size distributions, and aspect ratios. Produces detailed visualization plots.

## Generated Outputs

**Distribution plots:**
- `class_distribution.png` - Number of images per taxonomic class
- `family_distribution.png` - Number of images per family
- `order_distribution.png` - Number of images per order
- `split_distribution.png` - Dataset split proportions
- `taxonomic_diversity.png` - Overall taxonomic coverage

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/stats/taxonomic_diversity.png?raw=true" width="300"> <img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/stats/split_distribution.png?raw=true" width="300">

**`summary.txt`** - Text summary of all dataset statistics

## Subdirectories

**`ArthroNat/`** - Statistics for main arthropod dataset

**`flatbug/`** - Statistics for flatbug augmentation dataset

**`generalization_sets/`** - Statistics for generalization validation sets
