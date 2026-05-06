# Validation Scripts

Scripts for evaluating model performance and generating analysis plots.

## Prerequisites

1. [Set up Python environment](../README.md#set-up-python)
2. [Set up R](../README.md#set-up-r)
2. [Download the dataset](../README.md#download-the-dataset)

## Overview

Validation is done in two steps for efficiency:
1. **Generate metrics** - Run model inference and compute metrics, save to CSV files (time-intensive)
2. **Plot metrics** - Visualize pre-computed metrics from CSV files (fast, for experimentation)

## Main Scripts

**`get_validation_metrics.py`** - Runs inference on validation sets and computes performance metrics (mAP, precision, recall, F1).

## 1. Generate Metrics (Optional)

Pre-computed CSV files are included in `metrics/`. Skip this step unless regenerating metrics.

Run inference and compute metrics:
```bash
python validation/get_validation_metrics.py
```


## 2. Plot Metrics Using CSV

### Model Performance Comparison

Compare different models on test datasets using plots in the [`metrics/plots`](metrics/plots) folder:

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/metrics/plots/arthro_f1_comparison.png?raw=true" width="300"> <img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/metrics/plots/flatbug_f1_comparison.png?raw=true" width="300">

Run the R plotting script:
```bash
Rscript validation/metrics/compare_average_perfs.r
```

### Performance vs Bounding Box Properties

Compare model performance according to bbox size and bbox count in the [`bbox_properties/plots`](bbox_properties/plots) folder:

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/bbox_properties/plots/perf_vs_bbox_size_F1.png?raw=true" width="300"> <img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/bbox_properties/plots/perf_vs_bbox_number_F1.png?raw=true" width="300">

Run the plotting scripts:
```bash
Rscript validation/bbox_properties/perf_vs_bbox_size.r
Rscript validation/bbox_properties/perf_vs_bbox_number.r
```

### Hierarchical Performance Analysis

Analyze performance across taxonomic hierarchies (class, order, family) in the [`hierarchical_metrics`](hierarchical_metrics) folder:

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/plot_from_metrics/hierarchical_perfs/plots/class_perfs_yolo11l.png?raw=true" width="300"> <img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/plot_from_metrics/hierarchical_perfs/plots/order_perfs_yolo11l.png?raw=true" width="300">

R Script for Correlation analysis between training data quantity and performance:
```bash
Rscript validation/hierarchical_metrics/get_correlation_nb_perfs.r
```


### Taxon Generalization Performance

Evaluate model generalization on external datasets (new taxa) in the [`plot_from_metrics/generalization/plots`](plot_from_metrics/generalization/plots) folder:

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/plot_from_metrics/generalization/plots/generalization_11l.png?raw=true" width="300">

Run the plotting script:
```bash
python validation/plot_from_metrics/generalization/plot_generalization_metrics.py
```

### Fine-tuning Comparison

Compare fine-tuning results on external validation sets in the [`fine-tuning/plots`](fine-tuning/plots) folder:

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/fine-tuning/plots/OOD_f1_mean_iou_sidebyside.png?raw=true" width="500">

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/fine-tuning/plots/Lepinoc_f1_mean_iou_sidebyside.png?raw=true" width="500">

Run the plotting script:
```bash
python validation/fine-tuning/plot_finetuning.py
```

### Cross-validation Results

Inspect k-fold validation outputs in the [`cross-validation/`](cross-validation/) folder:

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/cross-validation/cross_validation_results.png?raw=true" width="300">

Run the plotting script:
```bash
python validation/cross-validation/plot-cross-validation.py
```

### Other Generalization Dataset Metrics

Compare validation metrics across additional external generalization datasets in [`other_datasets_metrics/`](other_datasets_metrics/). This folder currently stores CSV outputs and a LaTeX table generator rather than plots.

## Subdirectories

**`metrics/`** - Pre-computed performance metrics CSV files and R plotting scripts for model comparison.

**`plot_from_metrics/`** - Python plotting scripts for hierarchical performance, image properties, and generalization.

**`bbox_properties/`** - Scripts and plots for performance versus bounding box size and bbox count.

**`hierarchical_metrics/`** - R and Python scripts for analyzing performance across taxonomic hierarchies (class, order, family). Includes correlation analysis between training data quantity and performance.

**`generalization/`** - Validation on external datasets to test generalization capabilities to **new taxa**.

**`fine-tuning/`** - Fine-tuning validation metrics and plots for external test sets.

**`cross-validation/`** - K-fold cross-validation scripts and results.

**`other_datasets_metrics/`** - CSV summaries and comparison scripts for generalization datasets from the literature.

**`plot_from_metrics/hierarchical_perfs/`** - Python scripts and plots for hierarchical performance summaries.

**`plot_from_metrics/perfs_vs_img_properties/`** - Python scripts and plots for performance versus image properties.

**`plot_from_metrics/generalization/`** - Python scripts and plots for generalization across external datasets.

## Download Additional Validation Datasets

External validation datasets for new taxa generalization testing can be downloaded separately (see `generalization/` subdirectory).

#
# 🐞 🐜 🦋 🦗 🐝 🕷️ 🐛 🪰 🪲