# Validation Scripts

Scripts for evaluating model performance and generating analysis plots.

## Prerequisites

1. [Set up Python environment](../README.md#set-up-python)
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

Compare model performance according to bbox size and bbox count:

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/metrics/plots/compare_bbox_size_F1.png?raw=true" width="300"> <img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/metrics/plots/compare_bbox_nb_F1.png?raw=true" width="300">

Run the R plotting scripts:
```bash
Rscript validation/metrics/compare_bbox_size_perfs.r
Rscript validation/metrics/compare_bbox_nb_perfs.r
```

### Hierarchical Performance Analysis

Analyze performance across taxonomic hierarchies (class, order, family) in the [`hierarchical_metrics`](hierarchical_metrics) folder:

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/plot_from_metrics/hierarchical_perfs/plots/class_perfs_yolo11l.png?raw=true" width="300"> <img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/plot_from_metrics/hierarchical_perfs/plots/order_perfs_yolo11l.png?raw=true" width="300">

**R Script (recommended)** - Correlation analysis between training data quantity and performance:
```bash
Rscript validation/hierarchical_metrics/get_correlation_nb_perfs.r
```





### Generalization Performance

Evaluate model generalization on external datasets in the [`plot_from_metrics/generalization/plots`](plot_from_metrics/generalization/plots) folder:

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/plot_from_metrics/generalization/plots/generalization_11l.png?raw=true" width="300">

Run the plotting script:
```bash
python validation/plot_from_metrics/generalization/plot_generalization_metrics
```

## Subdirectories

**`metrics/`** - Pre-computed performance metrics CSV files and R plotting scripts for model comparison.

**`plot_from_metrics/`** - Python plotting scripts for hierarchical performance, image properties, and generalization.

**`hierarchical_metrics/`** - R and Python scripts for analyzing performance across taxonomic hierarchies (class, order, family). Includes correlation analysis between training data quantity and performance.

**`generalization/`** - Validation on external datasets to test generalization capabilities.

**`cross-validation/`** - K-fold cross-validation scripts and results.

## Download Additional Validation Datasets

External validation datasets for generalization testing can be downloaded separately (see `generalization/` subdirectory).

#
# 🐞 🐜 🦋 🦗 🐝 🕷️ 🐛 🪰 🪲