# Training Scripts

Scripts for training YOLO models with various configurations.

## Prerequisites

1. [Set up Python environment](../README.md#set-up-python)
2. [Download the dataset](../README.md#download-the-dataset)

## Training Scripts

**`train_default.py`** - Standard YOLO11 training with default augmentation parameters.

**`train_with_flatbug.py`** - Train on ArthroNat dataset augmented with flatbug detection data.

**`train_flatbugonly.py`** - Train exclusively on flatbug dataset.

**`train_ALL_DATASETS.py`** - Train on the combined ALL_DATASETS configuration.

**`train_with_flatbug_nano.py`** - Train YOLO11n (nano) model with flatbug augmentation.

**`train_without_mosaic.py`** - Train with mosaic augmentation disabled.

**`train_with_mosaic33.py`** - Train with custom 3x3 mosaic augmentation enabled.

**`train_with_mosaic33_nano.py`** - Train YOLO11n with custom 3x3 mosaic augmentation enabled.

**`train_with_mosaic44.py`** - Train with custom 4x4 mosaic augmentation enabled.

**`train_with_mosaic66.py`** - Train with custom 6x6 mosaic augmentation enabled.

**`train_with_flatbug_mosaic16.py`** - Train with flatbug data and custom 4x4 mosaic augmentation (Mosaic16) enabled.

## Fine-Tuning

The [`fine-tuning/`](fine-tuning/) subdirectory contains the scripts used for OOD and Lepinoc subset fine-tuning experiments. See [its README](fine-tuning/README.md) for the matrix launchers, wrappers, and dataset-specific commands.

Main entry points:

**`fine-tuning/finetune_on_OOD_subset_case.py`** - Run a single OOD subset fine-tuning case.

**`fine-tuning/finetune_on_Lepinoc_subset_case.py`** - Run a single Lepinoc subset fine-tuning case.

**`fine-tuning/run_matrix_ood_folds.sh`** - Run the OOD subset matrix over sizes and folds.

**`fine-tuning/run_matrix_lepinoc_folds.sh`** - Run the Lepinoc subset matrix over sizes and folds.

## Utilities

**`export_models.py`** - Export trained PyTorch models to ONNX format with proper NMS post-processing.

**`Mosaic16.py`**, **`Mosaic36.py`** - Custom mosaic augmentation implementations for 4x4 (16-image) and 6x6 (36-image) mosaicing. The 3x3 variant is implemented directly inside the dedicated training scripts.

## Usage Example

```bash
# Train YOLO11l with flatbug augmentation
python training/train_with_flatbug.py

# Train on the combined ALL_DATASETS configuration
python training/train_ALL_DATASETS.py

# Train YOLO11n (faster, smaller model)
python training/train_with_flatbug_nano.py

# Run a fine-tuning subset case
python training/fine-tuning/finetune_on_OOD_subset_case.py --mode transfer --size 500 --fold 2

# Export trained model to ONNX
python training/export_models.py
```

#
# 🐞 🐜 🦋 🦗 🐝 🕷️ 🐛 🪰 🪲