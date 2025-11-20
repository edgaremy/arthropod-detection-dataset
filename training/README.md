# Training Scripts

Scripts for training YOLO models with various configurations.

## Prerequisites

1. [Set up Python environment](../README.md#set-up-python)
2. [Download the dataset](../README.md#download-the-dataset)

## Training Scripts

**`train_default.py`** - Standard YOLO11 training with default augmentation parameters.

**`train_with_flatbug.py`** - Train on ArthroNat dataset augmented with flatbug detection data.

**`train_flatbugonly.py`** - Train exclusively on flatbug dataset.

**`train_with_flatbug_nano.py`** - Train YOLO11n (nano) model with flatbug augmentation.

**`train_without_mosaic.py`** - Train with mosaic augmentation disabled.

**`train_with_mosaic33.py`** - Train with mosaic scale parameter = 0.33.

**`train_with_mosaic33_nano.py`** - Train YOLO11n with mosaic scale = 0.33.

**`train_with_mosaic44.py`** - Train with mosaic scale parameter = 0.44.

**`train_with_mosaic66.py`** - Train with mosaic scale parameter = 0.66.

**`train_with_flatbug_mosaic16.py`** - Train with flatbug data and mosaic scale = 0.16.

## Utilities

**`export_models.py`** - Export trained PyTorch models to ONNX format with proper NMS post-processing.

**`Mosaic16.py`**, **`Mosaic36.py`** - Custom mosaic augmentation implementations for resp. 4x4 and 6x6 mosaicing (3x3 is directly implemented inside dedicated training scripts).

## Usage Example

```bash
# Train YOLO11l with flatbug augmentation
python training/train_with_flatbug.py

# Train YOLO11n (faster, smaller model)
python training/train_with_flatbug_nano.py

# Export trained model to ONNX
python training/export_models.py
```

#
# 🐞 🐜 🦋 🦗 🐝 🕷️ 🐛 🪰 🪲